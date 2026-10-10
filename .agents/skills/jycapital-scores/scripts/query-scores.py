"""Read BigQuant competition scores without exposing authentication or user profiles."""

import argparse
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
import json
import math
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request


ORIGIN = "https://bigquant.com"
LOGIN_URL = ORIGIN + "/bigapis/auth/v1/auth/login"
SUBMISSIONS_URL = ORIGIN + "/bigapis/alphathon/v1/submissions"
DEFAULT_COMPETITION = "45b41c7a-0ac8-42e9-9fe3-0c33dcfb9bca"
METRIC_FIELDS = {
    "ic_ir", "ic_mean", "turnover", "sharpe_ratio", "stress_stability",
    "rank_ic_mean", "rank_ic_ir", "rank_ic", "ic", "rank", "score",
}


class ScoreError(Exception):
    """An error whose text is safe to display without the original response."""


class AuthenticationError(ScoreError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def load_auth(args):
    """Select one credential source; never mix accounts from different sources."""
    if args.auth_stdin:
        try:
            auth = json.load(sys.stdin)
        except (ValueError, OSError):
            raise ScoreError("Authentication stdin must contain valid JSON.") from None
    else:
        auth_file = args.auth_file or os.environ.get("BIGQUANT_AUTH_FILE")
        env_auth = {
            "cookie": os.environ.get("BIGQUANT_COOKIE", ""),
            "username": os.environ.get("BIGQUANT_USERNAME", ""),
            "password": os.environ.get("BIGQUANT_PASSWORD", ""),
        }
        if not auth_file and any(env_auth.values()):
            auth = env_auth
        else:
            if not auth_file:
                for parent in (Path.cwd(), *Path.cwd().parents):
                    candidate = parent / ".bigquant" / "platform-auth.json"
                    if candidate.is_file():
                        auth_file = candidate
                        break
            if not auth_file:
                raise ScoreError(
                    "No authentication configured. Set BIGQUANT_COOKIE or "
                    "BIGQUANT_USERNAME/BIGQUANT_PASSWORD, or use --auth-file/--auth-stdin."
                )
            try:
                auth = json.loads(Path(auth_file).read_text(encoding="utf-8-sig"))
            except (ValueError, OSError):
                raise ScoreError("Cannot read authentication JSON file.") from None
    if not isinstance(auth, dict):
        raise ScoreError("Authentication JSON must be an object.")
    selected = {key: auth.get(key, "") for key in ("cookie", "username", "password")}
    if any(not isinstance(value, str) for value in selected.values()):
        raise ScoreError("Authentication fields must be strings.")
    return selected


def jwt_cookie(header):
    try:
        cookies = SimpleCookie()
        cookies.load(header)
        token = cookies["bigjwt"].value if "bigjwt" in cookies else ""
    except Exception:
        raise ScoreError("Invalid Cookie format.") from None
    if token and any(char.isspace() or char in ";\r\n" for char in token):
        raise ScoreError("Invalid bigjwt Cookie.")
    return "bigjwt=" + token if token else ""


class BigQuantClient:
    def __init__(self, auth, timeout=30, opener=None):
        self.cookie = jwt_cookie(auth.get("cookie", ""))
        self.username = auth.get("username", "")
        self.password = auth.get("password", "")
        self.timeout = timeout
        self.opener = opener or urllib.request.build_opener(NoRedirect())
        self.login_attempted = False

    def _request(self, url, payload=None):
        headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
        if payload is not None:
            if url != LOGIN_URL:
                raise ScoreError("Only the fixed login endpoint accepts POST requests.")
            headers.update({"Content-Type": "application/json;charset=UTF-8", "Origin": ORIGIN})
        elif not url.startswith(SUBMISSIONS_URL + "?"):
            raise ScoreError("Only the fixed submissions endpoint accepts authenticated GET requests.")
        if self.cookie and payload is None:
            headers["Cookie"] = self.cookie
        request = urllib.request.Request(
            url, headers=headers,
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            method="POST" if payload is not None else "GET",
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                # Never follow a login page, external redirect, or downgrade to HTTP.
                if response.geturl() != url:
                    raise ScoreError("Unexpected response URL; refusing authentication redirect.")
                raw = response.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    raise ScoreError("API response is too large; use a smaller --page-size.")
                result = json.loads(raw)
                response_headers = response.headers
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise AuthenticationError(f"Authentication rejected (HTTP {exc.code}).") from None
            raise ScoreError(f"API request failed (HTTP {exc.code}); no automatic retry.") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ScoreError("API network request failed or timed out; no automatic retry.") from None
        except (ValueError, UnicodeError):
            raise ScoreError("API did not return valid JSON.") from None
        if not isinstance(result, dict):
            raise ScoreError("Unexpected API response structure.")
        code = result.get("code")
        if str(code) in ("401", "403"):
            raise AuthenticationError("API reports expired or rejected authentication.")
        if type(code) is not int or code not in (0, 200):
            raise ScoreError("API reports a business error or unknown status; response omitted.")
        return result.get("data"), response_headers

    def login(self):
        if self.login_attempted:
            raise AuthenticationError("Authentication failed after one login attempt.")
        if not self.username or not self.password:
            raise AuthenticationError("Cookie is missing/invalid and no username/password is configured.")
        self.login_attempted = True
        data, headers = self._request(LOGIN_URL, {
            "payload": {"username": self.username, "password": self.password},
            "auth_name": "UserPass",
        })
        cookie = ""
        for value in headers.get_all("Set-Cookie", []):
            cookie = jwt_cookie(value) or cookie
        if not cookie and isinstance(data, dict) and isinstance(data.get("token"), str):
            cookie = jwt_cookie("bigjwt=" + data["token"])
        if not cookie:
            raise AuthenticationError("Login did not return a usable bigjwt Cookie or token.")
        self.cookie = cookie

    def get_page(self, competition_id, page, size):
        if not self.cookie:
            self.login()
        url = SUBMISSIONS_URL + "?" + urllib.parse.urlencode({
            "order_by": "-created_at", "competition_id": competition_id,
            "page": page, "size": size, "constraints": "{}",
        })
        try:
            data, _ = self._request(url)
        except AuthenticationError:
            self.login()
            data, _ = self._request(url)
        return data


def finite_number(value):
    if isinstance(value, bool) or value is None or not isinstance(value, (str, int, float)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def score_summary(value):
    number = finite_number(value)
    missing = value is None or (isinstance(value, str) and not value.strip())
    state = "available" if number is not None else "missing" if missing else "non_numeric"
    # Keep finite numeric values and string representations; never serialize NaN/Infinity.
    raw = value if isinstance(value, str) or number is not None else None
    return {"raw": raw, "value": number, "state": state}


def project_submission(item):
    result = {key: item[key] for key in ("id", "competition_id", "created_at", "updated_at")
              if isinstance(item.get(key), str)}
    if isinstance(item.get("status"), str):
        result["status"] = item["status"]
    if isinstance(item.get("selected_for_private"), bool):
        result["selected_for_private"] = item["selected_for_private"]
    for key in ("rank", "public_rank", "private_rank"):
        if finite_number(item.get(key)) is not None:
            result[key] = item[key]
    file_map = item.get("data", {}).get("files", {}) if isinstance(item.get("data"), dict) else {}
    result["files"] = []
    if isinstance(file_map, dict):
        for value in file_map.values():
            if isinstance(value, dict) and isinstance(value.get("name"), str):
                entry = {"name": value["name"]}
                if type(value.get("size")) is int and value["size"] >= 0:
                    entry["size"] = value["size"]
                result["files"].append(entry)
    for stage in ("public", "private"):
        result[stage + "_score"] = score_summary(item.get(stage + "_score"))
        metrics = item.get(stage + "_score_data")
        if metrics is None:
            metrics = {}
        if not isinstance(metrics, dict):
            raise ScoreError("Unexpected score metrics structure; refusing to discard it silently.")
        result[stage + "_score_data"] = {
            key: value for key, value in metrics.items()
            if key in METRIC_FIELDS and (value is None or finite_number(value) is not None)
        }
        result[stage + "_unrecognized_metric_fields"] = sorted(set(metrics) - METRIC_FIELDS)
        result[stage + "_invalid_metric_fields"] = sorted(
            key for key, value in metrics.items()
            if key in METRIC_FIELDS and value is not None and finite_number(value) is None
        )
    return result


def query_scores(client, competition_id=DEFAULT_COMPETITION, page_size=100, max_pages=100):
    items = []
    seen = set()
    total_pages = None
    for page in range(1, max_pages + 1):
        data = client.get_page(competition_id, page, page_size)
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            raise ScoreError("Unknown submissions response structure; expected data.items.")
        page_total = data.get("total")
        batch = data["items"]
        page_count = data.get("count")
        if (type(page_total) is not int or page_total < 0
                or type(data.get("page")) is not int or data["page"] != page
                or type(page_count) is not int or page_count != len(batch)):
            raise ScoreError("Invalid pagination metadata.")
        if total_pages is not None and total_pages != page_total:
            raise ScoreError("Page count changed during pagination; query again for a consistent list.")
        total_pages = page_total
        # BigQuant's total is the number of pages; count is this page's row count.
        if ((total_pages == 0 and batch)
                or (total_pages > 0 and (page > total_pages or not batch))):
            raise ScoreError("Inconsistent pagination; refusing a partial list.")
        for item in batch:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
                raise ScoreError("Submission is missing its platform ID.")
            if item["id"] in seen:
                raise ScoreError("Duplicate submission/page detected; refusing a partial list.")
            if item.get("competition_id") != competition_id:
                raise ScoreError("Submission belongs to an unexpected competition.")
            seen.add(item["id"])
            items.append(project_submission(item))
        if page == total_pages or total_pages == 0:
            return {
                "source": "bigquant_submissions_api", "endpoint": SUBMISSIONS_URL,
                "competition_id": competition_id,
                "queried_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
                "pages_fetched": page, "total_pages": total_pages,
                "total": len(items), "returned": len(items), "submissions": items,
            }
    raise ScoreError("Reached --max-pages before retrieving all submissions; no partial results saved.")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-id", default=DEFAULT_COMPETITION)
    parser.add_argument("--submission-id")
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument("--auth-file", type=Path)
    auth.add_argument("--auth-stdin", action="store_true")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not 1 <= args.page_size <= 1000 or args.max_pages < 1 or not 0 < args.timeout <= 120:
        parser.error("page-size must be 1..1000, max-pages positive, timeout in (0, 120].")
    try:
        client = BigQuantClient(load_auth(args), timeout=args.timeout)
        result = query_scores(client, args.competition_id, args.page_size, args.max_pages)
        if args.submission_id:
            result["submissions"] = [item for item in result["submissions"] if item["id"] == args.submission_id]
            result["returned"] = len(result["submissions"])
            if not result["returned"]:
                raise ScoreError("Requested submission ID is not present in the complete response.")
        output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
            print(json.dumps({"saved": True, "total": result["total"], "returned": result["returned"]}))
        else:
            print(output, end="")
        return 0
    except ScoreError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    except OSError:
        print('{"error": "Cannot write score output file."}', file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
