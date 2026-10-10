"""API contract, authentication boundaries, and complete score reporting."""

from email.message import Message
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.error


SCRIPT = Path(__file__).resolve().parents[1] / ".agents/skills/jycapital-scores/scripts/query-scores.py"
SPEC = importlib.util.spec_from_file_location("platform_scores", SCRIPT)
scores = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scores)


def submission(identifier, public="0.75", private=None):
    return {
        "id": identifier, "competition_id": scores.DEFAULT_COMPETITION,
        "public_score": public, "private_score": private,
        "public_score_data": {"ic_mean": 0.08, "turnover": 0.2}, "private_score_data": {},
        "selected_for_private": False,
        "user_id": "private-user", "data": {
            "description": "private text", "files": {"opaque-not-a-hash": {"name": "v1.ipynb", "size": 100}},
        },
    }


class FakeResponse:
    def __init__(self, request, payload, cookie=None):
        self.url = request.full_url
        self.body = json.dumps(payload).encode()
        self.headers = Message()
        if cookie:
            self.headers.add_header("Set-Cookie", cookie)

    def geturl(self):
        return self.url

    def read(self, limit):
        return self.body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeOpener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        response = next(self.responses)
        if isinstance(response, int):
            raise urllib.error.HTTPError(request.full_url, response, "secret response", Message(), io.BytesIO())
        if isinstance(response, tuple):
            return FakeResponse(request, *response)
        return FakeResponse(request, response)


def page(items, number=1, total=None):
    return {
        "page": number, "size": 100, "count": len(items),
        "total": (1 if items else 0) if total is None else total, "items": items,
    }


class PlatformScoresTests(unittest.TestCase):
    def test_zero_is_a_score_and_missing_is_not_zero(self):
        result = scores.project_submission(submission("one", "0", None))
        self.assertEqual(result["public_score"], {"raw": "0", "value": 0.0, "state": "available"})
        self.assertEqual(result["private_score"]["state"], "missing")
        self.assertIsNone(result["private_score"]["value"])
        self.assertEqual(scores.score_summary("NaN")["state"], "non_numeric")
        self.assertEqual(scores.score_summary("   ")["state"], "missing")

    def test_projection_excludes_profiles_descriptions_and_internal_ids(self):
        result = scores.project_submission(submission("one"))
        rendered = json.dumps(result)
        for private in ("private-user", "private text", "opaque-not-a-hash"):
            self.assertNotIn(private, rendered)
        self.assertEqual(result["files"], [{"name": "v1.ipynb", "size": 100}])
        self.assertNotIn("rank", result)
        self.assertNotIn("status", result)

    def test_public_private_and_unknown_metrics_stay_separate(self):
        item = submission("one", "0.1", "0.2")
        item["private_score_data"] = {"ic_mean": 0.3, "new_metric": 100, "turnover": "NaN"}
        result = scores.project_submission(item)
        self.assertEqual(result["public_score_data"]["ic_mean"], 0.08)
        self.assertEqual(result["private_score_data"], {"ic_mean": 0.3})
        self.assertEqual(result["private_unrecognized_metric_fields"], ["new_metric"])
        self.assertEqual(result["private_invalid_metric_fields"], ["turnover"])

    def test_total_drives_pagination_when_server_caps_page_size(self):
        client = unittest.mock.Mock()
        client.get_page.side_effect = [page([submission("one")], total=2), page([submission("two")], 2, 2)]
        result = scores.query_scores(client, page_size=100)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["pages_fetched"], 2)
        self.assertEqual(client.get_page.call_count, 2)

    def test_total_is_pages_and_count_is_current_page_rows(self):
        client = unittest.mock.Mock()
        client.get_page.return_value = page([submission("one"), submission("two")])
        result = scores.query_scores(client)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["total_pages"], 1)
        self.assertEqual(result["pages_fetched"], 1)
        self.assertEqual(client.get_page.call_count, 1)

    def test_inconsistent_pagination_does_not_return_partial_success(self):
        variants = [
            [page([submission("one")], total=2), page([submission("one")], 2, 2)],
            [page([submission("one")], total=2), page([], 2, 2)],
            [page([submission("one")], total=2), page([submission("two")], 2, 3)],
        ]
        for responses in variants:
            with self.subTest(responses=responses):
                client = unittest.mock.Mock()
                client.get_page.side_effect = responses
                with self.assertRaises(scores.ScoreError):
                    scores.query_scores(client)

    def test_page_limit_is_an_error(self):
        client = unittest.mock.Mock()
        client.get_page.return_value = page([submission("one")], total=2)
        with self.assertRaises(scores.ScoreError):
            scores.query_scores(client, max_pages=1)

    def test_cross_competition_response_is_rejected(self):
        item = submission("one")
        item["competition_id"] = "another-competition"
        client = unittest.mock.Mock()
        client.get_page.return_value = page([item])
        with self.assertRaises(scores.ScoreError):
            scores.query_scores(client)

    def test_cookie_expiry_triggers_one_login_and_retries_with_new_cookie(self):
        opener = FakeOpener([
            401,
            ({"code": 0, "data": {"token": "body-token", "user": {"phone": "secret"}}},
             "bigjwt=fresh-token; Path=/; Secure; HttpOnly"),
            {"code": 0, "data": page([])},
        ])
        client = scores.BigQuantClient({"cookie": "_ga=tracking; bigjwt=stale", "username": "test", "password": "secret"}, opener=opener)
        self.assertEqual(scores.query_scores(client)["total"], 0)
        self.assertEqual([request.method for request in opener.requests], ["GET", "POST", "GET"])
        self.assertEqual(opener.requests[0].get_header("Cookie"), "bigjwt=stale")
        self.assertEqual(opener.requests[2].get_header("Cookie"), "bigjwt=fresh-token")
        self.assertEqual(opener.requests[1].full_url, scores.LOGIN_URL)
        self.assertIsNone(opener.requests[1].get_header("Cookie"))

    def test_login_can_use_body_token_without_set_cookie(self):
        opener = FakeOpener([{"code": 200, "data": {"token": "fresh"}}, {"code": 200, "data": page([])}])
        client = scores.BigQuantClient({"username": "test", "password": "secret"}, opener=opener)
        self.assertEqual(scores.query_scores(client)["total"], 0)
        self.assertEqual(opener.requests[-1].get_header("Cookie"), "bigjwt=fresh")

    def test_rejected_new_cookie_does_not_loop_login(self):
        opener = FakeOpener([401, {"code": 0, "data": {"token": "fresh"}}, 401])
        client = scores.BigQuantClient({"cookie": "bigjwt=stale", "username": "test", "password": "secret"}, opener=opener)
        with self.assertRaises(scores.AuthenticationError):
            scores.query_scores(client)
        self.assertEqual(len(opener.requests), 3)

    def test_business_failure_is_redacted_and_not_retried(self):
        opener = FakeOpener([{"code": 500, "message": "secret-cookie", "data": {"password": "secret-password"}}])
        client = scores.BigQuantClient({"cookie": "bigjwt=current"}, opener=opener)
        with self.assertRaises(scores.ScoreError) as caught:
            scores.query_scores(client)
        self.assertNotIn("secret", str(caught.exception))
        self.assertEqual(len(opener.requests), 1)

    def test_rate_limit_does_not_trigger_login(self):
        opener = FakeOpener([429])
        client = scores.BigQuantClient({"cookie": "bigjwt=current", "username": "test", "password": "secret"}, opener=opener)
        with self.assertRaises(scores.ScoreError):
            scores.query_scores(client)
        self.assertFalse(client.login_attempted)

    def test_explicit_auth_file_does_not_merge_environment_credentials(self):
        args = scores.build_parser().parse_args(["--auth-stdin"])
        with patch.dict("os.environ", {"BIGQUANT_COOKIE": "bigjwt=other-account"}), patch("sys.stdin", io.StringIO('{"username":"test","password":"secret"}')):
            auth = scores.load_auth(args)
        self.assertEqual(auth["cookie"], "")
        self.assertEqual(auth["username"], "test")

    def test_redirect_is_not_followed(self):
        self.assertIsNone(scores.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.example"))


if __name__ == "__main__":
    unittest.main()
