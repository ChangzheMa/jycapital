"""验证导出包、阻断推理读标签，并与冻结运行的评估预测比较。"""

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from artifacts import read_json, sha256, write_json


def verify(package, run):
    package, run = Path(package).resolve(), Path(run).resolve()
    manifest = read_json(package / "submission-manifest.json")
    for name, digest in manifest["files"].items():
        if sha256(package / name) != digest:
            raise ValueError(f"提交文件哈希不一致: {name}")
    notebooks = list(package.glob("*.ipynb"))
    if len(notebooks) != 1 or (package / "dai.py").exists():
        raise ValueError("Notebook 数量或本地 DAI 文件错误")
    import nbformat
    notebook = nbformat.read(notebooks[0], as_version=4)
    nbformat.validate(notebook)
    spec = importlib.util.spec_from_file_location("local_validation_dai", ROOT / "dai.py")
    local_dai = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local_dai)

    class GuardedDAI:
        def __init__(self):
            self.queries = 0

        def query(self, sql, **kwargs):
            if any(token in sql.lower() for token in ["vwap", "exposure", "ret", "label"]):
                raise AssertionError("推理不能读取标签或风险暴露")
            self.queries += 1
            return local_dai.query(sql, **kwargs)

    guard = GuardedDAI()
    sys.modules["dai"] = guard
    sys.path.insert(0, str(package))
    namespace = {"__name__": "submission_validation"}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, str(notebooks[0]), "exec"), namespace)
    if guard.queries != 0 or "factor_data" in namespace:
        raise AssertionError("导入 Notebook 时不能自动运行调试推理")
    begin = time.perf_counter()
    tests = []
    for split, lo, hi in [("validation", "2024-01-31", "2024-02-02"), ("test", "2024-12-30", "2024-12-31")]:
        actual = namespace["main"]({"bar5m": "cpt_jyc_2026_stock_bar5m"}, lo, hi)
        expected = pd.read_parquet(run / "evaluation" / split / "predictions.parquet",
                                   filters=[("date", ">=", pd.Timestamp(lo)), ("date", "<", pd.Timestamp(hi) + pd.Timedelta(days=1))])
        expected = expected.sort_values(["date", "instrument"]).reset_index(drop=True)
        if list(actual.columns) != ["date", "instrument", "factor"] or actual["factor"].dtype != np.dtype("float64"):
            raise AssertionError("输出列或 dtype 不符")
        pd.testing.assert_frame_equal(actual[["date", "instrument"]], expected[["date", "instrument"]], check_dtype=False)
        np.testing.assert_allclose(actual["factor"], expected["factor"], atol=1e-8, rtol=1e-5)
        tests.append({"split": split, "start": lo, "end": hi, "rows": len(actual),
                      "max_abs_difference": float(np.abs(actual["factor"] - expected["factor"]).max())})
    # 精确分钟边界只应返回该分钟的股票池。
    exact = namespace["main"]({"bar5m": "cpt_jyc_2026_stock_bar5m"}, "2024-12-31 10:00:00", "2024-12-31 10:00:00")
    if len(exact) != 1000 or not (exact["date"] == pd.Timestamp("2024-12-31 10:00")).all():
        raise AssertionError("精确分钟边界处理错误")
    debug_namespace = {"__name__": "__main__", "display": lambda frame: None}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in notebook.cells:
            if cell.cell_type == "code":
                exec(compile(cell.source, str(notebooks[0]), "exec"), debug_namespace)
    debug_result = debug_namespace["factor_data"]
    pd.testing.assert_frame_equal(debug_result, actual)
    result = {"passed": True, "notebooks": 1, "queries": guard.queries,
              "torch_imported": "torch" in sys.modules, "tests": tests,
              "import_runs_debug": False, "debug_rows": len(debug_result),
              "exact_minute_rows": len(exact), "seconds": round(time.perf_counter() - begin, 3),
              "scope": "本地提交目录验证；平台运行与正式提交结论另行记录"}
    write_json(run / "submission-validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    verify(args.package, args.run_dir)
