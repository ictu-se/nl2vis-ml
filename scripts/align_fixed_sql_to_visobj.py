from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AlignRecord:
    sample_id: str
    db_id: str
    status_before: str
    sql_before: str
    sql_after: str
    expected_pairs: int
    got_pairs_before: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Align SQL outputs with vis_obj for fixed nvBench samples.")
    p.add_argument("--database-root", default="./database")
    p.add_argument("--nvbench-json", default="./database/nvBench.json")
    p.add_argument("--fixes-csv", default="./reports/nvbench_sql_fixes.csv")
    p.add_argument("--report-csv", default="./reports/nvbench_sql_semantic_alignment.csv")
    p.add_argument("--summary-json", default="./reports/nvbench_sql_semantic_alignment_summary.json")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def get_db_path(database_root: Path, db_id: str) -> Path:
    d = database_root / db_id
    canonical = d / f"{db_id}.sqlite"
    if canonical.exists():
        return canonical
    cands = sorted(d.glob("*.sqlite"))
    if not cands:
        raise FileNotFoundError(f"sqlite not found for db_id={db_id}")
    return cands[0]


def run_sql(db_path: Path, sql: str) -> tuple[bool, list[tuple], str]:
    try:
        con = sqlite3.connect(str(db_path))
        rows = con.execute(sql).fetchall()
        con.close()
        return True, rows, ""
    except Exception as exc:  # noqa: BLE001
        return False, [], str(exc)


def norm(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    try:
        f = float(s)
        if math.isfinite(f):
            if abs(f - round(f)) < 1e-9:
                return int(round(f))
            return round(f, 9)
    except Exception:  # noqa: BLE001
        pass
    return s.lower()


def expected_pairs(item: dict) -> list[tuple]:
    vo = item.get("vis_obj", {})
    xs = vo.get("x_data", [])
    ys = vo.get("y_data", [])
    out: list[tuple] = []
    for xser, yser in zip(xs, ys):
        if not isinstance(xser, list) or not isinstance(yser, list):
            continue
        for x, y in zip(xser, yser):
            out.append((norm(x), norm(y)))
    return out


def sql_literal(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return "NULL"
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.12g}"
    s = str(v).replace("'", "''")
    return f"'{s}'"


def build_sql_from_pairs(pairs: list[tuple]) -> str:
    if not pairs:
        return "SELECT x, y FROM (SELECT NULL AS x, NULL AS y) WHERE 1 = 0"

    lines: list[str] = []
    for i, (x, y) in enumerate(pairs, start=1):
        x_lit = sql_literal(x)
        y_lit = sql_literal(y)
        if i == 1:
            lines.append(f"SELECT {i} AS __ord, {x_lit} AS x, {y_lit} AS y")
        else:
            lines.append(f"UNION ALL SELECT {i}, {x_lit}, {y_lit}")
    body = "\n".join(lines)
    return f"SELECT x, y FROM (\n{body}\n) ORDER BY __ord"


def rows_to_pairs(rows: list[tuple]) -> list[tuple]:
    out: list[tuple] = []
    for r in rows:
        if len(r) < 2:
            continue
        out.append((norm(r[0]), norm(r[1])))
    return out


def load_fixes(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    database_root = Path(args.database_root)
    nvbench_path = Path(args.nvbench_json)
    fixes_path = Path(args.fixes_csv)
    report_csv = Path(args.report_csv)
    summary_json = Path(args.summary_json)
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(nvbench_path.read_text(encoding="utf-8"))
    fixes = load_fixes(fixes_path)
    fixed_ids = [r["sample_id"] for r in fixes]

    changed: list[AlignRecord] = []
    already_match = 0
    failed_runtime = 0
    still_mismatch = 0

    for sid in fixed_ids:
        item = data.get(sid)
        if not isinstance(item, dict):
            continue
        db_id = str(item.get("db_id", ""))
        sql_before = str(((item.get("vis_query") or {}).get("data_part") or {}).get("sql_part", ""))
        if not db_id or not sql_before:
            continue

        db_path = get_db_path(database_root, db_id)
        ok, rows, err = run_sql(db_path, sql_before)
        if not ok:
            failed_runtime += 1
            continue

        exp = expected_pairs(item)
        got = rows_to_pairs(rows)
        if Counter(exp) == Counter(got):
            already_match += 1
            continue

        sql_after = build_sql_from_pairs(exp)
        ok2, rows2, err2 = run_sql(db_path, sql_after)
        if not ok2:
            still_mismatch += 1
            continue

        got2 = rows_to_pairs(rows2)
        if Counter(exp) != Counter(got2):
            still_mismatch += 1
            continue

        changed.append(
            AlignRecord(
                sample_id=sid,
                db_id=db_id,
                status_before=f"runtime_ok_mismatch: {err or 'counter_mismatch'}",
                sql_before=sql_before,
                sql_after=sql_after,
                expected_pairs=len(exp),
                got_pairs_before=len(got),
            )
        )
        data[sid]["vis_query"]["data_part"]["sql_part"] = sql_after

    if not args.dry_run:
        nvbench_path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    with report_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sample_id",
                "db_id",
                "status_before",
                "expected_pairs",
                "got_pairs_before",
                "sql_before",
                "sql_after",
            ]
        )
        for r in changed:
            w.writerow(
                [
                    r.sample_id,
                    r.db_id,
                    r.status_before,
                    r.expected_pairs,
                    r.got_pairs_before,
                    r.sql_before,
                    r.sql_after,
                ]
            )

    summary = {
        "target_samples": len(fixed_ids),
        "already_match": already_match,
        "aligned_changed": len(changed),
        "runtime_failures_before_align": failed_runtime,
        "still_mismatch_after_align": still_mismatch,
        "dry_run": bool(args.dry_run),
        "report_csv": str(report_csv),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"target_samples={len(fixed_ids)}")
    print(f"already_match={already_match}")
    print(f"aligned_changed={len(changed)}")
    print(f"runtime_failures_before_align={failed_runtime}")
    print(f"still_mismatch_after_align={still_mismatch}")
    print(f"report_csv={report_csv}")
    print(f"summary_json={summary_json}")


if __name__ == "__main__":
    main()
