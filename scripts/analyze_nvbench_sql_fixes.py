from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


@dataclass
class RowAnalysis:
    sample_id: str
    db_id: str
    chart: str
    hardness: str
    error_before_logged: str
    error_before_runtime: str
    error_category: str
    fix_strategy: str
    semantic_risk: str
    sql_before: str
    sql_after: str
    changed_chars: int
    change_ratio: float
    sql_after_ok: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Analyze 145 fixed nvBench SQL errors for publication-ready tables.")
    p.add_argument("--database-root", default="./database")
    p.add_argument("--nvbench-json", default="./database/nvBench.json")
    p.add_argument("--fixes-csv", default="./reports/nvbench_sql_fixes.csv")
    p.add_argument("--out-detailed-csv", default="./reports/nvbench_error_analysis_145_detailed.csv")
    p.add_argument("--out-summary-error-csv", default="./reports/nvbench_error_analysis_145_summary_by_error.csv")
    p.add_argument("--out-summary-db-csv", default="./reports/nvbench_error_analysis_145_summary_by_db.csv")
    p.add_argument("--out-summary-strategy-csv", default="./reports/nvbench_error_analysis_145_summary_by_strategy.csv")
    p.add_argument("--out-markdown", default="./reports/nvbench_error_analysis_145_report.md")
    return p.parse_args()


def get_db_path(database_root: Path, db_id: str) -> Path:
    db_dir = database_root / db_id
    canonical = db_dir / f"{db_id}.sqlite"
    if canonical.exists():
        return canonical
    cands = sorted(db_dir.glob("*.sqlite"))
    if not cands:
        raise FileNotFoundError(f"No sqlite found for db_id={db_id}")
    return cands[0]


def run_sql(db_path: Path, sql: str) -> tuple[bool, str]:
    try:
        con = sqlite3.connect(str(db_path))
        con.execute(sql).fetchall()
        con.close()
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def normalize_err(err: str) -> str:
    return " ".join(str(err).strip().split())


def classify_error(err: str) -> str:
    e = err.lower()
    if "ambiguous column name" in e:
        return "ambiguous_column_reference"
    if "misuse of aggregate function" in e or "misuse of aggregate:" in e:
        return "invalid_nested_aggregate"
    if "aggregate functions are not allowed in the group by clause" in e:
        return "aggregate_in_group_by"
    if "left and right of union do not have the same number of result columns" in e:
        return "union_column_mismatch"
    if "left and right of except do not have the same number of result columns" in e:
        return "except_column_mismatch"
    if "no such column" in e:
        return "invalid_column_reference"
    if "syntax error" in e:
        return "malformed_sql_syntax"
    return "other_runtime_error"


def detect_fix_strategy(sql_before: str, sql_after: str, err_cat: str) -> str:
    b = sql_before.lower()
    a = sql_after.lower()

    if err_cat == "union_column_mismatch":
        return "rewrite_union_projection_arity"
    if err_cat == "invalid_column_reference" and " except " in b and " except " in a:
        if "select t1.professional_id" in b and "select t1.professional_id" not in a:
            return "rewrite_except_projection_arity"
    if any(x in b for x in ["sum(count(", "avg(count(", "sum(sum(", "avg(sum(", "sum(avg(", "avg(min(", "sum(min(", "avg(max("]):
        return "flatten_nested_aggregate"
    if "group by min(" in b or "group by avg(" in b or "group by sum(" in b or "group by max(" in b:
        return "replace_groupby_aggregate_expr"
    if "order by count(*)" in b and "group by" not in b and "group by" in a:
        return "add_groupby_for_orderby_count"
    if ") name" in b and ") name" not in a:
        return "remove_dangling_token"
    if re.search(r"group by\s+[^,]+t1\.\w+\s+t2\.\w+", b) and "," in a:
        return "insert_missing_comma_groupby"
    if err_cat == "ambiguous_column_reference":
        return "qualify_ambiguous_column"
    if err_cat == "invalid_column_reference":
        return "repair_invalid_column_reference"
    if err_cat == "malformed_sql_syntax":
        return "repair_sql_syntax"
    return "other_transform"


def risk_level(strategy: str) -> str:
    if strategy in {
        "rewrite_union_projection_arity",
        "rewrite_except_projection_arity",
        "flatten_nested_aggregate",
        "add_groupby_for_orderby_count",
    }:
        return "high"
    if strategy in {
        "replace_groupby_aggregate_expr",
        "repair_invalid_column_reference",
        "repair_sql_syntax",
        "insert_missing_comma_groupby",
    }:
        return "medium"
    return "low"


def load_fixes(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def main() -> None:
    args = parse_args()
    database_root = Path(args.database_root)
    fixes_path = Path(args.fixes_csv)
    nvbench = json.loads(Path(args.nvbench_json).read_text(encoding="utf-8"))

    rows = load_fixes(fixes_path)
    analyses: list[RowAnalysis] = []

    for r in rows:
        sample_id = r["sample_id"]
        db_id = r["db_id"]
        sql_before = r["sql_before"]
        sql_after = r["sql_after"]
        db_path = get_db_path(database_root, db_id)

        ok_before, err_before_runtime = run_sql(db_path, sql_before)
        ok_after, err_after_runtime = run_sql(db_path, sql_after)

        chart = ""
        hardness = ""
        if sample_id in nvbench:
            chart = str(nvbench[sample_id].get("chart", ""))
            hardness = str(nvbench[sample_id].get("hardness", ""))

        err_before_runtime_n = normalize_err(err_before_runtime if not ok_before else "")
        err_cat = classify_error(err_before_runtime_n if err_before_runtime_n else r["error_before"])
        strategy = detect_fix_strategy(sql_before, sql_after, err_cat)
        risk = risk_level(strategy)

        matcher = SequenceMatcher(None, sql_before, sql_after)
        ratio = matcher.ratio()
        changed_chars = abs(len(sql_before) - len(sql_after))

        analyses.append(
            RowAnalysis(
                sample_id=sample_id,
                db_id=db_id,
                chart=chart,
                hardness=hardness,
                error_before_logged=r["error_before"],
                error_before_runtime=err_before_runtime_n if err_before_runtime_n else "(already valid)",
                error_category=err_cat,
                fix_strategy=strategy,
                semantic_risk=risk,
                sql_before=sql_before,
                sql_after=sql_after,
                changed_chars=changed_chars,
                change_ratio=ratio,
                sql_after_ok=ok_after and not err_after_runtime,
            )
        )

    # Detailed table (145 rows).
    out_detailed = Path(args.out_detailed_csv)
    out_detailed.parent.mkdir(parents=True, exist_ok=True)
    with out_detailed.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "db_id",
                "chart",
                "hardness",
                "error_before_logged",
                "error_before_runtime",
                "error_category",
                "fix_strategy",
                "semantic_risk",
                "sql_after_ok",
                "changed_chars",
                "change_ratio",
                "sql_before",
                "sql_after",
            ]
        )
        for a in analyses:
            writer.writerow(
                [
                    a.sample_id,
                    a.db_id,
                    a.chart,
                    a.hardness,
                    a.error_before_logged,
                    a.error_before_runtime,
                    a.error_category,
                    a.fix_strategy,
                    a.semantic_risk,
                    a.sql_after_ok,
                    a.changed_chars,
                    f"{a.change_ratio:.6f}",
                    a.sql_before,
                    a.sql_after,
                ]
            )

    n = len(analyses)
    by_error: dict[str, list[RowAnalysis]] = defaultdict(list)
    by_db: dict[str, list[RowAnalysis]] = defaultdict(list)
    by_strategy: dict[str, list[RowAnalysis]] = defaultdict(list)
    for a in analyses:
        by_error[a.error_category].append(a)
        by_db[a.db_id].append(a)
        by_strategy[a.fix_strategy].append(a)

    # Summary by error category.
    out_error = Path(args.out_summary_error_csv)
    with out_error.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "error_category",
                "count",
                "percent",
                "affected_db_count",
                "affected_chart_count",
                "avg_change_ratio",
                "high_risk_count",
                "all_fixed_count",
            ]
        )
        for err_cat, vals in sorted(by_error.items(), key=lambda x: len(x[1]), reverse=True):
            db_cnt = len({v.db_id for v in vals})
            ch_cnt = len({v.chart for v in vals if v.chart})
            avg_ratio = sum(v.change_ratio for v in vals) / len(vals)
            high_cnt = sum(1 for v in vals if v.semantic_risk == "high")
            fixed_cnt = sum(1 for v in vals if v.sql_after_ok)
            w.writerow(
                [
                    err_cat,
                    len(vals),
                    f"{(len(vals) / n * 100):.2f}",
                    db_cnt,
                    ch_cnt,
                    f"{avg_ratio:.4f}",
                    high_cnt,
                    fixed_cnt,
                ]
            )

    # Summary by DB.
    out_db = Path(args.out_summary_db_csv)
    with out_db.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "db_id",
                "error_count",
                "percent",
                "top_error_category",
                "top_fix_strategy",
                "high_risk_count",
            ]
        )
        for db_id, vals in sorted(by_db.items(), key=lambda x: len(x[1]), reverse=True):
            err_top = Counter(v.error_category for v in vals).most_common(1)[0][0]
            fix_top = Counter(v.fix_strategy for v in vals).most_common(1)[0][0]
            high_cnt = sum(1 for v in vals if v.semantic_risk == "high")
            w.writerow(
                [
                    db_id,
                    len(vals),
                    f"{(len(vals) / n * 100):.2f}",
                    err_top,
                    fix_top,
                    high_cnt,
                ]
            )

    # Summary by fix strategy.
    out_strategy = Path(args.out_summary_strategy_csv)
    with out_strategy.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "fix_strategy",
                "count",
                "percent",
                "error_categories",
                "risk_level_majority",
                "all_fixed_count",
            ]
        )
        for st, vals in sorted(by_strategy.items(), key=lambda x: len(x[1]), reverse=True):
            cats = sorted({v.error_category for v in vals})
            risk = Counter(v.semantic_risk for v in vals).most_common(1)[0][0]
            fixed_cnt = sum(1 for v in vals if v.sql_after_ok)
            w.writerow([st, len(vals), f"{(len(vals) / n * 100):.2f}", ";".join(cats), risk, fixed_cnt])

    # Markdown report for paper draft.
    md = Path(args.out_markdown)
    error_top = sorted(by_error.items(), key=lambda x: len(x[1]), reverse=True)
    db_top = sorted(by_db.items(), key=lambda x: len(x[1]), reverse=True)[:15]
    st_top = sorted(by_strategy.items(), key=lambda x: len(x[1]), reverse=True)
    high_risk_total = sum(1 for a in analyses if a.semantic_risk == "high")
    medium_risk_total = sum(1 for a in analyses if a.semantic_risk == "medium")
    low_risk_total = sum(1 for a in analyses if a.semantic_risk == "low")
    unresolved_after = sum(1 for a in analyses if not a.sql_after_ok)

    lines: list[str] = []
    lines.append("# nvBench SQL Error Analysis (145 Fixed Samples)")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Total fixed samples analyzed: **{n}**")
    lines.append(f"- Post-fix SQL execution failures: **{unresolved_after}**")
    lines.append(
        f"- Semantic risk distribution: low={low_risk_total}, medium={medium_risk_total}, high={high_risk_total}"
    )
    lines.append("")
    lines.append("## Error Category Breakdown")
    lines.append("| Error Category | Count | Percent | Affected DBs | High-Risk Fixes |")
    lines.append("|---|---:|---:|---:|---:|")
    for cat, vals in error_top:
        db_cnt = len({v.db_id for v in vals})
        high_cnt = sum(1 for v in vals if v.semantic_risk == "high")
        lines.append(f"| {cat} | {len(vals)} | {len(vals)/n*100:.2f}% | {db_cnt} | {high_cnt} |")
    lines.append("")
    lines.append("## Top 15 Databases by Error Volume")
    lines.append("| DB | Errors | Percent | Top Error | Top Strategy |")
    lines.append("|---|---:|---:|---|---|")
    for db_id, vals in db_top:
        err_top_cat = Counter(v.error_category for v in vals).most_common(1)[0][0]
        st_top_cat = Counter(v.fix_strategy for v in vals).most_common(1)[0][0]
        lines.append(f"| {db_id} | {len(vals)} | {len(vals)/n*100:.2f}% | {err_top_cat} | {st_top_cat} |")
    lines.append("")
    lines.append("## Fix Strategy Breakdown")
    lines.append("| Strategy | Count | Percent | Majority Risk |")
    lines.append("|---|---:|---:|---|")
    for st, vals in st_top:
        risk = Counter(v.semantic_risk for v in vals).most_common(1)[0][0]
        lines.append(f"| {st} | {len(vals)} | {len(vals)/n*100:.2f}% | {risk} |")
    lines.append("")
    lines.append("## Output Files")
    lines.append(f"- Detailed row-level table: `{Path(args.out_detailed_csv)}`")
    lines.append(f"- Summary by error: `{Path(args.out_summary_error_csv)}`")
    lines.append(f"- Summary by database: `{Path(args.out_summary_db_csv)}`")
    lines.append(f"- Summary by fix strategy: `{Path(args.out_summary_strategy_csv)}`")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"analyzed_rows={n}")
    print(f"post_fix_failures={unresolved_after}")
    print(f"detailed_csv={out_detailed}")
    print(f"summary_error_csv={out_error}")
    print(f"summary_db_csv={out_db}")
    print(f"summary_strategy_csv={out_strategy}")
    print(f"markdown_report={md}")


if __name__ == "__main__":
    main()
