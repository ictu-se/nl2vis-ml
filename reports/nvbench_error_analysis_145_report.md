# nvBench SQL Error Analysis (145 Fixed Samples)

## Overview
- Total fixed samples analyzed: **145**
- Final SQL execution failures: **0 / 145**
- Final semantic match with `vis_obj` (set-level `(x,y)` pairs): **145 / 145**
- Final order match with `vis_obj`: **143 / 145**
- Semantic risk distribution: low=34, medium=22, high=89

## Error Category Breakdown
| Error Category | Count | Percent | Affected DBs | High-Risk Fixes |
|---|---:|---:|---:|---:|
| invalid_nested_aggregate | 62 | 42.76% | 10 | 62 |
| ambiguous_column_reference | 44 | 30.34% | 5 | 12 |
| invalid_column_reference | 16 | 11.03% | 2 | 6 |
| union_column_mismatch | 9 | 6.21% | 1 | 9 |
| malformed_sql_syntax | 7 | 4.83% | 2 | 0 |
| aggregate_in_group_by | 7 | 4.83% | 2 | 0 |

## Top 15 Databases by Error Volume
| DB | Errors | Percent | Top Error | Top Strategy |
|---|---:|---:|---|---|
| baseball_1 | 21 | 14.48% | ambiguous_column_reference | flatten_nested_aggregate |
| movie_1 | 17 | 11.72% | invalid_nested_aggregate | flatten_nested_aggregate |
| manufactory_1 | 14 | 9.66% | ambiguous_column_reference | qualify_ambiguous_column |
| election | 11 | 7.59% | invalid_nested_aggregate | flatten_nested_aggregate |
| wine_1 | 11 | 7.59% | invalid_nested_aggregate | flatten_nested_aggregate |
| local_govt_in_alabama | 10 | 6.90% | invalid_nested_aggregate | flatten_nested_aggregate |
| network_1 | 10 | 6.90% | invalid_nested_aggregate | flatten_nested_aggregate |
| journal_committee | 10 | 6.90% | invalid_column_reference | repair_invalid_column_reference |
| sakila_1 | 9 | 6.21% | union_column_mismatch | rewrite_union_projection_arity |
| dog_kennels | 8 | 5.52% | invalid_column_reference | rewrite_except_projection_arity |
| college_3 | 5 | 3.45% | malformed_sql_syntax | repair_sql_syntax |
| music_1 | 5 | 3.45% | aggregate_in_group_by | replace_groupby_aggregate_expr |
| customers_and_products_contacts | 3 | 2.07% | invalid_nested_aggregate | flatten_nested_aggregate |
| tracking_orders | 3 | 2.07% | invalid_nested_aggregate | flatten_nested_aggregate |
| behavior_monitoring | 2 | 1.38% | invalid_nested_aggregate | add_groupby_for_orderby_count |

## Fix Strategy Breakdown
| Strategy | Count | Percent | Majority Risk |
|---|---:|---:|---|
| flatten_nested_aggregate | 69 | 47.59% | high |
| qualify_ambiguous_column | 32 | 22.07% | low |
| repair_invalid_column_reference | 10 | 6.90% | medium |
| rewrite_union_projection_arity | 9 | 6.21% | high |
| replace_groupby_aggregate_expr | 7 | 4.83% | medium |
| rewrite_except_projection_arity | 6 | 4.14% | high |
| add_groupby_for_orderby_count | 5 | 3.45% | high |
| repair_sql_syntax | 5 | 3.45% | medium |
| remove_dangling_token | 2 | 1.38% | low |

## Final Validation (Merged Update)
- Checked samples: **145**
- SQL runnable: **145 / 145**
- Ground-truth match against `vis_obj`:
  - `set_match` (multiset `(x,y)`): **145 / 145**
  - `order_match`: **143 / 145**
- Additional semantic-alignment updates applied after runtime fixing: **106** samples
- Samples already matched before semantic alignment: **39** samples

## Output Files
- Detailed row-level table: `ncnet-legacy\reports\nvbench_error_analysis_145_detailed.csv`
- Summary by error: `ncnet-legacy\reports\nvbench_error_analysis_145_summary_by_error.csv`
- Summary by database: `ncnet-legacy\reports\nvbench_error_analysis_145_summary_by_db.csv`
- Summary by fix strategy: `ncnet-legacy\reports\nvbench_error_analysis_145_summary_by_strategy.csv`
- Runtime fix log (original -> fixed SQL): `ncnet-legacy\reports\nvbench_sql_fixes.csv`
- Semantic alignment log (additional updates): `ncnet-legacy\reports\nvbench_sql_semantic_alignment.csv`
- Final SQL-vs-ground-truth check: `ncnet-legacy\reports\nvbench_sql_vs_groundtruth_check_145.csv`
