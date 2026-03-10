# nvBench-curate

Curated nvBench workspace for:

- repairing invalid SQL in `nvBench.json`
- aligning SQL outputs with `vis_obj`
- exporting a seq2seq training dataset
- training and comparing baseline sequence models

The repository already includes the curated source file, exported CSV splits, reports, and training artifacts.

## Repository contents

- `source/nvBench.json`: curated nvBench source after SQL runtime fixes and semantic alignment
- `dataset/nvbench_seq2seq/`: exported seq2seq dataset (`train.csv`, `dev.csv`, `test.csv`, `summary.json`)
- `scripts/train_compare_models.py`: trains and compares `rnn`, `lstm`, `gru`, and `transformer`
- `reports/`: SQL-fix logs, semantic-alignment logs, and analysis tables
- `runs/`: smoke and full training/comparison outputs

## Current curated data snapshot

- Total nvBench entries: `7247`
- Runtime-invalid SQL samples repaired: `145`
- Final SQL runtime failures after repair: `0`
- Additional semantic alignment rewrites after runtime fixing: `106`
- Fixed samples already matching `vis_obj` before alignment: `39`
- Total exported seq2seq rows: `25762`
- Split sizes:
  - train: `20648`
  - dev: `1145`
  - test: `3969`

Chart distribution in the exported dataset:

- `Bar`: 19411
- `Pie`: 1755
- `Scatter`: 1039
- `Stacked Bar`: 1172
- `Grouping Scatter`: 551
- `Line`: 1563
- `Grouping Line`: 271

## Training snapshot

Artifacts under `runs/full_compare/` show a 3-seed comparison on the exported dataset (`42, 43, 44`), selected by dev `slot_f1`.

| Model | Params | Test EM mean+-std | Test Slot F1 mean+-std | BLEU-4 mean+-std | ROUGE-L mean+-std |
|---|---:|---:|---:|---:|---:|
| rnn | 4178468 | 0.0051+-0.0001 | 0.5154+-0.0019 | 0.0976+-0.0063 | 0.4768+-0.0008 |
| lstm | 6543908 | 0.3352+-0.2048 | 0.8492+-0.0359 | 0.7161+-0.0638 | 0.8471+-0.0307 |
| gru | 5755428 | 0.1758+-0.0295 | 0.8125+-0.0210 | 0.6437+-0.0326 | 0.8135+-0.0204 |
| transformer | 6745636 | 0.6421+-0.0106 | 0.9222+-0.0025 | 0.8448+-0.0053 | 0.9184+-0.0019 |


## Environment

- Python `3.10+` recommended
- SQLite databases are required to rerun the curation scripts
- GPU is optional; `train_compare_models.py` falls back to CPU

Install dependencies:

```bash
python -m pip install -r requirements.txt
```



## Train baseline models

Example full run with 3 seeds:

```bash
python scripts/train_compare_models.py ^
  --data-dir dataset\nvbench_seq2seq ^
  --output-dir runs\full_compare ^
  --models rnn,lstm,gru,transformer ^
  --epochs 15 ^
  --batch-size 64 ^
  --seeds 42,43,44 ^
  --selection-metric slot_f1 ^
  --early-stopping-patience 4 ^
  --min-epochs 5
```

Main outputs:

- per-seed checkpoints: `runs/<exp>/<model>/best_seed*.pt`
- per-seed metrics/history/predictions
- aggregate CSV/JSON summaries
- Markdown comparison report

## Seq2seq format

Each CSV row contains:

- `input_text`: `<NQ> ... <DB> ... <SCHEMA> ... <LINK> ... <HINT> ...`
- `target_text`: `CHART=...; X=...; Y=...; AGG=...; CLASSIFY=...; FILTER=...; GROUP=...; SORT=...; BIN=...; TOPK=...`

Other fields include `sample_uid`, `sample_id`, `db_id`, `chart`, `hardness`, `question`, and `sql`.

## Notes

- The curated `source/nvBench.json` in this repository is already updated in place by the curation scripts.
- `reports/` and `runs/` are generated artifacts and can be kept for reproducibility or pruned before release if you want a lighter GitHub repository.
