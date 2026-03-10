from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Vocab:
    PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"

    def __init__(self, seqs, min_freq=1):
        cnt = Counter()
        for s in seqs:
            cnt.update(s)
        self.itos = [self.PAD, self.SOS, self.EOS, self.UNK]
        for t, f in cnt.items():
            if f >= min_freq and t not in self.itos:
                self.itos.append(t)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @property
    def pad_idx(self): return self.stoi[self.PAD]
    @property
    def sos_idx(self): return self.stoi[self.SOS]
    @property
    def eos_idx(self): return self.stoi[self.EOS]
    @property
    def unk_idx(self): return self.stoi[self.UNK]
    def encode(self, toks): return [self.stoi.get(t, self.unk_idx) for t in toks]
    def __len__(self): return len(self.itos)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def tok(s: str): return str(s).strip().split()
def with_sp(x): return [Vocab.SOS] + x + [Vocab.EOS]


@dataclass
class Ex:
    uid: str; src: list[int]; trg: list[int]; src_text: str; trg_text: str; chart: str; hardness: str


class SeqDS(Dataset):
    def __init__(self, xs): self.xs = xs
    def __len__(self): return len(self.xs)
    def __getitem__(self, i): return self.xs[i]


def _pad(xs, p):
    m = max(len(x) for x in xs)
    return torch.tensor([x + [p] * (m - len(x)) for x in xs], dtype=torch.long)


def collate(batch, sp, tp):
    return {
        "sample_uid": [b.uid for b in batch],
        "src": _pad([b.src for b in batch], sp),
        "trg": _pad([b.trg for b in batch], tp),
        "src_text": [b.src_text for b in batch],
        "trg_text": [b.trg_text for b in batch],
        "chart": [b.chart for b in batch],
        "hardness": [b.hardness for b in batch],
    }


def load_csv_rows(path: Path, limit: int):
    out = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            out.append(dict(row))
            if limit > 0 and i + 1 >= limit: break
    return out


def build_data(data_dir: Path, bs: int, min_freq: int, nw: int, lt: int, ld: int, lte: int):
    tr = load_csv_rows(data_dir / "train.csv", lt); dv = load_csv_rows(data_dir / "dev.csv", ld); te = load_csv_rows(data_dir / "test.csv", lte)
    sv = Vocab([with_sp(tok(r["input_text"])) for r in tr], min_freq=min_freq)
    tv = Vocab([with_sp(tok(r["target_text"])) for r in tr], min_freq=min_freq)

    def mk(rows):
        xs = []
        for r in rows:
            xs.append(Ex(r.get("sample_uid", ""), sv.encode(with_sp(tok(r["input_text"]))), tv.encode(with_sp(tok(r["target_text"]))), r["input_text"], r["target_text"], r.get("chart", ""), r.get("hardness", "")))
        return SeqDS(xs)

    dtr, ddv, dte = mk(tr), mk(dv), mk(te)
    ldr = {
        "train": DataLoader(dtr, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=lambda b: collate(b, sv.pad_idx, tv.pad_idx)),
        "dev": DataLoader(ddv, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=lambda b: collate(b, sv.pad_idx, tv.pad_idx)),
        "test": DataLoader(dte, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=lambda b: collate(b, sv.pad_idx, tv.pad_idx)),
    }
    return ldr, sv, tv, {"train": len(dtr), "dev": len(ddv), "test": len(dte)}


class Enc(nn.Module):
    def __init__(self, vs, ed, hd, cell, dp, pad):
        super().__init__(); self.emb = nn.Embedding(vs, ed, padding_idx=pad); self.rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell](ed, hd, batch_first=True); self.dp = nn.Dropout(dp)
    def forward(self, x): return self.rnn(self.dp(self.emb(x)))


class Dec(nn.Module):
    def __init__(self, vs, ed, hd, cell, dp, pad):
        super().__init__(); self.emb = nn.Embedding(vs, ed, padding_idx=pad); self.rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell](ed, hd, batch_first=True); self.fc = nn.Linear(hd, vs); self.dp = nn.Dropout(dp)
    def step(self, x, h):
        o, h = self.rnn(self.dp(self.emb(x.unsqueeze(1))), h)
        return self.fc(o.squeeze(1)), h


class S2SRNN(nn.Module):
    def __init__(self, e, d): super().__init__(); self.e = e; self.d = d
    def forward(self, src, trg, tfr=1.0):
        b, L = trg.shape; V = self.d.fc.out_features; out = torch.zeros(b, L - 1, V, device=src.device); _, h = self.e(src); x = trg[:, 0]
        for t in range(1, L):
            lg, h = self.d.step(x, h); out[:, t - 1, :] = lg; x = trg[:, t] if random.random() < tfr else lg.argmax(1)
        return out
    @torch.no_grad()
    def greedy_decode(self, src, mx, sos, eos):
        b = src.size(0); _, h = self.e(src); x = torch.full((b,), sos, dtype=torch.long, device=src.device); ys = []
        for _ in range(mx): lg, h = self.d.step(x, h); x = lg.argmax(1); ys.append(x)
        return torch.stack(ys, dim=1)


class PosEnc(nn.Module):
    def __init__(self, d, mx=4096):
        super().__init__(); pe = torch.zeros(mx, d); pos = torch.arange(0, mx).float().unsqueeze(1); div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d)); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div); self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
    def forward(self, x): return x + self.pe[:, :x.size(1), :]


class S2STrans(nn.Module):
    def __init__(self, sv, tv, ed, ff, h, n, dp, sp, tp):
        super().__init__(); self.se = nn.Embedding(sv, ed, padding_idx=sp); self.te = nn.Embedding(tv, ed, padding_idx=tp); self.sp = PosEnc(ed); self.tp = PosEnc(ed); self.tr = nn.Transformer(d_model=ed, nhead=h, num_encoder_layers=n, num_decoder_layers=n, dim_feedforward=ff, dropout=dp, batch_first=True); self.fc = nn.Linear(ed, tv); self.dp = nn.Dropout(dp); self.spad, self.tpad = sp, tp
    def forward(self, src, tin):
        sk, tk = src.eq(self.spad), tin.eq(self.tpad); tm = torch.triu(torch.ones((tin.size(1), tin.size(1)), device=tin.device, dtype=torch.bool), diagonal=1)
        o = self.tr(self.dp(self.sp(self.se(src))), self.dp(self.tp(self.te(tin))), tgt_mask=tm, src_key_padding_mask=sk, tgt_key_padding_mask=tk, memory_key_padding_mask=sk)
        return self.fc(o)
    @torch.no_grad()
    def greedy_decode(self, src, mx, sos, eos):
        b = src.size(0); ys = torch.full((b, 1), sos, dtype=torch.long, device=src.device)
        for _ in range(mx): ys = torch.cat([ys, self.forward(src, ys)[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1)
        return ys[:, 1:]


def build_model(name, sv, tv, ed, hd, nh, nl, dp):
    if name in {"rnn", "lstm", "gru"}: return S2SRNN(Enc(len(sv), ed, hd, name, dp, sv.pad_idx), Dec(len(tv), ed, hd, name, dp, tv.pad_idx))
    if name == "transformer": return S2STrans(len(sv), len(tv), ed, hd, nh, nl, dp, sv.pad_idx, tv.pad_idx)
    raise ValueError(name)


def loss_batch(m, src, trg, ce):
    lg = m(src, trg, 1.0) if isinstance(m, S2SRNN) else m(src, trg[:, :-1]); V = lg.shape[-1]
    return ce(lg.contiguous().view(-1, V), trg[:, 1:].contiguous().view(-1))


def train_epoch(m, ld, opt, ce, dev, gc):
    m.train(); s = 0.0
    for b in tqdm(ld, desc="train", leave=False):
        src, trg = b["src"].to(dev), b["trg"].to(dev); opt.zero_grad(); l = loss_batch(m, src, trg, ce); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), gc); opt.step(); s += l.item()
    return s / max(len(ld), 1)


@torch.no_grad()
def eval_epoch(m, ld, ce, dev):
    m.eval(); s = 0.0
    for b in tqdm(ld, desc="eval", leave=False):
        s += loss_batch(m, b["src"].to(dev), b["trg"].to(dev), ce).item()
    return s / max(len(ld), 1)


def norm(s): return " ".join(str(s).strip().split()).lower()
def slots(s):
    o = {}
    for p in s.split(";"):
        if "=" in p:
            k, v = p.split("=", 1); o[k.strip().upper()] = v.strip().lower()
    return o


def ngrams(xs, n): return Counter(tuple(xs[i : i + n]) for i in range(len(xs) - n + 1)) if len(xs) >= n else Counter()


def corpus_bleu(hyps, refs):
    out = {}; hl, rl = sum(len(h) for h in hyps), sum(len(r) for r in refs); bp = 1.0 if hl > rl else math.exp(1.0 - float(rl) / max(hl, 1))
    ps = []
    for n in (1, 2, 3, 4):
        num = den = 0
        for h, r in zip(hyps, refs):
            hc, rc = ngrams(h, n), ngrams(r, n); den += sum(hc.values()); num += sum(min(c, rc.get(g, 0)) for g, c in hc.items())
        ps.append(float(num) / float(den) if den else 0.0)
    for n in (1, 2, 3, 4):
        p = ps[:n]; out[f"bleu{n}"] = 0.0 if any(x == 0.0 for x in p) else bp * math.exp(sum(math.log(x) for x in p) / n)
    return out


def lcs(a, b):
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prv = 0
        for j in range(1, len(b) + 1):
            cur = dp[j]; dp[j] = prv + 1 if a[i - 1] == b[j - 1] else max(dp[j], dp[j - 1]); prv = cur
    return dp[-1]


def rouge_n_f1(h, r, n):
    hc, rc = ngrams(h, n), ngrams(r, n)
    if not hc and not rc: return 1.0
    if not hc or not rc: return 0.0
    ov = sum(min(c, rc.get(g, 0)) for g, c in hc.items()); p = ov / max(sum(hc.values()), 1); rr = ov / max(sum(rc.values()), 1)
    return 2 * p * rr / (p + rr) if (p + rr) > 0 else 0.0


def rouge_l_f1(h, r):
    if not h and not r: return 1.0
    if not h or not r: return 0.0
    ll = lcs(h, r); p = ll / len(h); rr = ll / len(r)
    return 2 * p * rr / (p + rr) if (p + rr) > 0 else 0.0


@torch.no_grad()
def eval_gen(m, ld, tv, dev, mx, return_rows=True):
    m.eval(); rows = []; t0 = time.perf_counter(); ns = ex = 0; tp = fp = fn = 0; hyps = []; refs = []; r1 = r2 = rl = 0.0
    for b in tqdm(ld, desc="generate", leave=False):
        pid = m.greedy_decode(b["src"].to(dev), mx, tv.sos_idx, tv.eos_idx)
        for i in range(pid.size(0)):
            toks = []
            for t in pid[i].tolist():
                if t == tv.eos_idx: break
                toks.append(tv.itos[t] if t < len(tv) else Vocab.UNK)
            pred, gold = " ".join(toks).strip(), b["trg_text"][i]
            pn, gn = norm(pred), norm(gold); ns += 1; ex += int(pn == gn)
            ps, gs = slots(pred), slots(gold)
            for k, v in ps.items():
                if gs.get(k, None) == v: tp += 1
                else: fp += 1
            for k, v in gs.items():
                if ps.get(k, None) != v: fn += 1
            pt, gt = pn.split(), gn.split(); hyps.append(pt); refs.append(gt); r1 += rouge_n_f1(pt, gt, 1); r2 += rouge_n_f1(pt, gt, 2); rl += rouge_l_f1(pt, gt)
            if return_rows: rows.append({"sample_uid": b["sample_uid"][i], "chart": b["chart"][i], "hardness": b["hardness"][i], "prediction": pred, "target": gold, "exact_match": pn == gn})
    bleu = corpus_bleu(hyps, refs); p = tp / (tp + fp) if tp + fp > 0 else 0.0; rr = tp / (tp + fn) if tp + fn > 0 else 0.0; f1 = 2 * p * rr / (p + rr) if p + rr > 0 else 0.0; tt = time.perf_counter() - t0
    return {"n_samples": ns, "exact_match": ex / max(ns, 1), "slot_precision": p, "slot_recall": rr, "slot_f1": f1, "bleu1": bleu["bleu1"], "bleu2": bleu["bleu2"], "bleu3": bleu["bleu3"], "bleu4": bleu["bleu4"], "rouge1_f1": r1 / max(ns, 1), "rouge2_f1": r2 / max(ns, 1), "rougeL_f1": rl / max(ns, 1), "inference_total_sec": tt, "inference_ms_per_sample": tt * 1000.0 / max(ns, 1), "pred_rows": rows if return_rows else None}


def pcount(m): return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)


def write_template(path: Path):
    path.write_text("# Model Comparison Template\n\n- Fill per-seed and aggregate(mean/std) metrics.\n", encoding="utf-8")


def parse_seeds(s): return [int(x.strip()) for x in str(s).split(",") if x.strip()]
def mean_std(vals): return (float(vals[0]), 0.0) if len(vals) == 1 else (float(statistics.mean(vals)), float(statistics.stdev(vals)))


def agg_model(model, rows):
    ks = ["best_dev_loss", "best_dev_slot_f1", "test_exact_match", "test_slot_f1", "bleu1", "bleu2", "bleu3", "bleu4", "rouge1_f1", "rouge2_f1", "rougeL_f1", "train_time_sec", "inference_ms_per_sample", "best_epoch"]
    o = {"model": model, "n_seeds": len(rows), "params_total": rows[0]["params_total"], "params_trainable": rows[0]["params_trainable"], "src_vocab_size": rows[0]["src_vocab_size"], "trg_vocab_size": rows[0]["trg_vocab_size"], "selection_metric": rows[0]["selection_metric"], "early_stopping_patience": rows[0]["early_stopping_patience"], "min_epochs": rows[0]["min_epochs"]}
    for k in ks:
        m, s = mean_std([float(r[k]) for r in rows]); o[f"{k}_mean"], o[f"{k}_std"] = m, s
    return o


def args():
    p = argparse.ArgumentParser("Train and compare RNN/LSTM/GRU/Transformer on nvbench_seq2seq.")
    p.add_argument("--data-dir", default="./dataset/nvbench_seq2seq"); p.add_argument("--output-dir", default="./runs/model_compare"); p.add_argument("--models", default="rnn,lstm,gru,transformer")
    p.add_argument("--epochs", type=int, default=15); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--learning-rate", type=float, default=5e-4); p.add_argument("--min-freq", type=int, default=1); p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--seeds", default="42")
    p.add_argument("--emb-dim", type=int, default=256); p.add_argument("--hid-dim", type=int, default=512); p.add_argument("--n-heads", type=int, default=8); p.add_argument("--n-layers", type=int, default=3); p.add_argument("--dropout", type=float, default=0.2); p.add_argument("--grad-clip", type=float, default=1.0); p.add_argument("--max-gen-len", type=int, default=120)
    p.add_argument("--selection-metric", choices=["dev_loss", "slot_f1"], default="slot_f1"); p.add_argument("--early-stopping-patience", type=int, default=4); p.add_argument("--min-epochs", type=int, default=5)
    p.add_argument("--limit-train", type=int, default=0); p.add_argument("--limit-dev", type=int, default=0); p.add_argument("--limit-test", type=int, default=0)
    return p.parse_args()


def main():
    a = args(); seeds = parse_seeds(a.seeds) or [a.seed]
    data_dir, out_dir = Path(a.data_dir), Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True); write_template(out_dir / "comparison_template.md")
    ldr, sv, tv, sz = build_data(data_dir, a.batch_size, a.min_freq, a.num_workers, a.limit_train, a.limit_dev, a.limit_test)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); models = [m.strip().lower() for m in a.models.split(",") if m.strip()]
    print("=" * 90); print(f"sizes: {sz}"); print(f"vocab: src={len(sv)} trg={len(tv)}"); print(f"device: {dev}"); print(f"models: {models}"); print(f"seeds: {seeds}"); print(f"selection_metric: {a.selection_metric}"); print(f"early_stopping_patience: {a.early_stopping_patience}, min_epochs: {a.min_epochs}"); print("=" * 90)

    by_seed_csv = out_dir / "model_comparison_by_seed.csv"
    hdr = ["model", "params_total", "params_trainable", "src_vocab_size", "trg_vocab_size", "best_dev_loss", "test_exact_match", "test_slot_f1", "bleu1", "bleu2", "bleu3", "bleu4", "rouge1_f1", "rouge2_f1", "rougeL_f1", "train_time_sec", "inference_ms_per_sample", "epochs", "seed", "selection_metric", "best_metric_value", "best_dev_slot_f1", "early_stopping_patience", "min_epochs"]
    with by_seed_csv.open("w", encoding="utf-8", newline="") as f: csv.writer(f).writerow(hdr)
    by_seed, agg_rows = [], []

    for mn in models:
        mo = out_dir / mn; mo.mkdir(parents=True, exist_ok=True); print("-" * 90); print(f"Model: {mn}"); rows = []
        for sd in seeds:
            set_seed(sd); m = build_model(mn, sv, tv, a.emb_dim, a.hid_dim, a.n_heads, a.n_layers, a.dropout).to(dev); pt, pr = pcount(m); print(f"seed={sd} params total={pt:,} trainable={pr:,}")
            opt = torch.optim.Adam(m.parameters(), lr=a.learning_rate); ce = nn.CrossEntropyLoss(ignore_index=tv.pad_idx)
            best_ep, best_dl, best_sf = 0, float("inf"), 0.0; best_metric = float("-inf") if a.selection_metric == "slot_f1" else float("inf"); bad = 0; hist = []; t0 = time.perf_counter(); ck = mo / f"best_seed{sd}.pt"
            for ep in range(1, a.epochs + 1):
                e0 = time.perf_counter(); tl = train_epoch(m, ldr["train"], opt, ce, dev, a.grad_clip); dl = eval_epoch(m, ldr["dev"], ce, dev); ds = None
                if a.selection_metric == "slot_f1": ds = float(eval_gen(m, ldr["dev"], tv, dev, a.max_gen_len, return_rows=False)["slot_f1"])
                metric = float(ds if a.selection_metric == "slot_f1" else dl); imp = (metric > best_metric) if a.selection_metric == "slot_f1" else (metric < best_metric)
                if imp:
                    best_metric, best_dl, best_sf, best_ep, bad = metric, float(dl), float(ds if ds is not None else best_sf), ep, 0; torch.save({"model_state": m.state_dict()}, ck)
                else:
                    bad += 1
                es = time.perf_counter() - e0; hist.append({"epoch": ep, "train_loss": float(tl), "dev_loss": float(dl), "dev_slot_f1": float(ds) if ds is not None else None, "epoch_sec": es, "improved": bool(imp)})
                ext = f" dev_slot_f1={ds:.4f}" if ds is not None else ""
                print(f"[{mn}|seed={sd}] epoch={ep:03d} train_loss={tl:.4f} dev_loss={dl:.4f}{ext} best={best_metric:.4f} bad_epochs={bad} time={es:.2f}s")
                if ep >= a.min_epochs and bad >= a.early_stopping_patience:
                    print(f"[{mn}|seed={sd}] early stop at epoch={ep}, best_epoch={best_ep}, metric={best_metric:.4f}"); break

            tt = time.perf_counter() - t0; m.load_state_dict(torch.load(ck, map_location=dev)["model_state"]); gs = eval_gen(m, ldr["test"], tv, dev, a.max_gen_len, return_rows=True)
            with (mo / f"test_predictions_seed{sd}.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["sample_uid", "chart", "hardness", "prediction", "target", "exact_match"]); w.writeheader(); [w.writerow(r) for r in (gs["pred_rows"] or [])]
            res = {"model": mn, "params_total": pt, "params_trainable": pr, "src_vocab_size": len(sv), "trg_vocab_size": len(tv), "best_dev_loss": best_dl, "best_epoch": best_ep, "best_metric_value": best_metric, "best_dev_slot_f1": best_sf, "selection_metric": a.selection_metric, "early_stopping_patience": a.early_stopping_patience, "min_epochs": a.min_epochs, "test_exact_match": gs["exact_match"], "test_slot_f1": gs["slot_f1"], "bleu1": gs["bleu1"], "bleu2": gs["bleu2"], "bleu3": gs["bleu3"], "bleu4": gs["bleu4"], "rouge1_f1": gs["rouge1_f1"], "rouge2_f1": gs["rouge2_f1"], "rougeL_f1": gs["rougeL_f1"], "train_time_sec": tt, "inference_ms_per_sample": gs["inference_ms_per_sample"], "epochs": len(hist), "seed": sd}
            (mo / f"history_seed{sd}.json").write_text(json.dumps(hist, indent=2), encoding="utf-8"); (mo / f"metrics_seed{sd}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            with by_seed_csv.open("a", encoding="utf-8", newline="") as f: csv.writer(f).writerow([res[h] for h in hdr])
            rows.append(res); by_seed.append(res)
            print(f"{mn}-s{sd:6} | EM={res['test_exact_match']:.4f} | slot_f1={res['test_slot_f1']:.4f} | BLEU4={res['bleu4']:.4f} | ROUGE-L={res['rougeL_f1']:.4f} | infer={res['inference_ms_per_sample']:.2f}ms/sample")
        ag = agg_model(mn, rows); agg_rows.append(ag); print(f"[aggregate|{mn}] seeds={ag['n_seeds']} EM={ag['test_exact_match_mean']:.4f}+-{ag['test_exact_match_std']:.4f} slot_f1={ag['test_slot_f1_mean']:.4f}+-{ag['test_slot_f1_std']:.4f}")

    (out_dir / "model_comparison_by_seed.json").write_text(json.dumps(by_seed, indent=2), encoding="utf-8")
    agg_csv = out_dir / "model_comparison_aggregate.csv"
    cols = ["model", "n_seeds", "params_total", "params_trainable", "src_vocab_size", "trg_vocab_size", "selection_metric", "early_stopping_patience", "min_epochs", "best_dev_loss_mean", "best_dev_loss_std", "best_dev_slot_f1_mean", "best_dev_slot_f1_std", "test_exact_match_mean", "test_exact_match_std", "test_slot_f1_mean", "test_slot_f1_std", "bleu1_mean", "bleu1_std", "bleu2_mean", "bleu2_std", "bleu3_mean", "bleu3_std", "bleu4_mean", "bleu4_std", "rouge1_f1_mean", "rouge1_f1_std", "rouge2_f1_mean", "rouge2_f1_std", "rougeL_f1_mean", "rougeL_f1_std", "train_time_sec_mean", "train_time_sec_std", "inference_ms_per_sample_mean", "inference_ms_per_sample_std", "best_epoch_mean", "best_epoch_std"]
    with agg_csv.open("w", encoding="utf-8", newline="") as f: w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in agg_rows]
    (out_dir / "model_comparison_aggregate.json").write_text(json.dumps(agg_rows, indent=2), encoding="utf-8")
    rep = out_dir / "model_comparison_report.md"
    lines = ["# Model Comparison Report", "", f"- Data: `{data_dir.resolve()}`", f"- Sizes: train={sz['train']} dev={sz['dev']} test={sz['test']}", f"- Vocab sizes: src={len(sv)} trg={len(tv)}", f"- Seeds: {seeds}", f"- Selection metric: `{a.selection_metric}`", f"- Early stopping: patience={a.early_stopping_patience}, min_epochs={a.min_epochs}", "", "| Model | Seeds | Params | Best Dev Loss (mean+-std) | EM (mean+-std) | Slot F1 (mean+-std) | BLEU-4 (mean+-std) | ROUGE-L (mean+-std) | Train Time s (mean+-std) | Infer ms/sample (mean+-std) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in agg_rows: lines.append(f"| {r['model']} | {r['n_seeds']} | {r['params_total']} | {r['best_dev_loss_mean']:.4f}+-{r['best_dev_loss_std']:.4f} | {r['test_exact_match_mean']:.4f}+-{r['test_exact_match_std']:.4f} | {r['test_slot_f1_mean']:.4f}+-{r['test_slot_f1_std']:.4f} | {r['bleu4_mean']:.4f}+-{r['bleu4_std']:.4f} | {r['rougeL_f1_mean']:.4f}+-{r['rougeL_f1_std']:.4f} | {r['train_time_sec_mean']:.2f}+-{r['train_time_sec_std']:.2f} | {r['inference_ms_per_sample_mean']:.2f}+-{r['inference_ms_per_sample_std']:.2f} |")
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("=" * 90); print("Overall report")
    for r in agg_rows: print(f"{r['model']:11s} | EM={r['test_exact_match_mean']:.4f}+-{r['test_exact_match_std']:.4f} | slot_f1={r['test_slot_f1_mean']:.4f}+-{r['test_slot_f1_std']:.4f} | BLEU4={r['bleu4_mean']:.4f}+-{r['bleu4_std']:.4f}")
    print(f"summary_csv_by_seed: {by_seed_csv}"); print(f"summary_json_by_seed: {out_dir / 'model_comparison_by_seed.json'}"); print(f"summary_csv_aggregate: {agg_csv}"); print(f"summary_json_aggregate: {out_dir / 'model_comparison_aggregate.json'}"); print(f"report_md: {rep}"); print(f"template_md: {out_dir / 'comparison_template.md'}")


if __name__ == "__main__":
    main()
