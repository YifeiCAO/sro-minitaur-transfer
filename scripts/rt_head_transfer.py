#!/usr/bin/env python
"""Approach B (cleanest): CONTINUOUS RT-residual transfer.

Freeze M_pop. For each trial, read the model's hidden state at the choice and fit
a population RT model -- Ridge: hidden_state -> log-RT -- on TRAIN subjects. The
individual signal = the per-trial STANDARDISED residual (logRT - predicted)/sd:
"this person is faster / slower than the population expects, given the same
internal state". Summarise per person, then cross-task identification (within vs
across domain), the same rank metric as the surprise matrix.

No regression head, no custom training loop, no model surgery -- the "head" is a
Ridge on frozen hidden states. Needs output_nl_rtval (choice-only text + a
``rt_values`` sidecar of log-RT per response).

    python scripts/rt_head_transfer.py --mpop /content/drive/MyDrive/sro_minitaur/mpop_rt \
        --nl-dir /content/drive/MyDrive/sro_minitaur/output_nl_rtval --subset all
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from sro_transfer.data import make_splits
from sro_transfer.model.masking import build_labels
from sro_transfer.runtime import get_model
from sro_transfer.utils import load_config, load_tasks


def load_rtval(nl_dir, task, source="complete"):
    fp = Path(nl_dir) / source / f"{task}.all.jsonl"
    out = {}
    if fp.exists():
        for line in open(fp, encoding="utf-8"):
            o = json.loads(line)
            out[o["worker_id"]] = (o["text"], o.get("rt_values", []))
    return out


def _residual_rep(z):
    """Person vector from their standardised RT residuals (1-D)."""
    if len(z) == 0:
        return np.zeros(9, dtype=np.float32)
    q = np.quantile(z, [.1, .25, .5, .75, .9])
    return np.array([z.mean(), z.std(), *q, np.abs(z).mean(), (z > 0).mean()], dtype=np.float32)


def extract_task(model, tok, sess, max_len):
    """Per response: hidden state at the choice token + its log-RT target."""
    import torch
    H, rt, owner = [], [], []
    for w, (text, rtv) in sess.items():
        if not rtv:
            continue
        e = build_labels(text, tok, max_len)
        ids = torch.tensor([e["input_ids"]], device=model.device)
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states[-1][0]
        lab = e["labels"]
        groups, i, L = [], 0, len(lab)               # first token pos of each response
        while i < L:
            if lab[i] != -100:
                j = i
                while j + 1 < L and lab[j + 1] != -100:
                    j += 1
                groups.append(i); i = j + 1
            else:
                i += 1
        m = len(groups)
        if m == 0:
            continue
        tgt = rtv[-m:]                                # front-truncation -> align to the tail
        hsl = hs.float().cpu().numpy()
        for gi, p in enumerate(groups):
            v = tgt[gi]
            if v == v:                               # drop nan (no-response trials)
                H.append(hsl[p]); rt.append(float(v)); owner.append(w)
    return np.asarray(H, np.float32), np.asarray(rt), np.asarray(owner)


def residuals_via_head(model, tok, head, sess, max_len):
    """Per person: standardised residuals (logRT - mu)/sigma from the TRAINED head.
    Also returns flat (mu, y, owner) arrays for a heldout RT-prediction R^2 check
    (is the head TRIAL-CONDITIONAL, or just predicting the marginal mean?)."""
    import torch
    raw, MU, Y, OWN = {}, [], [], []
    for w, (text, rtv) in sess.items():
        if not rtv:
            continue
        e = build_labels(text, tok, max_len)
        ids = torch.tensor([e["input_ids"]], device=model.device)
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states[-1][0]
        lab = e["labels"]; groups, i, L = [], 0, len(lab)
        while i < L:
            if lab[i] != -100:
                j = i
                while j + 1 < L and lab[j + 1] != -100:
                    j += 1
                groups.append(i); i = j + 1
            else:
                i += 1
        m = len(groups)
        if m == 0:
            continue
        tgt = rtv[-m:]
        with torch.no_grad():
            mu = head(hs[groups].float()).cpu().numpy()      # point estimate of log-RT
        r = []
        for gi in range(m):
            if tgt[gi] == tgt[gi]:
                r.append(tgt[gi] - mu[gi])                   # residual = logRT - predicted
                MU.append(mu[gi]); Y.append(tgt[gi]); OWN.append(w)
        if len(r) >= 3:
            raw[w] = np.asarray(r)
    allr = np.concatenate(list(raw.values())) if raw else np.array([1.0])
    sd = allr.std() + 1e-6                                    # standardise by the task residual sd
    prof = {w: r / sd for w, r in raw.items()}
    return prof, np.asarray(MU), np.asarray(Y), np.asarray(OWN)


def _identify(pred, true_vecs, ids, K, seed):
    rng = np.random.RandomState(seed)
    T = np.stack([true_vecs[w] for w in ids]); T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    hits = 0
    for i in range(len(ids)):
        p = pred[i] / (np.linalg.norm(pred[i]) + 1e-8)
        others = [j for j in range(len(ids)) if j != i]
        cand = [i] + list(rng.choice(others, size=min(K - 1, len(others)), replace=False))
        if cand[int(np.argmax(T[cand] @ p))] == i:
            hits += 1
    return hits / len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mpop", default="/content/drive/MyDrive/sro_minitaur/mpop_rt")
    ap.add_argument("--trained", default=None,
                    help="dir of a trained RT head (adapter + rt_head.pt); else frozen-Ridge probe")
    ap.add_argument("--nl-dir", default="/content/drive/MyDrive/sro_minitaur/output_nl_rtval")
    ap.add_argument("--subset", default="all")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    args = ap.parse_args()
    from sklearn.linear_model import Ridge

    cfg = load_config(args.config)
    cfg["paths"]["nl_dir"] = args.nl_dir
    cfg["model"]["max_seq_len"] = args.max_seq_len
    seed, rdir = cfg["split"]["seed"], cfg["paths"]["results"]
    tax = load_tasks()
    tasks = sorted(tax["tasks"]) if args.subset == "all" else tax["subsets"][args.subset]
    domain = {t: tax["tasks"][t]["domain"] for t in tax["tasks"]}
    head = None
    if args.trained:
        from sro_transfer.model.rt_head import load_rt_model
        model, tok, head = load_rt_model(cfg, args.trained)
        print(f"using TRAINED RT head from {args.trained}")
    else:
        model, tok = get_model(cfg, args.mpop)
        print("using frozen-Ridge PROBE (no trained head)")

    reps, splits = {}, {}
    for t in tasks:
        sess = load_rtval(args.nl_dir, t)
        if len(sess) < 60:
            continue
        split = make_splits(list(sess), [], cfg["split"]["heldout_frac"], seed)
        if head is not None:                            # trained head: residual = (logRT-mu)/sigma
            prof_z, MU, Y, OWN = residuals_via_head(model, tok, head, sess, args.max_seq_len)
            reps[t] = {w: _residual_rep(z) for w, z in prof_z.items()}
            splits[t] = split
            hem = np.array([o in set(split.heldout) for o in OWN]) if len(OWN) else np.array([], bool)
            r2 = (np.corrcoef(MU[hem], Y[hem])[0, 1] ** 2) if hem.sum() > 2 else float("nan")
            print(f"  {t:<28} profiles={len(reps[t])}  heldout RT R^2={r2:.3f}")
            continue
        H, rt, owner = extract_task(model, tok, sess, args.max_seq_len)   # probe: fit a Ridge
        if len(rt) < 200:
            continue
        trm = np.array([o in set(split.train) for o in owner])
        mu, sd = H[trm].mean(0), H[trm].std(0) + 1e-6
        Hs = (H - mu) / sd
        rg = Ridge(alpha=args.alpha).fit(Hs[trm], rt[trm])
        resid = rt - rg.predict(Hs)
        z = resid / (resid[trm].std() + 1e-6)
        prof = {}
        for w in set(owner):
            zw = z[owner == w]
            if len(zw) >= 3:
                prof[w] = _residual_rep(zw)
        reps[t] = prof; splits[t] = split
        hem = ~trm
        r2 = (np.corrcoef(rg.predict(Hs[hem]), rt[hem])[0, 1] ** 2) if hem.any() else float("nan")
        print(f"  {t:<28} N={len(rt):6d}  heldout RT R^2={r2:.3f}  profiles={len(prof)}")

    # cross-task identification: map A-residual-rep -> B-residual-rep (train), identify heldout
    T = pd.DataFrame(index=tasks, columns=tasks, dtype=float)
    for a in tasks:
        for b in tasks:
            if a == b or a not in reps or b not in reps:
                continue
            common = set(reps[a]) & set(reps[b])
            tr = [w for w in splits[b].train if w in common]
            he = [w for w in splits[b].heldout if w in common]
            if len(tr) < 30 or len(he) < 20:
                continue
            Xtr = np.stack([reps[a][w] for w in tr]); Ytr = np.stack([reps[b][w] for w in tr])
            Xte = np.stack([reps[a][w] for w in he])
            mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-6
            my, sy = Ytr.mean(0), Ytr.std(0) + 1e-6
            pred = Ridge(alpha=10.0).fit((Xtr - mx) / sx, (Ytr - my) / sy).predict((Xte - mx) / sx)
            Tn = {w: (reps[b][w] - my) / sy for w in he}
            T.loc[a, b] = _identify(pred, Tn, he, args.K, seed)

    within = [T.loc[a, b] for a in tasks for b in tasks
              if a != b and not pd.isna(T.loc[a, b]) and domain[a] == domain[b]]
    across = [T.loc[a, b] for a in tasks for b in tasks
              if a != b and not pd.isna(T.loc[a, b]) and domain[a] != domain[b]]
    out = Path(rdir); out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "rt_head_matrix.csv")
    print(f"\n=== RT-residual transfer (chance {1/args.K:.2f}) ===")
    print(f"within-domain mean top1 = {np.nanmean(within):.3f}  (n={len(within)})")
    print(f"across-domain mean top1 = {np.nanmean(across):.3f}  (n={len(across)})")
    json.dump({"within": float(np.nanmean(within)), "across": float(np.nanmean(across)),
               "chance": 1 / args.K}, open(out / "rt_head_summary.json", "w"), indent=2)
    print(f"saved -> {out / 'rt_head_matrix.csv'}")


if __name__ == "__main__":
    main()
