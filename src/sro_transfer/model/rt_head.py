"""Approach B (trained): a continuous log-RT regression head trained JOINTLY with
the choice loss on top of a QLoRA base.

The head reads the last hidden state at each decision (captured via an lm_head
forward-pre-hook, so no output_hidden_states blow-up) and predicts (mu, log-sigma)
of log-RT. Loss = choice CE + lam * log-normal NLL. LoRA + head are trained
together, so the representation is shaped to encode RT. The individual signal at
eval = the standardised residual (logRT - mu)/sigma per trial.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .masking import build_labels


class RTHead(nn.Module):
    """Predicts mu = E[log-RT] (point estimate). MSE loss -> no sigma collapse."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, h):                                   # h [N,H] -> mu [N]
        return self.net(h).squeeze(-1)


class RTModel(nn.Module):
    """QLoRA base (frozen 4-bit + trainable LoRA) + RT head; joint loss."""

    def __init__(self, base, lam: float = 1.0):
        super().__init__()
        self.base = base
        self.head = RTHead(base.config.hidden_size)
        self.lam = lam
        self._h = None
        lm = base.get_base_model().get_output_embeddings()  # lm_head; its input = last hidden
        lm.register_forward_pre_hook(lambda m, a: setattr(self, "_h", a[0]))

    def gradient_checkpointing_enable(self, **kw):
        self.base.gradient_checkpointing_enable(**kw)

    def forward(self, input_ids, attention_mask=None, labels=None, rt_target=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.base(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        rt_mse = torch.tensor(0.0, device=loss.device)
        if rt_target is not None and self._h is not None:
            m = ~torch.isnan(rt_target)
            if bool(m.any()):
                mu = self.head(self._h[m].float())          # fp32 head, point estimate
                y = rt_target[m].float()
                rt_mse = ((mu - y) ** 2).mean()             # MSE on log-RT: stable, >= 0
                loss = loss + self.lam * rt_mse
        return {"loss": loss, "rt_mse": rt_mse.detach()}


def build_rt_rows(sessions_rtval: dict, tok, max_len: int):
    """One row per session: input_ids/attention_mask/labels + per-token rt_target
    (NaN except at each choice's first token = the trial's log-RT)."""
    rows = []
    for text, rtv in sessions_rtval.values():
        if not rtv:
            continue
        e = build_labels(text, tok, max_len)
        lab = e["labels"]; L = len(lab)
        groups, i = [], 0
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
        tgt = rtv[-m:]                                       # front-truncation -> tail align
        rt_target = [float("nan")] * L
        for gi, p in enumerate(groups):
            v = tgt[gi]
            if v == v:
                rt_target[p] = float(v)
        rows.append({"input_ids": e["input_ids"], "attention_mask": e["attention_mask"],
                     "labels": lab, "rt_target": rt_target})
    return rows


class RTCollator:
    def __init__(self, tok):
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    def __call__(self, feats):
        L = max(len(f["input_ids"]) for f in feats)

        def pad(key, val, dtype=torch.long):
            return torch.tensor([f[key] + [val] * (L - len(f[key])) for f in feats], dtype=dtype)

        return {
            "input_ids": pad("input_ids", self.pad),
            "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", -100),
            "rt_target": pad("rt_target", float("nan"), torch.float32),
        }


def _save(model, tok, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model.base.save_pretrained(str(out_dir))                # LoRA adapter
    tok.save_pretrained(str(out_dir))
    torch.save(model.head.state_dict(), out_dir / "rt_head.pt")


def train_rt_head(cfg, sessions_rtval, out_dir, lam=1.0, epochs=1, batch_size=4,
                  grad_accum=4, lr=1e-4, save_steps=200, max_len=4096, max_steps=0):
    """Manual QLoRA + head training loop (joint choice CE + log-RT NLL)."""
    from torch.utils.data import DataLoader

    from .mpop import load_base_model
    base, tok = load_base_model(cfg, use_liger=False)        # need lm_head input -> no liger
    dev = base.device
    model = RTModel(base, lam)
    model.head.to(dev).float()

    rows = build_rt_rows(sessions_rtval, tok, max_len)
    print(f"RT-head train rows: {len(rows)}")
    loader = DataLoader(rows, batch_size=batch_size, shuffle=True, collate_fn=RTCollator(tok))
    params = [p for p in base.parameters() if p.requires_grad] + list(model.head.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    base.train()

    step = 0
    for ep in range(epochs):
        for bi, batch in enumerate(loader):
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(**batch)
            (out["loss"] / grad_accum).backward()
            if (bi + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    print(f"  ep{ep} step{step}  loss {out['loss'].item():.3f}  rt_mse {out['rt_mse'].item():.3f}")
                if save_steps and step % save_steps == 0:
                    _save(model, tok, out_dir); print(f"  [saved @ step {step}]")
                if max_steps and step >= max_steps:
                    print(f"  [max_steps {max_steps} reached]"); _save(model, tok, out_dir); return out_dir
    _save(model, tok, out_dir)
    print(f"RT-head model saved -> {out_dir}")
    return out_dir


def load_rt_model(cfg, model_dir):
    """Load base + the trained RT LoRA adapter + the RT head, frozen for eval."""
    from .mpop import load_mpop
    model, tok = load_mpop(cfg, model_dir)                   # base + RT adapter, eval/frozen
    head = RTHead(model.config.hidden_size)
    head.load_state_dict(torch.load(Path(model_dir) / "rt_head.pt", weights_only=False))
    head.to(model.device).float().eval()
    return model, tok, head
