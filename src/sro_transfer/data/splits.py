"""Subject-level splits.

Splitting is by *subject*, never by trial: the whole study is about predicting a
held-out person, so no subject may appear in both train and eval. The 522
``complete`` subjects are split into train / heldout. The ``retest`` subjects
form a separate pool used for (a) the reliability ceiling and (b) same-task
time1->time2 identification (the upper bound for cross-task identification).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def _stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


@dataclass
class SubjectSplit:
    train: list[str] = field(default_factory=list)
    heldout: list[str] = field(default_factory=list)
    retest: list[str] = field(default_factory=list)
    unseen: str | None = None

    def summary(self) -> str:
        return (
            f"train={len(self.train)} heldout={len(self.heldout)} "
            f"retest={len(self.retest)} unseen={self.unseen}"
        )


def make_splits(
    complete_subjects: list[str],
    retest_subjects: list[str],
    heldout_frac: float = 0.2,
    seed: int = 0,
    unseen_subject: str | None = None,
) -> SubjectSplit:
    """Deterministic subject-level split.

    Uses a stable per-subject hash so the assignment is reproducible regardless
    of input ordering or platform RNG.
    """
    subs = sorted(set(map(str, complete_subjects)))
    if unseen_subject is not None and unseen_subject in subs:
        subs.remove(unseen_subject)

    # rank by salted hash -> bottom `heldout_frac` go to heldout
    ranked = sorted(subs, key=lambda s: _stable_hash(f"{seed}:{s}"))
    n_held = int(round(heldout_frac * len(ranked)))
    heldout = sorted(ranked[:n_held])
    train = sorted(ranked[n_held:])

    return SubjectSplit(
        train=train,
        heldout=heldout,
        retest=sorted(set(map(str, retest_subjects))),
        unseen=unseen_subject,
    )
