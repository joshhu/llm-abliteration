#!/usr/bin/env python3
"""Summarize a heretic abliteration run from its Optuna journal checkpoint.

Run this INSIDE the heretic container (it needs optuna), from the model work dir:

    python3 tools/summarize.py checkpoints/*.jsonl > results/best-trial.md

It prints a markdown report: baseline refusals (if `run.log` is available),
the Pareto front of completed trials (fewest refusals vs lowest KL divergence),
and the full parameter set of the recommended trial (front index 0).
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import optuna

try:  # optuna >= 4.0 moved the journal backend
    from optuna.storages.journal import JournalFileBackend
except ImportError:  # pragma: no cover - older optuna
    from optuna.storages import JournalFileBackend  # type: ignore


def load_study(journal: Path) -> optuna.study.Study:
    storage = optuna.storages.JournalStorage(JournalFileBackend(str(journal)))
    studies = storage.get_all_studies()
    if not studies:
        raise SystemExit(f"no studies found in {journal}")
    name = studies[0].study_name
    return optuna.load_study(study_name=name, storage=storage)


def pareto_front(trials: list[optuna.trial.FrozenTrial]) -> list[optuna.trial.FrozenTrial]:
    """Same ordering heretic uses for its trial menu: sort by (refusals, kl), keep
    trials with strictly decreasing KL divergence."""
    done = [
        t
        for t in trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and "refusals" in t.user_attrs
        and "kl_divergence" in t.user_attrs
    ]
    done.sort(key=lambda t: (t.user_attrs["refusals"], t.user_attrs["kl_divergence"]))
    front: list[optuna.trial.FrozenTrial] = []
    best_kl = math.inf
    for t in done:
        kl = t.user_attrs["kl_divergence"]
        if kl < best_kl:
            best_kl = kl
            front.append(t)
    return front


def baseline_from_log(log: Path) -> str | None:
    if not log.exists():
        return None
    text = log.read_text(errors="ignore")
    m = re.search(r"Initial refusals:\s*(\d+)/(\d+)", text)
    return f"{m.group(1)}/{m.group(2)}" if m else None


def main() -> None:
    journal = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints")
    if journal.is_dir():
        candidates = sorted(journal.glob("*.jsonl"))
        if not candidates:
            raise SystemExit(f"no .jsonl journal under {journal}")
        journal = candidates[0]

    study = load_study(journal)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    front = pareto_front(completed)

    print("# Best trial so far\n")
    print(f"- Study: `{study.study_name}`")
    print(f"- Journal: `{journal.name}`")
    print(f"- Trials completed: **{len(completed)}** / {len(study.trials)} recorded")
    base = baseline_from_log(journal.parent.parent / "run.log")
    if base:
        print(f"- Baseline refusals of the original model: **{base}**")
    print()

    if not front:
        print("_No completed trials yet._")
        return

    print("## Pareto front (sorted by refusals, then KL divergence)\n")
    print("| # | trial | refusals | KL divergence | direction_index |")
    print("|---:|---:|---:|---:|---|")
    for i, t in enumerate(front):
        print(
            f"| {i} | {t.user_attrs.get('index', t.number)} "
            f"| {t.user_attrs['refusals']}/{t.user_attrs.get('n_bad_prompts', '?')} "
            f"| {t.user_attrs['kl_divergence']:.4f} "
            f"| `{t.params.get('direction_index', '?')}` |"
        )
    print()

    best = front[0]
    print("## Recommended trial (front index 0)\n")
    print("```text")
    for key, value in best.params.items():
        print(f"* {key} = {value}")
    print("```")
    print()
    print(
        f"Refusals **{best.user_attrs['refusals']}/{best.user_attrs.get('n_bad_prompts', '?')}**, "
        f"KL divergence **{best.user_attrs['kl_divergence']:.4f}**."
    )


if __name__ == "__main__":
    main()
