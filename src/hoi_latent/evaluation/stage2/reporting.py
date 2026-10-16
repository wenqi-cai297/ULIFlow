from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hoi_latent.config.io import load_yaml


@dataclass(frozen=True)
class Stage2MetricRepeatSummary:
    metric: str
    values: list[float]
    value_mean: float
    value_std: float
    value_ci95: float
    num_repeats: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage2RepeatReport:
    run_dirs: list[str]
    seeds: list[int | None]
    num_repeats: int
    metrics: dict[str, Stage2MetricRepeatSummary]
    all_interact_comparable: bool
    blocking_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dirs": self.run_dirs,
            "seeds": self.seeds,
            "num_repeats": self.num_repeats,
            "metrics": {key: value.to_dict() for key, value in self.metrics.items()},
            "all_interact_comparable": self.all_interact_comparable,
            "blocking_reasons": self.blocking_reasons,
        }


def summarize_stage2_repeat_runs(run_dirs: Sequence[str | Path]) -> Stage2RepeatReport:
    roots = [Path(path) for path in run_dirs]
    if not roots:
        raise ValueError("run_dirs must contain at least one evaluation run.")
    summaries = [_read_summary(root) for root in roots]
    seeds = [_read_seed(root) for root in roots]
    metric_values: dict[str, list[float]] = {}
    blocking: list[str] = []
    all_comparable = True
    for root, summary in zip(roots, summaries):
        metrics = summary.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{root / 'metrics' / 'summary.json'} has no metrics mapping.")
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                metric_values.setdefault(str(key), []).append(float(value))
        comparability = summary.get("comparability", {})
        comparable = bool(isinstance(comparability, Mapping) and comparability.get("interact_comparable", False))
        all_comparable = all_comparable and comparable
        if isinstance(comparability, Mapping):
            for reason in comparability.get("blocking_reasons", []):
                blocking.append(str(reason))
        elif not comparable:
            blocking.append("missing_comparability")

    num_repeats = len(roots)
    table = {
        key: _summarize_metric(key, values, num_repeats=num_repeats)
        for key, values in sorted(metric_values.items())
        if len(values) == num_repeats
    }
    return Stage2RepeatReport(
        run_dirs=[str(root) for root in roots],
        seeds=seeds,
        num_repeats=num_repeats,
        metrics=table,
        all_interact_comparable=bool(all_comparable),
        blocking_reasons=sorted(set(blocking)),
    )


def write_stage2_repeat_report(report: Stage2RepeatReport, out_dir: str | Path) -> None:
    root = Path(out_dir)
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    (metrics_dir / "report_table.json").write_text(
        json.dumps(payload["metrics"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (metrics_dir / "report_table.md").write_text(_repeat_report_markdown(report), encoding="utf-8")
    (metrics_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_repeats": report.num_repeats,
                "seeds": report.seeds,
                "run_dirs": report.run_dirs,
                "all_interact_comparable": report.all_interact_comparable,
                "blocking_reasons": report.blocking_reasons,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _summarize_metric(key: str, values: Sequence[float], *, num_repeats: int) -> Stage2MetricRepeatSummary:
    arr = np.asarray(values, dtype=np.float64)
    # InterAct reports use numpy's default population std (ddof=0).
    std = float(np.std(arr)) if arr.size > 1 else 0.0
    return Stage2MetricRepeatSummary(
        metric=str(key),
        values=[float(v) for v in arr.tolist()],
        value_mean=float(np.mean(arr)),
        value_std=std,
        value_ci95=float(1.96 * std / math.sqrt(max(1, int(num_repeats)))),
        num_repeats=int(num_repeats),
    )


def _read_summary(run_dir: Path) -> Mapping[str, Any]:
    path = run_dir / "metrics" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing Stage2 evaluation summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_seed(run_dir: Path) -> int | None:
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.exists():
        return None
    config = load_yaml(config_path)
    try:
        return int(config["evaluation"]["stage2"]["seed"])
    except Exception:
        return None


def _repeat_report_markdown(report: Stage2RepeatReport) -> str:
    lines = [
        "# Stage2 Repeat Report",
        "",
        f"- num_repeats: {report.num_repeats}",
        f"- seeds: {report.seeds}",
        f"- all_interact_comparable: {report.all_interact_comparable}",
        f"- blocking_reasons: {report.blocking_reasons}",
        "",
        "| metric | mean | ci95 | std | values |",
        "|---|---:|---:|---:|---|",
    ]
    for key, item in report.metrics.items():
        values = ", ".join(f"{value:.6g}" for value in item.values)
        lines.append(
            f"| `{key}` | {item.value_mean:.6g} | {item.value_ci95:.6g} | "
            f"{item.value_std:.6g} | {values} |"
        )
    return "\n".join(lines) + "\n"
