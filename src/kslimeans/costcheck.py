"""Check the cost counters of a logged run against a rebuild from its inputs (#35, part 1).

A run logs, per summary step and per run, how many summarizer calls it made, how many tokens went in and out,
and how long `summarize_batch` took (`core.COST_KEYS`, `n_calls`). This module recomputes what it can from the
logged partitions and summaries, with no model and no GPU, and compares:

- **calls**: the clusters `runlog.sent_clusters` says were sent (non-empty, not memo hits) against the logged
  `n_calls`; the trace rows against the per-run totals in `metrics.jsonl`.
- **input tokens, truncated clusters, dropped texts**: each sent prompt is rebuilt through the same functions
  `HFSummarizer` uses (`hf.fit_and_render`) and re-tokenized with the model's tokenizer.
- **output tokens**: each logged summary is re-tokenized. The logged count is "up to and including the first
  EOS" over the raw generation, which is not recoverable from the decoded, stripped text, so the comparison is
  an expectation: re-tokenized length plus one EOS per summary (see `StepCheck.output_model`).
- **seconds**: the trace rows against the per-run total, and `summarizer_seconds <= runtime_s`.

Which trace row belongs to summary step `j` follows `runlog`'s indexing: the fit after step `j` is row
`summary_source_steps[j] + 1`. A final step that stopped on `tol` has no row; its counters are only in the
per-run totals.

What `summarizer_seconds` brackets is documented in `docs/results/cost_verification.md`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .core import COST_KEYS
from .runlog import RunLog, RunRecord, load_run_dataset, sent_clusters, step_inputs
from .summarizers.hf import (
    DEFAULT_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEXT_TYPE,
    fit_and_render,
    supports_system_role,
)

SECONDS_TOL = 1e-6
"""Float slack when comparing sums of per-step seconds with the per-run total."""

TOKEN_KEYS = ("input_tokens", "n_truncated_clusters", "n_dropped_texts")
"""The `COUNT_KEYS` that need a tokenizer to rebuild."""

COUNT_KEYS = ("n_calls", "input_tokens", "n_truncated_clusters", "n_dropped_texts", "n_empty_summaries")
"""Counters this module recomputes exactly (everything in `COST_KEYS` + `n_calls` except seconds and outputs)."""


@dataclass(frozen=True)
class PromptSettings:
    """The `HFSummarizer` settings that decide what a prompt looks like and how long it may be."""

    prompt: str
    text_type: str
    system_prompt: str
    max_input_tokens: int
    max_new_tokens: int
    chat_template_kwargs: dict
    max_words_per_text: int | None = None

    @classmethod
    def from_run(cls, summarizer_cfg: dict, dataset_prompt: str, dataset_text_type: str) -> PromptSettings:
        """Settings as `experiment.run_offline` + `build_summarizer` + `HFSummarizer.__init__` resolve them."""
        kwargs = summarizer_cfg.get("chat_template_kwargs")
        return cls(
            prompt=summarizer_cfg.get("prompt") or dataset_prompt or DEFAULT_PROMPT,
            text_type=summarizer_cfg.get("text_type") or dataset_text_type or DEFAULT_TEXT_TYPE,
            system_prompt=summarizer_cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            max_input_tokens=summarizer_cfg.get("max_input_tokens", 4096),
            max_new_tokens=summarizer_cfg.get("max_new_tokens", 128),
            chat_template_kwargs={"enable_thinking": False} if kwargs is None else kwargs,
            max_words_per_text=summarizer_cfg.get("max_words_per_text"),
        )


def embed_keys(row: dict) -> list[str]:
    """The embedder's optional `embed_*` counters in a trace or metrics row (e.g. TF-IDF's OOV tokens).

    Runs made before #54, and runs with an embedder that reports none, have no such keys.
    """
    return sorted(k for k, v in row.items() if k.startswith("embed_") and isinstance(v, (int, float)))


def count_tokens(tokenizer, text: str) -> int:
    """Tokens of `text` as `HFSummarizer` counts them: no special tokens added."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


@dataclass
class PromptCount:
    input_tokens: int
    n_dropped: int
    truncated: bool


def rebuild_prompt(tokenizer, settings: PromptSettings, texts: list[str], fold_system: bool) -> PromptCount:
    """Rebuild one cluster's prompt as `HFSummarizer._render` does and count its tokens."""
    rendered, n_dropped, truncated = fit_and_render(
        tokenizer,
        texts,
        system_prompt=settings.system_prompt,
        prompt=settings.prompt,
        text_type=settings.text_type,
        chat_template_kwargs=settings.chat_template_kwargs,
        fold_system=fold_system,
        max_input_tokens=settings.max_input_tokens,
        max_words_per_text=settings.max_words_per_text,
    )
    return PromptCount(count_tokens(tokenizer, rendered), n_dropped, truncated)


@dataclass
class StepCheck:
    """One summary step: the counters rebuilt from the log and, when the step has a trace row, the logged ones."""

    seed: int
    step: int
    logged: dict | None
    """The trace row's `n_calls` + `COST_KEYS`, or `None` for a final step that stopped on `tol`."""
    counts: dict
    """Rebuilt `COUNT_KEYS`; token and truncation entries are 0 for a summarizer without prompts (LSA)."""
    output_retokenized: int = 0
    """Sum over sent clusters of the re-tokenized logged summary."""
    output_capped: int = 0
    """Sent summaries whose re-tokenized length reaches `max_new_tokens - 1`: likely stopped by the cap, not an EOS."""

    @property
    def n_sent(self) -> int:
        return self.counts["n_calls"]

    @property
    def output_model(self) -> int:
        """Expected logged `output_tokens`: the re-tokenized summaries plus one EOS for each."""
        return self.output_retokenized + self.n_sent


@dataclass
class RecordCheck:
    """All summary steps of one (method, seed), plus the per-run totals from `metrics.jsonl`."""

    method: str
    seed: int
    steps: list[StepCheck]
    metrics: dict
    with_tokens: bool
    rebuild_seconds: float = 0.0
    """CPU seconds this module spent rebuilding and tokenizing the prompts (not part of any logged number)."""


def check_record(
    record: RunRecord,
    texts: list[str],
    X: np.ndarray,
    force_context_length: int,
    tokenizer=None,
    settings: PromptSettings | None = None,
) -> RecordCheck:
    """Rebuild the counters of every summary step of `record`.

    Without `tokenizer` and `settings` (an extractive summarizer) only the call and empty-summary counters are
    rebuilt. `X` and `force_context_length` are those of `runlog.step_inputs`.
    """
    with_tokens = tokenizer is not None and settings is not None
    fold_system = with_tokens and not supports_system_role(tokenizer, settings.chat_template_kwargs)
    rows = {row["step"]: row for row in record.trace}
    steps = []
    rebuild_seconds = 0.0
    for j in range(record.n_summary_steps):
        sent = sent_clusters(record, j, texts, X, force_context_length)
        inputs = step_inputs(record, j, texts, X, force_context_length)
        summaries = record.summaries_evolution[j]
        counts = dict.fromkeys(COUNT_KEYS, 0)
        counts["n_calls"] = len(sent)
        counts["n_empty_summaries"] = sum(1 for c in sent if not summaries[c].strip())
        out_tokens = capped = 0
        if with_tokens:
            start = time.perf_counter()
            for c in sent:
                prompt = rebuild_prompt(tokenizer, settings, inputs[c].texts, fold_system)
                counts["input_tokens"] += prompt.input_tokens
                counts["n_dropped_texts"] += prompt.n_dropped
                counts["n_truncated_clusters"] += int(bool(prompt.n_dropped or prompt.truncated))
                n_out = count_tokens(tokenizer, summaries[c])
                out_tokens += n_out
                capped += int(n_out >= settings.max_new_tokens - 1)
            rebuild_seconds += time.perf_counter() - start
        row = rows.get(record.summary_source_steps[j] + 1)
        logged = None if row is None else {key: row[key] for key in ("n_calls", *COST_KEYS, *embed_keys(row))}
        steps.append(StepCheck(record.seed, j, logged, counts, out_tokens, capped))
    return RecordCheck(record.method, record.seed, steps, record.metrics, with_tokens, rebuild_seconds)


@dataclass
class Check:
    """One pass/fail line of the report: `n_failed` of `n` comparisons differ, the largest by `max_abs`."""

    name: str
    n: int = 0
    n_failed: int = 0
    max_abs: float = 0.0
    note: str = ""
    informational: bool = False
    """Reported but not a test: the discrepancy is expected to be non-zero."""

    @property
    def passed(self) -> bool:
        return self.n_failed == 0

    def compare(self, a: float, b: float, tol: float = 0.0) -> None:
        diff = abs(a - b)
        self.n += 1
        self.max_abs = max(self.max_abs, diff)
        self.n_failed += int(diff > tol)


def _summary_steps_with_rows(rc: RecordCheck) -> list[StepCheck]:
    return [s for s in rc.steps if s.logged is not None]


def run_checks(checks: list[RecordCheck], output_tol: float = 0.01) -> list[Check]:
    """The pass/fail checks over every record (seed) of one method in one run.

    `output_tol` is the relative tolerance of the output-token comparison per record (the other counters must
    match exactly; seconds within `SECONDS_TOL`).
    """
    out: list[Check] = []

    def per_step(key: str, label: str) -> None:
        check = Check(f"per step: {label}")
        for rc in checks:
            for s in _summary_steps_with_rows(rc):
                check.compare(s.logged[key], s.counts[key])
        out.append(check)

    per_step("n_calls", "logged `n_calls` = clusters sent")
    per_step("n_empty_summaries", "logged `n_empty_summaries` = blank summaries sent")
    if checks and checks[0].with_tokens:
        per_step("input_tokens", "logged `input_tokens` = re-tokenized prompts")
        per_step("n_truncated_clusters", "logged `n_truncated_clusters` = rebuilt trims")
        per_step("n_dropped_texts", "logged `n_dropped_texts` = rebuilt trims")

    # Per run: the metrics row is the truth the tables use. It must equal the rebuild over ALL summary steps,
    # including a final step that stopped on `tol`, and the trace rows must sum to it except for that step.
    with_tokens = bool(checks) and checks[0].with_tokens
    for key in COUNT_KEYS:
        if key in TOKEN_KEYS and not with_tokens:
            continue
        check = Check(f"per run: `{key}` in metrics = rebuild over all steps")
        for rc in checks:
            check.compare(rc.metrics[key], sum(s.counts[key] for s in rc.steps))
        out.append(check)

    extra = sorted({k for rc in checks for k in embed_keys(rc.metrics)})
    for key in ("n_calls", *COST_KEYS, *extra):
        check = Check(f"per run: `{key}` in metrics = sum of trace rows")
        n_tol = 0
        for rc in checks:
            tol = SECONDS_TOL if key == "summarizer_seconds" else 0
            rowless = [s for s in rc.steps if s.logged is None]
            total = rc.metrics.get(key, 0)
            traced = sum(s.logged.get(key, 0) for s in _summary_steps_with_rows(rc))
            if rowless:
                n_tol += 1
                # The row-less step is in the total only: the total must exceed the trace by what it added.
                if key in COUNT_KEYS:
                    check.compare(total - traced, sum(s.counts[key] for s in rowless))
                else:
                    check.n += 1
                    check.n_failed += int(total - traced < -tol)
            else:
                check.compare(total, traced, tol)
        if n_tol:
            check.note = f"{n_tol} record(s) ended on `tol`: the stopped step is in the total only"
        out.append(check)

    seconds = Check("per run: `summarizer_seconds` <= `runtime_s`")
    for rc in checks:
        seconds.n += 1
        seconds.n_failed += int(rc.metrics["summarizer_seconds"] > rc.metrics["runtime_s"])
        seconds.max_abs = max(seconds.max_abs, rc.metrics["summarizer_seconds"] - rc.metrics["runtime_s"])
    out.append(seconds)

    if checks and checks[0].with_tokens:
        total = Check(f"per run: `output_tokens` in metrics ~ re-tokenized summaries + EOS (rel. tol {output_tol:.1%})")
        step = Check(
            "per step: logged `output_tokens` vs the same expectation (absolute difference in tokens)",
            informational=True,
        )
        for rc in checks:
            logged = rc.metrics["output_tokens"]
            expected = sum(s.output_model for s in rc.steps)
            total.n += 1
            total.max_abs = max(total.max_abs, abs(logged - expected) / max(logged, 1))
            total.n_failed += int(abs(logged - expected) > output_tol * max(logged, 1))
            for s in _summary_steps_with_rows(rc):
                step.n += 1
                step.max_abs = max(step.max_abs, abs(s.logged["output_tokens"] - s.output_model))
        total.note = "max_abs is a relative difference"
        out.extend([total, step])
    return out


@dataclass
class MethodReport:
    """The checks of one method (all its seeds) in one run directory."""

    method: str
    checks: list[Check]
    records: list[RecordCheck] = field(default_factory=list)


@dataclass
class RunReport:
    path: str
    name: str
    model: str | None
    batch_size: int | None
    max_new_tokens: int | None
    methods: list[MethodReport]
    seconds: float = 0.0
    """Wall seconds `verify_run` took, tokenization included."""


def verify_run(
    runlog: RunLog,
    cache_dir=None,
    load_tokenizer=None,
    output_tol: float = 0.01,
) -> RunReport:
    """Run every check on every summary method of `runlog`.

    `load_tokenizer(model_id)` supplies the tokenizer of an `hf` summarizer (the script loads it from the local
    Hugging Face cache only); it is not called for extractive summarizers. Dataset and embeddings come from the
    cache through `runlog.load_run_dataset`, which never encodes.
    """
    start = time.perf_counter()
    cfg = runlog.config
    scfg = cfg["summarizer"]
    ds, X = load_run_dataset(runlog, cache_dir)
    tokenizer = settings = None
    if scfg.get("type") == "hf":
        tokenizer = load_tokenizer(scfg["model_id"])
        settings = PromptSettings.from_run(scfg, ds.prompt, ds.text_type)
    fcl = cfg.get("force_context_length", 0)
    methods = []
    for method in runlog.methods:
        records = [r for r in runlog.method_records(method) if r.summaries_evolution]
        if not records:
            continue
        rcs = [check_record(r, ds.texts, X, fcl, tokenizer, settings) for r in records]
        methods.append(MethodReport(method, run_checks(rcs, output_tol), rcs))
    return RunReport(
        path=str(runlog.path),
        name=cfg["name"],
        model=scfg.get("model_id"),
        batch_size=scfg.get("batch_size"),
        max_new_tokens=scfg.get("max_new_tokens"),
        methods=methods,
        seconds=time.perf_counter() - start,
    )


def _fmt(x: float) -> str:
    return f"{x:.3g}" if x != int(x) or abs(x) < 1e-3 else f"{int(x)}"


def format_report(reports: list[RunReport]) -> str:
    """Markdown: one pass/fail table per run, then per-seed token totals for runs with a tokenizer."""
    lines = ["# Cost verification report", ""]
    for rep in reports:
        label = f"{rep.name} / {rep.path.rstrip('/').rsplit('/', 1)[-1]}"
        lines += [f"## {label}", ""]
        if rep.model:
            lines += [f"Model `{rep.model}`, batch size {rep.batch_size}, `max_new_tokens` {rep.max_new_tokens}.", ""]
        for mrep in rep.methods:
            n_seeds = len(mrep.records)
            lines += [
                f"### {mrep.method} ({n_seeds} seed{'s' if n_seeds != 1 else ''})",
                "",
                "| Check | Result | Compared | Failed | Max discrepancy | Note |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
            for c in mrep.checks:
                result = "INFO" if c.informational else "PASS" if c.passed else "FAIL"
                lines.append(f"| {c.name} | {result} | {c.n} | {c.n_failed} | {_fmt(c.max_abs)} | {c.note} |")
            lines.append("")
            if mrep.records and mrep.records[0].with_tokens:
                lines += [
                    "| seed | logged input | rebuilt input | logged output | re-tokenized output + EOS | "
                    "difference | summaries | likely capped |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
                for rc in mrep.records:
                    model = sum(s.output_model for s in rc.steps)
                    sent = sum(s.n_sent for s in rc.steps)
                    capped = sum(s.output_capped for s in rc.steps)
                    rebuilt = sum(s.counts["input_tokens"] for s in rc.steps)
                    lines.append(
                        f"| {rc.seed} | {rc.metrics['input_tokens']} | {rebuilt} "
                        f"| {rc.metrics['output_tokens']} | {model} | {rc.metrics['output_tokens'] - model:+d} "
                        f"| {sent} | {capped} |"
                    )
                lines.append("")
            tokenized = mrep.records[0].with_tokens
            head = "| seed | `summarizer_seconds` | `runtime_s` | runtime - summarizer (clustering, embedding, other) |"
            rule = "| ---: | ---: | ---: | ---: |"
            if tokenized:
                head += " rendering + tokenizing the prompts here (CPU s) |"
                rule += " ---: |"
            lines += [head, rule]
            for rc in mrep.records:
                s, t = rc.metrics["summarizer_seconds"], rc.metrics["runtime_s"]
                row = f"| {rc.seed} | {s:.1f} | {t:.1f} | {t - s:.1f} |"
                lines.append(row + (f" {rc.rebuild_seconds:.1f} |" if tokenized else ""))
            lines.append("")
        lines += [f"Verification time for this run: {rep.seconds:.1f} s.", ""]
    return "\n".join(lines)
