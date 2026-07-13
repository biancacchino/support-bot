"""Phase 11: measure retrieval, reranking and the confidence gate.

    python scripts/eval.py                  # score, print, write docs/benchmark.md
    python scripts/eval.py --no-report      # score and print only

Runs the frozen eval set (`eval/queries.jsonl`) through the same Retriever and
Reranker the bot uses, against an in-memory Qdrant built from the real KB. It
never calls Gemini: everything measured here - what is retrieved, how it ranks,
and whether the gate escalates - is decided before the LLM is reached. An eval
that burned quota to measure retrieval would be measuring the wrong thing anyway.

Ground truth is the `intents:` frontmatter in kb/, not a hand-written mapping. A
query labelled `track_order` is correct if it retrieves a document that claims
`track_order`. Nothing to keep in sync, and `scripts/check_kb_coverage.py`
already fails the build if that frontmatter is wrong.

The two numbers that matter are in tension, and that is the point:

- **deflection** - an in-scope query answered with the right document in front of
  the model. Every in-scope query the gate escalates is deflection the product
  does not get.
- **false answers** - a query answered with no correct document in front of the
  model, or a negative answered at all. The PRD caps this at 2%, and it is what a
  lower confidence threshold buys deflection with.

Both are printed at every threshold in the sweep, because a threshold chosen
without seeing what it costs is how Phase 3 arrived at a number that turned out to
be tuned on 17 queries and wrong on 320.
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import AsyncQdrantClient, models  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.ingestion import load_chunks, load_embedder  # noqa: E402
from app.reranker import Reranker, build_cross_encoder  # noqa: E402
from app.retrieval import Retriever, build_encoder  # noqa: E402
from check_kb_coverage import load_docs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "eval" / "queries.jsonl"
REPORT = ROOT / "docs" / "benchmark.md"
COLLECTION = "eval_kb"

SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]


@dataclass
class Scored:
    """One query, run through the pipeline, with the gate left undecided.

    `confidence` is kept rather than a boolean escalate/answer, so the threshold
    can be swept afterwards without re-running the models.
    """

    kind: str
    intent: str
    expected: set[str]
    stage1_docs: list[str]  # doc_ids, best first, deduped
    stage2_docs: list[str]
    stage2_chunk_docs: list[str]  # one per surviving chunk, for precision@k
    confidence: float

    @property
    def top_doc(self) -> str | None:
        return self.stage2_docs[0] if self.stage2_docs else None

    def top1_correct(self) -> bool:
        """The best chunk came from a document that covers the intent."""
        return self.top_doc in self.expected

    def grounded(self) -> bool:
        """A document covering the intent survived into the top-k handed to the LLM.

        This, not top-1, is what the pipeline actually does: all `rerank_top_k`
        chunks go into the prompt and the model may cite any of them. Scoring on
        top-1 alone would count "the right document ranked 2nd and was cited" as a
        false answer, which is not a failure the customer would ever see.

        Both are reported. top-1 is the honest ceiling on ranking quality; this is
        the honest measure of what reaches a customer.
        """
        return bool(set(self.stage2_docs) & self.expected)


def dedupe(doc_ids: list[str]) -> list[str]:
    """Chunk ranks are not document ranks. Four chunks of one document is one hit."""
    return list(dict.fromkeys(doc_ids))


def reciprocal_rank(ranked: list[str], expected: set[str]) -> float:
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in expected:
            return 1.0 / i
    return 0.0


def ground_truth() -> dict[str, set[str]]:
    """intent -> the documents that claim it."""
    mapping: dict[str, set[str]] = defaultdict(set)
    for _, fm in load_docs():
        for intent in fm.get("intents") or []:
            mapping[intent].add(fm["doc_id"])
    return mapping


async def build_pipeline(settings) -> tuple[Retriever, Reranker]:
    """The real models over the real corpus, in an in-memory Qdrant."""
    model = load_embedder(settings)

    def count_tokens(text: str) -> int:
        return len(model.tokenizer.tokenize(text))

    chunks = load_chunks(
        Path(settings.kb_dir), count_tokens, settings.chunk_size_tokens,
        settings.chunk_overlap_tokens,
    )
    vectors = model.encode(
        [c.embed_text for c in chunks], normalize_embeddings=True, show_progress_bar=False
    )

    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=len(vectors[0]), distance=models.Distance.COSINE
        ),
    )
    await client.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(id=c.point_id, vector=v.tolist(), payload=c.payload())
            for c, v in zip(chunks, vectors, strict=True)
        ],
        wait=True,
    )

    retriever = Retriever(
        client, build_encoder(settings), COLLECTION, settings.retrieval_top_n
    )
    # threshold=0: the gate is applied later, in the sweep. Scoring and deciding are
    # separate steps, so deciding again is free.
    reranker = Reranker(build_cross_encoder(settings), settings.rerank_top_k, 0.0)
    return retriever, reranker


async def score_all(retriever: Retriever, reranker: Reranker, truth: dict) -> list[Scored]:
    records = [json.loads(line) for line in QUERIES.read_text().splitlines()]
    scored: list[Scored] = []

    for i, record in enumerate(records, start=1):
        candidates = await retriever.search(record["query"])
        result = await reranker.rerank(record["query"], candidates)

        scored.append(
            Scored(
                kind=record["kind"],
                intent=record["intent"],
                expected=truth.get(record["intent"], set()),
                stage1_docs=dedupe([c.doc_id for c in candidates]),
                stage2_docs=dedupe([r.doc_id for r in result.ranked]),
                stage2_chunk_docs=[r.doc_id for r in result.ranked],
                confidence=result.confidence,
            )
        )
        if i % 40 == 0:
            print(f"  scored {i}/{len(records)}", flush=True)

    return scored


def retrieval_metrics(scored: list[Scored]) -> dict:
    """Recall, precision and MRR - stage 1 alone, then after reranking.

    Reported side by side because that comparison is the entire justification for
    running a cross-encoder at all. If reranking does not move MRR, it is a second
    model's worth of latency buying nothing.
    """
    in_scope = [s for s in scored if s.kind == "in_scope"]
    n = len(in_scope)

    return {
        "queries": n,
        "stage1_recall_at_10": sum(bool(set(s.stage1_docs) & s.expected) for s in in_scope) / n,
        "stage1_mrr": sum(reciprocal_rank(s.stage1_docs, s.expected) for s in in_scope) / n,
        "stage2_recall_at_4": sum(bool(set(s.stage2_docs) & s.expected) for s in in_scope) / n,
        "stage2_mrr": sum(reciprocal_rank(s.stage2_docs, s.expected) for s in in_scope) / n,
        "stage2_precision_at_4": sum(
            sum(d in s.expected for d in s.stage2_chunk_docs) / len(s.stage2_chunk_docs)
            for s in in_scope if s.stage2_chunk_docs
        ) / n,
        "top1_accuracy": sum(s.top1_correct() for s in in_scope) / n,
    }


def gate_metrics(scored: list[Scored], threshold: float) -> dict:
    """What the confidence gate does at one threshold.

    A false answer is either kind of confidently-wrong: an in-scope query answered
    with no document covering its intent in front of the model, or a negative
    answered at all. Both reach the customer as a cited, confident answer that is
    not true, which is the failure the whole system is built to prevent.

    Scored on `grounded()` (the right doc reached the LLM) rather than on top-1,
    because that is what the pipeline hands to generation. The stricter top-1
    numbers are reported alongside, since they are the ceiling on ranking quality
    and the honest thing to quote about the reranker.
    """
    in_scope = [s for s in scored if s.kind == "in_scope"]
    negatives = [s for s in scored if s.kind != "in_scope"]

    answered = [s for s in in_scope if s.confidence >= threshold]
    deflected = [s for s in answered if s.grounded()]
    ungrounded = [s for s in answered if not s.grounded()]
    leaked = [s for s in negatives if s.confidence >= threshold]

    total = len(scored)
    return {
        "threshold": threshold,
        "deflection_rate": len(deflected) / len(in_scope),
        "deflection_top1": sum(s.top1_correct() for s in answered) / len(in_scope),
        "escalated_in_scope": 1 - len(answered) / len(in_scope),
        "leak_rate": len(leaked) / len(negatives),
        "false_answer_rate": (len(ungrounded) + len(leaked)) / total,
        "false_answers": len(ungrounded) + len(leaked),
        "ungrounded": len(ungrounded),
        "leaked": len(leaked),
    }


def by_intent(scored: list[Scored], threshold: float) -> dict:
    out = {}
    for intent in sorted({s.intent for s in scored if s.kind == "in_scope"}):
        rows = [s for s in scored if s.kind == "in_scope" and s.intent == intent]
        answered = [s for s in rows if s.confidence >= threshold]
        out[intent] = {
            "queries": len(rows),
            "deflected": sum(s.grounded() for s in answered),
            "recall_at_4": sum(bool(set(s.stage2_docs) & s.expected) for s in rows) / len(rows),
            "mrr": sum(reciprocal_rank(s.stage2_docs, s.expected) for s in rows) / len(rows),
        }
    return out


def pick_threshold(scored: list[Scored], cap: float = 0.02) -> tuple[float, dict]:
    """The threshold that deflects most while keeping false answers under the cap.

    The PRD's targets are >=40% deflection and <2% false answers, and the second is
    the binding one: it is a promise to a customer, and deflection is a promise to
    a budget.
    """
    best = None
    for threshold in SWEEP:
        row = gate_metrics(scored, threshold)
        if row["false_answer_rate"] <= cap and (
            best is None or row["deflection_rate"] > best["deflection_rate"]
        ):
            best = row

    if best is None:
        # No threshold meets the cap. Return the safest one, flagged - a number that
        # merely looks like a winner is how a benchmark starts lying to the person
        # reading it.
        best = min(
            (gate_metrics(scored, t) for t in SWEEP), key=lambda r: r["false_answer_rate"]
        )
        best["meets_cap"] = False
    else:
        best["meets_cap"] = True

    return best["threshold"], best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    settings = settings.model_copy(update={"kb_dir": str(ROOT / "kb")})

    truth = ground_truth()
    retriever, reranker = asyncio.run(build_pipeline(settings))
    scored = asyncio.run(score_all(retriever, reranker, truth))

    retrieval = retrieval_metrics(scored)
    current = gate_metrics(scored, settings.confidence_threshold)
    best_threshold, best = pick_threshold(scored)

    print("\nRetrieval (220 in-scope queries)")
    print(f"  stage 1  recall@10 {retrieval['stage1_recall_at_10']:.3f}   MRR {retrieval['stage1_mrr']:.3f}")
    print(f"  stage 2  recall@4  {retrieval['stage2_recall_at_4']:.3f}   MRR {retrieval['stage2_mrr']:.3f}"
          f"   P@4 {retrieval['stage2_precision_at_4']:.3f}   top-1 {retrieval['top1_accuracy']:.3f}")

    print(f"\nGate at the shipped threshold ({settings.confidence_threshold})")
    print(f"  deflection {current['deflection_rate']:.3f}   false answers "
          f"{current['false_answer_rate']:.3f} ({current['false_answers']}/320)   "
          f"negatives leaked {current['leaked']}/100")

    print("\nThreshold sweep")
    print(f"  {'thr':>5}  {'deflect':>8}  {'false':>7}  {'ungrounded':>10}  {'leaked':>7}")
    for threshold in SWEEP:
        row = gate_metrics(scored, threshold)
        mark = ""
        if threshold == best_threshold:
            mark = (
                " <-- best under the 2% cap" if best["meets_cap"]
                else " <-- safest available; NO threshold meets the 2% cap"
            )
        print(f"  {threshold:>5.2f}  {row['deflection_rate']:>8.3f}  "
              f"{row['false_answer_rate']:>7.3f}  {row['ungrounded']:>10}  {row['leaked']:>7}{mark}")

    if not args.no_report:
        write_report(retrieval, current, best, scored, settings)
        print(f"\nwrote {REPORT.relative_to(ROOT)}")
    return 0


def _verdict(best: dict) -> str:
    if best["meets_cap"]:
        return (
            f"Best deflection under the 2% false-answer cap: **{best['threshold']}** "
            f"({best['deflection_rate']:.1%} deflection)."
        )
    return (
        f"**No threshold in the sweep meets the 2% false-answer cap.** "
        f"The safest available is **{best['threshold']}**, still at "
        f"{best['false_answer_rate']:.1%} false answers and only "
        f"{best['deflection_rate']:.1%} deflection.\n\n"
        "Tuning cannot fix this. The threshold trades deflection against false answers "
        "along one axis, and both targets are missed at every point on it - so the "
        "problem is upstream of the gate, in retrieval and in what the KB covers."
    )


def write_report(retrieval, current, best, scored, settings) -> None:
    rows = "\n".join(
        f"| {t:.2f} | {r['deflection_rate']:.1%} | {r['false_answer_rate']:.1%} "
        f"| {r['ungrounded']} | {r['leaked']} |"
        for t in SWEEP
        for r in [gate_metrics(scored, t)]
    )
    cats = "\n".join(
        f"| `{intent}` | {m['queries']} | {m['recall_at_4']:.0%} | {m['mrr']:.2f} | {m['deflected']} |"
        for intent, m in sorted(
            by_intent(scored, best["threshold"]).items(),
            key=lambda kv: kv[1]["mrr"],
        )
    )

    REPORT.write_text(f"""<!-- Generated by scripts/eval.py. Do not edit by hand; re-run it. -->
# Retrieval Benchmark (Phase 11)

320 queries sampled from the Bitext datasets, scored against the real KB through the
real retrieval and reranking pipeline.
Regenerate with `python scripts/eval.py`; resample with `python scripts/sample_eval_set.py`.

What these numbers mean for the product is in `tasks/todo.md` under Phase 11 - this file
is the measurement, not the argument about it.

The eval never calls Gemini.
Everything it measures - what is retrieved, how it ranks, and whether the gate escalates - is decided before the LLM is reached.

## The set

| Kind | Queries | What it proves |
|---|---|---|
| `in_scope` | 220 | 22 in-scope intents, 10 each. Retrieval must find the document and the gate must not escalate. |
| `out_of_scope` | 50 | Real Bitext support intents we deliberately do not cover (CONTACT, FEEDBACK, SUBSCRIPTION). Near-miss negatives: they are phrased in the KB's own vocabulary. |
| `out_of_domain` | 50 | Insurance queries, a different Bitext vertical. The easy negatives. |

Ground truth is the `intents:` frontmatter in `kb/`, so a query labelled `track_order` is correct if it retrieves a document claiming `track_order`.
Nothing is hand-mapped.

## Retrieval

| Stage | Recall | MRR | Precision@4 |
|---|---|---|---|
| Stage 1 (vector search, top 10) | {retrieval['stage1_recall_at_10']:.1%} @10 | {retrieval['stage1_mrr']:.3f} | - |
| Stage 2 (cross-encoder, top 4) | {retrieval['stage2_recall_at_4']:.1%} @4 | {retrieval['stage2_mrr']:.3f} | {retrieval['stage2_precision_at_4']:.1%} |

Top-1 accuracy after reranking: **{retrieval['top1_accuracy']:.1%}** - the share of in-scope queries whose best chunk comes from a document that genuinely covers the intent.

## The confidence gate

A **false answer** is either kind of confidently-wrong: an in-scope query answered with no document covering its intent in front of the model, or a negative answered at all.
Both reach the customer as a cited, confident answer that is not true, which is the failure the system exists to prevent.

It is scored on whether a correct document reached the top-4 handed to generation, not on whether it ranked first, because all four chunks go into the prompt and the model may cite any of them.
Scoring on top-1 would count "the right document ranked 2nd and was cited" as a false answer, which is not a failure a customer would ever see.

At the shipped threshold of **{settings.confidence_threshold}**:

- deflection **{current['deflection_rate']:.1%}** (target: >= 40%)
- false answers **{current['false_answer_rate']:.1%}** ({current['false_answers']}/320) (target: < 2%)
- negatives leaked: **{current['leaked']}/100**
- in-scope queries the gate escalated: **{current['escalated_in_scope']:.1%}**

### Sweep

| Threshold | Deflection | False answers | Ungrounded | Negatives leaked |
|---|---|---|---|---|
{rows}

{_verdict(best)}

## By intent

Sorted by MRR, worst first - the top of this table is where the KB or the chunking is weakest.

| Intent | Queries | Recall@4 | MRR | Deflected |
|---|---|---|---|---|
{cats}
""")


if __name__ == "__main__":
    raise SystemExit(main())
