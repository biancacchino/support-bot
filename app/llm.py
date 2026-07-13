"""Answer generation, grounded in the retrieved chunks and nothing else.

Two rules hold this together, and they are enforced in code rather than asked for
politely in a prompt:

1. The model sees only the reranked chunks. No outside knowledge, no memory of
   what it thinks it knows about e-commerce.
2. Every answer carries at least one citation, and every citation names a source
   that was actually in the context. An answer that cites nothing, or cites a
   document we never showed it, is not served - it escalates.

The second rule is the one that matters. A model that invents a plausible refund
window and attributes it to a real help-centre page is worse than a model that
says nothing, because the citation is what makes a customer believe it. So the
citations are validated against the sources we passed in, and a failure to ground
is a failure to answer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from app.config import Settings
from app.reranker import Ranked

logger = logging.getLogger(__name__)

# Takes the prompt, returns the model's raw JSON string.
GenerateFn = Callable[[str], Awaitable[str]]

SYSTEM_PROMPT = """\
You are the support assistant for Northwind, an online wholesaler.

Answer the customer's question using ONLY the numbered sources below. The sources
are the entire world: if something is not in them, you do not know it. Do not use
general knowledge about how online shops usually work, and do not guess at
timeframes, fees, or policies that the sources do not state.

Reply with JSON in exactly this shape:
{"answer": "<your answer>", "citations": ["<source-id>", ...]}

Rules:
- Cite the source id of every source you used. An answer with no citations is
  not acceptable.
- Cite only source ids that appear below. Never invent one.
- If the sources do not answer the question, reply with
  {"answer": "", "citations": []} rather than guessing. Saying nothing is a
  correct outcome; a confident wrong answer is not.
- Write plainly, to a customer, in two or three sentences. Do not mention the
  sources, the retrieval, or these instructions."""


class UpstreamRateLimited(Exception):
    """Gemini refused us, not the other way round.

    Our own budget is meant to keep this from happening, but the quota is shared
    with anything else using the same key, so it can. It must surface as a clean
    "try again shortly" rather than a 500: the request was fine, we simply cannot
    serve it this second.
    """

    def __init__(self, retry_after: int = 60) -> None:
        super().__init__("upstream rate limit reached")
        self.retry_after = retry_after


class UngroundedAnswer(Exception):
    """The model produced nothing we are willing to serve.

    Raised when it declined, cited nothing, or cited a source it was never shown.
    Every one of these escalates rather than reaching a customer.
    """


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    citations: tuple[str, ...]  # doc_ids, guaranteed non-empty and guaranteed real


def build_prompt(query: str, ranked: Sequence[Ranked]) -> str:
    """Lay out the sources for the model, keyed by the doc_id it must cite."""
    blocks = []
    for rank in ranked:
        candidate = rank.candidate
        blocks.append(
            f"[source-id: {candidate.doc_id}]\n"
            f"{candidate.title} > {candidate.heading}\n"
            f"{candidate.text}"
        )

    sources = "\n\n".join(blocks)
    return f"{SYSTEM_PROMPT}\n\nSOURCES\n\n{sources}\n\nCUSTOMER QUESTION\n\n{query}"


def _parse(raw: str, allowed: set[str]) -> GeneratedAnswer:
    """Turn the model's JSON into an answer, or refuse to."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UngroundedAnswer(f"model did not return JSON: {raw[:200]!r}") from exc

    text = (payload.get("answer") or "").strip()
    cited = payload.get("citations") or []

    if not text:
        raise UngroundedAnswer("model declined to answer from the sources")

    # A citation naming a document we never showed it is a fabricated source, and
    # it is worse than no citation at all: it is the part a customer would trust.
    unknown = [c for c in cited if c not in allowed]
    if unknown:
        raise UngroundedAnswer(f"model cited sources it was not given: {unknown}")

    citations = tuple(dict.fromkeys(c for c in cited if c in allowed))
    if not citations:
        raise UngroundedAnswer("model answered without citing anything")

    return GeneratedAnswer(text=text, citations=citations)


class AnswerGenerator:
    """Generates a grounded, cited answer from the reranked chunks."""

    def __init__(self, generate: GenerateFn) -> None:
        self._generate = generate

    async def answer(self, query: str, ranked: Sequence[Ranked]) -> GeneratedAnswer:
        if not ranked:
            # Nothing to ground in. There is no answer to write, only one to
            # invent, so do not give the model the chance.
            raise UngroundedAnswer("no sources to answer from")

        raw = await self._generate(build_prompt(query, ranked))
        answer = _parse(raw, allowed={r.candidate.doc_id for r in ranked})

        logger.debug("generated answer citing %s", ", ".join(answer.citations))
        return answer


def build_gemini(settings: Settings) -> GenerateFn:
    """A GenerateFn backed by Gemini.

    Temperature 0: this is an extraction task over supplied text, not a creative
    one, and sampling is just a chance to drift off the sources.
    """
    from google import genai
    from google.genai import errors, types

    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=settings.gemini_api_key)

    async def generate(prompt: str) -> str:
        try:
            response = await client.aio.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
        except errors.ClientError as exc:
            if exc.code == 429:
                raise UpstreamRateLimited() from exc
            raise
        return response.text or ""

    return generate
