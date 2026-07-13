"""Sample the Bitext datasets into a frozen evaluation set.

    python scripts/sample_eval_set.py            # writes eval/queries.jsonl
    python scripts/sample_eval_set.py --seed 7   # a different sample

The output is committed. A benchmark that resamples on every run measures a
different thing every run, and two numbers from two samples are not comparable -
so the sample is drawn once, written down, and the eval reads the file. Re-run
this only when you mean to change what is being measured.

Three kinds of query, because the bot has to get three different things right:

- `in_scope`     - the KB answers it. Retrieval must find the document, and the
                   gate must not escalate. Every one it escalates is deflection
                   the product does not get.
- `out_of_scope` - a real Bitext support intent we deliberately do not cover
                   (CONTACT, FEEDBACK, SUBSCRIPTION). These are the *near-miss*
                   negatives: they are phrased in the KB's own vocabulary and land
                   close to real documents in embedding space, which is exactly the
                   case the confidence gate exists for.
- `out_of_domain`- insurance queries, from a different Bitext vertical entirely.
                   The easy negatives. If the gate cannot escalate these it is not
                   a gate.

Queries are sampled per intent rather than uniformly at random. A uniform sample
of 26,872 rows would follow the dataset's own intent distribution and leave the
rarer intents with two or three queries each, which is not enough to say anything
about them.
"""

import argparse
import json
import random
import re
import ssl
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import certifi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_kb_coverage import IN_SCOPE, OUT_OF_SCOPE, TAXONOMY  # noqa: E402

# A framework Python on macOS ships no CA bundle, so urllib cannot verify the
# datasets-server certificate. certifi is already here (httpx pulls it in), and
# trusting its bundle beats the usual advice of turning verification off.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

SERVER = "https://datasets-server.huggingface.co"
SUPPORT_DS = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
INSURANCE_DS = "bitext/Bitext-insurance-llm-chatbot-training-dataset"

OUT_PATH = Path(__file__).resolve().parent.parent / "eval" / "queries.jsonl"

PER_INTENT = 10  # 22 in-scope intents -> 220, plus 5 out-of-scope -> 50
OUT_OF_DOMAIN = 50

# The dataset templates entities out: "cancel order {{Order Number}}". A customer
# does not type braces, and the placeholder is noise to an embedding model, so it
# goes. Removing it leaves "cancel order", which is what the query really is.
PLACEHOLDER_RE = re.compile(r"\s*\{\{[^}]*\}\}")

# The window the API is asked for in one call. Sampling from a window rather than
# from the whole intent keeps this to one request per intent instead of one per row.
WINDOW = 100

INTENT_TO_CATEGORY = {
    intent: category for category, intents in TAXONOMY.items() for intent in intents
}


def _get(path: str, params: dict) -> dict:
    url = f"{SERVER}/{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60, context=SSL_CONTEXT) as response:  # noqa: S310
        return json.loads(response.read())


def _clean(instruction: str) -> str:
    return PLACEHOLDER_RE.sub("", instruction).strip()


def sample_intent(intent: str, rng: random.Random, n: int) -> list[str]:
    """`n` distinct queries for one intent, drawn from a random window of its rows."""
    where = f"\"intent\"='{intent}'"
    total = _get(
        "filter",
        {
            "dataset": SUPPORT_DS,
            "config": "default",
            "split": "train",
            "where": where,
            "offset": 0,
            "length": 1,
        },
    )["num_rows_total"]

    offset = rng.randrange(max(1, total - WINDOW))
    rows = _get(
        "filter",
        {
            "dataset": SUPPORT_DS,
            "config": "default",
            "split": "train",
            "where": where,
            "offset": offset,
            "length": min(WINDOW, total),
        },
    )["rows"]

    # dict.fromkeys: the dataset repeats phrasings, and the same question twice is
    # one measurement pretending to be two.
    queries = list(dict.fromkeys(_clean(r["row"]["instruction"]) for r in rows))
    return rng.sample(queries, min(n, len(queries)))


def sample_out_of_domain(rng: random.Random, n: int) -> list[dict]:
    """Insurance queries. A different industry, so nothing in the KB is close."""
    total = _get(
        "rows",
        {
            "dataset": INSURANCE_DS,
            "config": "default",
            "split": "train",
            "offset": 0,
            "length": 1,
        },
    )["num_rows_total"]

    seen: dict[str, dict] = {}
    while len(seen) < n:
        offset = rng.randrange(max(1, total - WINDOW))
        rows = _get(
            "rows",
            {
                "dataset": INSURANCE_DS,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": WINDOW,
            },
        )["rows"]
        for r in rows:
            query = _clean(r["row"]["instruction"])
            if query and query not in seen:
                seen[query] = {
                    "query": query,
                    "kind": "out_of_domain",
                    "intent": r["row"]["intent"],
                    "category": r["row"]["category"],
                }

    return rng.sample(list(seen.values()), n)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records: list[dict] = []

    for intent in sorted(IN_SCOPE):
        for query in sample_intent(intent, rng, PER_INTENT):
            records.append(
                {
                    "query": query,
                    "kind": "in_scope",
                    "intent": intent,
                    "category": INTENT_TO_CATEGORY[intent],
                }
            )
        print(f"  {intent:<26} {len([r for r in records if r['intent'] == intent])}")

    for intent in sorted(OUT_OF_SCOPE):
        for query in sample_intent(intent, rng, PER_INTENT):
            records.append(
                {
                    "query": query,
                    "kind": "out_of_scope",
                    "intent": intent,
                    "category": None,
                }
            )
        print(f"  {intent:<26} {PER_INTENT} (negative)")

    records.extend(sample_out_of_domain(rng, OUT_OF_DOMAIN))
    print(f"  insurance                  {OUT_OF_DOMAIN} (out-of-domain)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    kinds = {k: sum(1 for r in records if r["kind"] == k) for k in
             ("in_scope", "out_of_scope", "out_of_domain")}
    print(f"\nwrote {len(records)} queries to {OUT_PATH.relative_to(Path.cwd())}: {kinds}")
    print(f"seed {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
