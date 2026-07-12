"""Verify the KB corpus against the Bitext taxonomy.

The Phase 11 eval joins queries to documents on the intent strings in each
doc's frontmatter. A typo there does not fail loudly, it just silently scores
every query for that intent as a miss. So check the mapping explicitly.

Run: python scripts/check_kb_coverage.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import yaml

# Derived from the dataset, not transcribed from memory. See
# docs/intent-taxonomy.md for how to reproduce.
TAXONOMY = {
    "ORDER": {"place_order", "track_order", "change_order", "cancel_order"},
    "REFUND": {"check_refund_policy", "get_refund", "track_refund"},
    "ACCOUNT": {
        "create_account",
        "edit_account",
        "delete_account",
        "recover_password",
        "registration_problems",
        "switch_account",
    },
    "PAYMENT": {"check_payment_methods", "payment_issue"},
    "SHIPPING": {"set_up_shipping_address", "change_shipping_address"},
    "DELIVERY": {"delivery_options", "delivery_period"},
    "INVOICE": {"check_invoice", "get_invoice"},
    "CANCEL": {"check_cancellation_fee"},
}

# Real Bitext intents we deliberately do not answer. These are the near-miss
# negatives: the system must escalate on them rather than retrieve. A KB doc
# claiming one of these is a bug, not an omission.
OUT_OF_SCOPE = {
    "contact_customer_service",
    "contact_human_agent",
    "complaint",
    "review",
    "newsletter_subscription",
}

IN_SCOPE = {i for intents in TAXONOMY.values() for i in intents}
KB_DIR = Path(__file__).resolve().parent.parent / "kb"


def load_docs() -> list[tuple[Path, dict]]:
    docs = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            raise SystemExit(f"{path.name}: missing YAML frontmatter")
        _, fm, _ = text.split("---", 2)
        docs.append((path, yaml.safe_load(fm)))
    return docs


def main() -> int:
    docs = load_docs()
    errors: list[str] = []
    covered: dict[str, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()

    for path, fm in docs:
        for field in ("title", "doc_id", "category", "intents"):
            if not fm.get(field):
                errors.append(f"{path.name}: missing frontmatter field '{field}'")

        doc_id = fm.get("doc_id")
        if doc_id in seen_ids:
            errors.append(f"{path.name}: duplicate doc_id '{doc_id}'")
        seen_ids.add(doc_id)

        if doc_id and doc_id != path.stem:
            errors.append(f"{path.name}: doc_id '{doc_id}' does not match filename")

        category = fm.get("category")
        if category not in TAXONOMY:
            errors.append(
                f"{path.name}: category '{category}' is not an in-scope category"
            )

        for intent in fm.get("intents") or []:
            if intent in OUT_OF_SCOPE:
                errors.append(
                    f"{path.name}: intent '{intent}' is deliberately out of scope. "
                    "It must escalate, not retrieve."
                )
            elif intent not in IN_SCOPE:
                errors.append(
                    f"{path.name}: intent '{intent}' is not a real Bitext intent"
                )
            else:
                covered[intent].append(path.stem)
                if category in TAXONOMY and intent not in TAXONOMY[category]:
                    errors.append(
                        f"{path.name}: intent '{intent}' does not belong to category '{category}'"
                    )

    uncovered = IN_SCOPE - covered.keys()
    for intent in sorted(uncovered):
        errors.append(f"intent '{intent}' has no KB document")

    print(f"documents:      {len(docs)}")
    print(f"intents in scope: {len(IN_SCOPE)}")
    print(f"intents covered:  {len(covered)}")

    if errors:
        print(f"\nFAILED ({len(errors)} problem(s)):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nCoverage by category:")
    for category, intents in TAXONOMY.items():
        print(f"  {category}")
        for intent in sorted(intents):
            print(f"    {intent:<26} {', '.join(covered[intent])}")
    print("\nOK: every in-scope intent has at least one document.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
