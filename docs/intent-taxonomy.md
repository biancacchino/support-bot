# Intent Taxonomy (Task 1.0)

The ground-truth taxonomy for retrieval evaluation.
Derived from the Bitext dataset, not invented.

## Source

`bitext/Bitext-customer-support-llm-chatbot-training-dataset` on HuggingFace.
~26,900 rows, 27 intents, 11 categories.

Derived on 2026-07-12 by querying the HuggingFace datasets-server directly rather than transcribing from the dataset card, so the intent and category strings below are exactly the literals present in the data.
This matters: the Phase 11 eval joins on these strings, and a typo would silently score every query as a miss.

Reproduce with:

```bash
DS="bitext%2FBitext-customer-support-llm-chatbot-training-dataset"
curl -sS "https://datasets-server.huggingface.co/statistics?dataset=$DS&config=default&split=train"
```

## The full taxonomy

Two category names are worth noting because they are easy to guess wrong: the category is `CANCEL` (not `CANCELLATION_FEE`) and `SHIPPING` (not `SHIPPING_ADDRESS`).

| Category | Intents | In scope |
|---|---|---|
| `ORDER` | `place_order`, `track_order`, `change_order`, `cancel_order` | Yes |
| `REFUND` | `check_refund_policy`, `get_refund`, `track_refund` | Yes |
| `ACCOUNT` | `create_account`, `edit_account`, `delete_account`, `recover_password`, `registration_problems`, `switch_account` | Yes |
| `PAYMENT` | `check_payment_methods`, `payment_issue` | Yes |
| `SHIPPING` | `set_up_shipping_address`, `change_shipping_address` | Yes |
| `DELIVERY` | `delivery_options`, `delivery_period` | Yes |
| `INVOICE` | `check_invoice`, `get_invoice` | Yes |
| `CANCEL` | `check_cancellation_fee` | Yes |
| `CONTACT` | `contact_customer_service`, `contact_human_agent` | No |
| `FEEDBACK` | `complaint`, `review` | No |
| `SUBSCRIPTION` | `newsletter_subscription` | No |

**Scoped in: 8 categories, 22 intents.**
**Scoped out: 3 categories, 5 intents.**

## What is in scope, and why

The vertical is e-commerce order lifecycle, so the scope is every intent a customer hits between placing an order and being made whole when it goes wrong.

`ORDER`, `SHIPPING`, `DELIVERY`, `PAYMENT`, `INVOICE`, `REFUND`, and `CANCEL` are that lifecycle end to end.
`ACCOUNT` is in because a customer who cannot log in cannot reach any of the above, so it is a precondition for the rest rather than a separate concern.

`CANCEL` is a single-intent category and it would have been tempting to drop it for tidiness.
It stays because `check_cancellation_fee` is the natural follow-up question to `cancel_order`, and a KB that can cancel an order but cannot say what it costs is incoherent.
It is also a good multi-turn test case for Phase 5: "cancel my order" then "will I be charged?" is only answerable if turn 1's context survives.

## What is out of scope, and why

**`CONTACT` (`contact_customer_service`, `contact_human_agent`).**
These are not questions with answers, they are requests to leave the bot.
Writing a KB doc for "I want to speak to a human" would train retrieval to *answer* an escalation request with a document, which directly fights the confidence-gated escalation design in Phase 3.
The correct system response to `contact_human_agent` is to escalate, not to retrieve. Giving it a doc to match against would be actively harmful.

**`FEEDBACK` (`complaint`, `review`).**
Also not informational.
A complaint needs a human to acknowledge it and a review needs somewhere to be recorded.
Neither is a retrieval problem, and a confidently-cited policy doc is close to the worst possible response to an angry customer.

**`SUBSCRIPTION` (`newsletter_subscription`).**
The Bitext `SUBSCRIPTION` category is only about newsletter opt-in and opt-out, not about recurring product subscriptions.
It is marketing preferences, orthogonal to the order lifecycle, and a single intent's worth of content.

Note: the original task list named "subscription changes" as a KB topic.
That appears to have assumed `SUBSCRIPTION` meant recurring orders.
In this dataset it does not, so there is nothing order-related to write.

## Why the excluded intents are still useful

The 5 out-of-scope intents are deliberately kept in the eval set as **near-miss negatives**.

They are real customer-support queries, phrased in the same register as the in-scope ones, drawn from the same distribution.
An embedding model will score them as highly similar to the KB.
But there is no correct document for them, so the system *must* escalate.

This is exactly the failure Phase 3's reranker gate exists to catch, and it is a sharper test than the out-of-domain (healthcare, insurance) queries Phase 11.3 calls for.
Out-of-domain queries are easy to escalate, since nothing in the KB looks remotely close.
In-domain-but-unanswerable queries are the hard case, and they are the ones that produce a confidently wrong answer in production.

Both sets are used in Phase 11:

- **Out-of-domain** (other Bitext verticals) measures the floor: escalation should be near-total.
- **Near-miss** (`CONTACT`, `FEEDBACK`, `SUBSCRIPTION`) measures the thing we actually care about: does the confidence gate hold when the query *looks* like it belongs?

## Mapping to the KB corpus

Every file in `kb/` declares its intents in YAML frontmatter:

```yaml
---
title: Cancelling an order
doc_id: cancelling-an-order
category: ORDER
intents: [cancel_order]
---
```

Ingestion (Task 1.2) carries `doc_id`, `category`, and `intents` through to Qdrant as chunk metadata.
Phase 11 scores a retrieved chunk as correct when the query's Bitext `intent` appears in the chunk's `intents` list.

All 22 in-scope intents are covered by at least one document.
Some documents map to more than one intent where the customer's question does not split cleanly, for example returns of damaged goods, which is both `get_refund` and `check_refund_policy`.
