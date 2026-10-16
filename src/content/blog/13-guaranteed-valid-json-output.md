---
title: "Guaranteeing Valid JSON Output from LLMs"
description: "Enforce JSON Schema at decode time, design schemas that keep quality high, and add validation, repair and truncation handling on top."
slug: "guaranteed-valid-json-output"
category: "agents-tool-use"
tags: ["structured-output", "tool-use", "serving", "evals"]
seriesOrder: 13
pubDate: 2026-10-16
sourceDoc: "1WwZ8yG-cd8f7_IZixvOLmzYPvB9hYJ554S4N4h26QE0"
---

## TL;DR

"Always valid" means enforcing the format at decode time, not asking for it in the prompt. Use **constrained decoding**: compile the JSON Schema into a grammar or finite-state machine and mask invalid tokens at each generation step. On hosted APIs that means OpenAI Structured Outputs with `strict: true`, or Anthropic's structured outputs (JSON outputs or strict tool use), both generally available; on self-hosted models it means Outlines, XGrammar or llama.cpp grammars. That guarantees syntax and schema validity, but not that the values are correct, so you still need semantic validation (Pydantic plus business rules), a bounded repair and retry path, and explicit handling of truncation and refusals. Finally, measure task accuracy with and without constraints, because whether format constraints hurt reasoning is disputed and depends on how you set them up.

## Requirements to pin down first

- **Which model and which serving stack?** A hosted API with native structured outputs, or self-hosted vLLM, SGLang or TensorRT-LLM where you control the logits? This decides whether you configure a feature or own the decoder.
- **How complex is the schema?** Flat objects are easy. Recursive schemas, `oneOf`, regex patterns and enums with thousands of values affect grammar compile time and whether a given provider supports them at all.
- **Does "valid" mean syntactically valid, schema-valid, or semantically correct?** A date that exists, or an `order_id` present in your database, is a semantic property no decoder can guarantee.
- **Is the output streamed?** If consumers need partial JSON, you need an incremental parser.
- **What are the throughput and latency SLOs?** Grammar masking adds per-token overhead, which matters at thousands of QPS (queries per second).

## Layer 1: Enforce the format at decode time

![Flowchart: a request with a JSON Schema goes to constrained decoding, which masks invalid tokens. The stop reason is then checked. A refusal goes to its own refusal path. Hitting the max token limit triggers a retry with a bigger budget or a more compact schema. A complete output goes to strict parsing and schema validation, then semantic checks. If the semantic checks pass, the object is returned. If they fail and retries remain, the validator error is sent back to the model for another constrained decode. If they fail after one or two retries, the system falls back to a deterministic default or human review.](/diagrams/q13-json-pipeline.svg "Decode-time guarantee plus a bounded validate-and-repair loop.")

Constrained decoding guarantees the shape; everything after the stop-reason check exists because it cannot guarantee completeness, compliance with refusals, or correct values.

### How constrained decoding works

1. Convert the JSON Schema into a regular expression or a context-free grammar (CFG).
2. Build an index from automaton states to the token IDs allowed in each state.
3. At each step, set the logits of disallowed tokens to −∞, so they can never be sampled.

Outlines pre-compiles the finite-state-machine-to-vocabulary index, which makes the per-token cost an O(1) lookup. XGrammar handles full context-free grammars (needed for arbitrarily nested JSON) with a pushdown automaton and pre-computes masks for context-independent tokens, which keeps overhead small. XGrammar is integrated into vLLM and SGLang.

### Hosted APIs

**OpenAI Structured Outputs.** In the Responses API, set `text.format` to `{"type": "json_schema", "name": ..., "schema": ..., "strict": true}`; in Chat Completions the equivalent is `response_format` with `type: "json_schema"`. Function calling also accepts `strict: true`. Strict mode guarantees schema adherence within a supported subset of JSON Schema: every field must be listed as required (model optional fields as a union with `null`), and objects need `additionalProperties: false`. The first request with a new schema has extra latency while the schema is processed. At launch, OpenAI reported that `gpt-4o-2024-08-06` with Structured Outputs scored 100% on its complex-schema eval, versus under 40% for `gpt-4-0613` with prompting alone.

**Anthropic structured outputs.** Generally available on the Claude API in two forms that share one grammar-compilation pipeline:

- **JSON outputs**: set `output_config.format` to `{"type": "json_schema", "schema": ...}` and the response text conforms to the schema.
- **Strict tool use**: add `"strict": true` to a tool definition and the tool call's `input` is guaranteed to match its `input_schema`, with a valid tool `name`.

The older `output_format` parameter is deprecated and only works with the `structured-outputs-2025-11-13` beta header. Unsupported schema features include recursive schemas, numeric constraints such as `minimum` and `maximum`, string length constraints, and `additionalProperties` set to anything but `false`; using one returns a 400 error. The first use of a schema adds compile latency, and compiled grammars are cached for 24 hours from last use. Forcing a non-strict tool with `tool_choice` still gives schema-*shaped* JSON, but without the guarantee.

**JSON mode is not schema adherence.** Plain "JSON mode" guarantees only parseable JSON, not conformance to your schema.

### Tokenization subtleties

Tokens can cross grammar boundaries; for example, `"}` is often a single token. A correct implementation must allow such tokens when the full string is valid. Naive character-level masking can force odd tokenizations that hurt quality. Mature libraries handle token healing correctly; home-grown implementations often get it wrong.

## Layer 2: Design the schema for model quality

### Give the model room to reason

Generation is left to right, so the model commits to each field as it writes it. Options, in order of preference:

- **Use the model's native reasoning channel.** On models with built-in thinking, let reasoning happen there and keep the schema for the answer. Anthropic's docs warn that a schema property asking for step-by-step reasoning may trigger a `reasoning_extraction` refusal and recommend asking for a short explanation instead.
- **Put a short rationale field before the answer** (`"rationale"` before `"answer"`) on models without a separate reasoning channel.
- **Two steps:** reason in an unconstrained call, then format in a second, constrained call.

### Whether constraints hurt reasoning is disputed

- **Evidence for harm.** Tam et al. (2024), "Let Me Speak Freely?", reported a significant decline in reasoning accuracy under format restrictions, with stricter formats hurting more, while classification tasks were unaffected or even improved with JSON mode.
- **Evidence against.** Will Kurt's response for .txt (the Outlines team), "Say What You Mean" (2024), argues the comparison was confounded: the prompts differed between conditions, the JSON prompts left out the schema, answers were parsed by another LLM, and JSON mode was conflated with true constrained decoding. With matched prompts on Llama-3-8B-Instruct, structured generation matched or beat unstructured on the same tasks (for example GSM8K 0.78 vs 0.77, Last Letter 0.77 vs 0.73, Shuffled Objects 0.44 vs 0.41).

The practical takeaway is the same either way: give the model space to reason, write the schema and its descriptions into the prompt, and measure your own task accuracy against an unconstrained baseline.

### Field order and naming

- **Order fields by dependency.** Put fields the others depend on first. Note that Anthropic's structured outputs emit required properties before optional ones, so mark fields required if their order matters.
- **Use enums for categorical fields**, plus descriptive field names and descriptions; the model reads them.
- **Use nullable fields instead of forcing invented values.** Include an explicit status so the model isn't forced to make something up to satisfy the schema.
- **Keep schemas shallow.** Deep nesting and large `anyOf` unions increase compile time and confuse the model.

A worked example:

```json
{
  "type": "object",
  "properties": {
    "status":   { "type": "string", "enum": ["ok", "insufficient_info"] },
    "order_id": { "type": ["string", "null"], "description": "Order ID quoted by the customer" },
    "quantity": { "type": ["integer", "null"] }
  },
  "required": ["status", "order_id", "quantity"],
  "additionalProperties": false
}
```

Every field is required, so the structure is fixed, but `null` and `"insufficient_info"` give the model an honest way to say it doesn't know.

## Layer 3: Validate and repair

Constrained decoding is the guarantee for shape. Defense in depth covers the rest.

### Parse, validate, check semantics

Parse with a strict parser, validate against Pydantic or JSON Schema, then apply semantic checks: IDs exist, sums match, dates are in range.

```python
from typing import Literal
from pydantic import BaseModel

class OrderUpdate(BaseModel):
    status: Literal["ok", "insufficient_info"]
    order_id: str | None
    quantity: int | None

def semantic_errors(u: OrderUpdate, db) -> list[str]:
    if u.status != "ok":
        return []
    if u.order_id is None or not db.order_exists(u.order_id):
        return [f"order_id {u.order_id!r} does not exist"]
    stock = db.stock(u.order_id)
    if u.quantity is None or u.quantity > stock:
        return [f"quantity must be ≤ stock ({stock}); you returned {u.quantity}"]
    return []
```

### Truncation

Constrained decoding can't stop the token limit from cutting the output short. Check the stop signal (`stop_reason == "max_tokens"` on Anthropic, `finish_reason == "length"` in Chat Completions, an `incomplete` status in the Responses API) and retry with a larger budget or a more compact schema.

### Refusals

Providers may return a refusal instead of the object: a `refusal` field on OpenAI, `stop_reason: "refusal"` on Anthropic, where the output may not match the schema. Handle it as its own code path, not as a parse error.

### Bounded repair loop

When semantic validation fails, send the validator error back to the model ("quantity must be ≤ stock (12); you returned 40") for at most 1-2 retries. Then fall back to a deterministic default or human review. For fallbacks that run without constraints, a lenient parser such as `json-repair` can fix trailing commas and markdown fences.

## Layer 4: Streaming

Use an incremental parser to emit partial objects. With constrained decoding every prefix is a valid JSON prefix, so the client can render fields as they arrive. Some SDKs instead accumulate the stream and deserialize at the end; pick the mode your UI needs.

## How to measure it

| Metric | What it tells you | Target |
|---|---|---|
| Parse success rate | Output is syntactically valid JSON | 100% with constraints |
| Schema-valid rate | Output matches the schema | 100% with constraints, excluding refusals and truncations |
| Semantic-valid rate | Values pass business rules | Track the trend; most real bugs live here |
| Retry rate | How often the repair loop runs | Alert on increases; it's a regression signal |
| Refusal and truncation rates | Outputs outside the guarantee | Low and stable |
| Task accuracy vs. unconstrained baseline | Whether constraints cost quality | No significant drop on the same eval set |

Prompt-only setups land below 100% parse success, and how far below depends heavily on the model and schema complexity; an illustrative range is 95-99.5% on modern models, and much lower on hard schemas for older ones (see OpenAI's under-40% figure above). Measure your own baseline before deciding constraints are unnecessary.

For latency, track TTFT (time to first token), which includes grammar compilation on a new schema (cache it per schema), and per-token time. An illustrative overhead target is under ~5%.

Track everything by schema version. Schema changes are deploys and need contract tests.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| Prompting + few-shot ("respond in JSON") | Zero infrastructure; any model | No guarantee; a few percent failures at scale is common | Prototypes, low volume |
| JSON mode (syntax only) | Always parseable | Not schema-conformant | Loose schemas, quick wins |
| Native structured outputs / strict tool schemas | Schema guarantee; no infrastructure | Schema-subset limits; provider lock-in; first-call compile latency | Default on hosted APIs |
| Self-hosted constrained decoding (Outlines, XGrammar, llama.cpp GBNF) | Full control; arbitrary grammars and regex | Infrastructure ownership; tokenizer edge cases; throughput tuning | Self-hosted or open-weights models, high volume |
| Validate + retry/repair loop | Catches semantic errors | Extra latency and cost; not a guarantee alone | Always, on top of constraints |
| Two-step: free-form reasoning, then format | Preserves reasoning quality | 2x calls and latency | Hard reasoning tasks with structured output |

- **Syntax is solved; semantics aren't.** Most remaining production bugs are valid JSON with wrong values, so put the effort into semantic validators and evals.
- **Check that constraints don't cost accuracy.** Compare task accuracy with and without them, especially on reasoning-heavy fields.

## Common mistakes

- **Saying "JSON mode" when the requirement is schema adherence.** They guarantee different things, and the gap shows up as missing or mistyped fields in production.
- **Forgetting truncation and refusals.** Constrained decoding protects against neither, so unhandled cases surface as parse errors.
- **Forcing the answer field first.** The model commits before it has reasoned.
- **No way to express "unknown".** The model makes up values to fill required fields.
- **Unbounded retry loops.** They are one cause of agent loops (see [Detecting and Recovering from Agent Loops](/blog/agent-loop-detection-recovery/)); also monitor retry rate as a regression signal.
- **Assuming every JSON Schema keyword is supported.** Providers support subsets, with limits on patterns, formats, numeric bounds and recursion; unsupported features fail the request.

## Thinking at scale

- **Treat schemas as versioned APIs.** Store them in a registry, review changes, and run contract tests in CI against recorded outputs.
- **Warm the grammar cache.** On deploy, send one request per new schema so users don't pay compile latency, and remember provider caches expire (24 hours from last use on Anthropic).
- **Keep provider portability in mind.** Each provider supports a different JSON Schema subset. Write schemas to the intersection you need, or keep an adapter layer per provider.
- **Own semantic validation centrally.** Business rules belong in shared validators that every LLM-backed endpoint uses, not in each prompt.

## Further reading

- [Efficient Guided Generation for Large Language Models (Willard and Louf, 2023)](https://arxiv.org/abs/2307.09702): The FSM-index approach behind Outlines, with O(1) per-token masking.
- [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models (Dong et al., 2024)](https://arxiv.org/abs/2411.15100): Fast CFG-constrained decoding via a pushdown automaton, used in vLLM and SGLang.
- [Structured Outputs guide (OpenAI)](https://developers.openai.com/api/docs/guides/structured-outputs): `json_schema` with `strict: true`, the supported schema subset, and refusal handling.
- [Introducing Structured Outputs in the API (OpenAI, 2024)](https://openai.com/index/introducing-structured-outputs-in-the-api/): How the guarantee is implemented (constrained decoding plus training) and the reliability numbers.
- [Structured outputs (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/structured-outputs): JSON outputs, schema limitations, grammar caching, and refusal and `max_tokens` behavior.
- [Strict tool use (Claude API docs)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use): Guaranteed schema-valid tool inputs with `strict: true`.
- [Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models (Tam et al., 2024)](https://arxiv.org/abs/2408.02442): Reports reasoning degradation under strict format constraints.
- [Say What You Mean: A Response to 'Let Me Speak Freely' (Kurt, .txt, 2024)](https://blog.dottxt.ai/say-what-you-mean.html): Rebuttal arguing the degradation came from confounded prompts and parsing, not constrained decoding.
- [Outlines (GitHub)](https://github.com/dottxt-ai/outlines): Production library for structured generation on open models.
