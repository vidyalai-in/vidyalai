---
title: "Building a PII Redaction Gateway in Front of Your LLM"
description: "A layered design for detecting, pseudonymizing and re-identifying PII before it reaches a model, and how to measure the leak rate honestly."
slug: "pii-redaction-gateway"
category: "security-privacy"
tags: ["privacy", "security", "architecture", "observability"]
seriesOrder: 18
pubDate: 2026-02-03
updatedDate: 2026-10-16
sourceDoc: "1peTYAI7ps58PRKJWERCzNUnsehYjGOwhNRHViG8g-bs"
---

## TL;DR

Put a pre-model redaction gateway in front of every model call, and make it impossible to bypass. It uses a layered detector: deterministic validators for structured identifiers (card numbers with Luhn, SSNs, IBANs, emails, phone numbers), an NER (named-entity recognition) model for names, addresses and free-text identifiers, and context rules on top. It replaces detected spans with consistent typed placeholders such as `<PERSON_1>` and keeps a per-request vault so it can re-identify the output when the caller is authorized. Optimize recall on the high-risk types, with targets such as 99%+ on structured identifiers and 95%+ on names, measured on a labeled in-domain test set, and accept some over-redaction. Because redaction is never perfect, treat it as one control among several, alongside data retention limits, zero-retention model endpoints, access logging and output scanning.

## Requirements to pin down first

**Why are you redacting?** The driver might be regulation (HIPAA Safe Harbor's 18 identifiers, GDPR, PCI-DSS for card data), a contractual limit with a third-party model vendor, or keeping PII out of training data and logs. The answer sets which entity types matter and how high recall has to be.

**Does the task need the PII?** Drafting a reply to "John" needs the name back at output time. Classifying a support ticket doesn't. This decides between irreversible redaction and pseudonymization with re-identification.

**Where does the data flow?** Into prompts, retrieval indices, tool outputs, logs, eval sets and fine-tuning corpora. Every one of these is a leak path, and logs are the one most commonly forgotten.

**Which languages and formats, and what latency budget?** Multilingual text, OCR'd PDFs, code and chat transcripts all behave very differently for detectors. A gateway typically gets a budget of 20-50 ms at p95 (95th-percentile latency).

## The gateway architecture

![Flowchart of the PII gateway. An incoming request, including prompts, retrieved documents and tool outputs, goes through deterministic detectors (regex plus checksums) and an NER model with context rules, with an optional LLM pass for high-risk traffic. Their spans are merged and pseudonymized into typed placeholders such as PERSON_1 and EMAIL_2. The placeholder map is written to an encrypted, request-scoped vault with a short TTL, and the redacted request goes to a zero-retention model endpoint. The model's output passes through an egress scan, then a check of whether the caller is authorized to re-identify. Authorized callers get the response with real values restored from the vault; others get the response with placeholders.](/diagrams/q18-pii-gateway.svg "The PII gateway: detect and pseudonymize on the way in, scan and conditionally re-identify on the way out.")

Every model call crosses one chokepoint on the way in and one on the way out, and the real values never leave the vault except to an authorized caller.

### 1. One ingress chokepoint

Run a single gateway service in front of all model calls, and in front of the embedding model for RAG (retrieval-augmented generation), since embeddings and vector stores are also data sinks. No team calls the model API directly. Enforce this with network policy and credentials, not convention: if the model API key only exists inside the gateway, bypassing it is impossible.

### 2. Layered detection

The first layer is deterministic validators, regexes plus checksums: Luhn for card numbers (PANs, primary account numbers), mod-97 for IBANs, SSN area-number rules, and libphonenumber for phone numbers. These are high precision and cheap, and checksums cut false positives on random digit strings.

The second layer is statistical NER, a fine-tuned token classifier (DeBERTa- or spaCy-class, roughly 100-400M parameters, under 10 ms on a GPU) for PERSON, LOCATION/ADDRESS, ORG, date of birth and medical record numbers.

Off-the-shelf NER recall varies a lot with the text. As an illustrative planning figure, expect something like 85-90% recall on names in clean, well-formed text, and noticeably less on chat, ALL-CAPS text or names from underrepresented cultures. Robustness studies of Presidio and spaCy-based detectors consistently report significant degradation on out-of-distribution inputs, non-standard formats and unseen surface forms (Zafar and Nowaczyk, 2026). Don't plan around a published number: measure recall on your own labeled data, and fine-tune on in-domain examples to close the gap.

Context rules sit on top. They boost scores near trigger phrases ("my SSN is", "DOB:"), and suppress them on allowlisted terms such as public figures, product names or your own company name.

As an optional extra pass, a small, self-hosted LLM can catch quasi-identifiers such as "the only female cardiologist at St. Mary's in Boise", which no token-level detector flags. Run it only on flagged or high-risk traffic, because of its cost and latency.

Frameworks such as Microsoft Presidio (open source) or Google Cloud Sensitive Data Protection (managed) provide this pluggable recognizer pipeline. Expect to customize them heavily.

### 3. Transformation

- **Typed, consistent pseudonyms** (`<PERSON_1>`, `<EMAIL_2>`) preserve coreference: the model can still reason that the same person appears twice. Masking everything as `[REDACTED]` destroys task quality.
- **Format-preserving surrogates** (a fake but valid-looking name or date) give better model fluency. The cost is that surrogates can leak into outputs as if they were real facts, and they make audits harder because redacted and real text look alike.
- **Date shifting** by a per-patient random offset preserves intervals between events, which clinical tasks often need.

A short example of typed pseudonymization:

```text
Input:   Hi, I'm Priya Shah. Email priya.shah@example.com, or ask Priya's manager Tom.
To LLM:  Hi, I'm <PERSON_1>. Email <EMAIL_1>, or ask <PERSON_1>'s manager <PERSON_2>.
```

Both mentions of Priya map to `<PERSON_1>`, so the model keeps track of who is who.

### 4. Vault and re-identification

Keep the placeholder-to-value map in an encrypted, request-scoped store with a short TTL (time to live). Re-identify only on the egress path, only for authorized callers, and never write the map to logs. The vault concentrates the most sensitive data in the system in one place, so it gets the strictest access controls.

### 5. Egress scanning

Scan the model's output too. Models can reconstruct or hallucinate PII from context, and tool outputs or retrieved documents may carry PII that bypassed ingress. The egress scanner runs the same detectors and either redacts or blocks.

### 6. Defense in depth

- Use zero-data-retention model endpoints where the vendor offers them.
- Encrypt in transit and at rest.
- Apply the same redaction to logs and traces.
- Keep a separate, stricter policy for training data. Models memorize and can regurgitate rare PII, as Carlini et al. (2021) demonstrated by extracting names, phone numbers and addresses from GPT-2.

## How to measure it

### The labeled test set

Build 2,000-5,000 in-domain documents with span-level annotations, double-annotated, and stratified by entity type and channel (chat, email, OCR, code). Include adversarial cases:

- spelled-out numbers ("four one one one...")
- identifiers split across tokens or lines
- PII inside JSON, code or URLs

### Metrics and targets

Measure per-entity-type recall (the leak rate is `1 − recall`) and precision, at both entity level and token level. Example targets:

| Entity type | Recall target |
|---|---|
| Card numbers (PAN), SSN | 99.5% or more |
| PERSON | 95% or more |
| Addresses | 90% or more |

Precision matters less, but over-redaction does hurt task quality, so track it.

### Downstream impact

Run the task eval with and without redaction. A good pseudonymization scheme should cost under 1-2 points of task quality. If it costs more, you are over-redacting or losing coreference.

### Residual risk accounting

Turn recall into an absolute number for stakeholders. At 1 million requests a day with an average of 2 PII spans each, that is 2 million spans a day. At 95% recall, about 100,000 spans a day still reach the model (`2,000,000 × 0.05`). Even at 99.5%, it is 10,000. State the number explicitly. It is why redaction is combined with vendor contractual controls and is never sold as a guarantee.

### Production monitoring

- Weekly human review of a privacy-safe sample of post-redaction traffic.
- Canary tests that inject synthetic PII into requests and confirm it never appears at the model endpoint or in logs.
- Gateway latency at p95 and p99 against the 20-50 ms budget.

## Tradeoffs

| Option | Pros | Cons | When to choose |
|---|---|---|---|
| Regex plus checksum only | Fast, precise, explainable | Misses names, addresses, free-text PII | Structured IDs, PCI scope; always the first layer |
| NER model (fine-tuned) | Good recall on names and locations, low latency | Domain shift, needs labeled data, multilingual gaps | General free-text traffic |
| LLM-based detector | Catches quasi-identifiers and context | Costly, slower, itself a data sink unless self-hosted | High-risk or regulated flows, offline batch |
| Irreversible masking | Simplest privacy story | Breaks tasks that need identity or coreference | Analytics, classification, training data |
| Pseudonymization plus vault | Preserves utility, enables re-identification | The vault is a crown-jewel asset; more complexity | Assistants that must reply with real names |
| Self-hosted model, no redaction | No leak to a vendor, full utility | Infrastructure cost, capability gap | When redaction recall can't meet the compliance bar |

- Tune thresholds per entity type by risk. Over-redact SSNs and accept some missed city names.
- Sometimes the right answer is to move the model inside the trust boundary, not to redact harder.

## Common mistakes

- **Redacting the prompt but not the rest of the data path.** Logs, traces, embeddings and the vector store, eval sets and fine-tuning data all keep the raw PII, so the leak just moves.
- **Quoting precision or overall F1.** Recall per high-risk type is what determines the leak rate; a high F1 can hide poor recall on the rare, dangerous types.
- **Destroying coreference with a generic `[REDACTED]`.** The model confuses people and task quality collapses.
- **Ignoring quasi-identifiers.** ZIP code, birth date and gender together re-identify most people even with no direct identifiers: Sweeney (2000) estimated that 87% of the US population is uniquely identified by 5-digit ZIP, gender and date of birth.
- **Ignoring non-text modalities.** Images, OCR'd scans, audio transcripts and file metadata all carry PII that a text-only detector never sees.
- **Assuming the output is safe.** Models can echo surrogates as real facts or reconstruct PII from context.

## Thinking at scale

- Own the gateway centrally. A platform or security team owns the detectors, the vault and the policies; product teams consume the gateway. Detector improvements then reach every product at once, and audits have one place to look.
- Treat the vault as tier-zero infrastructure. It is the one component that holds re-identifiable data for every request. It needs its own threat model, key management, access reviews and incident runbook.
- Version detectors like models. Each detector change re-runs the labeled test set, and recall regressions on any high-risk type block the release.
- Revisit the build-vs-move decision. As self-hosted open models improve, the cheapest way to meet a strict compliance bar can shift from "redact harder" to "move the model inside the trust boundary". Re-check that tradeoff when the capability gap narrows.

## Further reading

- [Microsoft Presidio documentation](https://microsoft.github.io/presidio/): an open-source PII detection and anonymization framework with pluggable recognizers.
- [Google Cloud Sensitive Data Protection documentation](https://docs.cloud.google.com/sensitive-data-protection/docs): a managed DLP (data loss prevention) service with infoTypes, de-identification transforms and date shifting.
- [NIST SP 800-122: Guide to Protecting the Confidentiality of PII (McCallister et al., 2010)](https://csrc.nist.gov/pubs/sp/800/122/final): the canonical PII definitions and risk-based safeguards.
- [Extracting Training Data from Large Language Models (Carlini et al., 2021)](https://arxiv.org/abs/2012.07805): shows that models memorize and regurgitate rare PII.
- [Analyzing Leakage of Personally Identifiable Information in Language Models (Lukas et al., 2023)](https://arxiv.org/abs/2302.00539): measures PII leakage and the limits of scrubbing and differential privacy.
- [Mind the Gap: Robustness Risks in PII Detection Systems (Zafar and Nowaczyk, 2026)](https://arxiv.org/abs/2609.03464): rule-based and NER detectors, including Presidio and spaCy, degrade significantly on out-of-distribution inputs.
- [Simple Demographics Often Identify People Uniquely (Sweeney, 2000)](https://dataprivacylab.org/projects/identifiability/paper1.pdf): the 87% ZIP, gender and birth date re-identification result.
- [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/): LLM02 Sensitive Information Disclosure as a top LLM risk.
