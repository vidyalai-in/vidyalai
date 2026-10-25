"""Budget guard and usage ledger for every Claude API call in the VidyalAI series.

Every call goes through `create()` (or `record()` for batch results). Before a call, the guard
estimates its worst-case cost (all input uncached, max_tokens of output) and refuses the call if
that could push the part's spend or the project's spend past its cap. After the call, the exact
cost from `response.usage` is appended to ledger.jsonl.

The API key is read from ~/.config/vidyalai/anthropic.env and never printed or logged.
"""

import json
import os
import time
from pathlib import Path

import anthropic

LEDGER = Path(__file__).with_name("ledger.jsonl")
KEY_FILE = Path.home() / ".config/vidyalai/anthropic.env"

PROJECT_CAP = 90.00  # USD, the user's hard limit for the whole project

# Per-part caps (USD). Agentic AI: $75 in total; Evals: $15, shared with v04 and t13.
PART_CAPS = {
    "a01": 1, "a02": 2, "a03": 5, "a04": 4, "a05": 5, "a06": 3, "a07": 6, "a08": 10,
    "a09": 8, "a10": 5, "a11": 8, "a12": 3, "a13": 2, "a14": 0, "a15": 5, "a16": 5,
    "a-reserve": 3,
    "evals": 15,  # e01-e11, v04, t13 draw on this shared pool
}
EVALS_POOL = {f"e{n:02d}" for n in range(1, 12)} | {"v04", "t13"}

# USD per million tokens, from platform.claude.com/docs/en/about-claude/pricing (checked 2026-10-24).
# Haiku 5.5's long-prompt rates apply when a request's prompt (input + cache reads + writes) exceeds 100K.
# Sonnet 5.5 cache reads are 0.05x input; Haiku 5.5's are 0.1x. 5-minute cache writes are 1.25x input.
PRICES = {
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00, "cache_read": 0.10, "cache_write": 2.50},
    "claude-haiku-5-5": {"in": 0.10, "out": 0.50, "cache_read": 0.01, "cache_write": 0.125},
    "claude-haiku-5-5@long": {"in": 0.50, "out": 2.50, "cache_read": 0.05, "cache_write": 0.625},
}
WEB_SEARCH_PER_CALL = 0.01  # $10 per 1,000 searches
ALLOWED_MODELS = {"claude-sonnet-5-5", "claude-haiku-5-5"}


class BudgetExceeded(RuntimeError):
    pass


def _pool(part):
    return "evals" if part in EVALS_POOL else part


def client():
    key = None
    for line in KEY_FILE.read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            key = line.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError(f"No ANTHROPIC_API_KEY in {KEY_FILE}")
    return anthropic.Anthropic(api_key=key)


def _entries():
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def spent(part=None):
    """Total USD spent, for the project or for one part's pool."""
    return sum(e["cost"] for e in _entries() if part is None or _pool(e["part"]) == _pool(part))


def _rates(model, input_tokens):
    if model == "claude-haiku-5-5" and input_tokens > 100_000:
        return PRICES["claude-haiku-5-5@long"]
    return PRICES[model]


def cost_of(model, usage, batch=False):
    u = usage if isinstance(usage, dict) else usage.model_dump()
    inp = u.get("input_tokens") or 0
    cr = u.get("cache_read_input_tokens") or 0
    cw = u.get("cache_creation_input_tokens") or 0
    out = u.get("output_tokens") or 0
    r = _rates(model, inp + cr + cw)
    c = (inp * r["in"] + cr * r["cache_read"] + cw * r["cache_write"] + out * r["out"]) / 1e6
    searches = ((u.get("server_tool_use") or {}).get("web_search_requests") or 0)
    c += searches * WEB_SEARCH_PER_CALL
    return c * (0.5 if batch else 1.0)


def guard(part, estimate):
    """Raise BudgetExceeded unless `estimate` USD more fits under both caps."""
    pool = _pool(part)
    if pool not in PART_CAPS:
        raise BudgetExceeded(f"Unknown part {part!r}: no budget assigned")
    total, part_spent = spent(), spent(part)
    if total + estimate > PROJECT_CAP:
        raise BudgetExceeded(f"Project cap: spent ${total:.2f} of ${PROJECT_CAP:.2f}, next call up to ${estimate:.4f}")
    if part_spent + estimate > PART_CAPS[pool]:
        raise BudgetExceeded(f"{pool} cap: spent ${part_spent:.2f} of ${PART_CAPS[pool]:.2f}, next call up to ${estimate:.4f}")


def record(part, model, usage, note="", batch=False):
    c = cost_of(model, usage, batch)
    u = usage if isinstance(usage, dict) else usage.model_dump()
    with LEDGER.open("a") as f:
        f.write(json.dumps({"ts": time.time(), "part": part, "model": model, "batch": batch,
                            "note": note, "usage": u, "cost": c}) + "\n")
    return c


def create(cl, part, note="", **kwargs):
    """messages.create with a pre-call budget check and a ledger entry. Returns the response."""
    model = kwargs["model"]
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Model {model} not allowed (use Sonnet 5.5 or Haiku 5.5)")
    est_in = cl.messages.count_tokens(**{k: v for k, v in kwargs.items()
                                         if k in ("model", "messages", "system", "tools", "thinking")}).input_tokens
    r = _rates(model, est_in)
    estimate = (est_in * r["cache_write"] + kwargs.get("max_tokens", 4096) * r["out"]) / 1e6
    guard(part, estimate)
    resp = cl.messages.create(**kwargs)
    record(part, model, resp.usage, note)
    return resp


def report():
    by = {}
    for e in _entries():
        by[_pool(e["part"])] = by.get(_pool(e["part"]), 0) + e["cost"]
    lines = [f"{p:10s} ${v:7.4f} of ${PART_CAPS.get(p, 0):.2f}" for p, v in sorted(by.items())]
    lines.append(f"{'TOTAL':10s} ${spent():7.4f} of ${PROJECT_CAP:.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
