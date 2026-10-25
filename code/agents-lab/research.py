"""Part 10: research agents on the Claude API's server-side web search and web fetch tools.

The API runs search and fetch itself, inside one request, so there is no client tool loop:
the client sends the question, the server loops (think, search, read, fetch, ...) and returns
the finished answer with citations. Long turns come back with stop_reason "pause_turn", and the
client sends the paused content back to continue. Every request goes through ledger.create.

  TOOLS        tool configurations for the experiment's arms
  research()   one question, one arm: handles pause_turn, records usage, cost and latency
  tool_calls() searches and fetches the server ran, read from the response content
  citations()  every cited span in the final answer: claim text, URL, cited text
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ledger  # noqa: E402
from harness import _TimedClient  # noqa: E402

SEARCH = {"type": "web_search_20250305", "name": "web_search"}            # basic search
FETCH = {"type": "web_fetch_20250910", "name": "web_fetch",               # basic fetch
         "citations": {"enabled": True}, "max_content_tokens": 12_000}
SEARCH_DF = {"type": "web_search_20260318", "name": "web_search"}         # dynamic filtering
FETCH_DF = {"type": "web_fetch_20260318", "name": "web_fetch",
            "citations": {"enabled": True}, "max_content_tokens": 12_000}

TOOLS = {
    "one-search": [{**SEARCH, "max_uses": 1}],
    "research": [{**SEARCH, "max_uses": 6}, {**FETCH, "max_uses": 4}],
    "research-df": [{**SEARCH_DF, "max_uses": 6}, {**FETCH_DF, "max_uses": 4}],
}

DATE = "Today's date is 2026-10-25."
ONE_SEARCH = f"""You answer factual questions with the help of one web search. {DATE}
Answer briefly, then end with a final line "ANSWER: <answer>" in the format the question asks for."""
RESEARCH = f"""You are a careful research assistant. {DATE}
1. Work out which facts the question needs.
2. Search for each fact. Prefer primary sources: the official site of the project, organization or event.
3. When a search snippet is not conclusive, fetch the primary page and read the exact value.
4. If sources disagree, trust the primary source and the most recent page.
5. Do any date or number arithmetic step by step.
Answer briefly, citing your sources, then end with a final line "ANSWER: <answer>" in the format the
question asks for."""
SYSTEM = {"one-search": ONE_SEARCH, "research": RESEARCH, "research-df": RESEARCH}


def research(question, arm, *, part="a10", model="claude-haiku-5-5", effort="medium",
             max_tokens=8000, max_continuations=4, transcript=None, client=None):
    """Answer one question with one arm's tools. Returns a dict of answer, usage, cost, latency."""
    request = dict(model=model, max_tokens=max_tokens, tools=TOOLS[arm],
                   system=[{"type": "text", "text": SYSTEM[arm], "cache_control": {"type": "ephemeral"}}],
                   output_config={"effort": effort}, thinking={"type": "adaptive"},
                   cache_control={"type": "ephemeral"}, timeout=600)
    cl = _TimedClient(client or ledger.client())
    messages = [{"role": "user", "content": question}]
    content, turn, calls, cost, latency, usage = [], [], 0, 0.0, 0.0, []
    for _ in range(1 + max_continuations):
        resp = ledger.create(cl, part, note=f"a10 {arm}", messages=messages, **request)
        calls += 1
        latency += cl.last
        cost += ledger.cost_of(model, resp.usage)
        usage.append(resp.usage.model_dump())
        content += resp.content
        turn += resp.content  # the assistant turn in progress
        if resp.stop_reason == "pause_turn":
            # A paused turn: send the assistant content back unchanged and let the server continue.
            if messages[-1]["role"] == "assistant":
                messages[-1] = {"role": "assistant", "content": turn}
            else:
                messages.append({"role": "assistant", "content": turn})
        elif resp.stop_reason == "tool_use":
            # The model called a client tool, but there are none here: say so and continue.
            messages += [{"role": "assistant", "content": turn},
                         {"role": "user", "content": [
                             {"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                              "content": f"There is no tool named {b.name}. Use web_search or web_fetch."}
                             for b in turn if b.type == "tool_use"]}]
            turn = []
        else:
            break
    text = "".join(b.text for b in content if b.type == "text")
    m = re.findall(r"ANSWER:\s*(.+)", text)
    out = dict(answer=m[-1].strip() if m else "", text=text, stop=resp.stop_reason, api_calls=calls,
               cost=cost, latency_s=round(latency, 2), usage=usage, **tool_calls(content))
    if transcript:
        with open(transcript, "a") as f:
            f.write(json.dumps({"question": question, "arm": arm, "model": model, "effort": effort,
                                "content": [b.model_dump(exclude_none=True) for b in content],
                                **{k: v for k, v in out.items() if k != "text"}}, default=str) + "\n")
    return out


def tool_calls(content):
    """Count the searches and fetches the server ran, with their queries and URLs."""
    searches, fetches, errors, client_calls = [], [], [], []
    for b in content:
        if b.type == "tool_use":  # a client tool call: none exist in this experiment
            client_calls.append(b.name)
        elif b.type == "server_tool_use" and b.name == "web_search":
            searches.append(b.input.get("query"))
        elif b.type == "server_tool_use" and b.name == "web_fetch":
            fetches.append(b.input.get("url"))
        elif b.type in ("web_search_tool_result", "web_fetch_tool_result"):
            code = getattr(b.content, "error_code", None)  # a result list has none
            if code:
                errors.append(f"{b.name if hasattr(b, 'name') else b.type}:{code}")
    return dict(searches=searches, fetches=fetches, tool_errors=errors, client_calls=client_calls)


def citations(content):
    """Every cited span in a response: the text it supports and where the citation points."""
    docs = {}  # fetched documents by order, to resolve char_location citations to a URL
    for b in content:
        if b.type == "web_fetch_tool_result" and getattr(b.content, "type", "") == "web_fetch_result":
            docs[len(docs)] = b.content.url
    out = []
    for b in content:
        for c in (getattr(b, "citations", None) or []):
            url = getattr(c, "url", None) or docs.get(getattr(c, "document_index", -1))
            out.append(dict(claim=b.text, type=c.type, url=url, title=getattr(c, "title", None)
                            or getattr(c, "document_title", None), cited_text=c.cited_text))
    return out
