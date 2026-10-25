# Code behind the articles

The code the articles run, kept here so it isn't lost. The site build ignores this folder.

## llm-from-scratch/ (Build an LLM From Scratch)

| File | Part | What it is |
|---|---|---|
| `train_tokenizer.py`, `tokenizer.py`, `tokenizer-32k.json` | 2 | BPE training, the tokenizer, and the trained 32K vocabulary |
| `data.py` | 3 | Download, MinHash dedup, tokenizing, packing into shards, `ShardLoader` |
| `model.py` | 4 | `GPT`, `GPTConfig`, `MAIN` (88M) and `SMALL` (22M) |
| `train.py` | 5 | The training loop (`--compile --fused` for speed, see Part 6) |

`experiments/t01` to `t06` hold each part's benchmark, ablation and chart-data scripts and small result files.

Not included (rebuildable or too large): the token shards in `data/` (about 2.5B tokens; rebuild with `data.py`), checkpoints, run logs, and downloaded papers and third-party source.

## agents-lab/ (Agentic AI)

| File | Part | What it is |
|---|---|---|
| `ledger.py` | 1 | API client, pricing, per-part and total budget caps, JSONL cost ledger |
| `harness.py` | 1 | The manual tool loop with prompt caching |
| `react.py` | 3 | `run_loop` with thinking, effort and timeout control; ReAct and plan-and-execute |
| `context.py` | 5 | Context strategies: sizing, context editing, compaction, memory file |
| `mcp_lab.py`, `mcp_demo.py`, `tickets_server.py` | 6 | MCP servers as harness tools, and the demo server |
| `multiagent.py` | 7 | A lead agent with a `delegate` tool that runs workers in fresh contexts |
| `coding.py` | 8 | Coding-agent tools (grep, read, string-replace edit, sandboxed test runner), repo map, arms |
| `computer.py` | 9 | Playwright executors for the computer and browser toolsets, localhost only |
| `research.py` | 10 | Web search and fetch server-tool loop with `pause_turn` handling |

`experiments/a01` to `a10` hold each part's task sets, runners and analysis scripts.

The API key is read from `~/.config/vidyalai/anthropic.env` and is never stored here. Not included: the cost ledger (`ledger.jsonl`), run transcripts, and fetched docs. Scripts use absolute paths from the machine they ran on (`/home/amit/.cache/vidyalai-scratch/...`); adjust them to rerun elsewhere.
