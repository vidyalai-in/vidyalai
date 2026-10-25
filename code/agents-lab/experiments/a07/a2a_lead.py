"""The Part 7 lead agent, delegating over A2A to the worker server in a2a_worker.py.

python a2a_lead.py TASK_ID [TASK_ID ...] -> runs the lead on each task, logs the wire to runs/a2a-wire.jsonl
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/a05")
import ledger  # noqa: E402
from harness import Tool  # noqa: E402
from multiagent import LEAD, Limited, Run, loop  # noqa: E402
from store import final_answer  # noqa: E402

HERE = Path(__file__).parent
WIRE = HERE / "runs" / "a2a-wire.jsonl"
URL = "http://127.0.0.1:9999"


async def _log_request(req):
    with WIRE.open("a") as f:
        f.write(json.dumps({"t": time.time(), "dir": "client→server", "method": req.method, "url": str(req.url),
                            "headers": {k: v for k, v in req.headers.items() if k.lower().startswith(("a2a", "content-type"))},
                            "body": req.content.decode() if req.content else ""}) + "\n")


async def _log_response(resp):
    await resp.aread()
    with WIRE.open("a") as f:
        f.write(json.dumps({"t": time.time(), "dir": "server→client", "status": resp.status_code,
                            "body": resp.text}) + "\n")


async def send(brief):
    async with httpx.AsyncClient(timeout=300, event_hooks={"request": [_log_request],
                                                           "response": [_log_response]}) as http:
        client = await create_client(URL, client_config=ClientConfig(streaming=False, httpx_client=http))
        request = SendMessageRequest(message=new_text_message(brief, role=Role.ROLE_USER))
        texts, state = [], None
        async for chunk in client.send_message(request):
            task = chunk.task if chunk.HasField("task") else None
            if task:
                state = task.status.state
                texts = [p.text for a in task.artifacts for p in a.parts if p.text]
        await client.close()
    return "\n".join(texts), state


def delegate(task):
    text, state = asyncio.run(send(task))
    return text if state == 3 else f"[worker task state {state}] {text}"  # 3 = TASK_STATE_COMPLETED


A2A_DELEGATE = Tool("delegate", "Start a worker agent with a fresh context and the incident tracker tools. "
                    "It sees only the brief you write here and returns its final report.",
                    {"task": {"type": "string", "description": "The complete, self-contained brief for the worker"}},
                    delegate)

if __name__ == "__main__":
    tasks = {t["id"]: t for t in json.loads((HERE / "tasks.json").read_text())}
    CL = Limited(ledger.client().with_options(max_retries=6), n=3)
    for tid in sys.argv[1:]:
        t0 = time.time()
        r = loop(tasks[tid]["question"] + f"\n\n(Request ID: a2a-{tid})", [A2A_DELEGATE], LEAD, part="a07",
                 role="a2a-lead", client=CL, max_steps=12, max_cost=0.25, run=Run("lead"),
                 transcript=HERE / "runs" / f"a2a-lead-{tid}.jsonl")
        print(tid, final_answer(r.answer), "expected", tasks[tid]["answer"], f"lead ${r.cost:.4f}",
              f"{time.time() - t0:.1f}s")
