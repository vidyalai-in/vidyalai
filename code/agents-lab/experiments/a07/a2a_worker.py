"""The Part 7 worker agent as an A2A server (a2a-sdk 1.2.3, A2A protocol 1.0, JSON-RPC binding).

python a2a_worker.py [full|sized] -> serves http://127.0.0.1:9999, Agent Card at /.well-known/agent-card.json
"""
import asyncio
import sys

import uvicorn
from a2a.helpers import get_message_text, new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TaskState
from starlette.applications import Starlette

sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/a05")
import ledger  # noqa: E402
from multiagent import WORKER, Limited, Run, loop  # noqa: E402
from store import NAIVE, SIZED  # noqa: E402

TOOLS = SIZED if sys.argv[1:] == ["sized"] else NAIVE
CL = Limited(ledger.client().with_options(max_retries=6), n=3)

card = AgentCard(
    name="Larkspur incident analyst",
    description="Answers one self-contained question about Larkspur Robotics' incident tracker.",
    version="1.0.0",
    default_input_modes=["text/plain"], default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=False),
    supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url="http://127.0.0.1:9999",
                                         protocol_version="1.0")],
    skills=[AgentSkill(id="incident-lookup", name="Incident lookup",
                       description="Search incidents by service and month, read records, compute totals.",
                       tags=["incidents"], examples=["Total customer impact minutes on map-tiles in August 2026"])],
)


class WorkerExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task_from_user_message(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)
        await updater.update_status(state=TaskState.TASK_STATE_WORKING)
        run = await asyncio.to_thread(loop, get_message_text(context.message), TOOLS, WORKER, part="a07",
                                      role="a2a-worker", client=CL, max_steps=30, max_cost=0.06, run=Run("worker"))
        await updater.add_artifact(parts=[new_text_part(text=run.answer, media_type="text/plain")], name="report")
        await updater.update_status(state=TaskState.TASK_STATE_COMPLETED if run.stop == "end_turn"
                                    else TaskState.TASK_STATE_FAILED)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


handler = DefaultRequestHandler(agent_executor=WorkerExecutor(), task_store=InMemoryTaskStore(), agent_card=card)
app = Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/"))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9999, log_level="warning")
