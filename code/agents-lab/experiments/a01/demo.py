import json, sys
from pathlib import Path
from docs_agent import TOOLS, SYSTEM, grade, run_agent

tasks = {t["id"]: t for t in json.loads(Path("tasks.json").read_text())}
t = tasks[sys.argv[1]]
result = run_agent(t["question"], TOOLS, SYSTEM, part="a01", thinking="summarized",
                   transcript=f"runs/demo-{t['id']}-summarized.jsonl")
print(result.answer)
print("correct:", grade(t, result.answer), "| steps:", len(result.steps), f"| cost: ${result.cost:.5f}")
