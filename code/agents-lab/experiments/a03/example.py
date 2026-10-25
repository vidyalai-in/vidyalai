import sys
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
from react import DISABLED, REACT_RULES, plan_and_execute, run_loop
from saltmere import SYSTEM, TASKS, grade, make_tools

task = next(t for t in TASKS if t["id"] == "route06")
q = task["question"]
arms = {
    "plain": lambda: run_loop(q, make_tools(task), SYSTEM, part="a03", thinking=DISABLED, max_tokens=4000),
    "react": lambda: run_loop(q, make_tools(task), SYSTEM + REACT_RULES, part="a03", thinking=DISABLED,
                              max_tokens=4000),
    "plan": lambda: plan_and_execute(q, make_tools(task), SYSTEM, part="a03"),
    "high": lambda: run_loop(q, make_tools(task), SYSTEM, part="a03", effort="high", max_tokens=20000),
}
for name, run in arms.items():
    r = run()
    print(f"{name:6s} correct={grade(task, r.answer)}  calls={len(r.steps)}  cost=${r.cost:.5f}")
