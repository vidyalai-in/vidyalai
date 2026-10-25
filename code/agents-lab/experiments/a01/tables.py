import json, statistics as st
from analyze import load, wilson, boot_tasks
def M(x): return st.mean(x)
main = {a: [r for r in load(a) if r["kind"] != "probe"] for a in ("agent", "rag", "stuff")}
print("ARM  acc  wilson  taskboot  calls/task  cost/task  model-s/task  in/task  out/task")
for a, rs in main.items():
    k=sum(r["correct"] for r in rs); n=len(rs); lo,hi=wilson(k,n); blo,bhi=boot_tasks(rs)
    print(a, f"{k}/{n}={k/n:.3f} [{lo:.3f},{hi:.3f}] boot[{blo:.3f},{bhi:.3f}]",
          f"{M([len(r['steps']) for r in rs]):.2f}", f"${M([r['cost'] for r in rs]):.6f}",
          f"{M([sum(s['latency_s'] for s in r['steps']) for r in rs]):.2f}s",
          f"{M([sum(s['input_tokens']+s['cache_read']+s['cache_write'] for s in r['steps']) for r in rs]):.0f}",
          f"{M([sum(s['output_tokens'] for s in r['steps']) for r in rs]):.0f}")
    for kind in ("lookup","multi-hop","aggregate","long"):
        q=[r for r in rs if r["kind"]==kind]; k=sum(r["correct"] for r in q); lo,hi=wilson(k,len(q))
        print(f"   {kind:9s} {k}/{len(q)} {k/len(q):.3f} [{lo:.3f},{hi:.3f}] cost ${M([r['cost'] for r in q]):.6f} calls {M([len(r['steps']) for r in q]):.2f} model-s {M([sum(s['latency_s'] for s in r['steps']) for r in q]):.2f}")
# tool calls per run, parallel calls
ag = main["agent"]
tc = [sum(len(s["tool_calls"]) for s in r["steps"]) for r in ag]
par = sum(1 for r in ag for s in r["steps"] if len(s["tool_calls"])>1); tot=sum(1 for r in ag for s in r["steps"] if s["tool_calls"])
print("agent tool calls/task", M(tc), "max", max(tc), "| steps with >1 call:", par, "of", tot)
from collections import Counter
print("steps dist", sorted(Counter(len(r["steps"]) for r in ag).items()))
allsteps=[s for r in ag for s in r["steps"]]
print("latency/step mean %.2f median %.2f p90 %.2f" % (M([s['latency_s'] for s in allsteps]), st.median([s['latency_s'] for s in allsteps]), sorted(s['latency_s'] for s in allsteps)[int(.9*len(allsteps))]))
print("output/step mean %.0f median %.0f; final-step out mean %.0f; tool-step out mean %.0f" % (M([s['output_tokens'] for s in allsteps]), st.median([s['output_tokens'] for s in allsteps]), M([r['steps'][-1]['output_tokens'] for r in ag]), M([s['output_tokens'] for s in allsteps if s['tool_calls']])))
print("latency tool steps %.2f final steps %.2f" % (M([s['latency_s'] for s in allsteps if s['tool_calls']]), M([r['steps'][-1]['latency_s'] for r in ag])))
# cache share
cr=sum(s['cache_read'] for s in allsteps); cw=sum(s['cache_write'] for s in allsteps); ui=sum(s['input_tokens'] for s in allsteps)
print(f"agent input split: read {cr} write {cw} uncached {ui} -> read share {cr/(cr+cw+ui):.3f}")
# caching on/off matched
off = [r for r in load("agent_nocache") if r["kind"]!="probe"]
ids = {r["task"] for r in off}
on = [r for r in ag if r["task"] in ids]
for lab, rs in (("on", on), ("off", off)):
    k=sum(r["correct"] for r in rs)
    print("cache", lab, len(rs), f"acc {k}/{len(rs)}", f"cost/task ${M([r['cost'] for r in rs]):.6f}", f"model-s {M([sum(s['latency_s'] for s in r['steps']) for r in rs]):.2f}",
          f"in/task {M([sum(s['input_tokens']+s['cache_read']+s['cache_write'] for s in r['steps']) for r in rs]):.0f}",
          f"first-step latency {M([r['steps'][0]['latency_s'] for r in rs]):.2f}", f"steps {M([len(r['steps']) for r in rs]):.2f}")
for a in ("agent","agent_nocache","agent_serial","agent_serial_nocache"):
    rs=[r for r in load(a) if r["task"]=="t31"]
    print("probe", a, len(rs), f"steps {M([len(r['steps']) for r in rs]):.1f}", f"cost ${M([r['cost'] for r in rs]):.5f}", f"model-s {M([sum(s['latency_s'] for s in r['steps']) for r in rs]):.1f}",
          f"in/task {M([sum(s['input_tokens']+s['cache_read']+s['cache_write'] for s in r['steps']) for r in rs]):.0f}", f"out/task {M([sum(s['output_tokens'] for s in r['steps']) for r in rs]):.0f}")
# sonnet vs haiku on long
so = load("sonnet"); hl=[r for r in ag if r["kind"]=="long"]
for lab, rs in (("haiku", hl), ("sonnet", so)):
    k=sum(r["correct"] for r in rs); lo,hi=wilson(k,len(rs))
    print(lab, f"{k}/{len(rs)} {k/len(rs):.3f} [{lo:.3f},{hi:.3f}]", f"cost ${M([r['cost'] for r in rs]):.5f}", f"steps {M([len(r['steps']) for r in rs]):.2f}",
          f"model-s {M([sum(s['latency_s'] for s in r['steps']) for r in rs]):.2f}", f"in {M([sum(s['input_tokens']+s['cache_read']+s['cache_write'] for s in r['steps']) for r in rs]):.0f}", f"out {M([sum(s['output_tokens'] for s in r['steps']) for r in rs]):.0f}",
          "per task", {t: sum(r['correct'] for r in rs if r['task']==t) for t in sorted({r['task'] for r in rs})})
