"""Exp 2, LLM stage: Claude Haiku 5.5 streaming, thinking disabled. Time to first token, to the first
complete sentence and to the end of the reply, for the 40 user turns (their reference text)."""
import sys, json, time, re
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/agents/lab")
import ledger
D = "/home/amit/.cache/vidyalai-scratch/v11"
MODEL = "claude-haiku-5-5"
SYSTEM = ("You are a voice assistant. Your reply is read aloud, so answer in plain spoken English: "
          "one to three short sentences, no lists, no markdown.")
END = re.compile(r"[.!?](?=\s)")
turns = [l.strip() for l in open(f"{D}/data/user_turns.txt") if l.strip()]
n_rep = int(sys.argv[1]); only = int(sys.argv[2]) if len(sys.argv) > 2 else len(turns)
out = open(f"{D}/runs/claude_stream.jsonl", "a")
cl = ledger.client()
est = ledger.estimate_cost(MODEL, 120, 200)
for rep in range(n_rep):
    for i, t in enumerate(turns[:only]):
        ledger.guard("v11", est)
        rec = {"turn": i, "rep": rep, "wall": time.time()}
        t0 = time.perf_counter()
        text, t_first, t_sent = "", None, None
        try:
            with cl.messages.stream(model=MODEL, max_tokens=200, system=SYSTEM,
                                    thinking={"type": "disabled"},
                                    messages=[{"role": "user", "content": t}]) as s:
                for d in s.text_stream:
                    now = time.perf_counter() - t0
                    if t_first is None and d:
                        t_first = now
                    text += d
                    if t_sent is None and END.search(text):
                        t_sent = now
                msg = s.get_final_message()
            t_end = time.perf_counter() - t0
            if t_sent is None:
                t_sent = t_end
            m = END.search(text)
            first = text[: m.end()] if m else text
            rec.update(ttft=t_first, t_first_sentence=t_sent, t_total=t_end, stop=msg.stop_reason,
                       usage=msg.usage.model_dump(), first_chars=len(first.strip()), full_chars=len(text.strip()),
                       n_sent=len(END.findall(text + " ")), text=text)
            rec["cost"] = ledger.record("v11", MODEL, msg.usage, note=f"v11 stream turn {i} rep {rep}")
        except Exception as e:
            rec["error"] = repr(e)
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(i, rep, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k in ("ttft", "t_first_sentence", "t_total", "first_chars", "full_chars", "stop", "error")}, flush=True)
