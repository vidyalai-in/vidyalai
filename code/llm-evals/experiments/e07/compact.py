"""Compact the extracted votes into one CSV of model names, outcome and style counts (no text)."""
import json, csv
F = ["id", "ts", "session", "order", "model_a", "model_b", "winner", "lang", "is_code", "turns",
     "tok_a", "tok_b", "hdr_a", "hdr_b", "list_a", "list_b", "bold_a", "bold_b", "n_criteria"]
with open("data/votes-140k.jsonl") as f, open("data/e07-votes.csv", "w", newline="") as g:
    w = csv.writer(g); w.writerow(F)
    for line in f:
        r = json.loads(line); m = r["conv_metadata"]
        crit = (r.get("category_tag") or {}).get("criteria_v0.1") or {}
        w.writerow([r["id"], r["timestamp"], r["evaluation_session_id"], r["evaluation_order"],
                    r["model_a"], r["model_b"], r["winner"], r["language"], int(bool(r["is_code"])),
                    m["turns"], m["sum_assistant_a_tokens"], m["sum_assistant_b_tokens"],
                    sum(m["header_count_a"].values()), sum(m["header_count_b"].values()),
                    sum(m["list_count_a"].values()), sum(m["list_count_b"].values()),
                    sum(m["bold_count_a"].values()), sum(m["bold_count_b"].values()),
                    sum(bool(v) for v in crit.values())])
