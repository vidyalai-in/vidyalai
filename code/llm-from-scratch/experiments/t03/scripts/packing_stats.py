"""Packing vs padding, measured on the packed train stream."""
import sys, json
import numpy as np
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/fromscratch")
import data
T = 1024
file_of, start, length = data.load_documents()
ids = np.load(f"{data.SHARD_DIR}/train_documents.npy")
L = length[ids].astype(np.int64)                         # tokens per doc incl. the trailing id 0
total = L.sum()
# Padding: one document per row, long documents cut into 1024-token rows, the last row padded.
rows = -(-L // T)
pad_frac = 1 - total / (rows.sum() * T)
# Packing: where do document boundaries fall in 1024-token windows?
ends = np.cumsum(L)                                      # position after each doc's id 0
n_win = total // T
win_of_end = (ends[ends <= n_win * T] - 1) // T          # window holding each doc's last token
eots_per_window = np.bincount(win_of_end, minlength=n_win)
# A token "sees another document" if an earlier token in its window belongs to a different document.
starts = ends - L
first_in_window = (starts % T == 0)                      # docs that start exactly at a window start
# tokens whose window started before their document did:
doc_start_off = starts % T                               # offset of the doc's first token in its window
cross = 0
for s, l in zip(doc_start_off, L):
    cross += min(l, T - s) if s > 0 else 0               # the doc's tokens in its first window, if it started mid-window
res = dict(train_docs=len(L), train_tokens=int(total), doc_len_median=float(np.median(L - 1)), doc_len_mean=float((L - 1).mean()),
           doc_len_p90=float(np.percentile(L - 1, 90)), docs_over_1024=float(((L - 1) > T).mean()), tokens_in_docs_over_1024=float(L[(L - 1) > T].sum() / total),
           padding_rows=int(rows.sum()), padding_fraction=float(pad_frac),
           windows=int(n_win), windows_with_boundary=float((eots_per_window > 0).mean()), mean_docs_per_window=float(eots_per_window.mean() + 1),
           tokens_after_a_boundary_in_window=float(cross / total))
print(json.dumps(res, indent=1)); json.dump(res, open("/home/amit/.cache/vidyalai-scratch/t03/packing.json", "w"), indent=1)
