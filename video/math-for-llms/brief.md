# Brief: The Math You Need for LLMs, and the Shapes Inside One

Article: `src/content/blog/m01-math-for-llms.md` (The Math Behind LLMs, Part 1)

**Promise.** After this, the viewer can look at a matrix multiply in model code and predict the shape
it produces and roughly what it costs.

**Viewer.** Anyone who has used a chatbot and remembers high-school algebra: programmers, students,
curious readers. No linear algebra assumed. The opening says "if you remember high-school algebra,
that's enough".

**The sentence to remember.** An LLM is dot products plus bookkeeping: track the shapes, and you can
follow the whole model.

**Teaching shape.** One mechanism, layer by layer: the four objects, then the one operation, then
the shape rule and its cost, then the same rule followed through a real model, then the notation
real code uses. A map of the series closes it.

## Claims, not sections

| # | Claim | Article section | Scene | Keep |
|---|---|---|---|---|
| 1 | One token costs Llama 3.1 8B about 15.5B FLOPs; more than 99.9% is dot products | Why it matters | cold-open `stats` | core |
| 2 | Scalar, vector, matrix, tensor = cell, column, sheet, workbook | The intuition, Step 1 | `bullets` | core |
| 3 | A shape is the list of axis sizes; multiply them to count numbers: (2, 5, 4096) = 40,960 | Tensors and shapes | `stats` | core |
| 4 | Where the spreadsheet analogy breaks (no names, 5 axes, all learned) | The intuition | `bullets` | cuttable |
| 5 | Dot product: multiply position by position, add: [1,2,3]·[7,9,11] = 58 | Step 2 | `stats` | core |
| 6 | A matrix multiply is a grid of dot products: AB = [[58, 64], [139, 154]] | Matrix multiplication, by hand | `stats` + quiz | core |
| 7 | Shape rule (m×k)(k×n) → (m×n); order matters | The shape rule | `statement` + quiz | core |
| 8 | Cost ≈ 2mkn FLOPs; 8,000 tokens reuse the same weights 8,000 times | Counting the work | `statement` + `stats` | core |
| 9 | Seven config numbers set every shape in Llama 3.1 8B | The numbers that set every shape | `bullets` | core |
| 10 | Shapes through one forward pass; every block is (B, T, 4096) in and out; scores are T×T per head, the only place tokens meet | Following the data | `diagram` flow | core |
| 11 | MLP 75%, attention projections 18%, output table 7% of weight arithmetic | Following the data | `chart` | core |
| 12 | "2 × parameters" = 16.06B; corrections give 15.53B, within 4% | Checking against the rule of thumb | quiz + `stats` | core |
| 13 | At long context the rule breaks: 128,000 tokens → 82B per token, 5× | Checking against the rule of thumb | `stats` | core |
| 14 | Einsum: letters name axes; letters missing from the output get summed | Naming shapes with einsum | `code` | core (pays off the open loop) |
| 15 | PyTorch stores weights (out, in); servers keep only the last logits | In the real world | `bullets` | cuttable |
| 16 | Misconceptions: advanced math, exotic tensors | Common misconceptions | `bullets` + skeptic | core |
| 17 | Four toolkits, 16 parts | Step 4 | `bullets` + `seriesmap` | core |

## Left out on purpose

- The transpose (one sentence at most), and the 12-multiplies-8-adds footnote to 2mkn.
- The batch totals (122 trillion FLOPs for 8 × 1,000 tokens) and the 256M-number scores tensor.
- The NumPy snippet and the 5-axis grouped-query einsum: too dense to read on video; the article has it.
- Hugging Face names (`q_proj`, `repeat_kv`, `logits_to_keep`) and the per-block parameter count.

## Shorts

1. **Dot products**: 15.5B operations per token, nearly all one sum a twelve-year-old can do (claims 1, 5).
2. **The shape rule**: the one matrix rule worth memorizing, with the order-matters twist (claim 7).
3. **Long context**: the "2 × parameters" rule is 5× off at 128,000 tokens (claims 12, 13).

## Figures

None.
