---
title: "Symbols and operations 2: interchanging, new operations and coded signs"
description: "Swap signs or numbers to make an equation true, work with newly defined operations such as a ★ b = 2a + b, and decode symbols for greater than, less than and equal to."
slug: symbols-operations-2
family: symbols-operations
part: 2
timeMinutes: 12
questions:
  - id: so2-star-op
    topic: Defined operations
    difficulty: 1
    q: "If a ★ b = 2a + b, what is 5 ★ 3?"
    options: ["8", "11", "13", "16"]
    answer: C
    explanation: |
      Put a = 5 and b = 3: 2 × 5 + 3 = 10 + 3 = **13**.
      11 swaps the numbers (2 × 3 + 5). 16 doubles both, 2 × (5 + 3).

  - id: so2-hash-op
    topic: Defined operations
    difficulty: 1
    q: "If a # b = a² − b, what is 5 # 8?"
    options: ["2", "9", "17", "59"]
    answer: C
    explanation: |
      a² − b = 5 × 5 − 8 = 25 − 8 = **17**.
      59 swaps the numbers (8² − 5). 2 doubles a instead of squaring it (2 × 5 − 8).

  - id: so2-swap-signs-easy
    topic: Interchanging
    difficulty: 1
    q: "Which two signs must be interchanged to make this equation correct? 6 + 4 × 2 − 3 = 23"
    options: ["+ and −", "+ and ×", "− and ×", "× and ÷"]
    answer: B
    explanation: |
      Swap + and ×: 6 × 4 + 2 − 3 = 24 + 2 − 3 = **23** ✓, so the answer is **+ and ×**.
      The others give: + and − → 6 − 4 × 2 + 3 = 1; − and × → 6 + 4 − 2 × 3 = 4; × and ÷ → 6 + 4 ÷ 2 − 3 = 5.

  - id: so2-coded-simple
    topic: Coded inequalities
    difficulty: 1
    q: "A @ B means A > B, A # B means A < B, and A $ B means A = B. If P @ Q and Q $ R, which of these must be true?"
    options: ["P @ R", "P # R", "P $ R", "R @ P"]
    answer: A
    explanation: |
      Decode: P > Q and Q = R. So P > R, which is written **P @ R**.
      P # R would mean P < R, the opposite.

  - id: so2-swap-numbers
    topic: Interchanging
    difficulty: 2
    q: "Which two numbers must be interchanged to make this equation correct? 12 ÷ 4 + 6 × 2 − 3 = 27"
    options: ["4 and 6", "4 and 2", "6 and 3", "2 and 3"]
    answer: B
    explanation: |
      Swap 4 and 2: 12 ÷ 2 + 6 × 4 − 3 = 6 + 24 − 3 = **27** ✓, so the answer is **4 and 2**.
      The others: 4 and 6 → 12 ÷ 6 + 4 × 2 − 3 = 7; 6 and 3 → 12 ÷ 4 + 3 × 2 − 6 = 3; 2 and 3 → 12 ÷ 4 + 6 × 3 − 2 = 19.

  - id: so2-nested-op
    topic: Defined operations
    difficulty: 2
    q: "If a ◇ b = a × b − a − b, what is (3 ◇ 4) ◇ 2?"
    options: ["1", "3", "5", "10"]
    answer: B
    explanation: |
      Brackets first: 3 ◇ 4 = 12 − 3 − 4 = 5. Then 5 ◇ 2 = 10 − 5 − 2 = **3**.
      5 stops after the first step. 1 works out 3 ◇ (4 ◇ 2) instead, which uses the wrong bracket.

  - id: so2-solve-op
    topic: Defined operations
    difficulty: 2
    q: "If a ▲ b = 3a − 2b and x ▲ 4 = 13, what is x?"
    options: ["3", "5", "7", "11"]
    answer: C
    explanation: |
      x ▲ 4 = 3x − 2 × 4 = 3x − 8. So 3x − 8 = 13, which gives 3x = 21 and x = **7**.
      Check: 3 × 7 − 8 = 21 − 8 = 13 ✓. 5 is 13 − 8, which subtracts the 8 instead of adding it and never divides by 3.

  - id: so2-coded-chain
    topic: Coded inequalities
    difficulty: 2
    q: "@ means >, # means <, % means ≥, & means ≤ and $ means =. If M % N, N @ K and K $ L, which of these must be true?"
    options: ["L @ M", "M $ L", "N # L", "M @ L"]
    answer: D
    explanation: |
      Decode: M ≥ N, N > K and K = L. Join them: M ≥ N > K = L, so M > L, which is **M @ L**.
      M $ L (M = L) is impossible, because the chain has a strict > in it. N # L means N < L, but N > L.

  - id: so2-swap-signs-hard
    topic: Interchanging
    difficulty: 2
    q: "Which two signs must be interchanged to make this equation correct? 16 ÷ 4 + 2 × 3 − 5 = 17"
    options: ["+ and −", "− and ×", "× and ÷", "+ and ÷"]
    answer: D
    explanation: |
      Swap + and ÷: 16 + 4 ÷ 2 × 3 − 5 = 16 + 6 − 5 = **17** ✓, so the answer is **+ and ÷**.
      The others: + and − → 16 ÷ 4 − 2 × 3 + 5 = 3; − and × → 16 ÷ 4 + 2 − 3 × 5 = −9; × and ÷ → 16 × 4 + 2 ÷ 3 − 5 is not even a whole number.

  - id: so2-coded-conclusions
    topic: Coded inequalities
    difficulty: 3
    q: "@ means >, # means <, % means ≥, & means ≤ and $ means =. Given A % B, B $ C, C @ D and D & E, which conclusion must be true? I: A @ D. II: A @ E."
    options: ["Only I", "Only II", "Both I and II", "Neither I nor II"]
    answer: A
    explanation: |
      Decode: A ≥ B, B = C, C > D, D ≤ E. So A ≥ B = C > D, which gives A > D: I is true.
      E is only known to be at least D. E could be bigger than A, so II need not be true. **Only I**.

  - id: so2-conditional-op
    topic: Defined operations
    difficulty: 3
    q: "a ⊕ b means a + b when a is greater than b, and a × b otherwise. What is (6 ⊕ 2) ⊕ (3 ⊕ 4)?"
    options: ["15", "20", "84", "96"]
    answer: D
    explanation: |
      6 ⊕ 2: 6 > 2, so add: 8. 3 ⊕ 4: 3 is not greater than 4, so multiply: 12.
      8 ⊕ 12: 8 is not greater than 12, so multiply: 8 × 12 = **96**.
      20 adds in the last step without checking the rule again. 84 uses the rule the wrong way round, and 15 just adds all four numbers.

  - id: so2-swap-both
    topic: Interchanging
    difficulty: 3
    q: "If the signs ÷ and + are interchanged, and the numbers 3 and 6 are interchanged, which equation becomes correct?"
    options: ["3 + 2 ÷ 6 × 4 = 15", "18 + 6 ÷ 4 = 7", "12 ÷ 6 + 5 = 9", "8 + 6 ÷ 2 = 11"]
    answer: A
    explanation: |
      Make both changes in each one. **3 + 2 ÷ 6 × 4 = 15** becomes 6 ÷ 2 + 3 × 4 = 3 + 12 = 15 ✓.
      18 + 6 ÷ 4 becomes 18 ÷ 3 + 4 = 10, not 7 (it works only if you swap the signs alone). 12 ÷ 6 + 5 becomes 12 + 3 ÷ 5, not a whole number. 8 + 6 ÷ 2 = 11 is true as written, but becomes 8 ÷ 3 + 2.
---

This set has three kinds of puzzles. Each one asks you to follow a new rule exactly.

## Interchanging signs or numbers

You are told an equation is wrong and that swapping two signs (or two numbers) will fix it. Try each option: make the swap **everywhere** in the equation, work out the left side with BODMAS, and compare it with the right side. Only one option will match.

A quick check helps: if the right side is much bigger than the left, the swap probably creates a ×.

## Defined operations

A new symbol is defined by a formula, like a ★ b = 2a + b. Put the first number in place of a and the second in place of b. **Order matters**: 5 ★ 3 = 13 but 3 ★ 5 = 11.

With brackets, work out the inside first, then use that answer as a new number. If you are given the answer and asked for a missing number, write the formula as an equation and solve it.

## Coded inequalities

Symbols stand for >, <, ≥, ≤ and =. Decode every statement, then join them into one chain, such as P > Q = R ≥ S.

| Chain | Can you conclude? |
|---|---|
| A > B = C | A > C, yes |
| A ≥ B > C | A > C, yes |
| A > B < C | nothing about A and C |

A conclusion **must** be true only if the chain links the two letters with signs pointing the same way. If the signs point opposite ways, you cannot say anything.

## Worked example

If a ♦ b = a × b + a, then 4 ♦ 3 = 12 + 4 = 16.

## Common traps

- Swapping a sign in one place but not everywhere.
- Changing the order of a and b in a defined operation.
- Thinking ≥ and > are the same. A ≥ B allows A = B.
