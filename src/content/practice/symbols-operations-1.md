---
title: "Symbols and operations 1: changed signs and BODMAS"
description: "Replace the signs as the question says, then work out the answer in the right order: brackets, division, multiplication, addition, subtraction."
slug: symbols-operations-1
family: symbols-operations
part: 1
timeMinutes: 12
questions:
  - id: so1-swap-plus-times
    topic: Two signs swapped
    difficulty: 1
    q: "If × means + and + means ×, what is the value of 7 × 5 + 3?"
    options: ["15", "22", "36", "38"]
    answer: B
    explanation: |
      Rewrite with the real signs: 7 + 5 × 3. Multiply first: 5 × 3 = 15, then 7 + 15 = **22**.
      36 works left to right, (7 + 5) × 3, which breaks the BODMAS order. 38 forgets to change the signs.

  - id: so1-swap-minus-plus
    topic: Two signs swapped
    difficulty: 1
    q: "If + means − and − means +, what is the value of 15 − 7 + 4?"
    options: ["4", "12", "18", "26"]
    answer: C
    explanation: |
      Rewrite: 15 + 7 − 4 = **18**.
      12 is the value without changing the signs.

  - id: so1-swap-divide-times
    topic: Two signs swapped
    difficulty: 1
    q: "If ÷ means × and × means ÷, what is the value of 12 ÷ 3 × 4?"
    options: ["9", "16", "36", "144"]
    answer: A
    explanation: |
      Rewrite: 12 × 3 ÷ 4. Division and multiplication are done left to right: 12 × 3 = 36, then 36 ÷ 4 = **9**.
      16 is the value without changing the signs.

  - id: so1-letter-signs
    topic: Letters and symbols
    difficulty: 1
    q: "If P means +, Q means − and R means ×, what is the value of 9 R 3 Q 4 P 2?"
    options: ["21", "25", "29", "33"]
    answer: B
    explanation: |
      Rewrite: 9 × 3 − 4 + 2. Multiply first: 27 − 4 + 2 = **25**.
      21 subtracts 4 + 2 instead of subtracting 4 and then adding 2.

  - id: so1-four-way
    topic: All four signs changed
    difficulty: 2
    q: "If + means ×, × means −, − means ÷ and ÷ means +, what is the value of 20 − 4 + 3 ÷ 6 × 2?"
    options: ["15", "17", "19", "23"]
    answer: C
    explanation: |
      Rewrite: 20 ÷ 4 × 3 + 6 − 2. Then 20 ÷ 4 = 5, 5 × 3 = 15, and 15 + 6 − 2 = **19**.
      17 is the value without changing the signs.

  - id: so1-shape-signs
    topic: Letters and symbols
    difficulty: 2
    q: "If ★ means ×, ◆ means −, ▲ means + and ■ means ÷, what is the value of 63 ■ 9 ★ 4 ◆ 12 ▲ 5?"
    options: ["7", "11", "17", "21"]
    answer: D
    explanation: |
      Rewrite: 63 ÷ 9 × 4 − 12 + 5. Then 63 ÷ 9 = 7, 7 × 4 = 28, and 28 − 12 + 5 = **21**.
      11 comes from subtracting 12 + 5 together, as if there were a bracket.

  - id: so1-which-correct
    topic: Two signs swapped
    difficulty: 2
    q: "If + means × and × means +, which equation is correct?"
    options: ["3 × 6 + 2 = 18", "2 + 7 × 3 = 23", "6 + 3 × 4 = 30", "4 + 5 × 2 = 22"]
    answer: D
    explanation: |
      Change the signs in each and use BODMAS. **4 + 5 × 2 = 22** becomes 4 × 5 + 2 = 20 + 2 = 22 ✓.
      3 × 6 + 2 becomes 3 + 6 × 2 = 15 (18 only if you work left to right). 2 + 7 × 3 becomes 2 × 7 + 3 = 17 (23 is true only *before* changing). 6 + 3 × 4 becomes 18 + 4 = 22.

  - id: so1-with-brackets
    topic: All four signs changed
    difficulty: 2
    q: "If + means −, − means ×, × means ÷ and ÷ means +, what is the value of 18 × (3 − 2) + 5 ÷ 4?"
    options: ["2", "3", "11", "13"]
    answer: A
    explanation: |
      Rewrite: 18 ÷ (3 × 2) − 5 + 4. Brackets first: 3 × 2 = 6. Then 18 ÷ 6 = 3, and 3 − 5 + 4 = **2**.
      11 ignores the bracket: 18 ÷ 3 × 2 − 5 + 4.

  - id: so1-word-signs
    topic: Letters and symbols
    difficulty: 2
    q: "If 'plus' means 'times', 'times' means 'minus', 'minus' means 'divided by' and 'divided by' means 'plus', what is 9 plus 4 times 18 minus 6 divided by 2?"
    options: ["5", "29", "35", "78"]
    answer: C
    explanation: |
      Rewrite: 9 × 4 − 18 ÷ 6 + 2. Then 9 × 4 = 36 and 18 ÷ 6 = 3, so 36 − 3 + 2 = **35**.
      5 works left to right: (36 − 18) ÷ 6 + 2.

  - id: so1-find-missing
    topic: All four signs changed
    difficulty: 3
    q: "If + means ÷, − means ×, × means + and ÷ means −, which number should replace the question mark in 24 + 4 − ? × 6 = 30?"
    options: ["3", "4", "6", "8"]
    answer: B
    explanation: |
      Rewrite: 24 ÷ 4 × ? + 6 = 30. So 6 × ? + 6 = 30, which gives 6 × ? = 24 and ? = **4**.
      Check: 24 ÷ 4 = 6, 6 × 4 = 24, and 24 + 6 = 30 ✓.

  - id: so1-order-trap
    topic: All four signs changed
    difficulty: 3
    q: "If × means ÷, ÷ means −, − means + and + means ×, what is the value of 48 × 6 ÷ 2 + 3 − 5?"
    options: ["7", "19", "23", "142"]
    answer: A
    explanation: |
      Rewrite: 48 ÷ 6 − 2 × 3 + 5. Do ÷ and × first: 48 ÷ 6 = 8 and 2 × 3 = 6. Then 8 − 6 + 5 = **7**.
      23 works left to right, ((8 − 2) × 3) + 5. 142 forgets to change the signs.

  - id: so1-largest
    topic: All four signs changed
    difficulty: 3
    q: "If + means ÷, ÷ means +, − means × and × means −, which expression has the largest value?"
    options: ["12 + 4 − 3", "18 ÷ 6 + 2", "8 − 3 × 10", "20 × 4 ÷ 6"]
    answer: D
    explanation: |
      Rewrite each one: 12 ÷ 4 × 3 = 9; 18 + 6 ÷ 2 = 18 + 3 = 21; 8 × 3 − 10 = 14; 20 − 4 + 6 = 22.
      The largest is **20 × 4 ÷ 6**, worth 22. 18 ÷ 6 + 2 is close (21), so work every option out fully.
---

In these questions the signs don't mean what they usually mean. First rewrite the question with the real signs, then work it out in the usual order.

## Step 1: rewrite

Copy the expression again, changing **every** sign using the rule. Do it one sign at a time and don't change a sign twice. A small table helps:

| Written | Means |
|---|---|
| + | × |
| × | + |

Keep the numbers where they are. Only the signs change.

## Step 2: BODMAS

After rewriting, work it out in this order:

1. **B**rackets first.
2. **D**ivision and **M**ultiplication next, from left to right.
3. **A**ddition and **S**ubtraction last, from left to right.

## Worked example

If + means ÷ and ÷ means +, find 16 + 4 ÷ 2.

Rewrite: 16 ÷ 4 + 2. Divide first: 16 ÷ 4 = 4. Then 4 + 2 = 6.

## Finding a missing number

Rewrite first, then work out everything you can. What is left is a small equation, such as 5 × ? + 3 = 23, so 5 × ? = 20 and ? = 4. Put your answer back in to check it.

## Common traps

- Working from left to right without following BODMAS. In 2 + 3 × 4, multiply first: the answer is 14, not 20.
- Changing a sign, then changing it again when you meet the new sign in the rule.
- Answering with the value of the expression as it was written, before changing the signs.
