---
title: "Number puzzles 2: cryptarithms and missing digits"
description: "Letters that stand for digits, missing digits in sums and products, and puzzles about a number and its reverse."
slug: number-puzzles-2
family: number-puzzles
part: 2
timeMinutes: 12
questions:
  - id: np2-star-subtract
    topic: Missing digits
    difficulty: 1
    q: "In **★2 − 2★ = 36**, the same digit goes in both ★ places. What is the digit?"
    options: ["6", "7", "8", "9"]
    answer: A
    explanation: |
      Try the options: 62 − 26 = 36 ✓. So ★ = **6**.
      With 7 you get 72 − 27 = 45, and with 8 you get 54. Each step up adds 9.

  - id: np2-triple-a
    topic: Letter puzzles
    difficulty: 1
    q: "In **A + A + A = BA**, A and B stand for two different digits, and BA is a 2-digit number. What is A?"
    options: ["3", "5", "6", "9"]
    answer: B
    explanation: |
      3 × A must end in the digit A. That happens only for A = 0 or A = 5, and A = 0 would make the sum 0.
      So A = **5**: 5 + 5 + 5 = 15, which gives B = 1.

  - id: np2-ab-ba-sum
    topic: Letter puzzles
    difficulty: 1
    q: "AB and BA are 2-digit numbers, where A and B are different digits. If **AB + BA = 132**, what is A + B?"
    options: ["12", "13", "24", "66"]
    answer: A
    explanation: |
      AB + BA = (10A + B) + (10B + A) = 11A + 11B = 11 × (A + B). So A + B = 132 ÷ 11 = **12**.
      Many pairs work (39 + 93, 48 + 84, 57 + 75), but every one has A + B = 12. 66 is 132 ÷ 2.

  - id: np2-column-add
    topic: Missing digits
    difficulty: 1
    q: "In this addition, P and Q are missing digits. What are they?"
    figure: { kind: grid, cells: [[null, 3, "P", 6], ["+", 2, 4, "Q"], [null, 6, 0, 3]], shade: [[2, 1], [2, 2], [2, 3]] }
    options: ["P = 5, Q = 7", "P = 7, Q = 5", "P = 6, Q = 7", "P = 5, Q = 3"]
    answer: A
    explanation: |
      Ones: 6 + Q ends in 3, so Q = 7 and 1 is carried. Tens: P + 4 + 1 ends in 0, so P = 5 and 1 is carried. Hundreds: 3 + 2 + 1 = 6 ✓.
      So **P = 5, Q = 7**: 356 + 247 = 603.

  - id: np2-ab-minus-ba
    topic: Letter puzzles
    difficulty: 2
    q: "AB and BA are 2-digit numbers, where A and B are different digits. If **AB − BA = 54** and A + B = 12, what is AB?"
    options: ["39", "84", "93", "96"]
    answer: C
    explanation: |
      AB − BA = (10A + B) − (10B + A) = 9 × (A − B). So A − B = 54 ÷ 9 = 6.
      Now A + B = 12 and A − B = 6 give A = 9 and B = 3, so AB = **93**. (93 − 39 = 54 ✓.) 39 is BA.

  - id: np2-times-seven
    topic: Missing digits
    difficulty: 2
    q: "In **2P × 7 = 1Q1**, P and Q are missing digits. What is the product?"
    options: ["131", "141", "151", "161"]
    answer: D
    explanation: |
      The product must be a multiple of 7. Of the options, only 161 is: 161 ÷ 7 = 23.
      So P = 3, Q = 6, and the product is **161**. (Ones digit check: 7 × P must end in 1, and only 7 × 3 = 21 does.)

  - id: np2-times-eight
    topic: Missing digits
    difficulty: 2
    q: "In **12P × 8 = Q84**, P and Q are missing digits. What is P + Q?"
    options: ["9", "10", "12", "13"]
    answer: C
    explanation: |
      8 × P must end in 4, so P = 3 or P = 8. With P = 8, 128 × 8 = 1024, which has four digits. With P = 3, 123 × 8 = 984 ✓.
      So P = 3, Q = 9 and P + Q = **12**.

  - id: np2-ab-times-3
    topic: Letter puzzles
    difficulty: 2
    q: "In **AB × 3 = CAB**, each letter stands for a different digit. AB is a 2-digit number and CAB is a 3-digit number. What is AB?"
    options: ["25", "50", "75", "98"]
    answer: B
    explanation: |
      CAB = C hundreds + AB. So 3 × AB = C00 + AB, which means 2 × AB = C00.
      2 × AB is at most 198, so C = 1 and AB = 100 ÷ 2 = **50**. Check: 50 × 3 = 150 ✓.
      75 × 3 = 225 ends in 25, not 75.

  - id: np2-reverse-digits
    topic: Digit puzzles
    difficulty: 2
    q: "The digits of a 2-digit number add up to 11. When the digits are reversed, the new number is 27 more than the old one. What is the old number?"
    options: ["38", "47", "56", "74"]
    answer: B
    explanation: |
      Reversing a 2-digit number changes it by 9 × (difference of the digits). 27 ÷ 9 = 3, so the digits differ by 3 and add up to 11: they are 4 and 7.
      The reverse is bigger, so the smaller digit comes first: **47** (74 − 47 = 27 ✓). 74 is the new number, not the old one.

  - id: np2-ab-times-b
    topic: Letter puzzles
    difficulty: 3
    q: "In **AB × B = CAB**, each letter stands for a different digit. AB is a 2-digit number and CAB is a 3-digit number. Which digit is B?"
    options: ["1", "2", "4", "5"]
    answer: D
    explanation: |
      CAB = C00 + AB, so AB × B − AB = C00, which means AB × (B − 1) must end in 00. B = 1 gives 0.
      B = 2 gives AB × 1, which ends in 2. B = 4 gives AB × 3, which ends in 2 (since 4 × 3 = 12). Neither ends in 00.
      With B = 5: AB × 4 = C00 gives AB = 25 (25 × 5 = 125) or AB = 75 (75 × 5 = 375). There are two solutions, but in both B = **5**.

  - id: np2-abc-times-3
    topic: Letter puzzles
    difficulty: 3
    q: "In **ABC + ABC + ABC = BBB**, each letter stands for a different digit. What is A + B + C?"
    options: ["9", "12", "13", "15"]
    answer: C
    explanation: |
      BBB = B × 111 = B × 3 × 37, so ABC = BBB ÷ 3 = 37 × B. Its middle digit must also be B.
      Try B = 1 to 9: 37, 74, 111, 148, ... Only 37 × 4 = 148 has middle digit 4. So ABC = 148 (148 × 3 = 444) and A + B + C = **13**.

  - id: np2-ab-bc-ca
    topic: Letter puzzles
    difficulty: 3
    q: "In **AB + BC + CA = ABC**, each letter stands for a different non-zero digit. AB, BC and CA are 2-digit numbers. What is ABC?"
    options: ["128", "189", "198", "981"]
    answer: C
    explanation: |
      The left side is 10A + B + 10B + C + 10C + A = 11 × (A + B + C). Set it equal to 100A + 10B + C: this gives 89A = B + 10C.
      B + 10C is at most 99, so A = 1 and 10C + B = 89: C = 8, B = 9. ABC = **198** (19 + 98 + 81 = 198 ✓).
      189 swaps B and C: 18 + 89 + 91 = 198, not 189.
---

This set tests cryptarithms, where letters or symbols stand for digits, and other puzzles with missing digits.

## The rules

- The same letter always means the same digit. Different letters mean **different** digits (unless the puzzle says otherwise).
- The first digit of a number is never 0.
- The sum of two 2-digit numbers is at most 99 + 99 = 198, so a 3-digit answer must start with 1.

## Useful facts

| Expression | Equals |
|---|---|
| AB (a 2-digit number) | 10 × A + B |
| AB + BA | 11 × (A + B) |
| AB − BA | 9 × (A − B) |
| AAA | A × 111 = A × 3 × 37 |

## How to solve

1. Start with the **ones column**. It has no carry coming in.
2. Move left, remembering any carry.
3. Use the size of the numbers: a 3-digit answer from small numbers often means the first digit is 1.
4. When stuck, test the options. Ones digits and divisibility (by 3, 9 or 11) rule out wrong options fast.

## Worked example

*In AB + B = BA, what is AB? (Different letters, different digits.)*

Write it out: 10A + B + B = 10B + A. Take A and 2B from both sides: 9A = 8B. Since A and B are single digits, A = 8 and B = 9. So AB = **89**, and 89 + 9 = 98 ✓. Always check your answer in the original sum.

## Common trap

If a puzzle has more than one solution, check what the question asks. Sometimes the asked quantity, such as A + B, is the same in every solution.
