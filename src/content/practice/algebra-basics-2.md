---
title: "Algebra 2: simple linear equations"
description: "Solving linear equations with brackets and fractions, and word problems on ages, coins, consecutive numbers, perimeter and money."
slug: algebra-basics-2
family: algebra-basics
part: 2
timeMinutes: 12
questions:
  - id: al2-solve-basic
    topic: Solving equations
    difficulty: 1
    q: "Solve 3x − 7 = 14."
    options: ["x = −7", "x = 7/3", "x = 7", "x = 21"]
    answer: C
    explanation: |
      Add 7 to both sides: 3x = 21. Divide by 3: **x = 7**.
      7/3 subtracts 7 from 14 instead of adding. 21 forgets to divide by 3.

  - id: al2-solve-fraction
    topic: Solving equations
    difficulty: 1
    q: "Solve x/4 + 5 = 2."
    options: ["x = −12", "x = −3/4", "x = 12", "x = 28"]
    answer: A
    explanation: |
      Subtract 5: x/4 = −3. Multiply by 4: **x = −12**.
      28 adds 5 to 2 instead of subtracting. −3/4 divides by 4 instead of multiplying.

  - id: al2-consecutive
    topic: Consecutive numbers
    difficulty: 1
    q: "The sum of three consecutive integers is 72. What is the smallest of them?"
    options: ["22", "23", "24", "25"]
    answer: B
    explanation: |
      Call them n, n + 1, n + 2. Then 3n + 3 = 72, so 3n = 69 and n = **23**. The numbers are 23, 24, 25.
      24 is the middle one (72 ÷ 3).

  - id: al2-mother
    topic: Ages
    difficulty: 1
    q: "Asha's mother is 3 times as old as Asha. Their ages add up to 48 years. How old is Asha?"
    options: ["12 years", "16 years", "24 years", "36 years"]
    answer: A
    explanation: |
      Asha = a, mother = 3a. So a + 3a = 4a = 48 and a = **12 years**.
      16 divides 48 by 3, forgetting Asha's own share. 36 is the mother's age.

  - id: al2-brackets
    topic: Solving equations
    difficulty: 2
    q: "Solve 5(x − 2) − 3(x + 1) = 7."
    options: ["x = −3", "x = 7", "x = 9", "x = 10"]
    answer: D
    explanation: |
      Open the brackets: 5x − 10 − 3x − 3 = 7, so 2x − 13 = 7, 2x = 20 and **x = 10**.
      x = 7 writes −3(x + 1) as −3x + 3. Check: 5 × 8 − 3 × 11 = 40 − 33 = 7 ✓.

  - id: al2-coins
    topic: Coins and money
    difficulty: 2
    q: "Farhan has only ₹5 and ₹2 coins. He has 25 coins worth ₹89 in all. How many ₹2 coins does he have?"
    options: ["12", "13", "15", "17"]
    answer: A
    explanation: |
      Let f be the number of ₹5 coins. Then 5f + 2(25 − f) = 89, so 3f + 50 = 89, f = 13.
      ₹2 coins = 25 − 13 = **12**. Check: 65 + 24 = 89 ✓. 13 is the number of ₹5 coins.

  - id: al2-rectangle
    topic: Perimeter
    difficulty: 2
    q: "The length of a rectangle is 4 cm more than twice its breadth. Its perimeter is 62 cm. What is its length?"
    options: ["9 cm", "18 cm", "22 cm", "31 cm"]
    answer: C
    explanation: |
      Breadth = b, length = 2b + 4. Then 2(2b + 4 + b) = 62, so 3b + 4 = 31 and b = 9.
      Length = 2 × 9 + 4 = **22 cm**. 9 cm is the breadth. 31 cm is half the perimeter.

  - id: al2-ages-twice
    topic: Ages
    difficulty: 2
    q: "Kabir is 6 years older than Neha. Five years ago, Kabir was twice as old as Neha. How old is Neha now?"
    options: ["6 years", "11 years", "12 years", "17 years"]
    answer: B
    explanation: |
      Neha = n, Kabir = n + 6. Five years ago: n + 1 = 2(n − 5), so n + 1 = 2n − 10 and n = **11 years**.
      Check: five years ago they were 12 and 6 ✓. 6 is Neha's age five years ago.

  - id: al2-pocket-money
    topic: Coins and money
    difficulty: 2
    q: "Priya spends 1/3 of her pocket money on books and 1/4 of it on snacks. She has ₹150 left. How much pocket money did she get?"
    options: ["₹200", "₹210", "₹225", "₹360"]
    answer: D
    explanation: |
      She spends 1/3 + 1/4 = 7/12, so 5/12 is left. 5/12 of the money = 150, so the money = 150 × 12/5 = **₹360**.
      ₹210 is what she spent. ₹225 counts only the books.

  - id: al2-consecutive-even
    topic: Consecutive numbers
    difficulty: 3
    q: "The sum of three consecutive even numbers is 30 more than the smallest of them. What is the largest?"
    options: ["12", "14", "16", "18"]
    answer: C
    explanation: |
      Call them n, n + 2, n + 4. Then 3n + 6 = n + 30, so 2n = 24 and n = 12. The numbers are 12, 14, 16, so the largest is **16**.
      Check: 12 + 14 + 16 = 42 = 12 + 30 ✓. 12 is the smallest.

  - id: al2-two-fractions
    topic: Solving equations
    difficulty: 3
    q: "Solve (x + 4)/3 − (x − 2)/5 = 2."
    options: ["x = −12", "x = 2", "x = 4", "x = 8"]
    answer: B
    explanation: |
      Multiply every term by 15: 5(x + 4) − 3(x − 2) = 30, so 5x + 20 − 3x + 6 = 30, 2x = 4 and **x = 2**.
      x = 8 writes −3(x − 2) as −3x − 6. x = −12 forgets to multiply the 2 on the right by 15.

  - id: al2-wire
    topic: Perimeter
    difficulty: 3
    q: "A wire is bent into a square of side (x + 3) cm. The same wire is then bent into a rectangle of length 2x cm and breadth (x − 1) cm. What is the length of the rectangle?"
    options: ["7 cm", "10 cm", "14 cm", "40 cm"]
    answer: C
    explanation: |
      Same wire, so the perimeters are equal: 4(x + 3) = 2(2x + x − 1). So 4x + 12 = 6x − 2 and x = 7.
      Length = 2x = **14 cm**. 7 is x itself, 10 cm is the square's side and 40 cm is the wire's length.
---

This set tests solving simple linear equations and turning word problems into equations.

## Solving

An equation is a balance: whatever you do to one side, do to the other.

1. Open the brackets. A minus sign in front of a bracket flips every sign inside: −3(x − 2) = −3x + 6.
2. If there are fractions, multiply **every term on both sides** by the LCM of the denominators.
3. Collect the letter terms on one side and the numbers on the other.
4. Divide by the number in front of x.
5. Check by putting your answer back into the original equation.

## Word problems

| Problem | Choose |
|---|---|
| consecutive numbers | n, n + 1, n + 2 |
| consecutive even or odd numbers | n, n + 2, n + 4 |
| ages "5 years ago" | subtract 5 from **every** person's age |
| coins of two kinds | f of one kind, (total − f) of the other |
| perimeter of a rectangle | 2 × (length + breadth) |

Write the unknown, form one equation, solve it, then answer the question that was **asked**. The question may want a different quantity from your x.

## Worked example

*The ages of Ravi and his sister add up to 31. In 4 years, Ravi will be twice as old as his sister. How old is the sister now?*

Sister = s, Ravi = 31 − s. In 4 years: 35 − s = 2(s + 4), so 35 − s = 2s + 8, 3s = 27 and **s = 9**. Check: now 9 and 22; in 4 years 13 and 26 ✓.

## Common trap

Stopping at x. If x is the breadth but the question asks for the length, one more step is needed.
