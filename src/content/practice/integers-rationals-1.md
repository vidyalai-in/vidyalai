---
title: "Integers and rational numbers 1: integers"
description: "Adding, subtracting, multiplying and dividing with signs, the number line, temperature and sea-level problems, properties of integers and the order of operations."
slug: integers-rationals-1
family: integers-rationals
part: 1
timeMinutes: 12
questions:
  - id: ir1-sum-signs
    topic: Sign rules
    difficulty: 1
    q: "What is the value of (−8) + 13 − (−5)?"
    options: ["−26", "−16", "0", "10"]
    answer: D
    explanation: |
      Subtracting −5 is the same as adding 5: (−8) + 13 + 5 = **10**.
      0 treats − (−5) as − 5. −16 subtracts 13 instead of adding it.

  - id: ir1-divide-multiply
    topic: Sign rules
    difficulty: 1
    q: "What is the value of (−48) ÷ (−6) × (−2)?"
    options: ["−16", "−4", "4", "16"]
    answer: A
    explanation: |
      Work from left to right. (−48) ÷ (−6) = 8, then 8 × (−2) = **−16**.
      −4 multiplies first, (−6) × (−2) = 12, and then divides; ÷ and × must go left to right.

  - id: ir1-leh-temperature
    topic: Word problems
    difficulty: 1
    q: "At 5 am the temperature in Leh was −7 °C. By 2 pm it had risen by 12 °C. By midnight it had fallen by 9 °C from the 2 pm reading. What was the temperature at midnight?"
    options: ["−28 °C", "−4 °C", "4 °C", "14 °C"]
    answer: B
    explanation: |
      −7 + 12 = 5 °C at 2 pm. Then 5 − 9 = **−4 °C** at midnight.
      4 °C drops the minus sign. 14 °C adds the fall instead of subtracting it.

  - id: ir1-number-line
    topic: Number line
    difficulty: 1
    q: "On a number line, which integer is 7 units to the left of −3?"
    options: ["−10", "−4", "4", "10"]
    answer: A
    explanation: |
      Moving left means subtracting: −3 − 7 = **−10**.
      4 is 7 units to the *right* of −3.

  - id: ir1-order-ops
    topic: Order of operations
    difficulty: 2
    q: "What is the value of 36 ÷ (−3) × 2 − (−4)?"
    options: ["−28", "−20", "−2", "28"]
    answer: B
    explanation: |
      ÷ and × go left to right: 36 ÷ (−3) = −12, and −12 × 2 = −24. Then −24 − (−4) = −24 + 4 = **−20**.
      −2 divides 36 by (−3 × 2) = −6, which breaks the left-to-right rule. −28 subtracts 4 instead of adding it.

  - id: ir1-closure
    topic: Properties
    difficulty: 2
    q: "Which statement is true for **all** integers?"
    options: ["Subtraction of integers is commutative.", "Dividing an integer by a non-zero integer always gives an integer.", "Integers are closed under subtraction.", "a − (b − c) = (a − b) − c for any integers a, b, c."]
    answer: C
    explanation: |
      The difference of two integers is always an integer, so **integers are closed under subtraction**.
      The others fail: 3 − 5 ≠ 5 − 3; 7 ÷ 2 is not an integer; 10 − (4 − 1) = 7 but (10 − 4) − 1 = 5.

  - id: ir1-negative-a
    topic: Properties
    difficulty: 2
    q: "a is a negative integer. Which of these is always positive?"
    options: ["a × (−1)", "a + (−1)", "a × a × a", "(−1) × (−1) × a"]
    answer: A
    explanation: |
      Multiplying by −1 flips the sign, so **a × (−1)** is positive.
      a × a × a has three negative factors, so it is negative. (−1) × (−1) × a = 1 × a = a, still negative.

  - id: ir1-submarine
    topic: Word problems
    difficulty: 2
    q: "A submarine is 120 m below sea level. A helicopter hovers 350 m above sea level, right above it. The submarine rises by 45 m. Now how far is the helicopter above the submarine?"
    options: ["185 m", "230 m", "275 m", "425 m"]
    answer: D
    explanation: |
      Take sea level as 0. The submarine moves from −120 to −120 + 45 = −75 m.
      Distance = 350 − (−75) = 350 + 75 = **425 m**. 275 m subtracts 75 instead of adding it.

  - id: ir1-cold-store
    topic: Word problems
    difficulty: 2
    q: "A cold store cools a room that starts at 18 °C. The temperature falls by 4 °C in each hour. After how many full hours is the temperature first **lower than** −10 °C?"
    options: ["2 hours", "7 hours", "8 hours", "28 hours"]
    answer: C
    explanation: |
      After h hours the temperature is 18 − 4h. After 7 hours it is 18 − 28 = −10 °C, which is not lower than −10 °C.
      After **8** hours it is −14 °C. 7 is the trap: equal is not lower.

  - id: ir1-brackets-trap
    topic: Order of operations
    difficulty: 3
    q: "What is the value of [−8 + (−4) × 3] ÷ (−4) − (−2) × (−3)?"
    options: ["−33", "−1", "3", "11"]
    answer: B
    explanation: |
      Inside the bracket, multiply first: −8 + (−12) = −20, and −20 ÷ (−4) = 5. Next (−2) × (−3) = 6. So 5 − 6 = **−1**.
      3 adds −8 and −4 before multiplying. 11 adds the 6 instead of subtracting it.

  - id: ir1-sign-clues
    topic: Sign rules
    difficulty: 3
    q: "a, b and c are non-zero integers. a is positive, a × b is negative and b × c is positive. Which is true?"
    options: ["b > 0 and c > 0", "b < 0 and c > 0", "b < 0 and c < 0", "b > 0 and c < 0"]
    answer: C
    explanation: |
      a is positive and a × b is negative, so b must be negative.
      b × c is positive, so c has the same sign as b: **b < 0 and c < 0**.

  - id: ir1-test-marks
    topic: Word problems
    difficulty: 3
    q: "A quiz gives +4 marks for a correct answer, −1 for a wrong answer and 0 for a question left out. Kabir answered 17 questions and scored 38. How many did he get right?"
    options: ["6", "9", "11", "17"]
    answer: C
    explanation: |
      If he got r right, then 17 − r were wrong: 4r − (17 − r) = 38, so 5r = 55 and r = **11**.
      Check: 11 × 4 − 6 = 38 ✓. 6 is the number he got wrong. 9 ignores the minus marks (38 ÷ 4 is about 9).
---

This set tests how signs behave when you add, subtract, multiply and divide integers, and how to use integers for temperatures, heights and scores.

## Rules for signs

- Subtracting a negative is adding: 6 − (−2) = 6 + 2 = 8.
- Multiplying or dividing two numbers with the **same** sign gives a positive answer. Different signs give a negative answer.
- With many factors, count the negative ones. An even number of them gives a positive product, an odd number a negative product.
- Multiplying by −1 flips the sign: (−1) × a = −a.

## Number line and real life

Moving right on the number line means adding, and moving left means subtracting. For heights, take sea level as 0: above is positive and below is negative. The distance between two heights is the larger one minus the smaller one, so a point 30 m up and a point 20 m down are 30 − (−20) = 50 m apart.

## Properties

Integers are **closed** under addition, subtraction and multiplication: the answer is always an integer. They are not closed under division, since 5 ÷ 2 is not an integer. Addition and multiplication are commutative and associative. Subtraction and division are neither.

## Order of operations

Brackets first, then × and ÷ from left to right, then + and − from left to right.

## Worked example

*Find 20 − (−6) × 2 ÷ (−3).*

(−6) × 2 = −12, and −12 ÷ (−3) = 4. Then 20 − 4 = **16**.

## Common trap

Doing the operations in the order they are written, ignoring the rules. In 20 − (−6) × 2, the × comes before the −. Working left to right would give 26 × 2 = 52, which is wrong.
