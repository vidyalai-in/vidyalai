---
title: "Algebra 3: exponents and powers"
description: "Laws of exponents, zero and negative powers, comparing large powers, very large and very small numbers in standard form, and the units digit of a power."
slug: algebra-basics-3
family: algebra-basics
part: 3
timeMinutes: 12
questions:
  - id: al3-same-base
    topic: Laws of exponents
    difficulty: 1
    q: "Write 2⁵ × 2³ ÷ 2⁴ as a single power of 2."
    options: ["2²", "2⁴", "2¹¹", "2¹²"]
    answer: B
    explanation: |
      With the same base, add the powers when multiplying and subtract them when dividing: 5 + 3 − 4 = 4, so **2⁴**.
      2¹¹ multiplies 5 × 3 instead of adding. 2² divides (5 + 3) by 4.

  - id: al3-negative-power
    topic: Zero and negative powers
    difficulty: 1
    q: "What is the value of 3⁻²?"
    options: ["−9", "−6", "1/9", "1/6"]
    answer: C
    explanation: |
      A negative power means "one over": 3⁻² = 1/3² = **1/9**.
      A negative power does not make the number negative, so −9 is wrong. −6 multiplies 3 by −2.

  - id: al3-equals-one
    topic: Zero and negative powers
    difficulty: 1
    q: "Which of these is equal to 1?"
    options: ["(−7)⁰", "7⁻¹", "0⁷", "(1/7)⁻¹"]
    answer: A
    explanation: |
      Any non-zero number to the power 0 is 1, even a negative one, so **(−7)⁰** = 1.
      7⁻¹ = 1/7, 0⁷ = 0 and (1/7)⁻¹ = 7.

  - id: al3-standard-small
    topic: Standard form
    difficulty: 1
    q: "Write 0.0000456 in standard form."
    options: ["4.56 × 10⁻⁶", "4.56 × 10⁻⁵", "4.56 × 10⁻⁴", "4.56 × 10⁵"]
    answer: B
    explanation: |
      Move the decimal point 5 places to the right to get 4.56. The number is small, so the power is negative: **4.56 × 10⁻⁵**.
      10⁻⁶ and 10⁻⁴ are one place out. 10⁵ forgets the minus sign.

  - id: al3-power-of-power
    topic: Laws of exponents
    difficulty: 2
    q: "What is the value of (3²)³ × 3⁻⁴?"
    options: ["1/9", "3", "9", "81"]
    answer: C
    explanation: |
      (3²)³ = 3⁶, since a power of a power multiplies. Then 3⁶ × 3⁻⁴ = 3² = **9**.
      3 adds 2 + 3 instead of multiplying. 81 works out 2³ = 8 and uses 3⁸.

  - id: al3-fraction-negative
    topic: Zero and negative powers
    difficulty: 2
    q: "What is the value of (2/3)⁻²?"
    options: ["−9/4", "−4/9", "4/9", "9/4"]
    answer: D
    explanation: |
      A negative power flips the fraction: (2/3)⁻² = (3/2)² = **9/4**.
      4/9 squares without flipping. −9/4 makes the answer negative, but a negative power never changes the sign.

  - id: al3-units-seven
    topic: Units digit
    difficulty: 2
    q: "What is the units digit of 7⁴⁴?"
    options: ["1", "3", "7", "9"]
    answer: A
    explanation: |
      Units digits of 7, 7², 7³, 7⁴ are 7, 9, 3, 1, and then the cycle repeats every 4.
      44 is a multiple of 4, so 7⁴⁴ ends like 7⁴: **1**.

  - id: al3-light
    topic: Standard form
    difficulty: 2
    q: "Light travels about 3 × 10⁸ m in one second. About how far does it travel in 5 minutes?"
    options: ["1.5 × 10⁹ m", "1.8 × 10¹⁰ m", "9 × 10¹⁰ m", "9 × 10¹¹ m"]
    answer: C
    explanation: |
      5 minutes = 300 s = 3 × 10². Distance = 3 × 10⁸ × 3 × 10² = **9 × 10¹⁰ m**.
      1.5 × 10⁹ m multiplies by 5 without changing minutes to seconds. 1.8 × 10¹⁰ m is the distance in one minute.

  - id: al3-sum-negative
    topic: Zero and negative powers
    difficulty: 2
    q: "What is the value of (1/2)⁻² + (1/3)⁻²?"
    options: ["−13", "13/36", "36/25", "13"]
    answer: D
    explanation: |
      (1/2)⁻² = 2² = 4 and (1/3)⁻² = 3² = 9, so the sum is **13**.
      36/25 adds the fractions first, (5/6)⁻², but powers come before addition. 13/36 forgets to flip.

  - id: al3-greatest
    topic: Comparing powers
    difficulty: 3
    q: "Which of these is the greatest?"
    options: ["2⁴⁰", "3³⁰", "5²⁰", "6¹⁰"]
    answer: B
    explanation: |
      Give each the same power, 10: 2⁴⁰ = (2⁴)¹⁰ = 16¹⁰, 3³⁰ = (3³)¹⁰ = 27¹⁰, 5²⁰ = (5²)¹⁰ = 25¹⁰, and 6¹⁰.
      27 is the largest base, so **3³⁰** is the greatest. 2⁴⁰ has the biggest power but only 16 as its base.

  - id: al3-find-n
    topic: Laws of exponents
    difficulty: 3
    q: "If 8 × 2ⁿ⁺¹ = 4⁵, what is n?"
    options: ["1", "6", "7", "9"]
    answer: B
    explanation: |
      Write everything as powers of 2: 2³ × 2ⁿ⁺¹ = 2¹⁰, so 3 + n + 1 = 10 and **n = 6**.
      1 treats 4⁵ as 2⁵. 7 forgets the +1, and 9 forgets the 8.

  - id: al3-units-sum
    topic: Units digit
    difficulty: 3
    q: "What is the units digit of 3²⁰²⁶ + 7²⁰²⁶?"
    options: ["0", "2", "8", "9"]
    answer: C
    explanation: |
      3 cycles 3, 9, 7, 1 and 7 cycles 7, 9, 3, 1. 2026 leaves remainder 2 when divided by 4, so both end in 9.
      9 + 9 = 18, so the units digit is **8**. 0 is the trap from adding 3 + 7 without using the powers.
---

This set tests the laws of exponents, zero and negative powers, standard form, and quick tricks for comparing powers and finding units digits.

## The laws (same base, a not zero)

| Law | Example |
|---|---|
| aᵐ × aⁿ = aᵐ⁺ⁿ | 5² × 5³ = 5⁵ |
| aᵐ ÷ aⁿ = aᵐ⁻ⁿ | 7⁶ ÷ 7⁴ = 7² |
| (aᵐ)ⁿ = aᵐⁿ | (2³)² = 2⁶ |
| aⁿ × bⁿ = (ab)ⁿ | 2⁴ × 5⁴ = 10⁴ |
| a⁰ = 1 | (−9)⁰ = 1 |
| a⁻ⁿ = 1/aⁿ | 4⁻² = 1/16 |

A negative power makes the number **smaller**, never negative. For a fraction, a negative power flips it: (3/4)⁻² = (4/3)² = 16/9.

## Standard form

Write a number as k × 10ⁿ with k from 1 up to (but not including) 10. Count how many places the decimal point moves. Big numbers get a positive power and small numbers a negative one: 52 000 = 5.2 × 10⁴ and 0.0031 = 3.1 × 10⁻³.

## Comparing powers

Make the powers equal, then compare the bases: 2³⁰ = (2³)¹⁰ = 8¹⁰ and 3²⁰ = (3²)¹⁰ = 9¹⁰, so 3²⁰ is bigger. Or make the bases equal and compare the powers.

## Units digit

The units digits of powers repeat in short cycles: 2 → 2, 4, 8, 6 and 3 → 3, 9, 7, 1. Numbers ending in 0, 1, 5 or 6 always end the same way. Divide the power by 4 and use the remainder (remainder 0 means the 4th in the cycle).

## Worked example

*Units digit of 2³⁷.* 37 = 4 × 9 + 1, so it ends like 2¹: **2**.

## Common trap

Adding the powers when the bases are different. 2³ × 3² is 8 × 9 = 72, not 6⁵.
