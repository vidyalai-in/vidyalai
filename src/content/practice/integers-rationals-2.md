---
title: "Integers and rational numbers 2: rational numbers"
description: "Standard form, comparing and ordering negative fractions, a rational number between two others, operations, reciprocals and decimals written as rational numbers."
slug: integers-rationals-2
family: integers-rationals
part: 2
timeMinutes: 12
questions:
  - id: ir2-standard-form
    topic: Standard form
    difficulty: 1
    q: "What is the standard form of the rational number 36/(−48)?"
    options: ["−3/4", "3/4", "(−9)/12", "3/(−4)"]
    answer: A
    explanation: |
      Divide top and bottom by the HCF, 12, and keep the denominator positive: 36/(−48) = **−3/4**.
      (−9)/12 is not in lowest terms, and 3/(−4) has a negative denominator, so neither is in standard form.

  - id: ir2-greatest
    topic: Comparing
    difficulty: 1
    q: "Which of these is the greatest?"
    options: ["−5/6", "−3/4", "−2/3", "−1/2"]
    answer: D
    explanation: |
      With a common denominator of 12: −10/12, −9/12, −8/12 and −6/12. The one closest to zero is the greatest: **−1/2**.
      −5/6 is the *smallest*. For negative numbers, a bigger fraction part means further below zero.

  - id: ir2-add
    topic: Operations
    difficulty: 1
    q: "What is (−5/8) + 3/4?"
    options: ["−11/8", "−1/6", "1/8", "11/8"]
    answer: C
    explanation: |
      3/4 = 6/8, so −5/8 + 6/8 = **1/8**.
      −1/6 adds tops and bottoms separately, (−5 + 3)/(8 + 4) = −2/12, which is not how fractions add.

  - id: ir2-reciprocal
    topic: Reciprocals
    difficulty: 1
    q: "What is the reciprocal of −2 3/5 (minus two and three-fifths)?"
    options: ["−13/5", "−5/13", "5/13", "13/5"]
    answer: B
    explanation: |
      First make it an improper fraction: −2 3/5 = −13/5. Flip it, keeping the sign: **−5/13**.
      A number and its reciprocal always have the same sign, since their product is +1. So 5/13 is wrong.

  - id: ir2-between
    topic: Comparing
    difficulty: 2
    q: "Which rational number lies between −1/3 and −1/4?"
    options: ["−2/5", "−2/7", "−1/5", "1/12"]
    answer: B
    explanation: |
      Compare decimals: −1/3 ≈ −0.333 and −1/4 = −0.25. Only **−2/7** ≈ −0.286 lies between them.
      −2/5 = −0.4 is below −1/3, and −1/5 = −0.2 is above −1/4. 1/12 is the gap between the two numbers, not a number between them.

  - id: ir2-mul-div
    topic: Operations
    difficulty: 2
    q: "What is (−3/4) × 8/9 ÷ (−2/3)?"
    options: ["−1", "−4/9", "4/9", "1"]
    answer: D
    explanation: |
      (−3/4) × 8/9 = −24/36 = −2/3. Dividing −2/3 by −2/3 gives **1**.
      4/9 multiplies by −2/3 instead of dividing by it. Two negatives in the working make the answer positive.

  - id: ir2-decimal
    topic: Standard form
    difficulty: 2
    q: "Write −0.375 as a rational number in standard form."
    options: ["−375/100", "−8/3", "−3/8", "−3/80"]
    answer: C
    explanation: |
      −0.375 = −375/1000. The HCF of 375 and 1000 is 125, so it is **−3/8**.
      −375/100 puts the decimal point in the wrong place (it equals −3.75). −8/3 is the reciprocal.

  - id: ir2-ascending
    topic: Comparing
    difficulty: 2
    q: "Which list is in **ascending** order (smallest first)?"
    options: ["−7/8, −5/6, −3/4", "−3/4, −5/6, −7/8", "−5/6, −7/8, −3/4", "−7/8, −3/4, −5/6"]
    answer: A
    explanation: |
      With denominator 24: −21/24, −20/24, −18/24. So the order is **−7/8, −5/6, −3/4**.
      The second list is the descending order. It is the trap if you order 3/4, 5/6, 7/8 as if they were positive.

  - id: ir2-not-equal
    topic: Standard form
    difficulty: 2
    q: "Which of these is **not** equal to −4/7?"
    options: ["4/(−7)", "(−8)/14", "(−4)/(−7)", "−(4/7)"]
    answer: C
    explanation: |
      A minus sign on top or bottom gives a negative number, but minus on both cancels out: (−4)/(−7) = +4/7.
      So **(−4)/(−7)** is the odd one. (−8)/14 is just −4/7 with top and bottom doubled.

  - id: ir2-other-factor
    topic: Reciprocals
    difficulty: 3
    q: "The product of two rational numbers is −14/27. One of them is 7/(−9). What is the other?"
    options: ["−2/3", "98/243", "2/3", "3/2"]
    answer: C
    explanation: |
      Other = (−14/27) ÷ (−7/9) = (−14/27) × (−9/7) = 126/189 = **2/3**.
      98/243 multiplies the two numbers instead of dividing. −2/3 loses track of the two minus signs.

  - id: ir2-count-twelfths
    topic: Comparing
    difficulty: 3
    q: "How many numbers of the form n/12, where n is an integer, lie strictly between −3/4 and 1/3?"
    options: ["11", "12", "13", "14"]
    answer: B
    explanation: |
      −3/4 = −9/12 and 1/3 = 4/12. Strictly between means n goes from −8 to 3.
      That is 8 negative values, zero, and 3 positive values: **12**. 14 wrongly counts the two ends, −9 and 4.

  - id: ir2-x-over-y
    topic: Operations
    difficulty: 3
    q: "If x = −2/3 and y = 3/4, what is x ÷ y − y ÷ x?"
    options: ["−145/72", "−17/72", "0", "17/72"]
    answer: D
    explanation: |
      x ÷ y = (−2/3) × (4/3) = −8/9 and y ÷ x = (3/4) × (−3/2) = −9/8.
      So −8/9 − (−9/8) = −64/72 + 81/72 = **17/72**. −145/72 forgets that y ÷ x is negative.
---

This set tests rational numbers: fractions with a sign, written as p/q where p and q are integers and q is not zero.

## Standard form

A rational number is in **standard form** when its denominator is positive and the top and bottom have no common factor except 1. Move any minus sign to the top, then divide by the HCF. A minus on both the top and the bottom cancels: (−a)/(−b) = a/b.

## Comparing

- Every negative number is less than every positive number.
- To compare two negatives, compare their sizes ignoring the sign. The one with the **bigger** size is the **smaller** number, so −7/8 is less than −1/2.
- Make the denominators equal, or change to decimals, then compare.
- To find a number between two others, write both with a large common denominator and pick a numerator in between. Their average, (a + b) ÷ 2, also always works.

## Operations

Add and subtract with a common denominator. Multiply tops and bottoms. To divide, multiply by the **reciprocal**. The reciprocal of p/q is q/p, and the sign stays the same, because a number times its reciprocal is +1. Zero has no reciprocal.

## Decimals

A terminating decimal is a rational number: 0.45 = 45/100 = 9/20. Count the decimal places to choose 10, 100 or 1000 as the denominator, then simplify.

## Worked example

*Find (−5/6) ÷ (10/9).*

Multiply by the reciprocal: (−5/6) × (9/10) = −45/60 = **−3/4**.

## Common trap

Treating negative fractions like positive ones when ordering them. On the number line, −2/3 is to the right of −3/4, so −2/3 is the larger number.
