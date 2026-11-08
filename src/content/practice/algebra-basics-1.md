---
title: "Algebra 1: algebraic expressions"
description: "Like terms, adding and subtracting expressions, finding the value of an expression, turning words into algebra, and the simple identities (a + b)² and (a + b)(a − b)."
slug: algebra-basics-1
family: algebra-basics
part: 1
timeMinutes: 12
questions:
  - id: al1-like-terms
    topic: Like terms
    difficulty: 1
    q: "Which pair are like terms?"
    options: ["3x²y and 3xy²", "−5ab and 7ba", "4x and 4x²", "2pq and 2p"]
    answer: B
    explanation: |
      Like terms have the same letters with the same powers; only the number in front may differ. ab and ba are the same, so **−5ab and 7ba** are like terms.
      3x²y and 3xy² look alike, but the powers of x and y are swapped.

  - id: al1-twice-less
    topic: Forming expressions
    difficulty: 1
    q: "Which expression means \"5 less than twice a number n\"?"
    options: ["2(n − 5)", "5 − 2n", "2n + 5", "2n − 5"]
    answer: D
    explanation: |
      Twice the number is 2n. "5 less than" it means take 5 away from 2n: **2n − 5**.
      5 − 2n takes 2n away from 5, the wrong way round.

  - id: al1-value-neg
    topic: Value of an expression
    difficulty: 1
    q: "What is the value of 3x² − 2x + 1 when x = −2?"
    options: ["−7", "9", "17", "41"]
    answer: C
    explanation: |
      x² = (−2)² = 4, so 3x² = 12. Also −2x = −2 × (−2) = +4. Total: 12 + 4 + 1 = **17**.
      −7 takes (−2)² as −4. 9 forgets that −2 × (−2) is positive.

  - id: al1-add
    topic: Adding and subtracting
    difficulty: 1
    q: "Add (4a − 3b + 2) and (−a + 5b − 7)."
    options: ["3a + 2b − 5", "5a + 2b − 5", "3a − 8b + 9", "3a + 8b − 5"]
    answer: A
    explanation: |
      Collect like terms: 4a − a = 3a, −3b + 5b = 2b, 2 − 7 = −5. The sum is **3a + 2b − 5**.
      5a + 2b − 5 reads −a as +a.

  - id: al1-subtract
    topic: Adding and subtracting
    difficulty: 2
    q: "Subtract 2x² − 3x + 4 from 5x² + x − 1."
    options: ["−3x² − 4x + 5", "3x² − 2x + 3", "3x² + 4x − 5", "3x² + 4x + 3"]
    answer: C
    explanation: |
      (5x² + x − 1) − (2x² − 3x + 4) = 5x² + x − 1 − 2x² + 3x − 4 = **3x² + 4x − 5**. Every sign in the bracket flips.
      3x² − 2x + 3 flips only the first sign. −3x² − 4x + 5 subtracts the wrong way round.

  - id: al1-together
    topic: Forming expressions
    difficulty: 2
    q: "Ravi has ₹x. Meera has ₹15 more than three times what Ravi has. How much money do they have together?"
    options: ["₹(3x + 15)", "₹(x + 18)", "₹(4x + 45)", "₹(4x + 15)"]
    answer: D
    explanation: |
      Meera has 3x + 15. Together: x + 3x + 15 = **₹(4x + 15)**.
      3x + 15 is Meera's money alone. 4x + 45 reads it as three times (x + 15).

  - id: al1-square
    topic: Identities
    difficulty: 2
    q: "Expand (2x + 3)²."
    options: ["4x² + 9", "2x² + 12x + 9", "4x² + 6x + 9", "4x² + 12x + 9"]
    answer: D
    explanation: |
      (a + b)² = a² + 2ab + b² with a = 2x, b = 3: 4x² + 2 × 2x × 3 + 9 = **4x² + 12x + 9**.
      4x² + 9 forgets the middle term. 4x² + 6x + 9 forgets to double it.

  - id: al1-product-trick
    topic: Identities
    difficulty: 2
    q: "Using (a + b)(a − b) = a² − b², find 47 × 53."
    options: ["2491", "2494", "2500", "2509"]
    answer: A
    explanation: |
      47 × 53 = (50 − 3)(50 + 3) = 50² − 3² = 2500 − 9 = **2491**.
      2509 adds the 9 instead of subtracting. 2494 subtracts 2 × 3 instead of 3².

  - id: al1-sum-squares
    topic: Identities
    difficulty: 2
    q: "If a + b = 9 and ab = 14, what is a² + b²?"
    options: ["53", "67", "81", "109"]
    answer: A
    explanation: |
      (a + b)² = a² + 2ab + b², so a² + b² = 9² − 2 × 14 = 81 − 28 = **53**.
      67 takes away ab only once. 109 adds 2ab instead of taking it away.

  - id: al1-what-added
    topic: Adding and subtracting
    difficulty: 3
    q: "What must be added to 3x² − 5x + 2 to get x² + 4x − 6?"
    options: ["2x² − 9x + 8", "−2x² − x − 4", "−2x² + 9x − 8", "−2x² + 9x − 4"]
    answer: C
    explanation: |
      Required = (x² + 4x − 6) − (3x² − 5x + 2) = x² − 3x² + 4x + 5x − 6 − 2 = **−2x² + 9x − 8**.
      2x² − 9x + 8 subtracts the wrong way round. −2x² − x − 4 adds the two expressions.

  - id: al1-reciprocal-square
    topic: Identities
    difficulty: 3
    q: "If x + 1/x = 5, what is x² + 1/x²?"
    options: ["10", "23", "25", "27"]
    answer: B
    explanation: |
      Square both sides: x² + 2 × x × (1/x) + 1/x² = 25. The middle term is 2, so x² + 1/x² = 25 − 2 = **23**.
      25 forgets the middle term. 27 adds the 2 instead of taking it away.

  - id: al1-perimeter-gap
    topic: Forming expressions
    difficulty: 3
    q: "A rectangle has length (3a + 2) cm and breadth (a − 1) cm. A square has side (a + 2) cm. Take a greater than 2. By how much is the rectangle's perimeter more than the square's?"
    options: ["(3a − 1) cm", "(4a − 6) cm", "(4a + 10) cm", "(12a + 10) cm"]
    answer: B
    explanation: |
      Rectangle: 2(3a + 2 + a − 1) = 8a + 2. Square: 4(a + 2) = 4a + 8.
      Difference: 8a + 2 − 4a − 8 = **(4a − 6) cm**. 4a + 10 forgets to flip the sign of +8. 3a − 1 uses only length + breadth and one side.
---

This set tests the basic skills of algebra: collecting like terms, working out values, writing words as expressions, and using the first identities.

## Like terms

Terms are **like** when they have exactly the same letters raised to the same powers: 5x²y and −2yx² are like, but 5x²y and 5xy² are not. Only like terms can be added: 3ab + 4ab = 7ab, but 3a + 4b stays as it is.

## Adding and subtracting

To add, drop the brackets and collect like terms. To subtract an expression, **change the sign of every term in it**, then add.

## Value of an expression

Put the number in place of the letter, using brackets: with x = −3, x² = (−3)² = 9 and −4x = −4 × (−3) = +12. Do powers first, then ×, then + and −.

## Words to algebra

| Words | Algebra |
|---|---|
| 7 more than n | n + 7 |
| 7 less than n | n − 7 |
| n less than 7 | 7 − n |
| three times the sum of n and 2 | 3(n + 2) |

## Identities

- (a + b)² = a² + 2ab + b²
- (a − b)² = a² − 2ab + b²
- (a + b)(a − b) = a² − b²

These also make mental sums quick: 51² = (50 + 1)² = 2500 + 100 + 1.

## Worked example

*Expand (x − 4)².*

a = x, b = 4: x² − 2 × x × 4 + 16 = **x² − 8x + 16**.

## Common trap

Writing (a + b)² as a² + b². The middle term 2ab is always there. Test it with numbers: (2 + 3)² = 25, but 2² + 3² = 13.
