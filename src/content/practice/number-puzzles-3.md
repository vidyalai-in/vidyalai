---
title: "Number puzzles 3: balances, pyramids and number tricks"
description: "Weigh shapes on balanced scales, fill number pyramids, split numbers into consecutive parts and see why think-of-a-number tricks always work."
slug: number-puzzles-3
family: number-puzzles
part: 3
timeMinutes: 12
questions:
  - id: np3-pyramid-top
    topic: Number pyramids
    difficulty: 1
    q: "In this number pyramid, each cell is the sum of the two cells just below it. What number goes at the top, in place of ?"
    figure: {"kind": "shapes", "w": 124, "h": 82, "alt": "A number pyramid: bottom row 4, 7, 2; middle row two empty cells; top cell marked ?", "items": [{"t": "rect", "p": [42.0, 2, 40, 26], "fill": "none"}, {"t": "text", "at": [62.0, 16.0], "s": "?", "size": 15}, {"t": "rect", "p": [22.0, 28, 40, 26], "fill": "none"}, {"t": "rect", "p": [62.0, 28, 40, 26], "fill": "none"}, {"t": "rect", "p": [2.0, 54, 40, 26], "fill": "none"}, {"t": "text", "at": [22.0, 68.0], "s": "4", "size": 15}, {"t": "rect", "p": [42.0, 54, 40, 26], "fill": "none"}, {"t": "text", "at": [62.0, 68.0], "s": "7", "size": 15}, {"t": "rect", "p": [82.0, 54, 40, 26], "fill": "none"}, {"t": "text", "at": [102.0, 68.0], "s": "2", "size": 15}]}
    options: ["13", "20", "22", "24"]
    answer: B
    explanation: |
      Middle row: 4 + 7 = 11 and 7 + 2 = 9. Top: 11 + 9 = **20**.
      13 just adds the bottom row. The 7 in the middle is used twice, once for each cell above it.

  - id: np3-consecutive-three
    topic: Consecutive numbers
    difficulty: 1
    q: "Three consecutive whole numbers add up to 84. What is the largest of them?"
    options: ["27", "28", "29", "30"]
    answer: C
    explanation: |
      The middle number is the average: 84 ÷ 3 = 28. The numbers are 27, 28 and 29, so the largest is **29**.
      28 is the middle number, not the largest.

  - id: np3-think-five
    topic: Think of a number
    difficulty: 1
    q: "Think of any number. Double it and add 10. Halve the result, then take away the number you first thought of. What is the answer?"
    options: ["5", "10", "The number you started with", "It depends on the number"]
    answer: A
    explanation: |
      Call the number n. Double and add 10: 2n + 10. Halve: n + 5. Take away n: **5**, whatever n was.
      10 forgets that the 10 was halved too.

  - id: np3-apple-orange
    topic: Balances
    difficulty: 1
    q: "Two apples and one orange together weigh 400 g. One apple and one orange together weigh 260 g. All the apples weigh the same. How much does one apple weigh?"
    options: ["70 g", "120 g", "130 g", "140 g"]
    answer: D
    explanation: |
      The first group has exactly one apple more than the second. So one apple = 400 − 260 = **140 g**.
      120 g is the orange (260 − 140), and 130 g just halves 260.

  - id: np3-scales-figure
    topic: Balances
    difficulty: 2
    q: "Both scales are balanced. All circles weigh the same, and so do all squares. How many circles weigh the same as one triangle?"
    figure: { kind: shapes, w: 200, h: 155, alt: "Two balanced scales. Scale 1: a triangle and a circle on the left, two squares on the right. Scale 2: one square on the left, two circles on the right.", items: [{ t: text, at: [100, 12], s: "Scale 1", size: 11 }, { t: line, p: [30, 46, 170, 46], w: 2 }, { t: poly, p: [[100, 46], [88, 70], [112, 70]], fill: grey }, { t: line, p: [78, 70, 122, 70], w: 2 }, { t: line, p: [40, 46, 40, 40] }, { t: line, p: [10, 40, 70, 40], w: 2 }, { t: line, p: [160, 46, 160, 40] }, { t: line, p: [130, 40, 190, 40], w: 2 }, { t: poly, p: [[23, 39], [39, 39], [31, 24]], fill: accent }, { t: circle, c: [49, 32], r: 7, fill: solid }, { t: rect, p: [144, 26, 14, 14], fill: grey }, { t: rect, p: [162, 26, 14, 14], fill: grey }, { t: text, at: [100, 92], s: "Scale 2", size: 11 }, { t: line, p: [30, 126, 170, 126], w: 2 }, { t: poly, p: [[100, 126], [88, 150], [112, 150]], fill: grey }, { t: line, p: [78, 150, 122, 150], w: 2 }, { t: line, p: [40, 126, 40, 120] }, { t: line, p: [10, 120, 70, 120], w: 2 }, { t: line, p: [160, 126, 160, 120] }, { t: line, p: [130, 120, 190, 120], w: 2 }, { t: rect, p: [33, 106, 14, 14], fill: grey }, { t: circle, c: [151, 112], r: 7, fill: solid }, { t: circle, c: [169, 112], r: 7, fill: solid }] }
    options: ["2", "3", "4", "5"]
    answer: B
    explanation: |
      Scale 2: one square = 2 circles, so the two squares on scale 1 weigh 4 circles.
      Scale 1: triangle + circle = 4 circles. Take one circle off each side: triangle = **3** circles.
      4 forgets the circle sitting next to the triangle.

  - id: np3-pyramid-side
    topic: Number pyramids
    difficulty: 2
    q: "Each cell is the sum of the two cells just below it. What number goes in place of ?"
    figure: {"kind": "shapes", "w": 124, "h": 82, "alt": "A number pyramid: top 30; middle row 13 and an empty cell; bottom row 6, an empty cell and ?", "items": [{"t": "rect", "p": [42.0, 2, 40, 26], "fill": "none"}, {"t": "text", "at": [62.0, 16.0], "s": "30", "size": 15}, {"t": "rect", "p": [22.0, 28, 40, 26], "fill": "none"}, {"t": "text", "at": [42.0, 42.0], "s": "13", "size": 15}, {"t": "rect", "p": [62.0, 28, 40, 26], "fill": "none"}, {"t": "rect", "p": [2.0, 54, 40, 26], "fill": "none"}, {"t": "text", "at": [22.0, 68.0], "s": "6", "size": 15}, {"t": "rect", "p": [42.0, 54, 40, 26], "fill": "none"}, {"t": "rect", "p": [82.0, 54, 40, 26], "fill": "none"}, {"t": "text", "at": [102.0, 68.0], "s": "?", "size": 15}]}
    options: ["10", "11", "17", "23"]
    answer: A
    explanation: |
      Top: 13 + right middle = 30, so the right middle cell is 17. Left: 6 + bottom middle = 13, so the bottom middle is 7.
      Right: 7 + ? = 17, so ? = **10**. 17 stops one step too early.

  - id: np3-even-five
    topic: Consecutive numbers
    difficulty: 2
    q: "Five consecutive even numbers add up to 120. What is the smallest of them?"
    options: ["18", "20", "22", "24"]
    answer: B
    explanation: |
      The middle number is 120 ÷ 5 = 24. Even numbers go up in 2s, so they are 20, 22, 24, 26, 28. The smallest is **20**.
      22 would be right for five consecutive whole numbers (22 to 26), not even ones.

  - id: np3-reverse-steps
    topic: Think of a number
    difficulty: 2
    q: "Ravi thinks of a number. He multiplies it by 4, takes away 6, then halves the result. He gets 25. What was his number?"
    options: ["7.75", "11", "14", "56"]
    answer: C
    explanation: |
      Work backwards, undoing each step: 25 × 2 = 50, then 50 + 6 = 56, then 56 ÷ 4 = **14**.
      Check: 14 × 4 = 56, 56 − 6 = 50, 50 ÷ 2 = 25 ✓. 11 takes away 6 instead of adding it back.

  - id: np3-three-shapes
    topic: Balances
    difficulty: 2
    q: "3 triangles weigh the same as 2 squares. A circle weighs the same as a triangle and a square together. One triangle, one square and one circle together weigh 50 g. How much does the circle weigh?"
    options: ["10 g", "15 g", "20 g", "25 g"]
    answer: D
    explanation: |
      Triangle + square = circle, so triangle + square + circle = circle + circle = 50 g. The circle weighs **25 g**.
      You don't even need the first clue. (It gives the triangle 10 g and the square 15 g.)

  - id: np3-reverse-198
    topic: Think of a number
    difficulty: 3
    q: "Priya thinks of a three-digit number whose hundreds digit is 2 more than its ones digit (and the ones digit is not 0). She writes the digits in reverse order and takes the smaller number away from the larger. What answer does she get?"
    options: ["99", "198", "297", "It depends on the number"]
    answer: B
    explanation: |
      Write the number as 100a + 10b + c. Its reverse is 100c + 10b + a. The difference is 99a − 99c = 99 × (a − c).
      Here a − c = 2, so she always gets 99 × 2 = **198**. The middle digit cancels out. (For example, 735 − 537 = 198.)

  - id: np3-pyramid-base
    topic: Number pyramids
    difficulty: 3
    q: "Each cell is the sum of the two cells just below it. Only the bottom row and the top are shown. What number goes in place of ?"
    figure: {"kind": "shapes", "w": 164, "h": 108, "alt": "A number pyramid with four rows: top 45; the two middle rows empty; bottom row 2, 5, ? and 4", "items": [{"t": "rect", "p": [62.0, 2, 40, 26], "fill": "none"}, {"t": "text", "at": [82.0, 16.0], "s": "45", "size": 15}, {"t": "rect", "p": [42.0, 28, 40, 26], "fill": "none"}, {"t": "rect", "p": [82.0, 28, 40, 26], "fill": "none"}, {"t": "rect", "p": [22.0, 54, 40, 26], "fill": "none"}, {"t": "rect", "p": [62.0, 54, 40, 26], "fill": "none"}, {"t": "rect", "p": [102.0, 54, 40, 26], "fill": "none"}, {"t": "rect", "p": [2.0, 80, 40, 26], "fill": "none"}, {"t": "text", "at": [22.0, 94.0], "s": "2", "size": 15}, {"t": "rect", "p": [42.0, 80, 40, 26], "fill": "none"}, {"t": "text", "at": [62.0, 94.0], "s": "5", "size": 15}, {"t": "rect", "p": [82.0, 80, 40, 26], "fill": "none"}, {"t": "text", "at": [102.0, 94.0], "s": "?", "size": 15}, {"t": "rect", "p": [122.0, 80, 40, 26], "fill": "none"}, {"t": "text", "at": [142.0, 94.0], "s": "4", "size": 15}]}
    options: ["4", "6", "7", "8"]
    answer: D
    explanation: |
      Call the bottom row a, b, c, d. The rows above are a + b, b + c, c + d, then a + 2b + c, b + 2c + d, and the top is a + 3b + 3c + d.
      So 2 + 15 + 3 × ? + 4 = 45, giving 3 × ? = 24 and ? = **8**. Check: 2, 5, 8, 4 → 7, 13, 12 → 20, 25 → 45 ✓.

  - id: np3-45-ways
    topic: Consecutive numbers
    difficulty: 3
    q: "In how many ways can 45 be written as a sum of two or more consecutive positive whole numbers? (For example, 22 + 23 is one way.)"
    options: ["3", "4", "5", "6"]
    answer: C
    explanation: |
      22 + 23, 14 + 15 + 16, 7 + 8 + 9 + 10 + 11, 5 + 6 + 7 + 8 + 9 + 10, and 1 + 2 + ... + 9. That makes **5** ways.
      Try every length: 4 terms fail (the middle would be 11.25), and so do 7 and 8 terms. 6 wrongly counts 45 on its own, which is only one number.
---

This set tests number puzzles you solve by careful reasoning: balanced scales, number pyramids, sums of consecutive numbers and think-of-a-number tricks.

## Balances

A balanced scale is an equation. You may take the **same thing off both sides**, or swap something for an equal weight. Change every shape into the smallest one (often a circle), then count.

## Number pyramids

Each cell is the sum of the two cells below it. Work from a cell where you know two of its three neighbours. In a pyramid with 4 numbers at the bottom, a, b, c, d, the top is **a + 3b + 3c + d**: the middle numbers count three times.

## Consecutive numbers

- For an **odd** count of consecutive numbers, the middle one is the average: sum ÷ count.
- Even and odd numbers go up in 2s, so their neighbours are 2 apart.
- To split a number into consecutive parts, try each length. With 3 terms the middle is sum ÷ 3; with 4 terms the middle two have an average of sum ÷ 4, which must end in .5.

## Think of a number

Call the secret number n and write each step: double gives 2n, add 6 gives 2n + 6, and so on. If n disappears at the end, the answer is the same for everyone. To find a secret number from the answer, **undo the steps in reverse order**.

## Worked example

*Kabir thinks of a number, adds 7, multiplies by 3 and gets 45. What was it?*

Undo: 45 ÷ 3 = 15, then 15 − 7 = **8**. Check: (8 + 7) × 3 = 45 ✓.

## Common trap

In a pyramid, a middle number is used by **two** cells above it. Don't count it only once.
