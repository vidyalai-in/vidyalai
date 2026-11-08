---
title: "Number series 2: missing numbers in grids and figures"
description: "Row and column rules in number grids, numbers around triangles, squares and circles, and series that mix letters with numbers."
slug: number-series-2
family: number-series
part: 2
timeMinutes: 12
questions:
  - id: ns2-grid-row-sum
    topic: Grids
    difficulty: 1
    q: "The same rule works in every row. Which number goes in place of the question mark?"
    figure: { kind: grid, cells: [[4, 7, 11], [6, 9, 15], [8, 5, "?"]], shade: [[2, 2]] }
    options: ["12", "13", "14", "40"]
    answer: B
    explanation: |
      In each row, the third number is the sum of the first two: 4 + 7 = 11, 6 + 9 = 15.
      So 8 + 5 = **13**. 40 multiplies instead of adding; check the first row: 4 × 7 is 28, not 11.

  - id: ns2-grid-row-product
    topic: Grids
    difficulty: 1
    q: "The same rule works in every row. Which number goes in place of the question mark?"
    figure: { kind: grid, cells: [[3, 4, 12], [5, 2, 10], [6, 7, "?"]], shade: [[2, 2]] }
    options: ["13", "36", "42", "49"]
    answer: C
    explanation: |
      In each row, the third number is the first times the second: 3 × 4 = 12, 5 × 2 = 10.
      So 6 × 7 = **42**. 13 adds instead; that fails the first row (3 + 4 = 7, not 12).

  - id: ns2-grid-column-sum
    topic: Grids
    difficulty: 1
    q: "In this grid every column adds up to the same total. Which number is missing?"
    figure: { kind: grid, cells: [[6, 9, "?"], [8, 3, 5], [6, 8, 7]], shade: [[0, 2]] }
    options: ["5", "6", "7", "8"]
    answer: D
    explanation: |
      First column: 6 + 8 + 6 = 20. Second column: 9 + 3 + 8 = 20. So every column adds to 20.
      Third column: ? + 5 + 7 = 20, so ? = 20 − 12 = **8**.

  - id: ns2-letter-odd
    topic: Letters and numbers
    difficulty: 1
    q: "What comes next? B2, D4, F6, H8, ?"
    options: ["J10", "K11", "J12", "L12"]
    answer: A
    explanation: |
      The letters skip one each time (B, D, F, H, then J), and the numbers go up by 2 (2, 4, 6, 8, then 10).
      Each number is also the letter's place in the alphabet. So the answer is **J10**. K11 skips two letters instead of one.

  - id: ns2-letter-double
    topic: Letters and numbers
    difficulty: 2
    q: "What comes next? B2, D4, F8, H16, ?"
    options: ["J32", "I32", "J18", "L32"]
    answer: A
    explanation: |
      Letters: B, D, F, H skip one letter each time, so next is J. Numbers: 2, 4, 8, 16 double each time, so next is 32.
      The answer is **J32**. J18 adds 2 to 16, but the numbers double: 4 to 8 is + 4, not + 2.

  - id: ns2-triangles
    topic: Figures
    difficulty: 2
    q: "The same rule links the four numbers in each triangle. Which number goes in place of the question mark?"
    figure:
      kind: shapes
      w: 360
      alt: "Three triangles with a number at each corner and one inside. Triangle 1: top 2, left 3, right 4, inside 14. Triangle 2: top 3, left 5, right 1, inside 18. Triangle 3: top 4, left 6, right 3, inside unknown."
      items:
        - { t: poly, p: [[60, 24], [26, 80], [94, 80]] }
        - { t: text, at: [60, 11], s: "2", size: 15 }
        - { t: text, at: [19, 91], s: "3", size: 15 }
        - { t: text, at: [101, 91], s: "4", size: 15 }
        - { t: text, at: [60, 62], s: "14", size: 16 }
        - { t: poly, p: [[180, 24], [146, 80], [214, 80]] }
        - { t: text, at: [180, 11], s: "3", size: 15 }
        - { t: text, at: [139, 91], s: "5", size: 15 }
        - { t: text, at: [221, 91], s: "1", size: 15 }
        - { t: text, at: [180, 62], s: "18", size: 16 }
        - { t: poly, p: [[300, 24], [266, 80], [334, 80]] }
        - { t: text, at: [300, 11], s: "4", size: 15 }
        - { t: text, at: [259, 91], s: "6", size: 15 }
        - { t: text, at: [341, 91], s: "3", size: 15 }
        - { t: text, at: [300, 62], s: "?", size: 16 }
    options: ["13", "18", "27", "36"]
    answer: D
    explanation: |
      Add the two bottom corners, then multiply by the top: (3 + 4) × 2 = 14 and (5 + 1) × 3 = 18.
      Third triangle: (6 + 3) × 4 = **36**. 13 is just the sum of the corners, which fails the first triangle (2 + 3 + 4 = 9, not 14).

  - id: ns2-wheel
    topic: Figures
    difficulty: 2
    q: "Look at the numbers in parts of the circle that are directly opposite each other. Which number goes in place of the question mark?"
    figure:
      kind: shapes
      alt: "A circle cut into 8 equal parts. Going clockwise from the top right: 2, 3, 5, 7, 4, 9, 25 and a question mark."
      items:
        - { t: circle, c: [50, 50], r: 46, fill: none }
        - { t: line, p: [50, 4, 50, 96], w: 1.5 }
        - { t: line, p: [82.5, 17.5, 17.5, 82.5], w: 1.5 }
        - { t: line, p: [96, 50, 4, 50], w: 1.5 }
        - { t: line, p: [82.5, 82.5, 17.5, 17.5], w: 1.5 }
        - { t: text, at: [61.5, 22.3], s: "2", size: 14 }
        - { t: text, at: [77.7, 38.5], s: "3", size: 14 }
        - { t: text, at: [77.7, 61.5], s: "5", size: 14 }
        - { t: text, at: [61.5, 77.7], s: "7", size: 14 }
        - { t: text, at: [38.5, 77.7], s: "4", size: 14 }
        - { t: text, at: [22.3, 61.5], s: "9", size: 14 }
        - { t: text, at: [22.3, 38.5], s: "25", size: 14 }
        - { t: text, at: [38.5, 22.3], s: "?", size: 14 }
    options: ["14", "21", "49", "343"]
    answer: C
    explanation: |
      Opposite pairs: 2 and 4, 3 and 9, 5 and 25. Each number is the square of the number opposite it.
      The question mark is opposite 7, so it is 7² = **49**. 14 doubles 7 (that works for 2 and 4 only); 343 is 7³.

  - id: ns2-grid-squares
    topic: Grids
    difficulty: 2
    q: "The same rule works in every row. Which number goes in place of the question mark?"
    figure: { kind: grid, cells: [[2, 3, 13], [1, 4, 17], [3, 5, "?"]], shade: [[2, 2]] }
    options: ["15", "34", "45", "64"]
    answer: B
    explanation: |
      Square the first two numbers and add: 2² + 3² = 4 + 9 = 13, and 1² + 4² = 1 + 16 = 17.
      Last row: 3² + 5² = 9 + 25 = **34**. 64 squares the sum, (3 + 5)², which fails the first row: (2 + 3)² = 25.

  - id: ns2-letter-grid
    topic: Letters and numbers
    difficulty: 2
    q: "The same rule works in every row (A = 1, B = 2, C = 3, ...). Which letter goes in place of the question mark?"
    figure: { kind: grid, cells: [[B, 5, G], [D, 6, J], [F, 8, "?"]], shade: [[2, 2]] }
    options: ["M", "N", "O", "P"]
    answer: B
    explanation: |
      Move forward in the alphabet by the middle number: B (2) + 5 = 7 = G, D (4) + 6 = 10 = J.
      Last row: F (6) + 8 = 14, which is **N**. M is one short (13): count carefully.

  - id: ns2-letter-back
    topic: Letters and numbers
    difficulty: 3
    q: "What comes next? Z1, X3, U6, Q10, ?"
    options: ["L15", "K15", "M15", "L14"]
    answer: A
    explanation: |
      Letters go back 2, 3, 4, then 5: Z (26), X (24), U (21), Q (17), so next is 17 − 5 = 12 = L.
      Numbers go up by 2, 3, 4, then 5: 1, 3, 6, 10, so next is 15. The answer is **L15**.
      M15 goes back only 4 letters again.

  - id: ns2-squares-cross
    topic: Figures
    difficulty: 3
    q: "The same rule links the five numbers in each square. Which number goes in place of the question mark?"
    figure:
      kind: shapes
      w: 360
      alt: "Three squares with a number at each corner and one in the middle. Square 1: top left 6, top right 2, bottom left 3, bottom right 5, middle 24. Square 2: 4, 1, 2, 7, middle 26. Square 3: 8, 3, 5, 4, middle unknown."
      items:
        - { t: rect, p: [33, 22, 54, 54], fill: none }
        - { t: text, at: [24, 12], s: "6", size: 15 }
        - { t: text, at: [96, 12], s: "2", size: 15 }
        - { t: text, at: [24, 88], s: "3", size: 15 }
        - { t: text, at: [96, 88], s: "5", size: 15 }
        - { t: text, at: [60, 49], s: "24", size: 17 }
        - { t: rect, p: [153, 22, 54, 54], fill: none }
        - { t: text, at: [144, 12], s: "4", size: 15 }
        - { t: text, at: [216, 12], s: "1", size: 15 }
        - { t: text, at: [144, 88], s: "2", size: 15 }
        - { t: text, at: [216, 88], s: "7", size: 15 }
        - { t: text, at: [180, 49], s: "26", size: 17 }
        - { t: rect, p: [273, 22, 54, 54], fill: none }
        - { t: text, at: [264, 12], s: "8", size: 15 }
        - { t: text, at: [336, 12], s: "3", size: 15 }
        - { t: text, at: [264, 88], s: "5", size: 15 }
        - { t: text, at: [336, 88], s: "4", size: 15 }
        - { t: text, at: [300, 49], s: "?", size: 17 }
    options: ["12", "15", "17", "47"]
    answer: C
    explanation: |
      Multiply along each diagonal and subtract: (top left × bottom right) − (top right × bottom left).
      Square 1: 6 × 5 − 2 × 3 = 30 − 6 = 24. Square 2: 4 × 7 − 1 × 2 = 28 − 2 = 26.
      Square 3: 8 × 4 − 3 × 5 = 32 − 15 = **17**. 47 adds the two products instead of subtracting; check square 1: 30 + 6 is 36, not 24.

  - id: ns2-grid-column-rule
    topic: Grids
    difficulty: 3
    q: "The same rule works in every column. Which number goes in place of the question mark?"
    figure: { kind: grid, cells: [[3, 5, 4], [6, 2, 7], [17, 9, "?"]], shade: [[2, 2]] }
    options: ["11", "27", "28", "29"]
    answer: B
    explanation: |
      Multiply the top two numbers of a column and subtract 1: 3 × 6 − 1 = 17, 5 × 2 − 1 = 9.
      Third column: 4 × 7 − 1 = **27**. 28 forgets to subtract 1; 11 adds, which fails the first column (3 + 6 = 9, not 17).
---

These questions hide a rule inside a grid or a picture. Find it from the complete parts, test it on **all** of them, then use it on the part with the question mark.

## Where to look

- **Grids:** read across the rows first, then down the columns. Common rules: the third number is the sum, difference or product of the other two; every row or column adds to the same total; squares of the numbers are added.
- **Shapes:** compare the number in the middle with the numbers around it. Try adding, multiplying, or a mix such as (a + b) × c. In a circle, also compare numbers that are opposite each other.
- **Letters with numbers:** turn letters into their place in the alphabet (A = 1, ..., Z = 26) and treat them like a number series. Letters and numbers often follow two separate rules.

| Letter | E | J | O | T | Y |
|---|---|---|---|---|---|
| Place | 5 | 10 | 15 | 20 | 25 |

Remembering every fifth letter makes counting much faster.

## Worked example

A grid has rows 2, 5, 10 and 3, 4, 12 and 6, 1, ?. Adding fails in the first row (2 + 5 = 7). Multiplying works: 2 × 5 = 10 and 3 × 4 = 12, so ? = 6 × 1 = 6.

## Common traps

- A rule that fits only the first row or the first figure. Always check the second one too.
- Reading a grid by rows when the rule goes down the columns.
- Counting letters off by one. G is the 7th letter, so "G + 3" is J (10), not K.
