---
title: "Number puzzles 1: magic squares and number grids"
description: "Fill the gap in magic squares where every row, column and diagonal has the same sum, and find the rule hiding in a number grid."
slug: number-puzzles-1
family: number-puzzles
part: 1
timeMinutes: 12
questions:
  - id: np1-loshu-centre
    topic: Magic squares
    difficulty: 1
    q: "In this magic square, every row, column and diagonal adds up to the same number. What number goes in place of ?"
    figure: { kind: grid, cells: [[2, 7, 6], [9, "?", 1], [4, 3, 8]] }
    options: ["3", "4", "5", "6"]
    answer: C
    explanation: |
      The top row gives the magic sum: 2 + 7 + 6 = 15. The middle row needs 9 + ? + 1 = 15, so ? = **5**.
      Check with the middle column: 7 + 5 + 3 = 15 ✓.

  - id: np1-powers-row
    topic: Grid rules
    difficulty: 1
    q: "Each row of this grid follows the same rule. What number goes in place of ?"
    figure: { kind: grid, cells: [[2, 4, 8], [3, 9, 27], [5, "?", 125]] }
    options: ["10", "25", "50", "125"]
    answer: B
    explanation: |
      Each row is a number, its square and its cube: 2, 4, 8 and 3, 9, 27. For 5: 5 × 5 = **25** and 5 × 5 × 5 = 125.
      10 doubles 5, which works only for the first row (2 × 2 = 4).

  - id: np1-column-product
    topic: Grid rules
    difficulty: 1
    q: "Find the rule that links the three numbers in the first and third columns. Use it to find the number in place of ?"
    figure: { kind: grid, cells: [[4, 6, 8], [3, 5, 2], [12, "?", 16]] }
    options: ["11", "16", "24", "30"]
    answer: D
    explanation: |
      In each column, top × middle = bottom: 4 × 3 = 12 and 8 × 2 = 16. So ? = 6 × 5 = **30**.
      11 adds the two numbers instead (6 + 5), but 4 + 3 is not 12.

  - id: np1-magic-corner
    topic: Magic squares
    difficulty: 1
    q: "Every row, column and diagonal of this square adds up to the same number. What number goes in place of ?"
    figure: { kind: grid, cells: [[5, 10, "?"], [12, 8, 4], [7, 6, 11]] }
    options: ["9", "10", "11", "12"]
    answer: A
    explanation: |
      The middle row gives the magic sum: 12 + 8 + 4 = 24. Top row: 5 + 10 + ? = 24, so ? = **9**.
      Check with the diagonal: 9 + 8 + 7 = 24 ✓.

  - id: np1-totals-chain
    topic: Grid rules
    difficulty: 2
    q: "The shaded cells give the total of each row (on the right) and each column (at the bottom). One number is marked ? and two are left blank. What number goes in place of ?"
    figure: { kind: grid, cells: [[4, "?", "", 20], [6, "", 2, 17], [8, 5, 8, 21], [18, 21, 19, null]], shade: [[0, 3], [1, 3], [2, 3], [3, 0], [3, 1], [3, 2]] }
    options: ["5", "7", "9", "11"]
    answer: B
    explanation: |
      Start where only one number is missing. Third column: blank + 2 + 8 = 19, so the blank in the top row is 9.
      Now the top row: 4 + ? + 9 = 20, so ? = **7**. (Check: the middle blank is 17 − 6 − 2 = 9, and the second column is 7 + 9 + 5 = 21 ✓.)

  - id: np1-double-sum-rule
    topic: Grid rules
    difficulty: 2
    q: "The same rule links the three numbers in each row. What number goes in place of ?"
    figure: { kind: grid, cells: [[5, 3, 16], [7, 2, 18], [6, "?", 22]] }
    options: ["5", "8", "10", "16"]
    answer: A
    explanation: |
      Third number = (first + second) × 2: (5 + 3) × 2 = 16 and (7 + 2) × 2 = 18.
      So 6 + ? must be 22 ÷ 2 = 11, giving ? = **5**. 16 is 22 − 6, which forgets the doubling.

  - id: np1-four-by-four
    topic: Magic squares
    difficulty: 2
    q: "In this 4 × 4 magic square, every row, column and diagonal has the same sum. What number goes in place of ?"
    figure: { kind: grid, cells: [[16, 3, 2, 13], [5, 10, 11, 8], ["", "?", 7, 12], [4, 15, 14, 1]] }
    options: ["5", "6", "7", "9"]
    answer: B
    explanation: |
      The top row gives the sum: 16 + 3 + 2 + 13 = 34. The second column has only one gap: 3 + 10 + ? + 15 = 34, so ? = **6**.
      The third row has two gaps, so it cannot be used first. Its blank is 9, which is the tempting wrong answer.

  - id: np1-odd-magic-sum
    topic: Magic squares
    difficulty: 2
    q: "A 3 × 3 magic square is made using each of the nine odd numbers 3, 5, 7, 9, 11, 13, 15, 17 and 19 once. What is its magic sum (the sum of each row)?"
    options: ["11", "27", "33", "99"]
    answer: C
    explanation: |
      All nine numbers add up to 99. The three rows share this total equally, so each row adds up to 99 ÷ 3 = **33**.
      11 is the number in the centre (one third of 33), and 99 is the sum of all nine numbers.

  - id: np1-squares-rule
    topic: Grid rules
    difficulty: 2
    q: "The same rule links the three numbers in each row. What number goes in place of ?"
    figure: { kind: grid, cells: [[2, 3, 13], [4, 1, 17], [6, 5, "?"]] }
    options: ["11", "30", "61", "121"]
    answer: C
    explanation: |
      Third number = first² + second²: 4 + 9 = 13 and 16 + 1 = 17. So ? = 36 + 25 = **61**.
      121 is (6 + 5)², which does not fit the first row: (2 + 3)² = 25, not 13.

  - id: np1-three-clues
    topic: Magic squares
    difficulty: 3
    q: "This is a magic square: every row, column and diagonal has the same sum. Only three numbers are shown. What number goes in place of ?"
    figure: { kind: grid, cells: [[13, "", ""], ["", 10, ""], ["?", 14, ""]] }
    options: ["6", "7", "8", "9"]
    answer: D
    explanation: |
      Call the magic sum S. The diagonal 13, 10, bottom-right gives bottom-right = S − 23.
      The bottom row: ? + 14 + (S − 23) = S, so ? − 9 = 0 and ? = **9**.
      (In fact S = 3 × 10 = 30, and the full square is 13 6 11 / 8 10 12 / 9 14 7.)

  - id: np1-corners-centre
    topic: Magic squares
    difficulty: 3
    q: "In a 3 × 3 magic square, the four corner numbers (shaded) add up to 48. What number is in the centre?"
    figure: { kind: grid, cells: [["", "", ""], ["", "?", ""], ["", "", ""]], shade: [[0, 0], [0, 2], [2, 0], [2, 2]] }
    options: ["12", "16", "24", "36"]
    answer: A
    explanation: |
      Each diagonal is corner + centre + corner = S, so the four corners add up to 2S − 2 × centre.
      In a 3 × 3 magic square S = 3 × centre, so the corners add up to 6 × centre − 2 × centre = 4 × centre. Centre = 48 ÷ 4 = **12**.
      16 divides 48 by 3, as if the corners made a row.

  - id: np1-never-corner
    topic: Magic squares
    difficulty: 3
    q: "The numbers 1 to 9 are placed in a 3 × 3 magic square. Which of these numbers can NEVER be in a corner (shaded)?"
    figure: { kind: grid, cells: [["", "", ""], ["", "", ""], ["", "", ""]], shade: [[0, 0], [0, 2], [2, 0], [2, 2]] }
    options: ["2", "4", "7", "8"]
    answer: C
    explanation: |
      The magic sum is 45 ÷ 3 = 15 and the centre is 5. A corner lies on three lines, so it needs three different pairs to finish them.
      For **7**, each pair must add up to 8: only 2 + 6 and 3 + 5 are possible (1 + 7 would use 7 twice). Two pairs are not enough.
      2, 4 and 8 all fit in corners, as in the square 2 7 6 / 9 5 1 / 4 3 8.
---

This set tests magic squares and number grids, where you find a hidden rule and use it to fill a gap.

## Magic squares

In a magic square every row, every column and both diagonals have the same total, the **magic sum**.

- Find the magic sum from any full line. Then look for a line with **only one gap** and fill it.
- If you know all the numbers in the square, magic sum = (total of all numbers) ÷ (number of rows).
- In every 3 × 3 magic square, the **centre is one third of the magic sum**. Here is why. The middle row, the middle column and the two diagonals together make 4 lines, so they add up to 4 × S. They use every cell once, plus the centre three extra times. So 3 × S + 3 × centre = 4 × S, which gives S = 3 × centre.

## Number grids

When each row (or column) follows one rule, try these in order: add, subtract, multiply, then squares and cubes, then mixes like (a + b) × 2. A rule must work for **every** full row, not just the first.

## Worked example

*Rows: 3, 4, 19 and 5, 2, 17 and 6, 3, ?*

Try a × b + something: 3 × 4 = 12 and 19 − 12 = 7; 5 × 2 = 10 and 17 − 10 = 7. The rule is a × b + 7, so ? = 6 × 3 + 7 = **25**.

## Common trap

In a totals grid or a magic square, don't start with a line that has two gaps. Look for a line with only one unknown, fill it, then use the new number to open up the next line.
