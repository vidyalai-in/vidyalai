---
title: "Figure counting 2: squares, rectangles, lines and parts"
description: "Count squares and rectangles in grids, the fewest straight lines that draw a figure, and the closed parts a figure is cut into."
slug: "figure-counting-2"
family: "figure-counting"
part: 2
timeMinutes: 12
questions:
  - id: fc2-squares-2x2
    topic: "Squares"
    difficulty: 1
    q: "How many squares are there in this figure?"
    figure: {"kind": "shapes", "w": 170, "h": 170, "items": [{"t": "line", "p": [10, 10, 160, 10]}, {"t": "line", "p": [10, 85, 160, 85]}, {"t": "line", "p": [10, 160, 160, 160]}, {"t": "line", "p": [10, 10, 10, 160]}, {"t": "line", "p": [85, 10, 85, 160]}, {"t": "line", "p": [160, 10, 160, 160]}], "alt": "A square divided into a 2 by 2 grid of small squares"}
    options: ["4", "5", "6", "8"]
    answer: B
    explanation: |
      There are 4 small squares and 1 big square made of all four: 4 + 1 = **5**.
      4 forgets the big outer square.

  - id: fc2-rectangles-strip
    topic: "Rectangles"
    difficulty: 1
    q: "How many rectangles are there in this figure?"
    figure: {"kind": "shapes", "w": 212, "h": 84, "items": [{"t": "line", "p": [10, 10, 202, 10]}, {"t": "line", "p": [10, 74, 202, 74]}, {"t": "line", "p": [10, 10, 10, 74]}, {"t": "line", "p": [58, 10, 58, 74]}, {"t": "line", "p": [106, 10, 106, 74]}, {"t": "line", "p": [154, 10, 154, 74]}, {"t": "line", "p": [202, 10, 202, 74]}], "alt": "A long rectangle divided into 4 boxes in a row"}
    options: ["4", "7", "9", "10"]
    answer: D
    explanation: |
      A rectangle here is fixed by choosing 2 of the 5 upright lines as its left and right sides: 5 × 4 ÷ 2 = **10**.
      By size: 4 of one box, 3 of two boxes, 2 of three boxes, 1 of four boxes. 9 forgets the whole strip.

  - id: fc2-lines-square
    topic: "Straight lines"
    difficulty: 1
    q: "What is the least number of straight lines needed to draw this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 200, "items": [{"t": "line", "p": [20, 20, 180, 20]}, {"t": "line", "p": [180, 20, 180, 180]}, {"t": "line", "p": [180, 180, 20, 180]}, {"t": "line", "p": [20, 180, 20, 20]}, {"t": "line", "p": [20, 20, 180, 180]}, {"t": "line", "p": [180, 20, 20, 180]}, {"t": "line", "p": [100, 20, 100, 180]}, {"t": "line", "p": [20, 100, 180, 100]}], "alt": "A square with both diagonals and both lines joining the midpoints of opposite sides"}
    options: ["8", "10", "12", "16"]
    answer: A
    explanation: |
      Draw each straight line in one go, even where other lines cross it: 4 sides, 2 diagonals and 2 middle lines: **8**.
      16 counts every small piece between crossing points as a separate line.

  - id: fc2-regions-triangle
    topic: "Closed regions"
    difficulty: 1
    q: "Into how many small closed parts do the lines divide this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 180, "items": [{"t": "line", "p": [100, 12, 12, 168]}, {"t": "line", "p": [12, 168, 188, 168]}, {"t": "line", "p": [188, 168, 100, 12]}, {"t": "line", "p": [100, 12, 72, 168]}, {"t": "line", "p": [100, 12, 130, 168]}, {"t": "line", "p": [56, 90, 144, 90]}], "alt": "A triangle with two lines from the top corner to the base and one line across it, parallel to the base"}
    options: ["5", "6", "7", "12"]
    answer: B
    explanation: |
      The line across makes two layers. In each layer the two lines from the top cut it into 3 parts: 3 + 3 = **6**.
      12 is the number of triangles in the figure, which is a different question: here each part is counted once, and parts are not joined.

  - id: fc2-squares-3x3
    topic: "Squares"
    difficulty: 2
    q: "How many squares are there in this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 200, "items": [{"t": "line", "p": [10, 10, 190, 10]}, {"t": "line", "p": [10, 70, 190, 70]}, {"t": "line", "p": [10, 130, 190, 130]}, {"t": "line", "p": [10, 190, 190, 190]}, {"t": "line", "p": [10, 10, 10, 190]}, {"t": "line", "p": [70, 10, 70, 190]}, {"t": "line", "p": [130, 10, 130, 190]}, {"t": "line", "p": [190, 10, 190, 190]}], "alt": "A square divided into a 3 by 3 grid of small squares"}
    options: ["9", "10", "13", "14"]
    answer: D
    explanation: |
      Count by size. 1 by 1: 9. 2 by 2: 2 across × 2 down = 4. 3 by 3: 1.
      9 + 4 + 1 = **14**. 10 forgets the 2 by 2 squares, and 13 forgets the big outer square.

  - id: fc2-rectangles-2x3
    topic: "Rectangles"
    difficulty: 2
    q: "How many rectangles are there in this figure? (A square is also a rectangle.)"
    figure: {"kind": "shapes", "w": 206, "h": 144, "items": [{"t": "line", "p": [10, 10, 196, 10]}, {"t": "line", "p": [10, 72, 196, 72]}, {"t": "line", "p": [10, 134, 196, 134]}, {"t": "line", "p": [10, 10, 10, 134]}, {"t": "line", "p": [72, 10, 72, 134]}, {"t": "line", "p": [134, 10, 134, 134]}, {"t": "line", "p": [196, 10, 196, 134]}], "alt": "A rectangle divided into a grid of 2 rows and 3 columns of boxes"}
    options: ["6", "12", "18", "24"]
    answer: C
    explanation: |
      Every rectangle needs 2 of the 3 lines across (3 ways) and 2 of the 4 upright lines (4 × 3 ÷ 2 = 6 ways): 3 × 6 = **18**.
      6 counts only the single boxes.

  - id: fc2-lines-two-squares
    topic: "Straight lines"
    difficulty: 2
    q: "What is the least number of straight lines needed to draw this figure?"
    figure: {"kind": "shapes", "w": 220, "h": 120, "items": [{"t": "line", "p": [10, 10, 210, 10]}, {"t": "line", "p": [210, 10, 210, 110]}, {"t": "line", "p": [210, 110, 10, 110]}, {"t": "line", "p": [10, 110, 10, 10]}, {"t": "line", "p": [110, 10, 110, 110]}, {"t": "line", "p": [10, 10, 110, 110]}, {"t": "line", "p": [110, 10, 10, 110]}, {"t": "line", "p": [110, 10, 210, 110]}, {"t": "line", "p": [210, 10, 110, 110]}], "alt": "A rectangle made of two squares side by side; each square has both its diagonals drawn"}
    options: ["9", "11", "12", "16"]
    answer: A
    explanation: |
      The top and the bottom are each one long line: 2. The upright lines on the left, in the middle and on the right: 3. The diagonals of the two squares do not line up with each other, so they are 4 separate lines.
      2 + 3 + 4 = **9**. 11 draws the top and the bottom as two pieces each.

  - id: fc2-regions-star
    topic: "Closed regions"
    difficulty: 2
    q: "This star is made of two triangles, one pointing up and one pointing down. Into how many small closed parts do the lines divide this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 204, "items": [{"t": "line", "p": [100, 10, 180, 148]}, {"t": "line", "p": [180, 148, 20, 148]}, {"t": "line", "p": [20, 148, 100, 10]}, {"t": "line", "p": [100, 194, 20, 56]}, {"t": "line", "p": [20, 56, 180, 56]}, {"t": "line", "p": [180, 56, 100, 194]}], "alt": "A six-pointed star made of two overlapping triangles"}
    options: ["6", "7", "8", "12"]
    answer: B
    explanation: |
      There is a small triangle at each of the 6 points of the star, and one six-sided part in the middle: 6 + 1 = **7**.
      6 forgets the middle part, and 8 is the number of triangles, not parts.

  - id: fc2-squares-3x4
    topic: "Squares"
    difficulty: 2
    q: "How many squares are there in this figure?"
    figure: {"kind": "shapes", "w": 220, "h": 170, "items": [{"t": "line", "p": [10, 10, 210, 10]}, {"t": "line", "p": [10, 60, 210, 60]}, {"t": "line", "p": [10, 110, 210, 110]}, {"t": "line", "p": [10, 160, 210, 160]}, {"t": "line", "p": [10, 10, 10, 160]}, {"t": "line", "p": [60, 10, 60, 160]}, {"t": "line", "p": [110, 10, 110, 160]}, {"t": "line", "p": [160, 10, 160, 160]}, {"t": "line", "p": [210, 10, 210, 160]}], "alt": "A rectangle divided into a grid of 3 rows and 4 columns of small squares"}
    options: ["12", "18", "20", "24"]
    answer: C
    explanation: |
      Count by size. 1 by 1: 3 × 4 = 12. 2 by 2: 2 × 3 = 6. 3 by 3: 1 × 2 = 2. There is no 4 by 4 square because there are only 3 rows.
      12 + 6 + 2 = **20**. 18 forgets the two 3 by 3 squares.

  - id: fc2-tilted-squares
    topic: "Squares"
    difficulty: 3
    q: "A square has both diagonals, and a tilted square joins the midpoints of its sides. How many squares are there in this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 200, "items": [{"t": "line", "p": [20, 20, 180, 20]}, {"t": "line", "p": [180, 20, 180, 180]}, {"t": "line", "p": [180, 180, 20, 180]}, {"t": "line", "p": [20, 180, 20, 20]}, {"t": "line", "p": [20, 20, 180, 180]}, {"t": "line", "p": [180, 20, 20, 180]}, {"t": "line", "p": [100, 20, 180, 100]}, {"t": "line", "p": [180, 100, 100, 180]}, {"t": "line", "p": [100, 180, 20, 100]}, {"t": "line", "p": [20, 100, 100, 20]}], "alt": "A square with both diagonals, and a tilted square joining the midpoints of its sides"}
    options: ["2", "4", "6", "8"]
    answer: C
    explanation: |
      The big square and the tilted square: 2. The two diagonals cross the tilted square through its middle, at right angles to its sides, and cut it into 4 small tilted squares: 4 more.
      2 + 4 = **6**. 2 misses the 4 small squares inside the tilted one.

  - id: fc2-rectangles-half-line
    topic: "Rectangles"
    difficulty: 3
    q: "How many rectangles are there in this figure? (A square is also a rectangle.)"
    figure: {"kind": "shapes", "w": 220, "h": 120, "items": [{"t": "line", "p": [10, 10, 210, 10]}, {"t": "line", "p": [10, 110, 210, 110]}, {"t": "line", "p": [10, 10, 10, 110]}, {"t": "line", "p": [210, 10, 210, 110]}, {"t": "line", "p": [60, 10, 60, 110]}, {"t": "line", "p": [110, 10, 110, 110]}, {"t": "line", "p": [160, 10, 160, 110]}, {"t": "line", "p": [10, 60, 110, 60]}], "alt": "A rectangle divided into 4 tall boxes; a line across the middle cuts only the two boxes on the left in half"}
    options: ["16", "18", "20", "30"]
    answer: A
    explanation: |
      Full-height rectangles use 2 of the 5 upright lines: 5 × 4 ÷ 2 = 10.
      Half-height rectangles must sit in the left part, where the middle line is. There, choose 2 of the 3 upright lines (3 ways) and the top or bottom half (2 ways): 6.
      10 + 6 = **16**. 30 treats the middle line as if it went all the way across.

  - id: fc2-regions-busy-square
    topic: "Closed regions"
    difficulty: 3
    q: "A square has both diagonals, both middle lines, and a tilted square joining the midpoints of its sides. Into how many small closed parts do the lines divide this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 200, "items": [{"t": "line", "p": [20, 20, 180, 20]}, {"t": "line", "p": [180, 20, 180, 180]}, {"t": "line", "p": [180, 180, 20, 180]}, {"t": "line", "p": [20, 180, 20, 20]}, {"t": "line", "p": [20, 20, 180, 180]}, {"t": "line", "p": [180, 20, 20, 180]}, {"t": "line", "p": [100, 20, 100, 180]}, {"t": "line", "p": [20, 100, 180, 100]}, {"t": "line", "p": [100, 20, 180, 100]}, {"t": "line", "p": [180, 100, 100, 180]}, {"t": "line", "p": [100, 180, 20, 100]}, {"t": "line", "p": [20, 100, 100, 20]}], "alt": "A square with both diagonals, both middle lines and a tilted square joining the midpoints of its sides"}
    options: ["12", "14", "15", "16"]
    answer: D
    explanation: |
      Inside the tilted square, the 2 diagonals and 2 middle lines all cross at the centre and make 8 parts.
      Outside it there are 4 corner triangles, and each diagonal cuts one of them in two: 8 parts. 8 + 8 = **16**.
      12 forgets that the diagonals cut the corner triangles.

---

This set tests three kinds of counting in figures: squares and rectangles, the straight lines needed to draw a figure, and the closed parts a figure is cut into.

## Squares in a grid

Count by size. In a grid with r rows and c columns of small squares, the number of 2 by 2 squares is (r − 1) × (c − 1), the number of 3 by 3 squares is (r − 2) × (c − 2), and so on, until one side runs out.

## Rectangles in a grid

A rectangle is fixed by its 2 upright sides and its 2 sides across. So choose 2 of the upright lines and 2 of the lines across, and multiply. Choosing 2 out of n lines can be done in n × (n − 1) ÷ 2 ways. A square is also a rectangle, so squares are included.

## Worked example

A rectangle is cut into a grid of 2 rows and 4 columns of small squares. Lines across: 3, so 3 × 2 ÷ 2 = 3 ways. Upright lines: 5, so 5 × 4 ÷ 2 = 10 ways. Rectangles: 3 × 10 = 30. Squares: 8 of size 1 by 1 and 1 × 3 = 3 of size 2 by 2, so 11.

## Straight lines

To find the least number of straight lines, draw each straight line in one stroke, even where other lines cross it. A line counts once, however many pieces it is cut into.

## Closed parts

Here every small closed part is counted once. Parts are not joined to make bigger ones, which is the difference from counting triangles or squares.

## Common trap

Lines that only touch at a corner, or that are parallel but not in line, are different lines. Check with a ruler in your head before joining two pieces into one line.
