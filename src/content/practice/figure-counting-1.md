---
title: "Figure counting 1: triangles"
description: "Count every triangle in a figure: lines drawn from a corner, squares with diagonals, stars and triangles cut by inside lines."
slug: "figure-counting-1"
family: "figure-counting"
part: 1
timeMinutes: 12
questions:
  - id: fc1-fan-two
    topic: "Lines from a corner"
    difficulty: 1
    q: "Two lines are drawn from the top corner of a triangle down to its base. How many triangles are there in this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 180, "items": [{"t": "line", "p": [100, 12, 12, 168]}, {"t": "line", "p": [12, 168, 188, 168]}, {"t": "line", "p": [188, 168, 100, 12]}, {"t": "line", "p": [100, 12, 72, 168]}, {"t": "line", "p": [100, 12, 130, 168]}], "alt": "A triangle with two lines drawn from the top corner down to the base"}
    options: ["3", "4", "6", "8"]
    answer: C
    explanation: |
      The base now has 4 points on it (2 corners and 2 feet of the lines). Every triangle uses the top corner and any 2 of these 4 points: 4 × 3 ÷ 2 = **6**.
      3 counts only the smallest triangles, and 4 adds just the whole triangle.

  - id: fc1-square-diagonals
    topic: "Squares with diagonals"
    difficulty: 1
    q: "A square has both its diagonals drawn. How many triangles are there in this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 200, "items": [{"t": "line", "p": [20, 20, 180, 20]}, {"t": "line", "p": [180, 20, 180, 180]}, {"t": "line", "p": [180, 180, 20, 180]}, {"t": "line", "p": [20, 180, 20, 20]}, {"t": "line", "p": [20, 20, 180, 180]}, {"t": "line", "p": [180, 20, 20, 180]}], "alt": "A square with both diagonals drawn"}
    options: ["4", "6", "8", "10"]
    answer: C
    explanation: |
      There are 4 small triangles meeting at the centre. Each diagonal also cuts the square into 2 big triangles, so there are 4 big ones. 4 + 4 = **8**.
      4 counts only the small triangles.

  - id: fc1-middle-triangle
    topic: "Lines inside a triangle"
    difficulty: 1
    q: "The midpoints of the three sides of a triangle are joined. How many triangles are there in this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 180, "items": [{"t": "line", "p": [100, 12, 12, 168]}, {"t": "line", "p": [12, 168, 188, 168]}, {"t": "line", "p": [188, 168, 100, 12]}, {"t": "line", "p": [56, 90, 144, 90]}, {"t": "line", "p": [56, 90, 100, 168]}, {"t": "line", "p": [144, 90, 100, 168]}], "alt": "A triangle with the midpoints of its sides joined, making four small triangles"}
    options: ["5", "6", "7", "8"]
    answer: A
    explanation: |
      There are 3 corner triangles, 1 upside-down triangle in the middle and the whole triangle: 3 + 1 + 1 = **5**.
      The middle triangle together with a corner triangle makes a four-sided shape, not a triangle, so 6, 7 and 8 count shapes that are not triangles.

  - id: fc1-fan-three
    topic: "Lines from a corner"
    difficulty: 1
    q: "Three lines are drawn from the top corner of a triangle down to its base. How many triangles are there in this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 180, "items": [{"t": "line", "p": [100, 12, 12, 168]}, {"t": "line", "p": [12, 168, 188, 168]}, {"t": "line", "p": [188, 168, 100, 12]}, {"t": "line", "p": [100, 12, 56, 168]}, {"t": "line", "p": [100, 12, 100, 168]}, {"t": "line", "p": [100, 12, 144, 168]}], "alt": "A triangle with three lines drawn from the top corner down to the base"}
    options: ["4", "7", "9", "10"]
    answer: D
    explanation: |
      The base has 5 points on it. Any 2 of them, joined to the top corner, make a triangle: 5 × 4 ÷ 2 = **10**.
      By size: 4 small + 3 made of two + 2 made of three + 1 whole = 10. 9 forgets the whole triangle.

  - id: fc1-fan-and-cross-line
    topic: "Lines from a corner"
    difficulty: 2
    q: "A triangle has two lines from its top corner to the base and one line across it, parallel to the base. How many triangles are there in this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 180, "items": [{"t": "line", "p": [100, 12, 12, 168]}, {"t": "line", "p": [12, 168, 188, 168]}, {"t": "line", "p": [188, 168, 100, 12]}, {"t": "line", "p": [100, 12, 72, 168]}, {"t": "line", "p": [100, 12, 130, 168]}, {"t": "line", "p": [56, 90, 144, 90]}], "alt": "A triangle with two lines from the top corner to the base and one line across the middle, parallel to the base"}
    options: ["6", "9", "11", "12"]
    answer: D
    explanation: |
      Every triangle has the top corner as its peak. Its bottom side lies either on the middle line or on the base.
      On the middle line there are 4 points, giving 4 × 3 ÷ 2 = 6 triangles. On the base there are also 4 points, giving 6 more: 6 + 6 = **12**.
      The pieces in the bottom strip have four sides, so they are not triangles. 6 counts only one of the two layers.

  - id: fc1-star
    topic: "Mixed figures"
    difficulty: 2
    q: "This star is made of two big triangles, one pointing up and one pointing down. How many triangles are there in this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 204, "items": [{"t": "line", "p": [100, 10, 180, 148]}, {"t": "line", "p": [180, 148, 20, 148]}, {"t": "line", "p": [20, 148, 100, 10]}, {"t": "line", "p": [100, 194, 20, 56]}, {"t": "line", "p": [20, 56, 180, 56]}, {"t": "line", "p": [180, 56, 100, 194]}], "alt": "A six-pointed star made of two overlapping triangles, one pointing up and one pointing down"}
    options: ["8", "10", "12", "14"]
    answer: A
    explanation: |
      There are 6 small triangles, one at each point of the star, and the 2 big triangles that make the star: 6 + 2 = **8**.
      The middle piece has six sides. Joining it to small triangles gives shapes with four or more sides, so 10, 12 and 14 count shapes that are not triangles.

  - id: fc1-house
    topic: "Mixed figures"
    difficulty: 2
    q: "This house is a square with both diagonals drawn and a triangular roof on top. How many triangles are there in this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 226, "items": [{"t": "line", "p": [30, 76, 100, 12]}, {"t": "line", "p": [100, 12, 170, 76]}, {"t": "line", "p": [30, 76, 170, 76]}, {"t": "line", "p": [170, 76, 170, 216]}, {"t": "line", "p": [170, 216, 30, 216]}, {"t": "line", "p": [30, 216, 30, 76]}, {"t": "line", "p": [30, 76, 170, 216]}, {"t": "line", "p": [170, 76, 30, 216]}], "alt": "A house shape: a triangular roof on top of a square, and the square has both diagonals drawn"}
    options: ["9", "10", "11", "13"]
    answer: A
    explanation: |
      The square with both diagonals has 8 triangles (4 small and 4 halves of the square). The roof adds 1 more: 8 + 1 = **9**.
      The roof together with the top small triangle makes a four-sided shape (like a kite), so it does not count. 10 counts that shape.

  - id: fc1-middle-triangle-and-line
    topic: "Lines inside a triangle"
    difficulty: 2
    q: "The midpoints of the sides of a triangle are joined, and a line is drawn from the top corner to the middle of the base. How many triangles are there in this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 180, "items": [{"t": "line", "p": [100, 12, 12, 168]}, {"t": "line", "p": [12, 168, 188, 168]}, {"t": "line", "p": [188, 168, 100, 12]}, {"t": "line", "p": [56, 90, 144, 90]}, {"t": "line", "p": [56, 90, 100, 168]}, {"t": "line", "p": [144, 90, 100, 168]}, {"t": "line", "p": [100, 12, 100, 168]}], "alt": "A triangle with the midpoints of its sides joined, and a line from the top corner straight down to the middle of the base"}
    options: ["11", "13", "14", "16"]
    answer: B
    explanation: |
      Count by size. Smallest: the line cuts the top triangle and the middle triangle in half, so 4 halves.
      The size of a corner triangle: the 4 small triangles, plus 2 more made of one half of the top triangle and one half of the middle triangle (left and right), so 6.
      Then the 2 halves of the whole triangle, and the whole triangle: 4 + 6 + 2 + 1 = **13**.

  - id: fc1-square-diagonals-midlines
    topic: "Squares with diagonals"
    difficulty: 2
    q: "A square has both diagonals, and two lines joining the midpoints of opposite sides. How many triangles are there in this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 200, "items": [{"t": "line", "p": [20, 20, 180, 20]}, {"t": "line", "p": [180, 20, 180, 180]}, {"t": "line", "p": [180, 180, 20, 180]}, {"t": "line", "p": [20, 180, 20, 20]}, {"t": "line", "p": [20, 20, 180, 180]}, {"t": "line", "p": [180, 20, 20, 180]}, {"t": "line", "p": [100, 20, 100, 180]}, {"t": "line", "p": [20, 100, 180, 100]}], "alt": "A square with both diagonals and both lines joining the midpoints of opposite sides"}
    options: ["8", "12", "14", "16"]
    answer: D
    explanation: |
      Count by size. There are 8 smallest triangles around the centre.
      Each quarter cut off by the two diagonals is made of 2 smallest triangles: 4 more. Each diagonal cuts the square into 2 halves: 4 more.
      8 + 4 + 4 = **16**. 12 forgets the halves (or the quarters).

  - id: fc1-fan-and-corner-line
    topic: "Lines from a corner"
    difficulty: 3
    q: "A triangle has two lines from its top corner to the base, and one slanting line from its bottom left corner to the middle of the right side. How many triangles are there in this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 180, "items": [{"t": "line", "p": [100, 12, 12, 168]}, {"t": "line", "p": [12, 168, 188, 168]}, {"t": "line", "p": [188, 168, 100, 12]}, {"t": "line", "p": [100, 12, 72, 168]}, {"t": "line", "p": [100, 12, 130, 168]}, {"t": "line", "p": [12, 168, 144, 90]}], "alt": "A triangle with two lines from the top corner to the base, and one line from the bottom left corner to the middle of the right side"}
    options: ["9", "12", "15", "16"]
    answer: C
    explanation: |
      Triangles that do not use the slanting line: the top corner with 2 of the 4 base points, 4 × 3 ÷ 2 = 6.
      Triangles that use it: 3 have the bottom left corner and lie along the base, 3 have the bottom left corner and lie along the left side, and 3 have the top corner as peak and their bottom on the slanting line. That is 9.
      6 + 9 = **15**.

  - id: fc1-two-squares
    topic: "Squares with diagonals"
    difficulty: 3
    q: "A rectangle is made of two squares side by side, and each square has both its diagonals drawn. How many triangles are there in this figure?"
    figure: {"kind": "shapes", "w": 220, "h": 120, "items": [{"t": "line", "p": [10, 10, 210, 10]}, {"t": "line", "p": [210, 10, 210, 110]}, {"t": "line", "p": [210, 110, 10, 110]}, {"t": "line", "p": [10, 110, 10, 10]}, {"t": "line", "p": [110, 10, 110, 110]}, {"t": "line", "p": [10, 10, 110, 110]}, {"t": "line", "p": [110, 10, 10, 110]}, {"t": "line", "p": [110, 10, 210, 110]}, {"t": "line", "p": [210, 10, 110, 110]}], "alt": "A rectangle made of two squares side by side; each square has both its diagonals drawn"}
    options: ["16", "18", "20", "24"]
    answer: B
    explanation: |
      Each square with both diagonals has 8 triangles, so 16 for the two squares.
      There are also 2 big triangles that use both squares: the top middle point with the two bottom corners, and the bottom middle point with the two top corners.
      16 + 2 = **18**. 16 misses the two big triangles that cross the middle line.

  - id: fc1-square-diagonals-diamond
    topic: "Squares with diagonals"
    difficulty: 3
    q: "A square has both diagonals, and a tilted square is drawn by joining the midpoints of its sides. How many triangles are there in this figure?"
    figure: {"kind": "shapes", "w": 200, "h": 200, "items": [{"t": "line", "p": [20, 20, 180, 20]}, {"t": "line", "p": [180, 20, 180, 180]}, {"t": "line", "p": [180, 180, 20, 180]}, {"t": "line", "p": [20, 180, 20, 20]}, {"t": "line", "p": [20, 20, 180, 180]}, {"t": "line", "p": [180, 20, 20, 180]}, {"t": "line", "p": [100, 20, 180, 100]}, {"t": "line", "p": [180, 100, 100, 180]}, {"t": "line", "p": [100, 180, 20, 100]}, {"t": "line", "p": [20, 100, 100, 20]}], "alt": "A square with both diagonals, and a tilted square (diamond) joining the midpoints of its sides"}
    options: ["16", "20", "22", "24"]
    answer: B
    explanation: |
      Count by size. Each corner triangle outside the tilted square is cut in two by a diagonal: 8 smallest. The 4 corner triangles themselves: 4.
      The two diagonals cut the square into 4 quarters: 4. Each diagonal cuts the square into 2 halves: 4.
      8 + 4 + 4 + 4 = **20**. The pieces inside the tilted square are small squares, not triangles.

---

This set tests whether you can count every triangle in a figure, including the big ones made of smaller pieces.

## Count in an order

Never count by just looking around. Pick an order and stick to it:
- **By size.** Count the smallest triangles first, then the ones made of 2 pieces, then 3, and so on, up to the whole figure.
- **By corner.** Pick one point, count all the triangles that have it as a corner, then move to the next point and do not count those triangles again.

## The fan rule

If lines are drawn from one corner of a triangle to the opposite side, that side gets some points on it (its two ends plus the feet of the lines). Every pair of these points, joined to the corner, gives one triangle.

So if the base has n points, count 1 + 2 + … + (n − 1). For example, 4 lines from the corner give 6 points on the base and 5 + 4 + 3 + 2 + 1 = 15 triangles.

## Worked example

A triangle has one line from its top corner to the base and one line across it, parallel to the base. Each triangle has the top corner as its peak. Its bottom lies on the cross line (3 points, so 3 triangles) or on the base (3 more). Total 6.

## Common trap

Check that every shape you count has **exactly three straight sides** made of drawn lines. Two triangles side by side often make a four-sided shape, and that does not count.
