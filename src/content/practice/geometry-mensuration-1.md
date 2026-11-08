---
title: "Geometry and mensuration 1: lines and angles"
description: "Complementary, supplementary and vertically opposite angles, angles on a straight line and round a point, and parallel lines cut by a transversal."
slug: geometry-mensuration-1
family: geometry-mensuration
part: 1
timeMinutes: 12
questions:
  - id: gm1-complement
    topic: Angle pairs
    difficulty: 1
    q: "What is the complement of an angle of 38°?"
    options: ["42°", "52°", "62°", "142°"]
    answer: B
    explanation: |
      Complementary angles add up to 90°, so the complement is 90 − 38 = **52°**.
      142° is the supplement (180 − 38). Supplementary angles add up to 180°.

  - id: gm1-linear-pair
    topic: Angle pairs
    difficulty: 1
    q: "Two straight lines cross each other. Find x."
    figure:
      kind: shapes
      w: 270
      h: 162
      alt: "Two straight lines crossing. One angle is 65°. The angle next to it, on the same straight line, is x."
      items:
        - { t: line, p: [16.2, 81, 253.8, 81] }
        - { t: line, p: [101.9, 152, 168.1, 10] }
        - { t: poly, p: [[144, 81], [143.9, 79.7], [143.6, 78.5], [143.2, 77.3], [142.6, 76.2], [141.8, 75.1], [140.9, 74.2], [139.9, 73.5], [138.8, 72.8]], open: true }
        - { t: text, at: [159.9, 65.1], s: "65°", size: 13 }
        - { t: poly, p: [[143.5, 62.9], [140.8, 61.9], [138, 61.2], [135.1, 61], [132.3, 61.2], [129.5, 61.8], [126.8, 62.8], [124.3, 64.1], [122, 65.8], [119.9, 67.9], [118.2, 70.1], [116.8, 72.7], [115.8, 75.3], [115.2, 78.1], [115, 81]], open: true }
        - { t: text, at: [118.6, 55.3], s: "x", size: 15 }
    options: ["25°", "65°", "115°", "125°"]
    answer: C
    explanation: |
      x and 65° sit side by side on a straight line, so together they make 180°.
      x = 180 − 65 = **115°**. 65° is the angle vertically opposite the marked 65°, not x.

  - id: gm1-straight-line
    topic: Straight line and point
    difficulty: 1
    q: "AOB is a straight line. Find x."
    figure:
      kind: shapes
      w: 270
      h: 148.5
      alt: "A straight line AOB with two rays from O. The three angles along the line, from B round to A, are 40°, x and 75°."
      items:
        - { t: line, p: [16.2, 118.8, 253.8, 118.8] }
        - { t: line, p: [135, 118.8, 209.5, 56.3] }
        - { t: line, p: [135, 118.8, 109.8, 24.9] }
        - { t: poly, p: [[144, 118.8], [143.9, 117.5], [143.7, 116.3], [143.2, 115.1], [142.6, 114], [141.9, 113]], open: true }
        - { t: text, at: [174.5, 104.4], s: "40°", size: 13 }
        - { t: poly, p: [[151.9, 104.7], [149.7, 102.4], [147.2, 100.5], [144.5, 99], [141.6, 97.8], [138.6, 97.1], [135.5, 96.8], [132.4, 97], [129.3, 97.5]], open: true }
        - { t: text, at: [144.5, 88.8], s: "x", size: 15 }
        - { t: poly, p: [[131.1, 104.3], [129.1, 105], [127.1, 106], [125.4, 107.3], [123.8, 108.8], [122.5, 110.6], [121.4, 112.5], [120.6, 114.5], [120.2, 116.6], [120, 118.8]], open: true }
        - { t: text, at: [111.2, 100.5], s: "75°", size: 13 }
        - { t: text, at: [16.2, 127.8], s: "A", size: 14 }
        - { t: text, at: [253.8, 127.8], s: "B", size: 14 }
        - { t: text, at: [135, 127.8], s: "O", size: 14 }
    options: ["65°", "105°", "115°", "245°"]
    answer: A
    explanation: |
      Angles on a straight line add up to 180°. x = 180 − 40 − 75 = **65°**.
      245° uses 360°, which is for angles all the way round a point.

  - id: gm1-alternate
    topic: Parallel lines
    difficulty: 1
    q: "Lines l and m are parallel (l ∥ m). Find x."
    figure:
      kind: shapes
      w: 270
      h: 175.5
      alt: "Two parallel lines l and m, marked with arrows, cut by a slanting line. Between the parallel lines, the angle at the lower crossing on the right of the slanting line is 72°. The angle at the upper crossing on the left is x."
      items:
        - { t: line, p: [13.5, 47.2, 243, 47.2] }
        - { t: line, p: [13.5, 128.2, 243, 128.2] }
        - { t: line, p: [99.9, 161.6, 147.9, 13.9] }
        - { t: arrow, p: [198.7, 47.2, 206.7, 47.2] }
        - { t: arrow, p: [198.7, 128.2, 206.7, 128.2] }
        - { t: text, at: [256.5, 47.2], s: "l", size: 13 }
        - { t: text, at: [256.5, 128.2], s: "m", size: 13 }
        - { t: poly, p: [[120.7, 128.2], [120.6, 126.9], [120.3, 125.5], [119.8, 124.2], [119.2, 123], [118.4, 121.8], [117.4, 120.8], [116.3, 120], [115.1, 119.3], [113.8, 118.7]], open: true }
        - { t: text, at: [132.5, 112.4], s: "72°", size: 13 }
        - { t: poly, p: [[127, 47.2], [127.1, 48.6], [127.4, 50], [127.9, 51.3], [128.5, 52.5], [129.4, 53.7], [130.3, 54.7], [131.4, 55.5], [132.6, 56.2], [133.9, 56.8]], open: true }
        - { t: text, at: [120.8, 59], s: "x", size: 15 }
    options: ["18°", "72°", "108°", "144°"]
    answer: B
    explanation: |
      x and 72° are **alternate angles**: both lie between the parallel lines, on opposite sides of the slanting line (a Z shape). Alternate angles are equal, so x = **72°**.
      108° would be right for co-interior angles, which are on the same side.

  - id: gm1-supp-twice
    topic: Angle pairs
    difficulty: 2
    q: "Two angles are supplementary. One of them is 30° more than twice the other. What is the smaller angle?"
    options: ["50°", "60°", "75°", "130°"]
    answer: A
    explanation: |
      Call the smaller angle a. Then a + (2a + 30) = 180, so 3a = 150 and a = **50°**.
      The other angle is 130°, the larger one. 60° forgets the extra 30°.

  - id: gm1-less-than-complement
    topic: Angle pairs
    difficulty: 2
    q: "An angle is 18° less than its complement. What is the angle?"
    options: ["36°", "54°", "72°", "81°"]
    answer: A
    explanation: |
      If the angle is a, its complement is 90 − a. So a = 90 − a − 18, which gives 2a = 72 and a = **36°**.
      Check: the complement is 54°, and 54 − 18 = 36 ✓. 54° is the complement itself.

  - id: gm1-round-point
    topic: Straight line and point
    difficulty: 2
    q: "Four rays start from the same point. The angles between them are marked in degrees. What is the size of the largest of the four angles?"
    figure:
      kind: shapes
      w: 270
      h: 175.5
      alt: "Four rays from a point O. Going round, the angles between them are 110°, 2x, 3x and x + 10, all in degrees."
      items:
        - { t: line, p: [135, 89.1, 216, 89.1] }
        - { t: line, p: [135, 89.1, 107.3, 13] }
        - { t: line, p: [135, 89.1, 55.2, 103.2] }
        - { t: line, p: [135, 89.1, 187.1, 151.1] }
        - { t: text, at: [148.8, 69.4], s: "110°", size: 13 }
        - { t: text, at: [115.9, 78.1], s: "2x", size: 13 }
        - { t: text, at: [127.5, 109.8], s: "3x", size: 13 }
        - { t: text, at: [186.7, 113.2], s: "x + 10", size: 13 }
    options: ["40°", "80°", "110°", "120°"]
    answer: D
    explanation: |
      Angles round a point add up to 360°: 110 + 2x + 3x + (x + 10) = 360, so 6x = 240 and x = 40.
      The angles are 110°, 80°, 120° and 50°, so the largest is 3x = **120°**. 110° only looks the largest until you work out x.

  - id: gm1-vert-opp-algebra
    topic: Angle pairs
    difficulty: 2
    q: "Two straight lines cross. The two marked angles are vertically opposite. What is the size of each marked angle?"
    figure:
      kind: shapes
      w: 270
      h: 175.5
      alt: "Two straight lines crossing. The top angle is (3x − 10)° and the bottom angle, opposite it, is (2x + 25)°."
      items:
        - { t: line, p: [206.7, 22.1, 63.3, 153.4] }
        - { t: line, p: [63.3, 22.1, 206.7, 153.4] }
        - { t: poly, p: [[142.4, 81], [141.3, 80], [140, 79.1], [138.7, 78.5], [137.2, 78], [135.8, 77.8], [134.2, 77.8], [132.8, 78], [131.3, 78.5], [130, 79.1], [128.7, 80], [127.6, 81]], open: true }
        - { t: text, at: [135, 45.8], s: "(3x − 10)°", size: 12 }
        - { t: poly, p: [[127.6, 94.5], [128.7, 95.5], [130, 96.4], [131.3, 97], [132.8, 97.5], [134.2, 97.7], [135.8, 97.7], [137.2, 97.5], [138.7, 97], [140, 96.4], [141.3, 95.5], [142.4, 94.5]], open: true }
        - { t: text, at: [135, 129.8], s: "(2x + 25)°", size: 12 }
    options: ["35°", "85°", "95°", "145°"]
    answer: C
    explanation: |
      Vertically opposite angles are equal: 3x − 10 = 2x + 25, so x = 35.
      Each angle is 3 × 35 − 10 = **95°**. 35° is x, not the angle, and 85° is the angle next to it.

  - id: gm1-co-interior
    topic: Parallel lines
    difficulty: 2
    q: "l ∥ m. Find the size of the angle marked (2x + 20)°."
    figure:
      kind: shapes
      w: 270
      h: 175.5
      alt: "Two parallel lines l and m cut by a slanting line. On the right of the slanting line, between the parallel lines, the angle at the upper crossing is (2x + 20)° and the angle at the lower crossing is (x + 40)°."
      items:
        - { t: line, p: [13.5, 24.3, 243, 24.3] }
        - { t: line, p: [13.5, 151.2, 243, 151.2] }
        - { t: line, p: [88.9, 183.1, 122.5, -7.6] }
        - { t: arrow, p: [38, 24.3, 46, 24.3] }
        - { t: arrow, p: [38, 151.2, 46, 151.2] }
        - { t: text, at: [256.5, 24.3], s: "l", size: 13 }
        - { t: text, at: [256.5, 151.2], s: "m", size: 13 }
        - { t: poly, p: [[104.5, 151.2], [104.4, 149.8], [104.1, 148.4], [103.6, 147.1], [103, 145.9], [102.2, 144.8], [101.2, 143.8], [100.1, 142.9], [98.9, 142.2], [97.6, 141.7], [96.2, 141.4]], open: true }
        - { t: text, at: [134.2, 133.5], s: "(x + 40)°", size: 12 }
        - { t: poly, p: [[115.1, 34.1], [116.6, 34.3], [118, 34.2], [119.5, 34], [120.8, 33.5], [122.1, 32.8], [123.3, 32], [124.3, 30.9], [125.2, 29.8], [125.9, 28.5], [126.5, 27.2], [126.8, 25.7], [126.9, 24.3]], open: true }
        - { t: text, at: [153.5, 43.8], s: "(2x + 20)°", size: 12 }
    options: ["40°", "60°", "80°", "100°"]
    answer: D
    explanation: |
      The two marked angles are between the parallel lines, on the same side of the slanting line. These **co-interior** angles add up to 180°: (2x + 20) + (x + 40) = 180, so 3x = 120 and x = 40.
      The angle is 2 × 40 + 20 = **100°**. 60° comes from making the two angles equal, which is only true for alternate or corresponding angles.

  - id: gm1-bent-angle
    topic: Parallel lines
    difficulty: 3
    q: "AB ∥ CD. Find angle BED."
    figure:
      kind: shapes
      w: 270
      h: 175.5
      alt: "AB parallel to CD, AB above. Point E lies between the lines, to the left of B and D. Angle ABE is 40°, angle CDE is 35°, and angle BED is marked with a question mark."
      items:
        - { t: line, p: [16.2, 29.7, 253.8, 29.7] }
        - { t: line, p: [16.2, 151.2, 253.8, 151.2] }
        - { t: line, p: [205.2, 29.7, 126.3, 95.9] }
        - { t: line, p: [126.3, 95.9, 205.2, 151.2] }
        - { t: arrow, p: [51.2, 29.7, 59.2, 29.7] }
        - { t: arrow, p: [51.2, 151.2, 59.2, 151.2] }
        - { t: text, at: [16.2, 20.7], s: "A", size: 14 }
        - { t: text, at: [205.2, 20.7], s: "B", size: 14 }
        - { t: text, at: [16.2, 160.2], s: "C", size: 14 }
        - { t: text, at: [205.2, 160.2], s: "D", size: 14 }
        - { t: text, at: [117.3, 95.9], s: "E", size: 14 }
        - { t: poly, p: [[195.2, 29.7], [195.3, 31.1], [195.6, 32.5], [196.1, 33.8], [196.7, 35], [197.5, 36.1]], open: true }
        - { t: text, at: [165.7, 44.1], s: "40°", size: 13 }
        - { t: poly, p: [[197, 145.5], [196.2, 146.8], [195.7, 148.2], [195.3, 149.7], [195.2, 151.2]], open: true }
        - { t: text, at: [161.8, 137.5], s: "35°", size: 13 }
        - { t: poly, p: [[134.5, 101.7], [135.2, 100.4], [135.8, 99.1], [136.1, 97.7], [136.3, 96.2], [136.2, 94.8], [135.9, 93.3], [135.5, 92], [134.8, 90.7], [133.9, 89.5]], open: true }
        - { t: text, at: [145.3, 95.1], s: "?", size: 15 }
    options: ["5°", "75°", "105°", "140°"]
    answer: B
    explanation: |
      Draw a line through E parallel to AB. It splits angle BED into two parts. The top part equals angle ABE = 40° and the bottom part equals angle CDE = 35° (alternate angles).
      So angle BED = 40 + 35 = **75°**. 105° is 180 − 75, which mixes up alternate and co-interior angles.

  - id: gm1-bisectors
    topic: Parallel lines
    difficulty: 3
    q: "l ∥ m. The slanting line makes co-interior angles of 70° at Q and 110° at R. The dashed lines bisect these two angles and meet at P. Find angle QPR."
    figure:
      kind: shapes
      w: 270
      h: 175.5
      alt: "Parallel lines l (top) and m (bottom) cut by a slanting line at R (on l) and Q (on m). The angle at Q on the right is 70° and the angle at R on the right is 110°. Dashed bisectors of these angles meet at P, to the right."
      items:
        - { t: line, p: [13.5, 35.1, 245.7, 35.1] }
        - { t: line, p: [13.5, 140.4, 245.7, 140.4] }
        - { t: line, p: [69.1, 165.8, 125.9, 9.7] }
        - { t: line, p: [78.3, 140.4, 153.5, 87.7], dash: true }
        - { t: line, p: [116.6, 35.1, 153.5, 87.7], dash: true }
        - { t: arrow, p: [224.1, 35.1, 232.1, 35.1] }
        - { t: arrow, p: [224.1, 140.4, 232.1, 140.4] }
        - { t: text, at: [259.2, 35.1], s: "l", size: 13 }
        - { t: text, at: [259.2, 140.4], s: "m", size: 13 }
        - { t: text, at: [162.5, 87.7], s: "P", size: 14 }
        - { t: poly, p: [[102.3, 140.4], [102, 136.7], [101.2, 133.2], [99.8, 129.8], [98, 126.6], [95.6, 123.8], [92.9, 121.4], [89.8, 119.4], [86.5, 117.8]], open: true }
        - { t: text, at: [116, 132.5], s: "70°", size: 12 }
        - { t: text, at: [66.8, 148.4], s: "Q", size: 14 }
        - { t: poly, p: [[108.4, 57.7], [111.8, 58.6], [115.3, 59.1], [118.9, 59], [122.4, 58.4], [125.7, 57.3], [128.9, 55.7], [131.8, 53.7], [134.4, 51.3], [136.6, 48.5], [138.3, 45.4], [139.6, 42.1], [140.4, 38.6], [140.6, 35.1]], open: true }
        - { t: text, at: [157.6, 46.4], s: "110°", size: 12 }
        - { t: text, at: [97.4, 28.1], s: "R", size: 14 }
    options: ["35°", "55°", "70°", "90°"]
    answer: D
    explanation: |
      The bisectors cut the angles into 35° at Q and 55° at R. In triangle PQR, angle P = 180 − 35 − 55 = **90°**.
      This always happens: co-interior angles add up to 180°, so their halves add up to 90°, leaving 90° at P.

  - id: gm1-supp-4-comp
    topic: Angle pairs
    difficulty: 3
    q: "The supplement of an angle is 4 times its complement. What is the angle?"
    options: ["30°", "45°", "60°", "72°"]
    answer: C
    explanation: |
      Let the angle be a. Then 180 − a = 4(90 − a), so 180 − a = 360 − 4a, 3a = 180 and a = **60°**.
      Check: supplement 120°, complement 30°, and 120 = 4 × 30 ✓. 30° is the complement, not the angle.
---
This set tests the angle facts you use in almost every geometry question: angle pairs, angles at a point, and the angles made when a line cuts two parallel lines.

## The rules

| Angles | Add up to or are |
|---|---|
| complementary pair | 90° |
| supplementary pair | 180° |
| on a straight line, at one point | 180° |
| all the way round a point | 360° |
| vertically opposite (where two lines cross) | equal |

When a slanting line (a **transversal**) cuts two parallel lines:

- **corresponding** angles (an F shape) are equal;
- **alternate** angles (a Z shape, between the lines, on opposite sides) are equal;
- **co-interior** angles (a C or U shape, between the lines, on the same side) add up to 180°.

Arrow marks on two lines, or the sign ∥, tell you the lines are parallel.

## Worked example

*Two angles are complementary and one is 4 times the other. Find them.*

Call the smaller angle a. Then a + 4a = 90, so 5a = 90 and a = 18°. The angles are **18° and 72°**. Check: 18 + 72 = 90 ✓.

When an angle is written as something like (3x + 5)°, first use a rule to make an equation, then solve for x, then put x back in. The question may ask for x or for the angle itself, so read the last line again before you choose.

## Common traps

- Mixing up complement (90°) and supplement (180°).
- Treating co-interior angles as equal. They are equal only when both are 90°.
- A "bent" line between two parallel lines: draw a third parallel line through the bend. It splits the bend angle into two alternate-angle parts.
