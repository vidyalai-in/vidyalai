---
title: "Geometry and mensuration 2: triangles and polygons"
description: "Angle sum and exterior angle of a triangle, isosceles and equilateral triangles, Pythagoras, regular polygons, and which three lengths can make a triangle."
slug: geometry-mensuration-2
family: geometry-mensuration
part: 2
timeMinutes: 12
questions:
  - id: gm2-angle-sum
    topic: Angle sum
    difficulty: 1
    q: "Find x."
    figure:
      kind: shapes
      w: 243
      h: 178.2
      alt: "Triangle ABC with angle B = 52°, angle C = 71° and angle A = x."
      items:
        - { t: poly, p: [[149.7, 30.7], [51.3, 156.6], [193.1, 156.6]] }
        - { t: text, at: [149.7, 21.7], s: "A", size: 14 }
        - { t: text, at: [43.5, 161.1], s: "B", size: 14 }
        - { t: text, at: [200.8, 161.1], s: "C", size: 14 }
        - { t: poly, p: [[61.3, 156.6], [61.2, 155.1], [60.8, 153.6], [60.3, 152.2], [59.5, 150.9], [58.6, 149.7], [57.5, 148.7]], open: true }
        - { t: text, at: [82.8, 141.3], s: "52°", size: 13 }
        - { t: poly, p: [[189.8, 147.1], [188.4, 147.8], [187.1, 148.6], [185.9, 149.6], [184.9, 150.8], [184.1, 152.1], [183.5, 153.6], [183.2, 155.1], [183.1, 156.6]], open: true }
        - { t: text, at: [171.1, 140.9], s: "71°", size: 13 }
        - { t: poly, p: [[143.5, 38.5], [144.7, 39.3], [146, 40], [147.3, 40.4], [148.7, 40.6], [150.2, 40.7], [151.6, 40.5], [152.9, 40.1]], open: true }
        - { t: text, at: [146.1, 51.9], s: "x", size: 15 }
    options: ["47°", "57°", "67°", "123°"]
    answer: B
    explanation: |
      The three angles of a triangle add up to 180°. x = 180 − 52 − 71 = **57°**.
      123° is just 52 + 71.

  - id: gm2-exterior
    topic: Exterior angle
    difficulty: 1
    q: "Side BC of triangle ABC is extended to D. Find x."
    figure:
      kind: shapes
      w: 270
      h: 148.5
      alt: "Triangle ABC with BC extended to D. Angle B = 45°, angle A = 70°, and the exterior angle ACD is x."
      items:
        - { t: poly, p: [[121.8, 32.1], [29.7, 124.2], [164.7, 124.2]] }
        - { t: line, p: [164.7, 124.2, 256.5, 124.2] }
        - { t: text, at: [121.8, 23.1], s: "A", size: 14 }
        - { t: text, at: [29.7, 133.2], s: "B", size: 14 }
        - { t: text, at: [164.7, 133.2], s: "C", size: 14 }
        - { t: text, at: [256.5, 133.2], s: "D", size: 14 }
        - { t: poly, p: [[39.7, 124.2], [39.6, 122.6], [39.2, 121.1], [38.6, 119.7], [37.8, 118.3], [36.8, 117.1]], open: true }
        - { t: text, at: [65.3, 109.5], s: "45°", size: 13 }
        - { t: poly, p: [[114.7, 39.2], [115.9, 40.2], [117.2, 41], [118.6, 41.6], [120, 42], [121.5, 42.1], [123.1, 42], [124.6, 41.7], [126, 41.2]], open: true }
        - { t: text, at: [117, 59.2], s: "70°", size: 13 }
        - { t: poly, p: [[174.7, 124.2], [174.6, 122.8], [174.3, 121.4], [173.8, 120], [173.1, 118.8], [172.2, 117.6], [171.2, 116.6], [170.1, 115.8], [168.8, 115.1], [167.5, 114.6], [166.1, 114.3], [164.6, 114.2], [163.2, 114.3], [161.8, 114.6], [160.5, 115.1]], open: true }
        - { t: text, at: [175.7, 106.9], s: "x", size: 15 }
    options: ["45°", "65°", "70°", "115°"]
    answer: D
    explanation: |
      An exterior angle equals the sum of the two interior angles opposite it: x = 45 + 70 = **115°**.
      65° is the interior angle at C, which sits next to x on the straight line.

  - id: gm2-which-triangle
    topic: Triangle inequality
    difficulty: 1
    q: "Which three lengths can be the sides of a triangle?"
    options: ["2 cm, 9 cm, 11 cm", "3 cm, 4 cm, 8 cm", "5 cm, 6 cm, 12 cm", "4 cm, 7 cm, 10 cm"]
    answer: D
    explanation: |
      The two shorter sides must add up to **more** than the longest side. 4 + 7 = 11, which is more than 10, so **4 cm, 7 cm, 10 cm** works.
      2 + 9 is exactly 11, so those three sticks would lie flat in a straight line.

  - id: gm2-hypotenuse
    topic: Pythagoras
    difficulty: 1
    q: "The two shorter sides of a right-angled triangle are 9 cm and 12 cm. How long is the longest side?"
    options: ["15 cm", "17 cm", "21 cm", "225 cm"]
    answer: A
    explanation: |
      By Pythagoras, longest² = 9² + 12² = 81 + 144 = 225, so the longest side is √225 = **15 cm**.
      225 cm forgets the square root, and 21 cm just adds the two sides.

  - id: gm2-isosceles
    topic: Special triangles
    difficulty: 2
    q: "In triangle PQR, PQ = PR (the marks show the equal sides). Find x."
    figure:
      kind: shapes
      w: 202.5
      h: 175.5
      alt: "Isosceles triangle PQR with PQ = PR (marked). Angle P at the top is 40°. The base angle at Q is x."
      items:
        - { t: poly, p: [[101.3, 33.2], [55.4, 159.3], [147.2, 159.3]] }
        - { t: line, p: [73.6, 94.5, 83, 98], w: 1.8 }
        - { t: line, p: [119.5, 98, 128.9, 94.5], w: 1.8 }
        - { t: text, at: [101.3, 24.2], s: "P", size: 14 }
        - { t: text, at: [47.6, 163.8], s: "Q", size: 14 }
        - { t: text, at: [154.9, 163.8], s: "R", size: 14 }
        - { t: poly, p: [[97.8, 42.6], [99.2, 43], [100.6, 43.2], [101.9, 43.2], [103.3, 43], [104.7, 42.6]], open: true }
        - { t: text, at: [101.3, 79.2], s: "40°", size: 13 }
        - { t: poly, p: [[65.3, 159.3], [65.2, 157.8], [64.9, 156.3], [64.3, 154.9], [63.5, 153.6], [62.6, 152.4], [61.4, 151.4], [60.2, 150.5], [58.8, 149.9]], open: true }
        - { t: text, at: [71.7, 147.8], s: "x", size: 15 }
    options: ["40°", "70°", "110°", "140°"]
    answer: B
    explanation: |
      The angles opposite the equal sides are equal. The two base angles share 180 − 40 = 140°, so each is 140 ÷ 2 = **70°**.
      140° forgets to split the total between the two base angles.

  - id: gm2-polygon-sides
    topic: Polygons
    difficulty: 2
    q: "Each interior angle of a regular polygon is 160°. How many sides does it have?"
    options: ["9", "18", "20", "36"]
    answer: B
    explanation: |
      Each exterior angle is 180 − 160 = 20°. The exterior angles of any polygon add up to 360°, so it has 360 ÷ 20 = **18** sides.
      20 is the exterior angle, not the number of sides. 9 divides 180 instead of 360.

  - id: gm2-ladder
    topic: Pythagoras
    difficulty: 2
    q: "A 17 m long ladder leans against a wall. Its foot is 8 m from the wall. How high up the wall does it reach?"
    figure:
      kind: shapes
      w: 256.5
      h: 175.5
      alt: "A ladder 17 m long leaning against a wall. Its foot is 8 m from the wall on flat ground. The height it reaches up the wall is marked with a question mark."
      items:
        - { t: rect, p: [189, 10.8, 12.2, 140.4], fill: grey }
        - { t: line, p: [27, 151.2, 237.6, 151.2] }
        - { t: line, p: [122, 151.2, 189, 25.7], w: 3.5 }
        - { t: poly, p: [[181, 151.2], [181, 143.2], [189, 143.2]], open: true }
        - { t: text, at: [139.2, 79.7], s: "17 m", size: 13 }
        - { t: text, at: [155.5, 164.7], s: "8 m", size: 13 }
        - { t: text, at: [208.2, 88.4], s: "?", size: 14 }
    options: ["9 m", "12 m", "15 m", "25 m"]
    answer: C
    explanation: |
      The wall, the ground and the ladder make a right-angled triangle, with the ladder as the longest side.
      Height² = 17² − 8² = 289 − 64 = 225, so the height is **15 m**. 9 m is just 17 − 8.

  - id: gm2-exterior-algebra
    topic: Exterior angle
    difficulty: 2
    q: "Side BC of triangle ABC is extended to D. The angles are marked in degrees. What is the size of angle A?"
    figure:
      kind: shapes
      w: 270
      h: 164.7
      alt: "Triangle ABC with BC extended to D. Angle A is 2x, angle B is x + 15, and the exterior angle ACD is 120°, all in degrees."
      items:
        - { t: poly, p: [[112.2, 26], [16.2, 140.4], [178.2, 140.4]] }
        - { t: line, p: [178.2, 140.4, 259.2, 140.4] }
        - { t: text, at: [112.2, 17], s: "A", size: 14 }
        - { t: text, at: [16.2, 149.4], s: "B", size: 14 }
        - { t: text, at: [178.2, 149.4], s: "C", size: 14 }
        - { t: text, at: [259.2, 149.4], s: "D", size: 14 }
        - { t: poly, p: [[26.2, 140.4], [26.1, 139], [25.8, 137.5], [25.3, 136.2], [24.6, 134.9], [23.7, 133.8], [22.6, 132.7]], open: true }
        - { t: text, at: [64.7, 117.8], s: "x + 15", size: 12 }
        - { t: poly, p: [[105.7, 33.7], [107, 34.6], [108.3, 35.3], [109.8, 35.7], [111.3, 36], [112.8, 36], [114.3, 35.8], [115.8, 35.3], [117.2, 34.7]], open: true }
        - { t: text, at: [110.4, 46.5], s: "2x", size: 12 }
        - { t: poly, p: [[188.2, 140.4], [188.1, 139], [187.8, 137.6], [187.3, 136.3], [186.7, 135.1], [185.9, 134], [184.9, 133], [183.8, 132.1], [182.6, 131.4], [181.3, 130.9], [179.9, 130.6], [178.5, 130.4], [177.2, 130.5], [175.8, 130.7], [174.5, 131.1], [173.2, 131.7]], open: true }
        - { t: text, at: [188.5, 122.6], s: "120°", size: 12 }
    options: ["35°", "50°", "60°", "70°"]
    answer: D
    explanation: |
      Exterior angle = sum of the two opposite interior angles: 2x + (x + 15) = 120, so 3x = 105 and x = 35.
      Angle A = 2 × 35 = **70°**. 35° is x, 50° is angle B, and 60° is the interior angle at C.

  - id: gm2-equilateral
    topic: Special triangles
    difficulty: 2
    q: "ABC is an equilateral triangle. BC is extended to D so that CD = CA. Find angle BAD."
    figure:
      kind: shapes
      w: 270
      h: 148.5
      alt: "Equilateral triangle ABC with BC extended to D so that CD equals CA. A is joined to D. All four equal lengths AB, BC, CA and CD are marked."
      items:
        - { t: poly, p: [[82.4, 35.7], [29.7, 126.9], [135, 126.9]] }
        - { t: line, p: [135, 126.9, 240.3, 126.9] }
        - { t: line, p: [82.4, 35.7, 240.3, 126.9] }
        - { t: line, p: [51.7, 78.8, 60.4, 83.8], w: 1.8 }
        - { t: line, p: [82.3, 131.9, 82.3, 121.9], w: 1.8 }
        - { t: line, p: [113, 78.8, 104.3, 83.8], w: 1.8 }
        - { t: line, p: [187.7, 131.9, 187.7, 121.9], w: 1.8 }
        - { t: text, at: [82.4, 26.7], s: "A", size: 14 }
        - { t: text, at: [29.7, 135.9], s: "B", size: 14 }
        - { t: text, at: [135, 135.9], s: "C", size: 14 }
        - { t: text, at: [240.3, 135.9], s: "D", size: 14 }
    options: ["30°", "60°", "90°", "120°"]
    answer: C
    explanation: |
      Angle ACB = 60°, so angle ACD = 180 − 60 = 120°. Triangle ACD is isosceles (CA = CD), so angles CAD and CDA are each (180 − 120) ÷ 2 = 30°.
      Angle BAD = 60 + 30 = **90°**. 30° is only angle CAD.

  - id: gm2-bisector-36
    topic: Special triangles
    difficulty: 3
    q: "In triangle ABC, AB = AC and angle A = 36°. BD bisects angle B and meets AC at D. Find angle BDC."
    figure:
      kind: shapes
      w: 202.5
      h: 183.6
      alt: "Isosceles triangle ABC with AB = AC (marked) and angle A = 36° at the top. BD bisects angle B and meets AC at D."
      items:
        - { t: poly, p: [[101.2, 35.9], [59.4, 164.7], [143.1, 164.7]] }
        - { t: line, p: [59.4, 164.7, 127.1, 115.5] }
        - { t: line, p: [75.6, 98.8, 85.1, 101.8], w: 1.8 }
        - { t: line, p: [117.4, 101.8, 126.9, 98.8], w: 1.8 }
        - { t: text, at: [101.2, 26.9], s: "A", size: 14 }
        - { t: text, at: [51.6, 169.2], s: "B", size: 14 }
        - { t: text, at: [150.9, 169.2], s: "C", size: 14 }
        - { t: text, at: [136.1, 115.5], s: "D", size: 14 }
        - { t: poly, p: [[98.2, 45.4], [99.7, 45.8], [101.2, 45.9], [102.8, 45.8], [104.3, 45.4]], open: true }
        - { t: text, at: [101.2, 83.9], s: "36°", size: 12 }
    options: ["72°", "90°", "108°", "144°"]
    answer: A
    explanation: |
      Base angles: B = C = (180 − 36) ÷ 2 = 72°, so angle ABD = 72 ÷ 2 = 36°.
      Angle BDC is an exterior angle of triangle ABD, so it is 36 + 36 = **72°**. 108° is angle ADB, on the other side of D.

  - id: gm2-polygon-ratio
    topic: Polygons
    difficulty: 3
    q: "In a regular polygon, each interior angle is 5 times each exterior angle. How many sides does the polygon have?"
    options: ["6", "10", "12", "30"]
    answer: C
    explanation: |
      An interior angle and its exterior angle add up to 180°. So 5e + e = 180 and e = 30°.
      Number of sides = 360 ÷ 30 = **12**. 10 comes from 5e = 180, which forgets to add the exterior angle itself.

  - id: gm2-third-side
    topic: Triangle inequality
    difficulty: 3
    q: "Two sides of a triangle are 6 cm and 11 cm. The third side is a whole number of centimetres. How many different lengths can it have?"
    options: ["11", "12", "13", "16"]
    answer: A
    explanation: |
      The third side must be more than 11 − 6 = 5 and less than 11 + 6 = 17. So it can be 6, 7, 8, … up to 16 cm: **11** lengths.
      13 also counts 5 and 17, but those would make a flat triangle. 16 is the longest side it can have, not the count.
---
This set tests the facts about triangles and polygons: angle sums, exterior angles, special triangles, Pythagoras and which lengths can make a triangle.

## The rules

- The three angles of a triangle add up to **180°**.
- An **exterior angle** (made by extending one side) equals the sum of the two interior angles opposite it.
- **Isosceles** triangle: two equal sides, and the angles opposite them are equal. **Equilateral** triangle: all sides equal and every angle is 60°.
- **Pythagoras**: in a right-angled triangle, longest side² = sum of the squares of the other two. Useful whole-number sets: 3, 4, 5; 5, 12, 13; 8, 15, 17; 7, 24, 25, and their multiples such as 6, 8, 10.
- **Triangle inequality**: any two sides together must be longer than the third. To test three lengths, check that the two shorter ones add up to more than the longest.

## Regular polygons

| Fact | Formula |
|---|---|
| sum of exterior angles | always 360° |
| each exterior angle (n sides) | 360 ÷ n |
| interior + exterior | 180° |
| sum of interior angles | (n − 2) × 180° |

## Worked example

*Each exterior angle of a regular polygon is 40°. Find its interior angle and its number of sides.*

Sides = 360 ÷ 40 = 9. Interior angle = 180 − 40 = **140°**. Check: 9 × 140 = 1260 = (9 − 2) × 180 ✓.

## Common traps

- In an isosceles triangle, decide first which angle is the odd one out (the angle between the equal sides).
- For Pythagoras, square first, add or subtract, and only then take the square root.
- In the triangle inequality, "equal to" is not enough: 3, 4 and 7 lie flat in a straight line.
