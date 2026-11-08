---
title: "Geometry and mensuration 3: perimeter, area and volume"
description: "Rectangles, triangles, parallelograms and circles (π = 22/7), paths round a garden or pond, and the volume and surface area of cubes and cuboids."
slug: geometry-mensuration-3
family: geometry-mensuration
part: 3
timeMinutes: 12
questions:
  - id: gm3-breadth
    topic: Perimeter and area
    difficulty: 1
    q: "The perimeter of a rectangular field is 46 m. Its length is 14 m. What is its breadth?"
    options: ["9 m", "18 m", "23 m", "32 m"]
    answer: A
    explanation: |
      Perimeter = 2 × (length + breadth), so length + breadth = 46 ÷ 2 = 23 m. Breadth = 23 − 14 = **9 m**.
      18 m forgets to halve: 46 − 2 × 14 = 18 is the two breadths together.

  - id: gm3-triangle-area
    topic: Perimeter and area
    difficulty: 1
    q: "A triangle has a base of 16 cm and a height of 9 cm. What is its area?"
    options: ["36 cm²", "72 cm²", "144 cm²", "288 cm²"]
    answer: B
    explanation: |
      Area of a triangle = ½ × base × height = ½ × 16 × 9 = **72 cm²**.
      144 cm² forgets the ½; that would be the area of a rectangle 16 cm by 9 cm.

  - id: gm3-circumference
    topic: Circles
    difficulty: 1
    q: "Find the circumference of a circle of radius 14 cm. (Take π = 22/7.)"
    options: ["44 cm", "88 cm", "176 cm", "616 cm"]
    answer: B
    explanation: |
      Circumference = 2πr = 2 × 22/7 × 14 = **88 cm**.
      616 is the area (πr²), and 176 cm uses the diameter, 28 cm, in place of the radius.

  - id: gm3-cube-surface
    topic: Cubes and cuboids
    difficulty: 1
    q: "What is the total surface area of a cube of edge 5 cm?"
    options: ["25 cm²", "100 cm²", "125 cm²", "150 cm²"]
    answer: D
    explanation: |
      A cube has 6 square faces, each 5 × 5 = 25 cm². Total = 6 × 25 = **150 cm²**.
      125 is the volume (5 × 5 × 5, in cm³), and 100 cm² counts only the 4 side faces.

  - id: gm3-parallelogram
    topic: Perimeter and area
    difficulty: 2
    q: "Find the area of this parallelogram."
    figure:
      kind: shapes
      w: 270
      h: 135
      alt: "A parallelogram with base 15 cm, slanting side 10 cm, and a dashed height of 8 cm drawn from the top left corner down to the base."
      items:
        - { t: poly, p: [[18.9, 113.4], [172.8, 113.4], [234.4, 31.3], [80.5, 31.3]] }
        - { t: line, p: [80.5, 31.3, 80.5, 113.4], dash: true }
        - { t: poly, p: [[86.5, 113.4], [86.5, 107.4], [80.5, 107.4]], open: true }
        - { t: text, at: [106.7, 125.6], s: "15 cm", size: 13 }
        - { t: text, at: [229.6, 72.4], s: "10 cm", size: 13 }
        - { t: text, at: [102.5, 72.4], s: "8 cm", size: 13 }
    options: ["60 cm²", "80 cm²", "120 cm²", "150 cm²"]
    answer: C
    explanation: |
      Area of a parallelogram = base × **height**, where the height is the straight-up distance (the dashed line), not the slanting side.
      Area = 15 × 8 = **120 cm²**. 150 cm² uses the slanting side: 15 × 10.

  - id: gm3-path-outside
    topic: Paths and borders
    difficulty: 2
    q: "A rectangular garden is 30 m long and 20 m wide. A path 2.5 m wide runs all round it, on the outside. What is the area of the path?"
    figure:
      kind: shapes
      w: 243
      h: 197.1
      alt: "A rectangular garden 30 m by 20 m with a shaded path 2.5 m wide all round the outside."
      items:
        - { t: rect, p: [13.5, 10.8, 217.4, 15.5], fill: grey }
        - { t: rect, p: [13.5, 150.5, 217.4, 15.5], fill: grey }
        - { t: rect, p: [13.5, 26.3, 15.5, 124.2], fill: grey }
        - { t: rect, p: [215.3, 26.3, 15.5, 124.2], fill: grey }
        - { t: rect, p: [13.5, 10.8, 217.4, 155.2] }
        - { t: rect, p: [29, 26.3, 186.3, 124.2] }
        - { t: text, at: [122.2, 88.4], s: "Garden", size: 13 }
        - { t: text, at: [122.2, 37.1], s: "30 m", size: 13 }
        - { t: text, at: [47.9, 88.4], s: "20 m", size: 13 }
        - { t: text, at: [155.9, 183.6], s: "path, 2.5 m wide", size: 12 }
        - { t: arrow, p: [155.9, 175.5, 155.9, 158.3] }
    options: ["250 m²", "275 m²", "300 m²", "875 m²"]
    answer: B
    explanation: |
      The outer rectangle is (30 + 2 × 2.5) m by (20 + 2 × 2.5) m = 35 m by 25 m = 875 m².
      Path = 875 − 30 × 20 = 875 − 600 = **275 m²**. 250 m² (garden perimeter × 2.5) misses the four corner squares; 875 m² includes the garden.

  - id: gm3-circle-area
    topic: Circles
    difficulty: 2
    q: "The circumference of a circle is 132 cm. What is its area? (Take π = 22/7.)"
    options: ["441 cm²", "693 cm²", "1386 cm²", "5544 cm²"]
    answer: C
    explanation: |
      2 × 22/7 × r = 132, so r = 132 × 7 ÷ 44 = 21 cm.
      Area = 22/7 × 21 × 21 = **1386 cm²**. 5544 cm² uses the diameter, 42 cm, as the radius; 441 forgets π.

  - id: gm3-tank-litres
    topic: Cubes and cuboids
    difficulty: 2
    q: "A water tank is 2 m long, 1.5 m wide and 80 cm deep. How many litres of water can it hold? (1 m³ = 1000 litres)"
    options: ["2.4 litres", "24 litres", "240 litres", "2400 litres"]
    answer: D
    explanation: |
      First make all the units metres: 80 cm = 0.8 m. Volume = 2 × 1.5 × 0.8 = 2.4 m³.
      In litres: 2.4 × 1000 = **2400 litres**. 240 comes from 2 × 1.5 × 80, which mixes metres and centimetres.

  - id: gm3-wire-circle
    topic: Circles
    difficulty: 2
    q: "A wire is bent into a square of side 22 cm. It is then straightened and bent into a circle. What is the radius of the circle? (Take π = 22/7.)"
    options: ["14 cm", "22 cm", "28 cm", "44 cm"]
    answer: A
    explanation: |
      Length of wire = perimeter of the square = 4 × 22 = 88 cm. This becomes the circumference: 2 × 22/7 × r = 88, so r = 88 × 7 ÷ 44 = **14 cm**.
      28 cm is the diameter, and 44 cm forgets π.

  - id: gm3-cross-roads
    topic: Paths and borders
    difficulty: 3
    q: "A lawn is 60 m long and 40 m wide. Two roads, each 3 m wide, cross in the middle, one parallel to each side, as shown. What is the total area of the two roads?"
    figure:
      kind: shapes
      w: 270
      h: 189
      alt: "A rectangular lawn 60 m by 40 m with two shaded roads, each 3 m wide, crossing in the middle: one runs along the length and one along the width."
      items:
        - { t: rect, p: [10.8, 75.7, 210.6, 10.5], fill: grey }
        - { t: rect, p: [110.8, 10.8, 10.5, 140.4], fill: grey }
        - { t: rect, p: [10.8, 10.8, 210.6, 140.4] }
        - { t: arrow, p: [97.2, 162, 10.8, 162] }
        - { t: arrow, p: [135, 162, 221.4, 162] }
        - { t: text, at: [116.1, 162], s: "60 m", size: 13 }
        - { t: arrow, p: [232.2, 70.2, 232.2, 10.8] }
        - { t: arrow, p: [232.2, 91.8, 232.2, 151.2] }
        - { t: text, at: [252.5, 81], s: "40 m", size: 13 }
        - { t: text, at: [63.5, 45.9], s: "Lawn", size: 13 }
        - { t: text, at: [168.8, 63.6], s: "road 3 m", size: 12 }
    options: ["291 m²", "300 m²", "309 m²", "2109 m²"]
    answer: A
    explanation: |
      Road along the length: 60 × 3 = 180 m². Road along the width: 40 × 3 = 120 m².
      The 3 m × 3 m square where they cross is counted twice, so the total is 180 + 120 − 9 = **291 m²**. 300 m² counts that square twice; 2109 m² is the lawn left over.

  - id: gm3-cut-cubes
    topic: Cubes and cuboids
    difficulty: 3
    q: "A cuboid 12 cm × 8 cm × 6 cm is cut into small cubes of edge 2 cm. What is the total surface area of all the small cubes together?"
    options: ["432 cm²", "576 cm²", "864 cm²", "1728 cm²"]
    answer: D
    explanation: |
      Number of cubes = (12 ÷ 2) × (8 ÷ 2) × (6 ÷ 2) = 6 × 4 × 3 = 72. Each has surface area 6 × 2 × 2 = 24 cm².
      Total = 72 × 24 = **1728 cm²**. 432 cm² is the cuboid before cutting; every cut makes new faces. 576 is the volume, in cm³.

  - id: gm3-ring-path
    topic: Circles
    difficulty: 3
    q: "A circular pond has a radius of 35 m. A path 7 m wide runs all round it, on the outside. What is the area of the path? (Take π = 22/7.)"
    figure:
      kind: shapes
      w: 270
      h: 168.8
      alt: "A circular pond of radius 35 m with a path 7 m wide all round it, drawn as two circles with the same centre."
      items:
        - { t: circle, c: [89.1, 83.7], r: 70.9 }
        - { t: circle, c: [89.1, 83.7], r: 59.1 }
        - { t: circle, c: [89.1, 83.7], r: 1.6, fill: solid }
        - { t: line, p: [89.1, 83.7, 30, 83.7], dash: true }
        - { t: text, at: [59.6, 75.6], s: "35 m", size: 13 }
        - { t: text, at: [89.1, 106.7], s: "pond", size: 13 }
        - { t: line, p: [154.1, 83.7, 169.4, 83.7] }
        - { t: text, at: [203.2, 83.7], s: "path 7 m", size: 13 }
    options: ["154 m²", "1540 m²", "1694 m²", "5544 m²"]
    answer: C
    explanation: |
      Outer radius = 35 + 7 = 42 m. Path = big circle − pond = 22/7 × (42 × 42 − 35 × 35) = 22/7 × 539 = **1694 m²**.
      1540 m² (2πr × width) uses the pond's edge for the whole path and comes out too small; 5544 m² is the big circle without taking away the pond.
---
This set tests perimeter, area and volume: flat shapes, circles, paths and borders, and boxes.

## The formulas

| Shape | Perimeter | Area |
|---|---|---|
| rectangle, l by b | 2(l + b) | l × b |
| square, side a | 4a | a² |
| triangle | add the sides | ½ × base × height |
| parallelogram | add the sides | base × height |
| circle, radius r | 2πr | πr² |

Use π = 22/7 when the question says so. Choose radii that are multiples of 7 in your head: 7, 14, 21, 28, 35.

| Solid | Volume | Total surface area |
|---|---|---|
| cuboid, l × b × h | l × b × h | 2(lb + bh + hl) |
| cube, edge a | a³ | 6a² |

1 m³ = 1000 litres, and 1000 cm³ = 1 litre.

## Paths and borders

For a path **outside** a rectangle, add twice the path width to each side to get the outer rectangle. Path area = outer area − inner area. For a ring round a circle, path area = π(R² − r²), with R the outer radius.

## Worked example

*A square lawn of side 20 m has a 1 m path round it, outside. Find the path's area.*

Outer square: 22 m by 22 m = 484 m². Path = 484 − 400 = **84 m²**. Multiplying the perimeter by the width, 80 × 1 = 80 m², misses the four 1 m × 1 m corners.

## Common traps

- The height of a triangle or parallelogram is the straight-up distance, not a slanting side.
- Change every length to the same unit before you multiply.
- Two crossing roads share a square in the middle. Count it only once.
- Cutting a solid into pieces keeps the volume the same but adds surface area.
