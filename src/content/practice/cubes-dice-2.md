---
title: "Cubes and dice 2: painted cubes and piles"
description: "A painted cube cut into small cubes, counting cubes in a pile, and how many more make a full cube."
slug: cubes-dice-2
family: cubes-dice
part: 2
timeMinutes: 15
questions:
  - id: painted-3x3-three
    topic: Painted cubes
    difficulty: 1
    q: "A wooden cube is painted on all sides, then cut into 27 equal small cubes as shown. How many small cubes have paint on exactly 3 faces?"
    figure: { kind: cubegrid, n: 3 }
    options: ["8", "9", "12", "16"]
    answer: A
    explanation: |
      Only the **corner** cubes touch three painted faces. A cube has 8 corners, so there are **8**, whatever the size of the big cube.

  - id: painted-3x3-none
    topic: Painted cubes
    difficulty: 1
    q: "In the same 27 small cubes, how many have no paint at all?"
    figure: { kind: cubegrid, n: 3 }
    options: ["0", "3", "1", "6"]
    answer: C
    explanation: |
      Unpainted cubes are hidden in the middle, away from every outside face. Remove the outer layer and a 1 × 1 × 1 cube is left: **1** cube.

  - id: painted-3x3-two
    topic: Painted cubes
    difficulty: 2
    q: "A painted cube is cut into 27 equal small cubes. How many small cubes have paint on exactly 2 faces?"
    figure: { kind: cubegrid, n: 3 }
    options: ["6", "8", "9", "12"]
    answer: D
    explanation: |
      Two painted faces means the cube sits on an **edge** of the big cube but not at a corner.
      Each of the 12 edges has 3 small cubes. Leave out the 2 corners and 1 is left per edge: 12 × 1 = **12**.

  - id: painted-3x3-one
    topic: Painted cubes
    difficulty: 2
    q: "In the same 27 small cubes, how many have paint on exactly 1 face?"
    figure: { kind: cubegrid, n: 3 }
    options: ["6", "8", "9", "12"]
    answer: A
    explanation: |
      One painted face means the cube is in the **middle of a face**, away from the edges. Each of the 6 faces has 1 such cube: **6**.
      Check: 8 (three faces) + 12 (two) + 6 (one) + 1 (none) = 27 ✓.

  - id: painted-4x4-one
    topic: Painted cubes
    difficulty: 2
    q: "A painted cube is cut into 64 equal small cubes (4 × 4 × 4). How many small cubes have paint on exactly 1 face?"
    figure: { kind: cubegrid, n: 4 }
    options: ["12", "16", "18", "24"]
    answer: D
    explanation: |
      On each face, the cubes with one painted face are the middle ones: remove the border of the 4 × 4 face and a 2 × 2 = 4 square is left.
      6 faces × 4 = **24**.
      General rule for an n × n × n cube: 8 with three faces, 12 × (n − 2) with two, 6 × (n − 2)² with one, (n − 2)³ with none.

  - id: red-blue-cube
    topic: Painted cubes
    difficulty: 3
    q: "A cube is painted red on the top and bottom and blue on the other four faces. It is cut into 27 equal small cubes. How many small cubes have both red and blue paint?"
    figure: { kind: cubegrid, n: 3 }
    options: ["12", "16", "18", "20"]
    answer: B
    explanation: |
      A small cube with red paint is in the top layer or the bottom layer (9 cubes each).
      In the top layer, every cube except the centre one also touches a side face, so 8 of them have blue too. The bottom layer is the same.
      8 + 8 = **16**.

  - id: count-stack
    topic: Counting cubes
    difficulty: 2
    q: "How many cubes are in this pile? (Every cube rests on the floor or on another cube, so some cubes are hidden.)"
    figure: { kind: stack, heights: [[3, 2, 1], [2, 1, 1], [1, 1, 0]] }
    options: ["10", "11", "12", "13"]
    answer: C
    explanation: |
      Count column by column, from the back row: 3 + 2 + 1, then 2 + 1 + 1, then 1 + 1 + 0.
      6 + 4 + 2 = **12**. The cubes under the tall columns are hidden but must be there, or the top cubes would float.

  - id: painted-4x4-any
    topic: Painted cubes
    difficulty: 2
    q: "A painted cube is cut into 64 equal small cubes. How many small cubes have at least one painted face?"
    options: ["48", "56", "60", "64"]
    answer: B
    explanation: |
      It is easier to count the cubes with **no** paint and subtract. Removing the outer layer of a 4 × 4 × 4 cube leaves 2 × 2 × 2 = 8 unpainted cubes.
      64 − 8 = **56**.

  - id: painted-5x5-two
    topic: Painted cubes
    difficulty: 3
    q: "A cube with 5 cm sides is painted on all faces and cut into 1 cm cubes. How many of the small cubes have paint on exactly 2 faces?"
    options: ["24", "27", "30", "36"]
    answer: D
    explanation: |
      Cubes with two painted faces lie along the edges, not at the corners. Each edge has 5 cubes; leaving out the 2 corners gives 3.
      12 edges × 3 = **36**.

  - id: complete-the-cube
    topic: Counting cubes
    difficulty: 3
    q: "How many more cubes are needed to turn this pile into a solid 3 × 3 × 3 cube?"
    figure: { kind: stack, heights: [[3, 2, 1], [2, 1, 1], [1, 1, 0]] }
    options: ["15", "16", "17", "18"]
    answer: A
    explanation: |
      A 3 × 3 × 3 cube has 27 cubes. The pile has 3 + 2 + 1 + 2 + 1 + 1 + 1 + 1 + 0 = 12.
      27 − 12 = **15** more cubes.

  - id: painted-3x3-two-or-more
    topic: Painted cubes
    difficulty: 3
    q: "A painted cube is cut into 27 equal small cubes. How many small cubes have paint on at least 2 faces?"
    options: ["8", "12", "20", "26"]
    answer: C
    explanation: |
      "At least 2" means 2 faces or 3 faces. 3 faces: the 8 corners. 2 faces: 1 on each of 12 edges = 12.
      8 + 12 = **20**.

  - id: painted-area-total
    topic: Painted cubes
    difficulty: 3
    q: "A cube with 3 cm sides is painted on all faces and cut into 27 cubes of 1 cm. What is the total painted area of all 27 small cubes added together?"
    options: ["27 cm²", "54 cm²", "81 cm²", "162 cm²"]
    answer: B
    explanation: |
      Cutting does not add paint. The painted area is still the outside of the big cube: 6 faces × (3 × 3) cm² = **54 cm²**.
      You can check by counting: 8 × 3 + 12 × 2 + 6 × 1 = 54 painted 1 cm squares.
      162 cm² is the total area of all faces of the small cubes, painted or not.
---

These questions are about a big cube cut into small cubes, and about piles of cubes. One table and one habit solve almost all of them.

## Painted cube cut into n × n × n small cubes

| Small cubes with | Where they are | How many |
|---|---|---|
| 3 painted faces | corners | 8 |
| 2 painted faces | edges, not corners | 12 × (n − 2) |
| 1 painted face | middle of each face | 6 × (n − 2)² |
| no paint | hidden inside | (n − 2)³ |

For n = 3 that is 8, 12, 6 and 1, which add up to 27.

## Counting cubes in a pile

Count **column by column** (how tall each stack is), not the cubes you can see. Every cube must rest on something, so a cube on top means there are cubes under it, even when they are hidden.
