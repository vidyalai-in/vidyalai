---
title: "Cubes and dice 1: dice and nets"
description: "Opposite faces from views of a die, standard dice, rolling a die, and folding cube nets."
slug: cubes-dice-1
family: cubes-dice
part: 1
timeMinutes: 15
questions:
  - id: cube-edges
    topic: Cube basics
    difficulty: 1
    q: "How many edges does a cube have?"
    figure: { kind: dice, cubes: [{ top: "", front: "", right: "" }] }
    options: ["12", "16", "18", "24"]
    answer: A
    explanation: |
      A cube has 4 edges around the top, 4 around the bottom and 4 upright edges joining them: 4 + 4 + 4 = **12**.
      It also has 6 faces and 8 corners (vertices).

  - id: std-die-bottom
    topic: Dice
    difficulty: 1
    q: "This is a standard die: opposite faces always add up to 7. Which number is on the bottom face?"
    figure: { kind: dice, pips: true, cubes: [{ top: 2, front: 1, right: 3 }] }
    options: ["1", "3", "4", "5"]
    answer: D
    explanation: |
      The top shows 2. The bottom is opposite the top, so it is 7 − 2 = **5**.

  - id: std-die-hidden-sum
    topic: Dice
    difficulty: 2
    q: "On the same standard die, what is the total of the dots on the three faces you cannot see?"
    figure: { kind: dice, pips: true, cubes: [{ top: 2, front: 1, right: 3 }] }
    options: ["12", "13", "14", "15"]
    answer: D
    explanation: |
      Each hidden face is opposite a visible one: bottom 7 − 2 = 5, back 7 − 1 = 6, left 7 − 3 = 4. Total 5 + 6 + 4 = **15**.
      Shortcut: all six faces add to 21, and the visible ones add to 2 + 1 + 3 = 6, so the hidden ones add to 21 − 6 = 15.

  - id: cross-net-opposite
    topic: Nets
    difficulty: 1
    q: "This net is folded into a cube. Which number ends up opposite 3?"
    figure: { kind: net, rows: [[null, 1, null, null], [2, 3, 4, 5], [null, 6, null, null]] }
    options: ["5", "1", "2", "6"]
    answer: A
    explanation: |
      In a straight row of faces, a face is opposite the one **two steps away**, skipping one between.
      In the row 2 3 4 5, the face two steps from 3 is 5, so **5** is opposite 3. In the same way 2 is opposite 4, and 1 (above 3) is opposite 6 (below 3).

  - id: not-a-net
    topic: Nets
    difficulty: 2
    q: "Which of these can NOT be folded into a cube?"
    optionFigures:
      - { kind: net, rows: [[null, "", null, null], ["", "", "", ""], [null, "", null, null]] }
      - { kind: net, rows: [["", "", "", ""], [null, "", "", null]] }
      - { kind: net, rows: [["", "", ""], [null, "", null], [null, "", null], [null, "", null]] }
      - { kind: net, rows: [["", null, null, null], ["", "", "", ""], ["", null, null, null]] }
    answer: B
    explanation: |
      In **B**, the row of four folds into a ring of four faces. The two squares under it would both fold onto the same open end, and the other end is left with nothing.
      A, C and D are all real cube nets: each has a row of four (the ring) with exactly one square on each side of it.

  - id: faces-at-corner
    topic: Cube basics
    difficulty: 1
    q: "How many faces of a cube meet at each corner?"
    options: ["2", "4", "3", "6"]
    answer: C
    explanation: |
      Look at any corner of the cube in the pictures on this page: the top, the front and a side face meet there, which is **3** faces.

  - id: letter-views-opposite
    topic: Dice
    difficulty: 2
    q: "Three views of the same cube are shown. Which letter is opposite Q?"
    figure:
      kind: dice
      cubes:
        - { top: Q, front: R, right: P }
        - { top: T, front: U, right: Q }
        - { top: S, front: T, right: P }
      labels: ["(i)", "(ii)", "(iii)"]
    options: ["S", "P", "R", "U"]
    answer: A
    explanation: |
      Faces you see together are next to each other, so none of them can be opposite.
      Q appears with R and P in view (i), and with T and U in view (ii). That is four neighbours of Q.
      The only letter never seen with Q is **S**, so S is opposite Q.

  - id: number-views-opposite
    topic: Dice
    difficulty: 2
    q: "Three views of the same die are shown (it is not a standard die). Which number is opposite 2?"
    figure:
      kind: dice
      cubes:
        - { top: 1, front: 2, right: 3 }
        - { top: 1, front: 5, right: 2 }
        - { top: 2, front: 4, right: 3 }
      labels: ["(i)", "(ii)", "(iii)"]
    options: ["1", "4", "5", "6"]
    answer: D
    explanation: |
      List what appears next to 2: 1 and 3 in view (i), 1 and 5 in view (ii), 4 and 3 in view (iii).
      So 1, 3, 4 and 5 are neighbours of 2. The only number left is **6**, which must be opposite 2.

  - id: net-to-cube
    topic: Nets
    difficulty: 2
    q: "The net on the left is folded into a cube. Which cube could it make? (The letters are only labels: ignore which way they face.)"
    figure: { kind: net, rows: [[null, K, null, null], [L, M, N, O], [null, P, null, null]] }
    optionFigures:
      - { kind: dice, cubes: [{ top: K, front: P, right: L }] }
      - { kind: dice, cubes: [{ top: M, front: O, right: N }] }
      - { kind: dice, cubes: [{ top: L, front: N, right: K }] }
      - { kind: dice, cubes: [{ top: K, front: M, right: N }] }
    answer: D
    explanation: |
      First find the opposite pairs in the net (two steps apart in a line): K–P, L–N and M–O.
      Opposite faces can never be seen together.
      A shows K and P, B shows M and O, C shows L and N, so none of them can be right. **D** shows K, M and N, three faces that do meet at one corner.

  - id: staircase-net
    topic: Nets
    difficulty: 3
    q: "This net is folded into a cube. Which number is opposite 1?"
    figure: { kind: net, rows: [[1, 2, null, null], [null, 3, 4, null], [null, null, 5, 6]] }
    options: ["3", "4", "5", "6"]
    answer: B
    explanation: |
      Fold it in your mind with 3 as the front face. 2 is above 3, so it folds to the top, and 1 is to the left of 2, so it folds down to the left side.
      4 is to the right of 3, so it is the right side, opposite 1. 5 is below 4, so it folds to the bottom, and 6 ends up at the back.
      So **4** is opposite 1 (also 2–5 and 3–6).

  - id: roll-right
    topic: Dice
    difficulty: 2
    q: "This standard die (opposite faces add up to 7) is tipped over once to the right, so it lands on the face that now shows 3. Which number is on top after the roll?"
    figure: { kind: dice, pips: true, cubes: [{ top: 1, front: 2, right: 3 }] }
    options: ["2", "3", "4", "6"]
    answer: C
    explanation: |
      When a die tips to the right, the right face goes to the bottom and the **left** face comes up to the top.
      The left face is opposite the right face (3), so it shows 7 − 3 = **4**.

  - id: standard-die-net
    topic: Nets
    difficulty: 3
    q: "Which net folds into a standard die, where opposite faces add up to 7?"
    optionFigures:
      - { kind: net, pips: true, rows: [[null, 1, null, null], [2, 3, 5, 4], [null, 6, null, null]] }
      - { kind: net, pips: true, rows: [[null, 1, null, null], [2, 3, 4, 5], [null, 6, null, null]] }
      - { kind: net, pips: true, rows: [[null, 1, null, null], [2, 3, 6, 4], [null, 5, null, null]] }
      - { kind: net, pips: true, rows: [[null, 2, null, null], [1, 3, 5, 4], [null, 6, null, null]] }
    answer: A
    explanation: |
      In these nets the square above the row is opposite the square below it, and in the row of four, the 1st is opposite the 3rd and the 2nd is opposite the 4th.
      **A**: 1–6, 2–5, 3–4 all add to 7 ✓.
      B: 2 is opposite 4 ✗. C: 1 is opposite 5 ✗. D: 2 is opposite 6 ✗.
---

Cube questions test whether you can turn a shape over in your head. A few facts and a careful habit go a long way.

## Facts about a cube

A cube has **6 faces, 12 edges and 8 corners**. Three faces meet at each corner. Each face touches four others and is opposite exactly one.

On a **standard die**, opposite faces add up to 7: 1 is opposite 6, 2 is opposite 5, 3 is opposite 4. All six faces add up to 21.

## Finding opposite faces from several views

Faces seen together in one view are **neighbours**, never opposite. For the number you are asked about, list every number seen next to it across all the views. A face has four neighbours, so once you have found four, the one left over is opposite.

## Nets

A net is a cube unfolded flat. In any straight line of squares in a net, a square is **opposite the one two places away** (skip one, take the next). Squares that touch along an edge in the net become neighbours.

A real cube net usually has a row of four squares (which folds into a ring) with one square on each side of that row. If two squares sit on the same side, they fold onto the same face and the net fails.
