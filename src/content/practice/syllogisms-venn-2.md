---
title: "Syllogisms and Venn diagrams 2: drawing and reading Venn diagrams"
description: "Pick the circle diagram that shows how three groups are related, and read numbers and regions from a three-circle Venn diagram."
slug: syllogisms-venn-2
family: syllogisms-venn
part: 2
timeMinutes: 12
stems:
  - id: sv2-clubs
    title: "Games in Class 8"
    text: "Every student in Class 8 was asked which of three games they play. The diagram shows the number of students in each part. **3 students** play none of the three games, and they are not shown in the circles."
    figure:
      kind: venn
      w: 240
      h: 182
      circles: [{ c: [95, 68], r: 50 }, { c: [150, 68], r: 50 }, { c: [122, 115], r: 50 }]
      marks:
        - { at: [45, 12], s: "Cricket" }
        - { at: [200, 12], s: "Football" }
        - { at: [122, 174], s: "Chess" }
        - { at: [75, 55], s: "12" }
        - { at: [170, 56], s: "9" }
        - { at: [122, 139], s: "7" }
        - { at: [122, 49], s: "5" }
        - { at: [92, 101], s: "4" }
        - { at: [153, 101], s: "3" }
        - { at: [122, 83], s: "2" }
questions:
  - id: sv2-sparrows
    topic: Which diagram
    difficulty: 1
    q: "Which diagram best shows the relation between **Sparrows, Birds and Animals**?"
    optionFigures: 
      - { kind: venn, circles: [{ c: [100, 70], r: 64 }, { c: [100, 70], r: 43 }, { c: [100, 70], r: 22 }] }
      - { kind: venn, circles: [{ c: [100, 70], r: 64 }, { c: [72, 70], r: 22 }, { c: [128, 70], r: 22 }] }
      - { kind: venn, circles: [{ c: [65, 70], r: 50 }, { c: [65, 78], r: 22 }, { c: [162, 70], r: 30 }] }
      - { kind: venn, circles: [{ c: [75, 52], r: 38 }, { c: [125, 52], r: 38 }, { c: [100, 95], r: 38 }] }
    answer: A
    explanation: |
      Every sparrow is a bird and every bird is an animal. So sparrows sit inside birds, and birds sit inside animals: three circles one inside the other, **diagram A**.

  - id: sv2-rivers
    topic: Which diagram
    difficulty: 1
    q: "Which diagram best shows the relation between **Rivers, Pencils and Elephants**?"
    optionFigures: 
      - { kind: venn, circles: [{ c: [50, 70], r: 35 }, { c: [100, 70], r: 35 }, { c: [170, 70], r: 25 }] }
      - { kind: venn, circles: [{ c: [65, 70], r: 50 }, { c: [65, 78], r: 22 }, { c: [162, 70], r: 30 }] }
      - { kind: venn, circles: [{ c: [35, 70], r: 25 }, { c: [100, 70], r: 25 }, { c: [165, 70], r: 25 }] }
      - { kind: venn, circles: [{ c: [100, 70], r: 64 }, { c: [100, 70], r: 43 }, { c: [100, 70], r: 22 }] }
    answer: C
    explanation: |
      Nothing is both a river and a pencil, or a pencil and an elephant, or a river and an elephant. So the three circles do not touch at all: **diagram C**.

  - id: sv2-cows
    topic: Which diagram
    difficulty: 1
    q: "Which diagram best shows the relation between **Cows, Goats and Animals**?"
    optionFigures: 
      - { kind: venn, circles: [{ c: [100, 70], r: 64 }, { c: [100, 70], r: 43 }, { c: [100, 70], r: 22 }] }
      - { kind: venn, circles: [{ c: [100, 70], r: 64 }, { c: [72, 70], r: 22 }, { c: [128, 70], r: 22 }] }
      - { kind: venn, circles: [{ c: [100, 70], r: 64 }, { c: [82, 70], r: 30 }, { c: [118, 70], r: 30 }] }
      - { kind: venn, circles: [{ c: [65, 70], r: 50 }, { c: [65, 78], r: 22 }, { c: [162, 70], r: 30 }] }
    answer: B
    explanation: |
      Cows and goats are both animals, so both small circles go inside the big one. No cow is a goat, so the small circles stay apart: **diagram B**.
      The circles one inside the other would mean every cow is a goat.

  - id: sv2-cricket-total
    stem: sv2-clubs
    topic: Reading numbers
    difficulty: 1
    q: "How many students play cricket?"
    options: ["12", "17", "21", "23"]
    answer: D
    explanation: |
      Add every number inside the Cricket circle: 12 + 5 + 4 + 2 = **23**.
      12 counts only the students who play cricket and nothing else.

  - id: sv2-exactly-two
    stem: sv2-clubs
    topic: Reading numbers
    difficulty: 2
    q: "How many students play exactly two of the three games?"
    options: ["2", "10", "12", "14"]
    answer: C
    explanation: |
      Exactly two means the parts where two circles meet, but not the middle: 5 + 4 + 3 = **12**.
      14 also adds the 2 in the middle, who play all three.

  - id: sv2-football-not-chess
    stem: sv2-clubs
    topic: Reading numbers
    difficulty: 2
    q: "How many students play football but not chess?"
    options: ["9", "14", "16", "19"]
    answer: B
    explanation: |
      Take the Football circle and leave out every part that is also inside Chess: 9 + 5 = **14**.
      19 is everyone who plays football, chess or not.

  - id: sv2-class-size
    stem: sv2-clubs
    topic: Reading numbers
    difficulty: 2
    q: "How many students are there in Class 8?"
    options: ["42", "45", "48", "52"]
    answer: B
    explanation: |
      The circles hold 12 + 9 + 7 + 5 + 4 + 3 + 2 = 42 students. Add the 3 who play none: 42 + 3 = **45**.

  - id: sv2-neither
    stem: sv2-clubs
    topic: Reading numbers
    difficulty: 3
    q: "How many students play neither cricket nor football?"
    options: ["3", "7", "10", "14"]
    answer: C
    explanation: |
      These students are outside both the Cricket and Football circles. That is the 7 who play only chess **and** the 3 who play nothing: 7 + 3 = **10**.
      7 forgets the students outside all the circles.

  - id: sv2-region-letter
    topic: Regions
    difficulty: 2
    q: "In this diagram, which letter stands for **girls in Class 8 who are not swimmers**?"
    figure:
      kind: venn
      w: 220
      h: 165
      circles: [{ c: [85, 62], r: 42 }, { c: [135, 62], r: 42 }, { c: [110, 104], r: 42 }]
      marks:
        - { at: [50, 14], s: "Girls" }
        - { at: [170, 14], s: "Swimmers" }
        - { at: [110, 156], s: "Class 8" }
        - { at: [70, 51], s: "c" }
        - { at: [150, 51], s: "f" }
        - { at: [110, 49], s: "a" }
        - { at: [110, 75], s: "d" }
        - { at: [86, 90], s: "g" }
        - { at: [134, 90], s: "b" }
        - { at: [110, 123], s: "e" }
    options: ["a", "b", "d", "g"]
    answer: D
    explanation: |
      The part must be inside Girls and inside Class 8, but outside Swimmers. That is **g**.
      d is in all three circles, so those girls do swim. a is girls who swim but are not in Class 8.

  - id: sv2-mothers
    topic: Which diagram
    difficulty: 2
    q: "Which diagram best shows the relation between **Mothers, Women and Doctors**?"
    optionFigures: 
      - { kind: venn, circles: [{ c: [80, 70], r: 55 }, { c: [88, 75], r: 24 }, { c: [130, 75], r: 40 }] }
      - { kind: venn, circles: [{ c: [80, 70], r: 55 }, { c: [60, 70], r: 25 }, { c: [150, 70], r: 40 }] }
      - { kind: venn, circles: [{ c: [100, 70], r: 64 }, { c: [100, 70], r: 43 }, { c: [100, 70], r: 22 }] }
      - { kind: venn, circles: [{ c: [100, 70], r: 64 }, { c: [82, 70], r: 30 }, { c: [118, 70], r: 30 }] }
    answer: A
    explanation: |
      Every mother is a woman, so Mothers goes inside Women. Some women are doctors, some mothers are doctors, and some doctors are men.
      So the Doctors circle must cut across both circles and also stick out of Women: **diagram A**.

  - id: sv2-fish
    topic: Which diagram
    difficulty: 3
    q: "Which diagram best shows the relation between **Fish, Water animals and Mammals**? (Whales and dolphins are mammals that live in water.)"
    optionFigures: 
      - { kind: venn, circles: [{ c: [75, 52], r: 38 }, { c: [125, 52], r: 38 }, { c: [100, 95], r: 38 }] }
      - { kind: venn, circles: [{ c: [80, 70], r: 55 }, { c: [88, 75], r: 24 }, { c: [130, 75], r: 40 }] }
      - { kind: venn, circles: [{ c: [65, 70], r: 50 }, { c: [65, 78], r: 22 }, { c: [162, 70], r: 30 }] }
      - { kind: venn, circles: [{ c: [80, 70], r: 55 }, { c: [60, 70], r: 25 }, { c: [150, 70], r: 40 }] }
    answer: D
    explanation: |
      All fish live in water, so Fish goes inside Water animals. Some mammals live in water (whales) and most do not, so Mammals overlaps Water animals.
      No fish is a mammal, so Mammals must not touch Fish: **diagram D**. In the diagram where the third circle also cuts Fish, some fish would be mammals.

  - id: sv2-squares
    topic: Which diagram
    difficulty: 3
    q: "Which diagram best shows the relation between **Squares, Rectangles and Rhombuses**? (Every square is both a rectangle and a rhombus.)"
    optionFigures: 
      - { kind: venn, circles: [{ c: [75, 70], r: 50 }, { c: [125, 70], r: 50 }, { c: [100, 70], r: 16 }] }
      - { kind: venn, circles: [{ c: [80, 70], r: 55 }, { c: [88, 75], r: 24 }, { c: [130, 75], r: 40 }] }
      - { kind: venn, circles: [{ c: [75, 52], r: 38 }, { c: [125, 52], r: 38 }, { c: [100, 95], r: 38 }] }
      - { kind: venn, circles: [{ c: [100, 70], r: 64 }, { c: [100, 70], r: 43 }, { c: [100, 70], r: 22 }] }
    answer: A
    explanation: |
      Some rectangles are rhombuses and some are not, so those two circles overlap. A square is both a rectangle and a rhombus.
      So the Squares circle sits inside the shared part of both: **diagram A**. Three circles one inside the other would make every rectangle a rhombus.
---

This set tests how to show groups with circles, and how to read numbers from a Venn diagram.

## Three ways two groups can meet

| How they are related | Example | Circles |
|---|---|---|
| One is part of the other | Mangoes, Fruits | small circle inside big |
| Some are both, some are not | Girls, Swimmers | overlapping |
| Nothing is both | Spoons, Rivers | apart |

For three groups, take them in pairs. Decide each pair, then find the one diagram that gets **all three pairs** right.

## Worked example

*Tea drinkers, Coffee drinkers, People.* Everyone who drinks tea is a person, and so is everyone who drinks coffee, so both circles go inside People. Some people drink both, so the two small circles overlap each other inside the big one.

## Reading a Venn diagram

Each number counts only the students in that exact part.
- "Plays cricket": add **every** number inside the Cricket circle.
- "Only cricket": the part of Cricket outside the other circles.
- "Exactly two": the parts where two circles meet, without the middle.
- "At least two": exactly two, plus the middle.

## Common trap

Do not forget the people outside all the circles. They count in the total, and they count in "neither … nor".
