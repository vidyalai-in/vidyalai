---
title: "Mock test 3: mixed reasoning"
description: "A third timed 15-minute test: fractions, HCF and ages, idioms and letter codes, a grid to remember, turning figures, painted cubes, heights and if-then logic."
slug: mock-tests-3
family: mock-tests
part: 3
timeMinutes: 15
stems:
  - id: mk3-grid
    title: "Memory: the letter and number grid"
    text: "Look at the grid row by row. Notice where the letters and the numbers are."
    studySeconds: 35
    figure:
      kind: grid
      cells:
        - [7, K, 3, M]
        - [B, 9, R, 2]
        - [5, T, 8, A]
        - [H, 4, E, 6]
questions:
  - id: mk3-facing-turns
    topic: Analytical
    difficulty: 1
    q: "Kabir is facing north. He turns to his right, then to his right again, and then to his left. Which direction is he facing now?"
    options: ["North", "East", "South", "West"]
    answer: B
    explanation: |
      North → right turn → east → right turn → south → left turn → **east**.
      Two rights and a left come to one right turn in total.

  - id: mk3-fraction-left
    topic: Numerical
    difficulty: 1
    q: "Meera spends 1/4 of her pocket money on a book and 1/3 of it on a gift. What fraction of her pocket money is left?"
    options: ["2/7", "5/12", "1/2", "7/12"]
    answer: B
    explanation: |
      Spent: 1/4 + 1/3 = 3/12 + 4/12 = 7/12. Left: 1 − 7/12 = **5/12**.
      7/12 is what she spent. 2/7 comes from wrongly adding tops and bottoms: (1 + 1)/(4 + 3).

  - id: mk3-blue-moon
    topic: Verbal
    difficulty: 1
    q: "Ravi's uncle visits them only *once in a blue moon*. What does this mean?"
    options: ["Every month", "At night", "Very quickly", "Very rarely"]
    answer: D
    explanation: |
      *Once in a blue moon* means **very rarely**, hardly ever.

  - id: mk3-turn-card
    topic: Visual
    difficulty: 2
    q: "Ravi turns this card a quarter turn (90°) clockwise, then another quarter turn clockwise, and then a quarter turn anticlockwise. Which option shows the card now?"
    figure: { kind: shapes, frame: true, items: [{ t: poly, p: [[20, 18], [20, 82], [52, 82]], open: true }, { t: poly, p: [[60, 18], [86, 18], [86, 44]], fill: solid }, { t: circle, c: [70, 70], r: 7, fill: none }], alt: "A card with an L-shaped line on the left, a solid triangle in the top-right corner and a small circle near the bottom-right." }
    optionFigures:
      - { kind: shapes, frame: true, transform: rot270, items: [{ t: poly, p: [[20, 18], [20, 82], [52, 82]], open: true }, { t: poly, p: [[60, 18], [86, 18], [86, 44]], fill: solid }, { t: circle, c: [70, 70], r: 7, fill: none }] }
      - { kind: shapes, frame: true, transform: rot180, items: [{ t: poly, p: [[20, 18], [20, 82], [52, 82]], open: true }, { t: poly, p: [[60, 18], [86, 18], [86, 44]], fill: solid }, { t: circle, c: [70, 70], r: 7, fill: none }] }
      - { kind: shapes, frame: true, transform: rot90, items: [{ t: poly, p: [[20, 18], [20, 82], [52, 82]], open: true }, { t: poly, p: [[60, 18], [86, 18], [86, 44]], fill: solid }, { t: circle, c: [70, 70], r: 7, fill: none }] }
      - { kind: shapes, frame: true, transform: flipH, items: [{ t: poly, p: [[20, 18], [20, 82], [52, 82]], open: true }, { t: poly, p: [[60, 18], [86, 18], [86, 44]], fill: solid }, { t: circle, c: [70, 70], r: 7, fill: none }] }
    answer: C
    explanation: |
      Two quarter turns clockwise and one back anticlockwise leave **one quarter turn clockwise** in total.
      After a quarter turn clockwise, the top-right corner moves to the bottom-right, so the solid triangle ends up there and the L-shape wraps round the top-left corner: **C**.
      A is a quarter turn the other way, B is a half turn and D is a mirror image.

  - id: mk3-ribbon-pieces
    topic: Numerical
    difficulty: 2
    q: "Two ribbons, 84 cm and 126 cm long, are cut into pieces that are all the same length, with nothing left over. What is the smallest possible total number of pieces?"
    options: ["5", "10", "15", "42"]
    answer: A
    explanation: |
      Fewest pieces means the longest possible pieces: the HCF of 84 and 126, which is 42 cm.
      84 ÷ 42 = 2 and 126 ÷ 42 = 3, so **5** pieces. 42 is the length of a piece, not the number of pieces.

  - id: mk3-heights-middle
    topic: Analytical
    difficulty: 2
    q: "Five friends compare their heights. Joseph is shorter than Neha but taller than Farhan. Vikram is taller than Neha. Anjali is the shortest of all. Who is exactly in the middle (the third tallest)?"
    options: ["Neha", "Farhan", "Joseph", "Vikram"]
    answer: C
    explanation: |
      From the clues: Vikram > Neha > Joseph > Farhan, and Anjali is below everyone.
      So the order from tallest is Vikram, Neha, **Joseph**, Farhan, Anjali.

  - id: mk3-letter-analogy
    topic: Verbal
    difficulty: 2
    q: "ACE : FHJ :: KMO : ?"
    options: ["OQS", "PQR", "QSU", "PRT"]
    answer: D
    explanation: |
      Each letter moves 5 places forward: A→F, C→H, E→J.
      So K→P, M→R, O→T, giving **PRT**. OQS moves only 4 places.

  - id: mk3-grid-corner
    stem: mk3-grid
    topic: Memory
    difficulty: 1
    q: "What was in the top-left corner of the grid?"
    options: ["3", "5", "7", "9"]
    answer: C
    explanation: "The first row was 7, K, 3, M, so the top-left corner held **7**."

  - id: mk3-grid-bottom-left
    stem: mk3-grid
    topic: Memory
    difficulty: 2
    q: "Which letter was in the bottom-left corner of the grid?"
    options: ["H", "K", "M", "T"]
    answer: A
    explanation: "The bottom row was H, 4, E, 6, so the bottom-left corner held **H**. Letters and numbers took turns, like the squares of a chessboard."

  - id: mk3-grid-below-r
    stem: mk3-grid
    topic: Memory
    difficulty: 3
    q: "Which number was directly below the letter R?"
    options: ["2", "4", "8", "9"]
    answer: C
    explanation: "R was third in the second row (B, 9, R, 2). Directly below it, third in the third row (5, T, 8, A), was **8**."

  - id: mk3-green-yellow
    topic: Visual
    difficulty: 2
    q: "A wooden cube is painted green on its top and bottom faces and yellow on its four side faces. It is then cut into 64 equal small cubes (4 × 4 × 4). How many small cubes have green paint but no yellow paint?"
    figure: { kind: cubegrid, n: 4 }
    options: ["4", "8", "16", "24"]
    answer: B
    explanation: |
      Green cubes are in the top and bottom layers. To have no yellow, a cube must not touch any side face, so it is one of the middle 2 × 2 = 4 in its layer.
      Top layer 4 + bottom layer 4 = **8**. 16 counts every cube in the top layer, including the edge ones that touch the yellow sides.

  - id: mk3-ages
    topic: Numerical
    difficulty: 2
    q: "Arjun's father is 3 times as old as Arjun. Five years ago, he was 4 times as old as Arjun was then. How old is Arjun now?"
    options: ["10", "15", "20", "45"]
    answer: B
    explanation: |
      Let Arjun be A now, so his father is 3A. Five years ago: 3A − 5 = 4 × (A − 5) = 4A − 20, so A = **15**.
      Check: now 15 and 45; five years ago 10 and 40, and 40 = 4 × 10 ✓. 45 is the father's age.

  - id: mk3-code-reverse
    topic: Verbal
    difficulty: 3
    q: "In a code, FRIEND is written as IRFDNE. How is CANDLE written in the same code?"
    options: ["NACELD", "ELDNAC", "NACDLE", "DLECAN"]
    answer: A
    explanation: |
      Split FRIEND into FRI and END. Each half is written backwards: FRI → IRF and END → DNE, giving IRFDNE.
      CANDLE → CAN and DLE → NAC and ELD → **NACELD**. ELDNAC reverses the whole word, and NACDLE reverses only the first half.

  - id: mk3-dice-two-views
    topic: Visual
    difficulty: 3
    q: "Two views of the same die are shown (it is not a standard die). Which number is opposite 2?"
    figure:
      kind: dice
      cubes:
        - { top: 4, front: 1, right: 2 }
        - { top: 4, front: 6, right: 5 }
      labels: ["(i)", "(ii)"]
    options: ["1", "3", "4", "5"]
    answer: D
    explanation: |
      4 is on top in both views, so view (ii) is view (i) turned round on the table. 1, 2, 6 and 5 are the four side faces, and 3 (never seen with 4) is the bottom.
      Going round the sides in view (i): front 1, right 2, then the back, then the left. Turning the die keeps this order, so in view (ii) front 6 and right 5 must be the back and the left: back 6, left 5.
      2 is on the right and 5 is on the left, so **5** is opposite 2. 3 is opposite 4, not 2.

  - id: mk3-if-then-fair
    topic: Analytical
    difficulty: 3
    q: "Two rules are always followed.\nIf Asha goes to the fair, then Ravi goes too.\nIf Ravi goes to the fair, then Meera stays at home.\nYesterday Meera did not stay at home. Which of these must be true?"
    options: ["Asha went to the fair", "Ravi went, but Asha did not", "Ravi did not go, but Asha may have gone", "Neither Asha nor Ravi went to the fair"]
    answer: D
    explanation: |
      If Ravi had gone, Meera would have stayed at home. She didn't, so Ravi did not go.
      If Asha had gone, Ravi would have gone too. He didn't, so Asha did not go either: **neither Asha nor Ravi went**.
---

This is a timed mock test: **15 questions in 15 minutes**, three from each of the five areas (numbers, words, memory, shapes and cubes, and logic), mixed together.

## How it works

Press Start to begin. A timer counts down, and when it reaches zero the test is **submitted automatically**, with whatever answers you have marked. There is **no negative marking**, so it is always worth choosing an option, even when you are not sure.

## Pacing

- Plan on about **one minute per question**. Many questions here can be done in 30 seconds; bank that time for the harder ones near the end.
- If a question looks long, read the last line first so you know what you are looking for. If it still takes too long, **skip it and come back**.
- The memory block takes about a minute: 35 seconds to study a grid, then three questions. Read the grid row by row and look for a pattern in where the letters sit.
- For turning figures, follow one corner of the card through each turn instead of trying to turn the whole picture at once.
