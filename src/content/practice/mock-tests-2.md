---
title: "Mock test 2: mixed reasoning"
description: "A second timed 15-minute test: series, clocks, codes, a shopping list to remember, cube nets and piles, seating and truth-tellers."
slug: mock-tests-2
family: mock-tests
part: 2
timeMinutes: 15
stems:
  - id: mk2-list
    title: "Memory: Neha's shopping list"
    text: "Neha bought these things at the market. Notice the quantities as well as the prices."
    studySeconds: 35
    lines:
      - "Rice, 5 kg: ₹310"
      - "Sugar, 2 kg: ₹96"
      - "Tea, 250 g: ₹145"
      - "Soap, 4 bars: ₹120"
      - "Biscuits, 3 packs: ₹90"
      - "Cooking oil, 1 litre: ₹165"
questions:
  - id: mk2-odd-potato
    topic: Verbal
    difficulty: 1
    q: "Which word is the odd one out?"
    options: ["Mango", "Guava", "Potato", "Papaya"]
    answer: C
    explanation: |
      Mango, guava and papaya are fruits that grow on trees. A **potato** is a vegetable that grows under the ground.

  - id: mk2-series-double
    topic: Numerical
    difficulty: 1
    q: "What comes next? 4, 9, 19, 39, 79, ?"
    options: ["119", "158", "159", "160"]
    answer: C
    explanation: |
      Each number is double the one before, plus 1: 4 × 2 + 1 = 9, 9 × 2 + 1 = 19, and so on.
      Next: 79 × 2 + 1 = **159**. 158 forgets the +1.

  - id: mk2-laddoo
    topic: Analytical
    difficulty: 2
    q: "One of three friends, Ravi, Meera and Kabir, ate the last laddoo.\nRavi says, 'Meera ate it.'\nMeera says, 'I did not eat it.'\nKabir says, 'I did not eat it.'\nExactly one of them is telling the truth. Who ate the laddoo?"
    options: ["Kabir", "Ravi", "Meera", "It cannot be decided"]
    answer: A
    explanation: |
      Try each person. If Ravi ate it, Meera and Kabir are both telling the truth: two truths ✗. If Meera ate it, Ravi and Kabir tell the truth: two truths ✗.
      If **Kabir** ate it, only Meera tells the truth ✓. Exactly one case works, so it can be decided.

  - id: mk2-water-surat
    topic: Visual
    difficulty: 1
    q: "Which option is the water image of the text below (its reflection in a pond below it)?\n**SURAT 64**"
    optionFigures:
      - { kind: glyphs, text: "SURAT 64", transform: flipH }
      - { kind: glyphs, text: "SURAT 64", transform: rot180 }
      - { kind: glyphs, text: "46 TARUS", transform: none }
      - { kind: glyphs, text: "SURAT 64", transform: flipV }
    answer: D
    explanation: |
      In a water image, top and bottom swap, but left and right stay the same. The letters keep their order and are turned upside down as if flipped over a line below them: **D**.
      A is the mirror image (left and right swapped). B is a half turn, which also swaps left and right. C just writes the text backwards.

  - id: mk2-list-tea
    stem: mk2-list
    topic: Memory
    difficulty: 1
    q: "How much did the tea cost?"
    options: ["₹96", "₹120", "₹145", "₹165"]
    answer: C
    explanation: "The tea (250 g) cost **₹145**. ₹165 was the cooking oil."

  - id: mk2-list-biscuits
    stem: mk2-list
    topic: Memory
    difficulty: 2
    q: "How many packs of biscuits did Neha buy?"
    options: ["2", "3", "4", "5"]
    answer: B
    explanation: "She bought **3** packs of biscuits for ₹90. The 4 was the number of soap bars."

  - id: mk2-list-sugar-oil
    stem: mk2-list
    topic: Memory
    difficulty: 2
    q: "What did the sugar and the cooking oil cost together?"
    options: ["₹241", "₹261", "₹265", "₹275"]
    answer: B
    explanation: "Sugar cost ₹96 and the oil cost ₹165. Together: 96 + 165 = **₹261**."

  - id: mk2-clock-340
    topic: Numerical
    difficulty: 2
    q: "What is the smaller angle between the hour hand and the minute hand of a clock at 3:40?"
    options: ["120°", "130°", "140°", "150°"]
    answer: B
    explanation: |
      Minute hand: 40 × 6° = 240° from 12. Hour hand: 3 × 30° + 40 × 0.5° = 110° from 12.
      The difference is 240° − 110° = **130°**. 150° forgets that the hour hand has moved 20° past the 3.

  - id: mk2-code-dog
    topic: Verbal
    difficulty: 2
    q: "In a code, each letter stands for its place in the alphabet (A = 1, B = 2, …, Z = 26), and a word's code is the total. So CAT = 3 + 1 + 20 = 24. What is the code for DOG?"
    options: ["24", "25", "26", "28"]
    answer: C
    explanation: |
      D = 4, O = 15, G = 7. Total 4 + 15 + 7 = **26**.
      25 comes from taking O as 14 (O is the 15th letter: count on from J = 10).

  - id: mk2-stack-count
    topic: Visual
    difficulty: 2
    q: "How many cubes are in this pile? Every cube rests on the floor or on another cube, so some cubes are hidden."
    figure: { kind: stack, heights: [[3, 3, 2], [3, 2, 1], [2, 1, 1]] }
    options: ["15", "16", "17", "18"]
    answer: D
    explanation: |
      Count each column from the top and add. Back row: 3 + 3 + 2 = 8. Middle row: 3 + 2 + 1 = 6. Front row: 2 + 1 + 1 = 4.
      Total 8 + 6 + 4 = **18**. Counting only the cubes you can see gives a smaller number.

  - id: mk2-great-grandmother
    topic: Analytical
    difficulty: 2
    q: "Tara is Rohan's sister. Lata is Rohan's mother. Mohan is Lata's father. Kamala is Mohan's mother. How is Kamala related to Tara?"
    options: ["Grandmother", "Great-grandmother", "Aunt", "Mother"]
    answer: B
    explanation: |
      Tara and Rohan have the same mother, Lata. Lata's father Mohan is Tara's grandfather.
      Mohan's mother Kamala is one generation older again, so she is Tara's **great-grandmother**.

  - id: mk2-average-speed
    topic: Numerical
    difficulty: 3
    q: "Farhan cycles to school at 12 km/h and comes back along the same road at 8 km/h. What is his average speed for the whole trip?"
    options: ["9.6 km/h", "10 km/h", "10.4 km/h", "20 km/h"]
    answer: A
    explanation: |
      Suppose the school is 24 km away (any distance gives the same answer). Going takes 24 ÷ 12 = 2 h and coming back takes 24 ÷ 8 = 3 h.
      Average speed = total distance ÷ total time = 48 ÷ 5 = **9.6 km/h**.
      10 km/h is the trap: he spends longer at the slower speed, so the average is below the middle value.

  - id: mk2-meticulous
    topic: Verbal
    difficulty: 3
    q: "Choose the word closest in meaning to METICULOUS.\nAnjali is *meticulous*: she checks every line of her homework twice."
    options: ["Thorough", "Careless", "Generous", "Nervous"]
    answer: A
    explanation: |
      *Meticulous* means paying great attention to every small detail, so **thorough** is closest.
      Careless is the opposite. Checking twice is not a sign of being nervous; it is a sign of care.

  - id: mk2-round-table
    topic: Analytical
    difficulty: 3
    q: "Six friends sit around a round table, evenly spaced and facing the centre.\nKabir sits directly opposite Asha.\nRavi sits to Asha's immediate right.\nFarhan sits to Kabir's immediate left.\nMeera does not sit next to Kabir.\nThe sixth friend is Priya. Who sits directly opposite Meera?"
    options: ["Farhan", "Priya", "Ravi", "Kabir"]
    answer: A
    explanation: |
      Place Asha and Kabir opposite each other. The two seats next to Kabir are not for Meera, and Ravi takes one of the seats next to Asha, so Meera takes the other seat next to Asha.
      Farhan is on Kabir's left, which is the seat between Kabir and Ravi. Priya takes the last seat, between Meera and Kabir.
      Going round the table: Asha, Meera, Priya, Kabir, Farhan, Ravi. Meera's opposite seat is **Farhan's**.

  - id: mk2-net-not-possible
    topic: Visual
    difficulty: 3
    q: "The net is folded into a cube. Which cube can NOT be made from it? (The letters are only labels: ignore which way they face.)"
    figure: { kind: net, rows: [[null, W, null, null], [S, T, U, null], [null, null, V, R]] }
    optionFigures:
      - { kind: dice, cubes: [{ top: T, front: S, right: V }] }
      - { kind: dice, cubes: [{ top: W, front: R, right: S }] }
      - { kind: dice, cubes: [{ top: V, front: R, right: U }] }
      - { kind: dice, cubes: [{ top: T, front: W, right: R }] }
    answer: D
    explanation: |
      Fold with T as the front. W folds up to the top, S to the left and U to the right. V, below U, folds under to the bottom, and R, joined to V, comes up at the back.
      So the opposite pairs are S–U, W–V and **T–R**. Opposite faces are never seen together, and **D** shows both T and R, so it is impossible.
      A, B and C each show three faces that really do meet at one corner.
---

This is a timed mock test: **15 questions in 15 minutes**, three from each of the five areas (numbers, words, memory, shapes and cubes, and logic), mixed together.

## How it works

The clock starts when you press Start and the test is **submitted automatically** when time is up. There is **no negative marking**, so a guess can only help you. Make sure every question has an answer before the end.

## Pacing

- You have about **one minute per question**. Quick ones, like a water image or an odd one out, take less, which saves time for the harder puzzles.
- If you are stuck after a minute, **skip the question and come back** to it later.
- The memory block takes about a minute: 35 seconds to study a shopping list, then three questions. Link each item with its number, for example "rice, five kilos, 310".
- For seating and truth-teller puzzles, draw a quick sketch or try each case in turn rather than working it all in your head.
