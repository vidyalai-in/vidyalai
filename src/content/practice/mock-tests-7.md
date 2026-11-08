---
title: "Mock test 7: mixed reasoning"
description: "A timed 15-minute test with 15 mixed questions: a ticket table, ratios, a word list to remember, counting squares, a painted block, paper folded three times and counting digits."
slug: "mock-tests-7"
family: "mock-tests"
part: 7
timeMinutes: 15
stems:
  - id: mk7-words
    title: "Memorize: ten words"
    studySeconds: 30
    text: "Read the words in order, left to right and then the second line. They will be hidden when the time is up."
    lines:
      - "kettle · parrot · compass · umbrella · pumpkin"
      - "violin · camel · bucket · scissors · rocket"
questions:
  - id: mk7-play-tickets
    topic: "Numerical"
    difficulty: 1
    q: "The table shows the tickets sold for a school play on three days. How many **child** tickets were sold in all?"
    figure: {"kind": "table", "caption": "Tickets sold for the school play", "header": ["Day", "Adult", "Child"], "rows": [["Thursday", 40, 65], ["Friday", 55, 80], ["Saturday", 90, 120]]}
    options: ["185", "255", "265", "450"]
    answer: C
    explanation: |
      Add the Child column: 65 + 80 + 120 = **265**.
      185 is the Adult total and 450 is all the tickets together.

  - id: mk7-count-squares
    topic: "Visual"
    difficulty: 1
    q: "How many squares of any size are there in this figure?"
    figure: {"kind": "shapes", "w": 100, "h": 70, "items": [{"t": "line", "p": [5, 5, 95, 5]}, {"t": "line", "p": [5, 35, 95, 35]}, {"t": "line", "p": [5, 65, 95, 65]}, {"t": "line", "p": [5, 5, 5, 65]}, {"t": "line", "p": [35, 5, 35, 65]}, {"t": "line", "p": [65, 5, 65, 65]}, {"t": "line", "p": [95, 5, 95, 65]}], "alt": "A rectangle divided into 2 rows of 3 equal squares"}
    options: ["6", "7", "8", "9"]
    answer: C
    explanation: |
      There are 6 small squares. Two small squares on top of two others make a bigger 2 × 2 square, and there are 2 of those (left and right).
      6 + 2 = **8**. The whole figure is a rectangle, not a square, so it does not count.

  - id: mk7-humble
    topic: "Verbal"
    difficulty: 1
    q: "Even after winning the gold medal, Sana stayed *humble*. Which word is most nearly **opposite** in meaning to *humble*?"
    options: ["Modest", "Proud", "Polite", "Gentle"]
    answer: B
    explanation: |
      A humble person does not show off. The opposite is **proud**, in the sense of thinking too highly of yourself.
      *Modest* means almost the same as humble.

  - id: mk7-coded-relation
    topic: "Analytical"
    difficulty: 2
    q: "In a code, P + Q means P is the mother of Q, P − Q means P is the brother of Q, and P × Q means P is the father of Q. If A + B − C × D, how is A related to D?"
    options: ["Grandmother", "Mother", "Aunt", "Sister"]
    answer: A
    explanation: |
      A + B: A is the mother of B. B − C: B is the brother of C, so A is the mother of C too. C × D: C is the father of D.
      A is the mother of D's father, so A is D's **grandmother**.

  - id: mk7-ratio-gift
    topic: "Numerical"
    difficulty: 2
    q: "₹1,800 is shared among Lata, Mohan and Yash in the ratio 2 : 3 : 4. Then Yash gives ₹200 of his share to Lata. What is now the ratio of Lata's money to Yash's money?"
    options: ["1 : 2", "2 : 3", "1 : 1", "3 : 2"]
    answer: C
    explanation: |
      2 + 3 + 4 = 9 parts, so one part is ₹200. Lata gets ₹400, Mohan ₹600 and Yash ₹800.
      After the gift, Lata has ₹600 and Yash has ₹600: the ratio is **1 : 1**. 1 : 2 is the ratio before the gift.

  - id: mk7-words-in-list
    stem: mk7-words
    topic: "Memory"
    difficulty: 1
    q: "Which of these words was in the list?"
    options: ["Lantern", "Blanket", "Rabbit", "Violin"]
    answer: D
    explanation: |
      **Violin** was the first word of the second line. Lantern, blanket and rabbit were not in the list.

  - id: mk7-words-animals
    stem: mk7-words
    topic: "Memory"
    difficulty: 2
    q: "How many of the words were animals or birds?"
    options: ["1", "2", "3", "4"]
    answer: B
    explanation: |
      Only parrot and camel: **2**.

  - id: mk7-words-after
    stem: mk7-words
    topic: "Memory"
    difficulty: 3
    q: "Which word came just after *umbrella*?"
    options: ["Compass", "Pumpkin", "Violin", "Bucket"]
    answer: B
    explanation: |
      The first line was kettle, parrot, compass, umbrella, **pumpkin**. Compass came just *before* umbrella.

  - id: mk7-painted-cuboid
    topic: "Visual"
    difficulty: 2
    q: "A wooden block 5 cm long, 4 cm wide and 3 cm high is painted on all six faces. It is then cut into 1 cm cubes. How many of the small cubes have paint on exactly two faces?"
    options: ["22", "24", "28", "36"]
    answer: B
    explanation: |
      Cubes with two painted faces lie along the edges, but not at the corners. The block has 4 edges of each length.
      Edges of 5: 4 × (5 − 2) = 12. Edges of 4: 4 × (4 − 2) = 8. Edges of 3: 4 × (3 − 2) = 4.
      12 + 8 + 4 = **24**. 22 is the number with exactly one painted face, and 36 uses the rule for a 5 × 5 × 5 cube.

  - id: mk7-odd-instrument
    topic: "Verbal"
    difficulty: 2
    q: "Which is the odd one out?"
    options: ["Sitar", "Veena", "Sarod", "Tabla"]
    answer: D
    explanation: |
      The sitar, veena and sarod are played by plucking strings. The **tabla** is a pair of drums.

  - id: mk7-shadow-turns
    topic: "Analytical"
    difficulty: 2
    q: "At 7 o'clock one morning, Gauri stands so that her shadow falls straight in front of her. She turns 90° clockwise, and then 135° anticlockwise. Which direction is she facing now?"
    options: ["North-east", "North-west", "South-east", "South-west"]
    answer: D
    explanation: |
      In the morning the sun is in the east, so shadows fall towards the west. Gauri faces west.
      90° clockwise from west is north. 135° anticlockwise from north is 90° to west and 45° more, to **south-west**.

  - id: mk7-grid-squares
    topic: "Numerical"
    difficulty: 2
    q: "The numbers in each row follow the same rule. Which number replaces the question mark?"
    figure: {"kind": "grid", "cells": [[2, 3, 13], [4, 1, 17], [3, 5, "?"]], "shade": [[2, 2]]}
    options: ["25", "30", "34", "64"]
    answer: C
    explanation: |
      In each row, the third number is the sum of the squares of the first two: 2² + 3² = 4 + 9 = 13, and 4² + 1² = 16 + 1 = 17.
      So ? = 3² + 5² = 9 + 25 = **34**. 64 is (3 + 5)², which does not fit the first two rows.

  - id: mk7-fold-three
    topic: "Visual"
    difficulty: 3
    q: "A square sheet of paper is folded in half from top to bottom, then from left to right. The small square is then folded along its diagonal so that its bottom left corner meets its top right corner. A hole is punched as shown in the last picture. Which option shows the sheet when it is opened out?"
    figure: {"kind": "shapes", "w": 442, "h": 100, "items": [{"t": "line", "p": [12, 12, 88, 12]}, {"t": "line", "p": [88, 12, 88, 88]}, {"t": "line", "p": [88, 88, 12, 88]}, {"t": "line", "p": [12, 88, 12, 12]}, {"t": "line", "p": [12.0, 50.0, 88.0, 50.0], "dash": true}, {"t": "poly", "p": [[50.0, 34.8], [52.04, 38.6], [53.76, 42.4], [54.92, 46.2], [55.32, 50.0], [54.92, 53.8], [53.76, 57.6], [52.04, 61.4]], "open": true}, {"t": "arrow", "p": [52.04, 61.4, 50.0, 65.2]}, {"t": "line", "p": [202.0, 50.0, 202, 88]}, {"t": "line", "p": [202, 88, 126, 88]}, {"t": "line", "p": [126, 88, 126.0, 50.0]}, {"t": "line", "p": [126.0, 50.0, 202.0, 50.0]}, {"t": "line", "p": [164.0, 50.0, 164.0, 88.0], "dash": true}, {"t": "poly", "p": [[148.8, 69.0], [152.6, 71.04], [156.4, 72.76], [160.2, 73.92], [164.0, 74.32], [167.8, 73.92], [171.6, 72.76], [175.4, 71.04]], "open": true}, {"t": "arrow", "p": [175.4, 71.04, 179.2, 69.0]}, {"t": "line", "p": [316.0, 50.0, 316, 88]}, {"t": "line", "p": [316, 88, 278.0, 88.0]}, {"t": "line", "p": [278.0, 88.0, 278.0, 50.0]}, {"t": "line", "p": [278.0, 50.0, 316.0, 50.0]}, {"t": "line", "p": [278.0, 50.0, 316, 88], "dash": true}, {"t": "poly", "p": [[286.25, 79.75], [290.38, 78.5], [294.29, 77.03], [297.79, 75.16], [300.76, 72.76], [303.16, 69.79], [305.03, 66.29], [306.5, 62.38]], "open": true}, {"t": "arrow", "p": [306.5, 62.38, 307.75, 58.25]}, {"t": "line", "p": [430.0, 50.0, 430, 88]}, {"t": "line", "p": [430, 88, 392.0, 50.0], "dash": true}, {"t": "line", "p": [392.0, 50.0, 430.0, 50.0], "dash": true}, {"t": "circle", "c": [422.4, 65.2], "r": 3.6, "fill": "solid"}]}
    optionFigures:
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [99.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [1.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [99.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [1.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "circle", "c": [89.2, 69.6], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [69.6, 89.2], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [10.8, 69.6], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [30.4, 89.2], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [89.2, 30.4], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [69.6, 10.8], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [10.8, 30.4], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [30.4, 10.8], "r": 5.0, "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [99.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [1.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [99.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [1.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "circle", "c": [89.2, 69.6], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [10.8, 69.6], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [89.2, 30.4], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [10.8, 30.4], "r": 5.0, "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [99.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [1.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [99.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [1.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "circle", "c": [89.2, 69.6], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [79.4, 59.8], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [10.8, 69.6], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [20.6, 59.8], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [89.2, 30.4], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [79.4, 40.2], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [10.8, 30.4], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [20.6, 40.2], "r": 5.0, "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [99.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [1.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [99.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [1.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "circle", "c": [89.2, 69.6], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [69.6, 89.2], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [10.8, 69.6], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [30.4, 89.2], "r": 5.0, "fill": "solid"}]}
    answer: A
    explanation: |
      Unfold one step at a time, last fold first. The diagonal fold gives the hole a partner reflected across the diagonal of the small square (2 holes). Undoing the left-right fold doubles them (4), and undoing the top-bottom fold doubles them again: **8 holes**, as in **A**.
      B forgets the diagonal fold, C reflects across the wrong diagonal, and D forgets the first fold, so the top half has no holes.

  - id: mk7-code-halves
    topic: "Verbal"
    difficulty: 3
    q: "In a code, SILVER is written as MJTSFW. How is PICKLE written in the same code?"
    options: ["DJQFML", "FMLDJQ", "CIPELK", "BHODKJ"]
    answer: A
    explanation: |
      Split the word into two halves and write each half backwards: SIL VER → LIS REV. Then move every letter one place forward: LISREV → MJTSFW.
      PICKLE → PIC KLE → CIP ELK → **DJQFML**.
      FMLDJQ reverses the whole word instead of each half, and CIPELK forgets to move the letters.

  - id: mk7-count-threes
    topic: "Analytical"
    difficulty: 3
    q: "Kabir writes all the whole numbers from 1 to 50. How many times does he write the digit 3?"
    options: ["5", "14", "15", "16"]
    answer: C
    explanation: |
      As a units digit: 3, 13, 23, 33, 43 gives 5 threes. As a tens digit: 30 to 39 gives 10 threes.
      5 + 10 = **15**. 14 counts the numbers that contain a 3, but 33 has two of them.
---

This mock test mixes 15 questions from all five areas: numbers, words, memory, pictures and logic.

## How the test works

- You have **15 minutes**. The clock starts when you press Start, and the test is **submitted automatically** when the time is up.
- There is **no negative marking**, so a guess can only help you.
- Questions 6 to 8 are about a list of ten words. You get 30 seconds to study it, and then it is hidden.
- After you submit, you see your score, the right answers and an explanation for each question.

## Pacing

You have about one minute per question, but they are not equal. The early ones are short; the paper folding question and the last two need more care.

- Read each question once, fully, before you look at the options.
- If one approach is not working after a minute, choose your best option and move on.
- For memory lists, link the words into a quick story or picture. It is easier to remember a story than a list.
- Keep an eye on the clock at the halfway mark. By then you should be near question 8.
