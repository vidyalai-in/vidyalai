---
title: "Mock test 6: mixed reasoning"
description: "A timed 15-minute test with 15 mixed questions: a pie chart, shares of wages, vowel codes, symbols to remember, a figure series, a quarter turn and a digit puzzle."
slug: "mock-tests-6"
family: "mock-tests"
part: 6
timeMinutes: 15
stems:
  - id: mk6-symbols
    title: "Memorize: the symbol grid"
    studySeconds: 35
    text: "Study the grid of symbols. It will be hidden when the time is up."
    figure: {"kind": "grid", "cells": [["★", "●", "▲", "■"], ["▲", "★", "■", "●"], ["■", "▲", "★", "♥"], ["●", "■", "▲", "★"]], "size": 48}
questions:
  - id: mk6-walk-west
    topic: "Analytical"
    difficulty: 1
    q: "Rohan leaves home and walks 4 km south, then 6 km west, and then 4 km north. Where is he now?"
    options: ["6 km west of home", "6 km east of home", "4 km south of home", "14 km west of home"]
    answer: A
    explanation: |
      The 4 km south and the 4 km north cancel out. Only the 6 km west is left, so he is **6 km west of home**.
      14 km is the total distance he walked, not how far he is from home.

  - id: mk6-reluctant
    topic: "Verbal"
    difficulty: 1
    q: "Aditya was *reluctant* to jump into the cold pool. Which word is closest in meaning to *reluctant*?"
    options: ["Eager", "Ready", "Careful", "Unwilling"]
    answer: D
    explanation: |
      *Reluctant* means not wanting to do something: **unwilling**. *Eager* is its opposite.

  - id: mk6-mirror-kota
    topic: "Visual"
    difficulty: 1
    q: "A mirror is placed on the right of the figure. Which option is its **mirror image**?"
    figure: {"kind": "glyphs", "text": "KOTA 59"}
    optionFigures:
      - {"kind": "glyphs", "text": "KOTA 59", "transform": "flipV"}
      - {"kind": "glyphs", "text": "KOTA 59", "transform": "flipH"}
      - {"kind": "glyphs", "text": "KOTA 95", "transform": "flipH"}
      - {"kind": "glyphs", "text": "KOTA 59", "transform": "rot180"}
    answer: B
    explanation: |
      A mirror at the side swaps left and right: the order of the characters is reversed and each one faces the other way, but nothing turns upside down. That is **B**.
      A is the water image (upside down). C has the 5 and 9 in the wrong order. D is the figure turned upside down.

  - id: mk6-pie-food
    topic: "Numerical"
    difficulty: 2
    q: "The pie chart shows how Riya's family uses its income of ₹24,000 a month. How much more is spent on food than on education?"
    figure: {"kind": "pie", "title": "Monthly income: ₹24,000", "labels": ["Food", "Rent", "Education", "Travel", "Savings"], "values": [35, 25, 15, 10, 15]}
    options: ["₹3,600", "₹4,800", "₹6,000", "₹8,400"]
    answer: B
    explanation: |
      Food is 35% and education is 15%, a difference of 20%.
      20% of ₹24,000 = ₹24,000 × 20 ÷ 100 = **₹4,800**.
      ₹8,400 is the whole food amount and ₹3,600 is the whole education amount.

  - id: mk6-vowel-code
    topic: "Verbal"
    difficulty: 2
    q: "In a code, MELON is written as MILUN and SUGAR is written as SAGER. How is TULIP written in the same code?"
    options: ["TOLEP", "TULOP", "TALOP", "TALIP"]
    answer: C
    explanation: |
      Consonants stay the same. Each vowel changes to the next vowel: A → E → I → O → U, and U goes back round to A (SUGAR → SAGER shows this).
      In TULIP, U → A and I → O, giving **TALOP**. TOLEP moves the vowels backwards; TULOP and TALIP change only one vowel.

  - id: mk6-row-dev
    topic: "Analytical"
    difficulty: 2
    q: "In a row of 37 students, Zoya is 14th from the left end. Dev is 6 places to the right of Zoya. What is Dev's position from the right end?"
    options: ["17th", "18th", "19th", "24th"]
    answer: B
    explanation: |
      Dev is 14 + 6 = 20th from the left.
      From the right he is 37 − 20 + 1 = **18th**. 17th forgets the + 1, and 24th is Zoya's place from the right, not Dev's.

  - id: mk6-symbols-diagonal
    stem: mk6-symbols
    topic: "Memory"
    difficulty: 1
    q: "Which symbol ran all along the diagonal from the top left corner to the bottom right corner?"
    options: ["● circle", "▲ triangle", "■ square", "★ star"]
    answer: D
    explanation: |
      The diagonal was four stars: **★**.

  - id: mk6-symbols-once
    stem: mk6-symbols
    topic: "Memory"
    difficulty: 2
    q: "Which symbol appeared only once in the grid?"
    options: ["♥ heart", "● circle", "▲ triangle", "■ square"]
    answer: A
    explanation: |
      There was a single heart, **♥**, at the right end of the third row. Every other symbol appeared three or four times.

  - id: mk6-symbols-circles
    stem: mk6-symbols
    topic: "Memory"
    difficulty: 3
    q: "How many circles (●) were in the grid?"
    options: ["1", "2", "3", "4"]
    answer: C
    explanation: |
      There were **3** circles: second in the top row, at the right end of the second row, and at the left end of the bottom row. Stars, triangles and squares each appeared 4 times.

  - id: mk6-wages
    topic: "Numerical"
    difficulty: 2
    q: "Pooja can paint a wall alone in 6 days. Manav can paint it alone in 12 days. They paint it together and are paid ₹1,800 for the job. If the money is shared by the work each one did, what is Pooja's share?"
    options: ["₹600", "₹900", "₹1,200", "₹1,500"]
    answer: C
    explanation: |
      In one day Pooja does 1/6 of the wall and Manav does 1/12. Pooja works twice as fast, so the work (and the money) is shared 2 : 1.
      Pooja gets 2/3 of ₹1,800 = **₹1,200**. ₹600 is Manav's share; ₹900 splits the money equally.

  - id: mk6-turn-clockwise
    topic: "Visual"
    difficulty: 2
    q: "Which option shows this figure after it is turned a quarter turn (90°) **clockwise**?"
    figure: {"kind": "shapes", "frame": true, "transform": "none", "items": [{"t": "poly", "p": [[18, 18], [46, 18], [46, 30], [30, 30], [30, 78], [18, 78]], "fill": "none"}, {"t": "circle", "c": [72, 28], "r": 8, "fill": "solid"}, {"t": "poly", "p": [[56, 58], [84, 58], [56, 84]], "fill": "grey"}]}
    optionFigures:
      - {"kind": "shapes", "frame": true, "transform": "rot270", "items": [{"t": "poly", "p": [[18, 18], [46, 18], [46, 30], [30, 30], [30, 78], [18, 78]], "fill": "none"}, {"t": "circle", "c": [72, 28], "r": 8, "fill": "solid"}, {"t": "poly", "p": [[56, 58], [84, 58], [56, 84]], "fill": "grey"}]}
      - {"kind": "shapes", "frame": true, "transform": "flipH", "items": [{"t": "poly", "p": [[18, 18], [46, 18], [46, 30], [30, 30], [30, 78], [18, 78]], "fill": "none"}, {"t": "circle", "c": [72, 28], "r": 8, "fill": "solid"}, {"t": "poly", "p": [[56, 58], [84, 58], [56, 84]], "fill": "grey"}]}
      - {"kind": "shapes", "frame": true, "transform": "rot180", "items": [{"t": "poly", "p": [[18, 18], [46, 18], [46, 30], [30, 30], [30, 78], [18, 78]], "fill": "none"}, {"t": "circle", "c": [72, 28], "r": 8, "fill": "solid"}, {"t": "poly", "p": [[56, 58], [84, 58], [56, 84]], "fill": "grey"}]}
      - {"kind": "shapes", "frame": true, "transform": "rot90", "items": [{"t": "poly", "p": [[18, 18], [46, 18], [46, 30], [30, 30], [30, 78], [18, 78]], "fill": "none"}, {"t": "circle", "c": [72, 28], "r": 8, "fill": "solid"}, {"t": "poly", "p": [[56, 58], [84, 58], [56, 84]], "fill": "grey"}]}
    answer: D
    explanation: |
      In a quarter turn clockwise, the top of the figure goes to the right side. The L shape at the top left moves to the top right, and the black dot at the top right moves to the bottom right: **D**.
      A is a quarter turn anticlockwise, B is a mirror image and C is a half turn.

  - id: mk6-syllogism-pencils
    topic: "Analytical"
    difficulty: 2
    q: "Statements: All pencils are sticks. Some sticks are green.\nConclusions: I. Some pencils are green. II. Some green things are sticks.\nWhich conclusions follow from the statements?"
    options: ["Only conclusion I follows", "Only conclusion II follows", "Both I and II follow", "Neither I nor II follows"]
    answer: B
    explanation: |
      II is just "some sticks are green" turned round, so it must be true.
      I need not be true: the green sticks could all be sticks that are not pencils. So **only conclusion II follows**.

  - id: mk6-series-dot-arrow
    topic: "Visual"
    difficulty: 3
    q: "The figures change by a rule from left to right. Which option should replace the question mark?"
    figure: {"kind": "shapes", "w": 436, "h": 100, "items": [{"t": "rect", "p": [2, 2, 96, 96], "fill": "none"}, {"t": "arrow", "p": [50.0, 64.0, 50.0, 22.0]}, {"t": "circle", "c": [20, 20], "r": 7, "fill": "solid"}, {"t": "rect", "p": [114, 2, 96, 96], "fill": "none"}, {"t": "arrow", "p": [152.1, 59.9, 181.8, 30.2]}, {"t": "circle", "c": [132, 80], "r": 7, "fill": "solid"}, {"t": "rect", "p": [226, 2, 96, 96], "fill": "none"}, {"t": "arrow", "p": [260.0, 50.0, 302.0, 50.0]}, {"t": "circle", "c": [304, 80], "r": 7, "fill": "solid"}, {"t": "rect", "p": [338, 2, 96, 96], "fill": "none"}, {"t": "text", "at": [386, 52], "s": "?", "size": 40}], "alt": "Three boxes, each with an arrow from the centre and a black dot in one corner, then a box with a question mark"}
    optionFigures:
      - {"kind": "shapes", "items": [{"t": "rect", "p": [2, 2, 96, 96], "fill": "none"}, {"t": "arrow", "p": [36.0, 50.0, 78.0, 50.0]}, {"t": "circle", "c": [80, 20], "r": 7, "fill": "solid"}]}
      - {"kind": "shapes", "items": [{"t": "rect", "p": [2, 2, 96, 96], "fill": "none"}, {"t": "arrow", "p": [40.1, 40.1, 69.8, 69.8]}, {"t": "circle", "c": [20, 20], "r": 7, "fill": "solid"}]}
      - {"kind": "shapes", "items": [{"t": "rect", "p": [2, 2, 96, 96], "fill": "none"}, {"t": "arrow", "p": [40.1, 59.9, 69.8, 30.2]}, {"t": "circle", "c": [80, 20], "r": 7, "fill": "solid"}]}
      - {"kind": "shapes", "items": [{"t": "rect", "p": [2, 2, 96, 96], "fill": "none"}, {"t": "arrow", "p": [40.1, 40.1, 69.8, 69.8]}, {"t": "circle", "c": [80, 20], "r": 7, "fill": "solid"}]}
    answer: D
    explanation: |
      Two things change at each step. The black dot moves one corner **anticlockwise**: top left, bottom left, bottom right, so next is top right. The arrow turns 45° **clockwise**: up, up-right, right, so next is down-right.
      Only **D** has both. A does not turn the arrow, B puts the dot back at the start, and C turns the arrow the wrong way.

  - id: mk6-butter-words
    topic: "Verbal"
    difficulty: 3
    q: "Which word can be put in front of all three of these to make three new words?\n___fly    ___cup    ___milk"
    options: ["Butter", "Honey", "Sun", "Tea"]
    answer: A
    explanation: |
      **Butter** gives butterfly, buttercup and buttermilk.
      Tea works only for teacup, and honey and sun make none of the three.

  - id: mk6-ab-ba
    topic: "Numerical"
    difficulty: 3
    q: "AB is a two-digit number with different digits A and B. When AB is added to BA (the number with its digits reversed), the sum is 154. A is 2 more than B. What is the number AB?"
    options: ["68", "77", "86", "95"]
    answer: C
    explanation: |
      AB + BA = (10A + B) + (10B + A) = 11 × (A + B). So 11 × (A + B) = 154 and A + B = 14.
      Two digits that add to 14 and differ by 2 are 8 and 6. A is the bigger one, so AB = **86**.
      68 is BA, 77 repeats a digit, and 95 has digits that differ by 4.
---

This mock test mixes 15 questions from all five areas: numbers, words, memory, pictures and logic.

## How the test works

- You have **15 minutes**. The clock starts when you press Start, and the test is **submitted automatically** when the time runs out.
- There is **no negative marking**. Never leave a question blank.
- Questions 7 to 9 share one grid of symbols. You get 35 seconds to study it before it is hidden. Look for a pattern you can hold on to: a diagonal, a row, a symbol that is different.
- After you submit, every question shows the right answer and a short explanation.

## Pacing

A steady speed is about one minute per question.

- The first few questions are quick. Use them to bank time for the harder ones near the end.
- If you are stuck for more than a minute, choose the option that looks most likely and move on. You can change it later.
- For codes and number puzzles, test your rule on the example before you use it.
- Save the last two minutes to check that you have chosen an option in every question.
