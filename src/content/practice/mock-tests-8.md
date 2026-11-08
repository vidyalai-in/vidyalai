---
title: "Mock test 8: mixed reasoning"
description: "A timed 15-minute test with 15 mixed questions: a train timetable, changed signs, a score table to remember, two reflections, a magic triangle and a hockey league."
slug: "mock-tests-8"
family: "mock-tests"
part: 8
timeMinutes: 15
stems:
  - id: mk8-quiz
    title: "Memorize: the inter-house quiz"
    studySeconds: 40
    text: "Study the scores of the four houses. They will be hidden when the time is up."
    figure: {"kind": "table", "caption": "Quiz scores", "header": ["House", "Round 1", "Round 2"], "rows": [["Ruby", 14, 9], ["Emerald", 11, 16], ["Sapphire", 17, 8], ["Topaz", 10, 13]]}
questions:
  - id: mk8-idiom-chew
    topic: "Verbal"
    difficulty: 1
    q: "Farhan joined the quiz team, the choir and the football team in the same week. Soon he had *bitten off more than he could chew*. What does this mean?"
    options: ["He took on more than he could manage", "He ate too quickly", "He spoke rudely to others", "He wasted a lot of food"]
    answer: A
    explanation: |
      To *bite off more than you can chew* is to **take on more work than you can manage**. It is not about food at all.

  - id: mk8-cricket-football
    topic: "Analytical"
    difficulty: 1
    q: "In a class, 18 students play cricket and 12 play football. 5 of them play both games. How many students play at least one of the two games?"
    options: ["13", "25", "30", "35"]
    answer: B
    explanation: |
      Adding 18 + 12 = 30 counts the 5 who play both games twice. Take them away once: 30 − 5 = **25**.

  - id: mk8-train-journey
    topic: "Numerical"
    difficulty: 1
    q: "Here is part of a train timetable. How long does the journey take, from leaving Pune to reaching Thane?"
    figure: {"kind": "table", "caption": "Train timetable", "header": ["Station", "Arrives", "Departs"], "rows": [["Pune", "—", "6:10 am"], ["Lonavala", "7:05 am", "7:08 am"], ["Karjat", "7:50 am", "7:52 am"], ["Thane", "9:02 am", "—"]]}
    options: ["2 h 47 min", "2 h 52 min", "3 h 2 min", "3 h 52 min"]
    answer: B
    explanation: |
      The train leaves Pune at 6:10 am and reaches Thane at 9:02 am. From 6:10 to 9:10 is 3 hours, so to 9:02 it is 8 minutes less: **2 h 52 min**.
      2 h 47 min leaves out the 5 minutes the train stands at Lonavala and Karjat, but those minutes are part of the journey.

  - id: mk8-mirror-then-water
    topic: "Visual"
    difficulty: 2
    q: "First the figure is reflected in a mirror placed on its right. Then that mirror image is reflected in water placed below it. Which option shows the final picture?"
    figure: {"kind": "glyphs", "text": "FLAG 27"}
    optionFigures:
      - {"kind": "glyphs", "text": "FLAG 27", "transform": "none"}
      - {"kind": "glyphs", "text": "FLAG 27", "transform": "flipH"}
      - {"kind": "glyphs", "text": "FLAG 27", "transform": "rot180"}
      - {"kind": "glyphs", "text": "FLAG 27", "transform": "flipV"}
    answer: C
    explanation: |
      The mirror swaps left and right. The water then turns that picture upside down. Doing both is the same as turning the figure **half a turn**: the order of the characters is reversed and every one is upside down: **C**.
      A is the trap: two reflections do not cancel when they are in different directions. B is only the mirror image and D is only the water image.

  - id: mk8-quiz-seventeen
    stem: mk8-quiz
    topic: "Memory"
    difficulty: 1
    q: "Which house scored 17 in Round 1?"
    options: ["Ruby", "Emerald", "Sapphire", "Topaz"]
    answer: C
    explanation: |
      **Sapphire** scored 17 in Round 1 (and 8 in Round 2).

  - id: mk8-quiz-topaz
    stem: mk8-quiz
    topic: "Memory"
    difficulty: 2
    q: "What did Topaz score in Round 2?"
    options: ["9", "10", "13", "16"]
    answer: C
    explanation: |
      Topaz scored 10 in Round 1 and **13** in Round 2. 16 was Emerald's Round 2 score.

  - id: mk8-quiz-winner
    stem: mk8-quiz
    topic: "Memory"
    difficulty: 3
    q: "Which house had the highest total for the two rounds?"
    options: ["Ruby", "Emerald", "Sapphire", "Topaz"]
    answer: B
    explanation: |
      Totals: Ruby 14 + 9 = 23, Emerald 11 + 16 = 27, Sapphire 17 + 8 = 25, Topaz 10 + 13 = 23.
      **Emerald** won. Sapphire had the best single round, but not the best total.

  - id: mk8-changed-signs
    topic: "Numerical"
    difficulty: 2
    q: "If + means ×, × means −, − means ÷ and ÷ means +, what is the value of 18 − 3 + 4 ÷ 6 × 5?"
    options: ["19", "25", "27", "31"]
    answer: B
    explanation: |
      Change the signs first: 18 ÷ 3 × 4 + 6 − 5.
      Divide and multiply first, left to right: 18 ÷ 3 = 6, then 6 × 4 = 24. Now 24 + 6 − 5 = **25**.

  - id: mk8-dictionary-third
    topic: "Verbal"
    difficulty: 2
    q: "These words are put in dictionary order: Plantain, Planet, Plate, Planner, Plank. Which word comes **third**?"
    options: ["Planet", "Plank", "Plantain", "Planner"]
    answer: D
    explanation: |
      All five start with PLA. Plate (PLAT) comes last; the other four start with PLAN.
      After PLAN, compare the 5th letter: E (Planet), K (Plank), N (Planner), T (Plantain).
      Order: Planet, Plank, **Planner**, Plantain, Plate.

  - id: mk8-syllogism-rivers
    topic: "Analytical"
    difficulty: 2
    q: "Statements: All rivers are long. No long thing is a pond.\nConclusions: I. No river is a pond. II. Some long things are rivers.\nWhich conclusions follow from the statements?"
    options: ["Only conclusion I follows", "Only conclusion II follows", "Both I and II follow", "Neither I nor II follows"]
    answer: C
    explanation: |
      Every river is inside the "long" circle, and no pond touches that circle, so no river can be a pond: I follows.
      All the rivers are long, so at least some long things are rivers: II follows too. **Both I and II follow.**

  - id: mk8-turn-anticlockwise
    topic: "Visual"
    difficulty: 2
    q: "Which option shows this figure after it is turned a quarter turn (90°) **anticlockwise**?"
    figure: {"kind": "shapes", "frame": true, "transform": "none", "items": [{"t": "line", "p": [24, 14, 24, 86]}, {"t": "poly", "p": [[24, 14], [66, 26], [24, 40]], "fill": "solid"}, {"t": "rect", "p": [58, 62, 22, 22], "fill": "grey"}, {"t": "circle", "c": [42, 76], "r": 5, "fill": "none"}]}
    optionFigures:
      - {"kind": "shapes", "frame": true, "transform": "rot270", "items": [{"t": "line", "p": [24, 14, 24, 86]}, {"t": "poly", "p": [[24, 14], [66, 26], [24, 40]], "fill": "solid"}, {"t": "rect", "p": [58, 62, 22, 22], "fill": "grey"}, {"t": "circle", "c": [42, 76], "r": 5, "fill": "none"}]}
      - {"kind": "shapes", "frame": true, "transform": "rot90", "items": [{"t": "line", "p": [24, 14, 24, 86]}, {"t": "poly", "p": [[24, 14], [66, 26], [24, 40]], "fill": "solid"}, {"t": "rect", "p": [58, 62, 22, 22], "fill": "grey"}, {"t": "circle", "c": [42, 76], "r": 5, "fill": "none"}]}
      - {"kind": "shapes", "frame": true, "transform": "flipV", "items": [{"t": "line", "p": [24, 14, 24, 86]}, {"t": "poly", "p": [[24, 14], [66, 26], [24, 40]], "fill": "solid"}, {"t": "rect", "p": [58, 62, 22, 22], "fill": "grey"}, {"t": "circle", "c": [42, 76], "r": 5, "fill": "none"}]}
      - {"kind": "shapes", "frame": true, "transform": "rot180", "items": [{"t": "line", "p": [24, 14, 24, 86]}, {"t": "poly", "p": [[24, 14], [66, 26], [24, 40]], "fill": "solid"}, {"t": "rect", "p": [58, 62, 22, 22], "fill": "grey"}, {"t": "circle", "c": [42, 76], "r": 5, "fill": "none"}]}
    answer: A
    explanation: |
      In a quarter turn anticlockwise, the top of the figure goes to the left side. The flag pole, which was on the left, now lies flat along the bottom with the black flag pointing up from its left end, and the grey square moves to the top right: **A**.
      B is a quarter turn clockwise, C is a water image and D is a half turn.

  - id: mk8-magic-triangle
    topic: "Numerical"
    difficulty: 3
    q: "The numbers 1, 2, 3, 4, 5 and 6 are written in the six circles, one in each, so that the three numbers along each side of the triangle add up to 12. Which of these numbers must be in a corner circle?"
    figure: {"kind": "shapes", "items": [{"t": "poly", "p": [[50, 12], [12, 84], [88, 84]], "fill": "none"}, {"t": "circle", "c": [50, 12], "r": 9, "fill": "grey"}, {"t": "circle", "c": [12, 84], "r": 9, "fill": "grey"}, {"t": "circle", "c": [88, 84], "r": 9, "fill": "grey"}, {"t": "circle", "c": [31, 48], "r": 9, "fill": "grey"}, {"t": "circle", "c": [50, 84], "r": 9, "fill": "grey"}, {"t": "circle", "c": [69, 48], "r": 9, "fill": "grey"}], "alt": "A triangle with an empty circle at each corner and one in the middle of each side."}
    options: ["1", "2", "3", "5"]
    answer: D
    explanation: |
      Add the three side totals: 12 × 3 = 36. This adds every number once (1 + 2 + … + 6 = 21) and the three corners a second time, because each corner is on two sides.
      So the corners add up to 36 − 21 = 15. The only three different numbers from 1 to 6 with a sum of 15 are 4, 5 and 6. So **5** must be in a corner.
      One way: corners 4, 5, 6 with 3 between 4 and 5, 1 between 5 and 6, and 2 between 6 and 4.

  - id: mk8-opposite-letters
    topic: "Verbal"
    difficulty: 2
    q: "FISH : URHS :: BIRD : ?"
    options: ["YRIW", "CJSE", "XRIW", "YRJW"]
    answer: A
    explanation: |
      Each letter is swapped with its partner from the other end of the alphabet: A ↔ Z, B ↔ Y, C ↔ X, and so on. The two partners' places always add up to 27.
      F (6) ↔ U (21), I (9) ↔ R (18), S ↔ H. So BIRD → B ↔ Y, I ↔ R, R ↔ I, D ↔ W: **YRIW**.

  - id: mk8-fold-strip
    topic: "Visual"
    difficulty: 3
    q: "A square sheet of paper is folded in half from top to bottom, and then folded in half from top to bottom again. A hole is punched as shown in the last picture. Which option shows the sheet when it is opened out?"
    figure: {"kind": "shapes", "w": 328, "h": 100, "items": [{"t": "line", "p": [12, 12, 88, 12]}, {"t": "line", "p": [88, 12, 88, 88]}, {"t": "line", "p": [88, 88, 12, 88]}, {"t": "line", "p": [12, 88, 12, 12]}, {"t": "line", "p": [12.0, 50.0, 88.0, 50.0], "dash": true}, {"t": "poly", "p": [[50.0, 34.8], [52.04, 38.6], [53.76, 42.4], [54.92, 46.2], [55.32, 50.0], [54.92, 53.8], [53.76, 57.6], [52.04, 61.4]], "open": true}, {"t": "arrow", "p": [52.04, 61.4, 50.0, 65.2]}, {"t": "line", "p": [202.0, 50.0, 202, 88]}, {"t": "line", "p": [202, 88, 126, 88]}, {"t": "line", "p": [126, 88, 126.0, 50.0]}, {"t": "line", "p": [126.0, 50.0, 202.0, 50.0]}, {"t": "line", "p": [126.0, 69.0, 202.0, 69.0], "dash": true}, {"t": "poly", "p": [[164.0, 53.8], [166.04, 57.6], [167.76, 61.4], [168.92, 65.2], [169.32, 69.0], [168.92, 72.8], [167.76, 76.6], [166.04, 80.4]], "open": true}, {"t": "arrow", "p": [166.04, 80.4, 164.0, 84.2]}, {"t": "line", "p": [316.0, 69.0, 316, 88]}, {"t": "line", "p": [316, 88, 240, 88]}, {"t": "line", "p": [240, 88, 240.0, 69.0]}, {"t": "line", "p": [240.0, 69.0, 316.0, 69.0], "dash": true}, {"t": "circle", "c": [262.8, 81.92], "r": 3.6, "fill": "solid"}]}
    optionFigures:
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [1.0, 74.5, 99.0, 74.5], "dash": true, "w": 1.5}, {"t": "line", "p": [1.0, 25.5, 99.0, 25.5], "dash": true, "w": 1.5}, {"t": "circle", "c": [30.4, 91.16], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [30.4, 66.66], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [30.4, 42.16], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [30.4, 17.66], "r": 5.0, "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [1.0, 74.5, 99.0, 74.5], "dash": true, "w": 1.5}, {"t": "line", "p": [1.0, 25.5, 99.0, 25.5], "dash": true, "w": 1.5}, {"t": "circle", "c": [30.4, 91.16], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [30.4, 57.84], "r": 5.0, "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [1.0, 74.5, 99.0, 74.5], "dash": true, "w": 1.5}, {"t": "line", "p": [1.0, 25.5, 99.0, 25.5], "dash": true, "w": 1.5}, {"t": "circle", "c": [30.4, 91.16], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [30.4, 8.84], "r": 5.0, "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [1.0, 74.5, 99.0, 74.5], "dash": true, "w": 1.5}, {"t": "line", "p": [1.0, 25.5, 99.0, 25.5], "dash": true, "w": 1.5}, {"t": "circle", "c": [30.4, 91.16], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [30.4, 57.84], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [30.4, 8.84], "r": 5.0, "fill": "solid"}, {"t": "circle", "c": [30.4, 42.16], "r": 5.0, "fill": "solid"}]}
    answer: D
    explanation: |
      Undo the second fold: the hole near the bottom edge gets a partner reflected across the lower crease, just above it. Undo the first fold: both holes are reflected across the middle line into the top half.
      So the holes come in two pairs: one pair near the top and bottom edges, and one pair close to the middle line: **D**.
      A spaces the holes evenly, which is what you get by sliding instead of flipping. B and C undo only one fold.

  - id: mk8-league-teams
    topic: "Analytical"
    difficulty: 3
    q: "In a school hockey league, every team plays every other team twice: once at home and once away. 42 matches are played in all. How many teams are in the league?"
    options: ["6", "7", "8", "21"]
    answer: B
    explanation: |
      Each team plays every other team twice. With n teams, each one plays 2 × (n − 1) matches, and each match involves 2 teams, so the number of matches is n × (n − 1).
      n × (n − 1) = 42 = 7 × 6, so there are **7** teams. 21 is the number of different pairs of teams, not the number of teams.
---

This mock test mixes 15 questions from all five areas: numbers, words, memory, pictures and logic.

## How the test works

- You have **15 minutes**. The clock starts when you press Start, and the test is **submitted automatically** when the time runs out.
- There is **no negative marking**, so answer every question, even if you have to guess.
- Questions 5 to 7 are about one table of scores. You get 40 seconds to study it, and then it is hidden.
- After you submit, each question shows the right answer and a short explanation.

## Pacing

Aim for about one minute per question, and do not let any single question take more than two.

- Start with a quick pass and answer everything you are sure of. Then come back to the ones you skipped.
- For folding and logic questions, a tiny sketch on rough paper saves time and mistakes.
- When you study a table, read across each row and say the numbers to yourself. Notice the biggest and smallest values.
- With one minute left, make sure no question is blank.
