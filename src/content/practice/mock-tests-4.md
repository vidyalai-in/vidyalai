---
title: "Mock test 4: mixed reasoning"
description: "A fourth timed 15-minute test: ratio, series and a slow clock, word making and coded words, a timetable to remember, mirror images, cube piles, a rolling die, syllogisms and a row puzzle."
slug: mock-tests-4
family: mock-tests
part: 4
timeMinutes: 15
stems:
  - id: mk4-timetable
    title: "Memory: Class 8B's Wednesday timetable"
    text: "Remember the order of the periods and who teaches each one."
    studySeconds: 35
    lines:
      - "Period 1: Maths, Mr Joseph"
      - "Period 2: Science, Ms Anjali"
      - "Period 3: Hindi, Mr Vikram"
      - "Break"
      - "Period 4: Art, Ms Neha"
      - "Period 5: English, Ms Simran"
      - "Period 6: Games, Mr Farhan"
questions:
  - id: mk4-ratio-share
    topic: Numerical
    difficulty: 1
    q: "Ravi and Simran share ₹450 in the ratio 4 : 5. How much does Simran get?"
    options: ["₹200", "₹225", "₹250", "₹360"]
    answer: C
    explanation: |
      There are 4 + 5 = 9 equal parts, so one part is 450 ÷ 9 = ₹50. Simran gets 5 parts: **₹250**.
      ₹200 is Ravi's share. ₹360 wrongly takes 4/5 of ₹450.

  - id: mk4-mirror-boat
    topic: Visual
    difficulty: 1
    q: "A mirror is placed to the right of this picture of a boat. Which option is its mirror image?"
    figure: { kind: shapes, items: [{ t: poly, p: [[12, 70], [88, 70], [76, 86], [24, 86]], fill: grey }, { t: line, p: [40, 12, 40, 70] }, { t: poly, p: [[40, 16], [40, 62], [74, 62]], fill: none }, { t: poly, p: [[40, 12], [24, 18], [40, 24]], fill: solid }], alt: "A boat with a grey hull. Its mast is left of centre, the sail is on the right of the mast and a small flag at the top points left." }
    optionFigures:
      - { kind: shapes, transform: flipH, items: [{ t: poly, p: [[12, 70], [88, 70], [76, 86], [24, 86]], fill: grey }, { t: line, p: [40, 12, 40, 70] }, { t: poly, p: [[40, 16], [40, 62], [74, 62]], fill: none }, { t: poly, p: [[40, 12], [24, 18], [40, 24]], fill: solid }] }
      - { kind: shapes, transform: flipV, items: [{ t: poly, p: [[12, 70], [88, 70], [76, 86], [24, 86]], fill: grey }, { t: line, p: [40, 12, 40, 70] }, { t: poly, p: [[40, 16], [40, 62], [74, 62]], fill: none }, { t: poly, p: [[40, 12], [24, 18], [40, 24]], fill: solid }] }
      - { kind: shapes, transform: rot180, items: [{ t: poly, p: [[12, 70], [88, 70], [76, 86], [24, 86]], fill: grey }, { t: line, p: [40, 12, 40, 70] }, { t: poly, p: [[40, 16], [40, 62], [74, 62]], fill: none }, { t: poly, p: [[40, 12], [24, 18], [40, 24]], fill: solid }] }
      - { kind: shapes, transform: rot90, items: [{ t: poly, p: [[12, 70], [88, 70], [76, 86], [24, 86]], fill: grey }, { t: line, p: [40, 12, 40, 70] }, { t: poly, p: [[40, 16], [40, 62], [74, 62]], fill: none }, { t: poly, p: [[40, 12], [24, 18], [40, 24]], fill: solid }] }
    answer: A
    explanation: |
      A mirror on the right swaps left and right but keeps top and bottom. The boat stays upright, the sail moves to the left of the mast and the flag now points right: **A**.
      B is the water image (upside down), C is a half turn and D is a quarter turn.

  - id: mk4-expand-opposite
    topic: Verbal
    difficulty: 1
    q: "Choose the word most nearly opposite in meaning to EXPAND."
    options: ["Grow", "Shrink", "Spread", "Open"]
    answer: B
    explanation: |
      To *expand* is to become bigger. The opposite is to **shrink**, to become smaller.
      Grow and spread mean nearly the same as expand.

  - id: mk4-cousin-isha
    topic: Analytical
    difficulty: 2
    q: "Arjun is Meena's son. Meena is the only daughter of Gopal. Gopal's son Suresh has a daughter, Isha. How is Isha related to Arjun?"
    options: ["Sister", "Niece", "Aunt", "Cousin"]
    answer: D
    explanation: |
      Meena and Suresh are both Gopal's children, so Suresh is Meena's brother and Arjun's uncle.
      Isha is the uncle's daughter, so she is Arjun's **cousin**.

  - id: mk4-series-double-two
    topic: Numerical
    difficulty: 2
    q: "What comes next? 1, 4, 10, 22, 46, ?"
    options: ["70", "92", "94", "96"]
    answer: C
    explanation: |
      Each number is double the one before, plus 2: 1 × 2 + 2 = 4, 4 × 2 + 2 = 10, 10 × 2 + 2 = 22, 22 × 2 + 2 = 46.
      Next: 46 × 2 + 2 = **94**. (The gaps 3, 6, 12, 24 double too, so the next gap is 48.) 92 forgets the +2.

  - id: mk4-celebration
    topic: Verbal
    difficulty: 2
    q: "Which word can NOT be made from the letters of CELEBRATION? Use each letter only as many times as it appears in CELEBRATION."
    options: ["CARTOON", "RELATION", "BRACELET", "NOTABLE"]
    answer: A
    explanation: |
      CELEBRATION has C, E, L, E, B, R, A, T, I, O, N: two Es, but only one O.
      **CARTOON** needs two Os, so it can't be made. BRACELET needs two Es, and CELEBRATION has two, so it is fine.

  - id: mk4-tt-period-2
    stem: mk4-timetable
    topic: Memory
    difficulty: 1
    q: "Which subject was in Period 2?"
    options: ["Maths", "Science", "Hindi", "Art"]
    answer: B
    explanation: "Period 2 was **Science**, with Ms Anjali."

  - id: mk4-tt-english
    stem: mk4-timetable
    topic: Memory
    difficulty: 2
    q: "Who teaches English?"
    options: ["Ms Neha", "Ms Anjali", "Mr Vikram", "Ms Simran"]
    answer: D
    explanation: "English, in Period 5, was taught by **Ms Simran**. Ms Neha taught Art just before it."

  - id: mk4-tt-after-break
    stem: mk4-timetable
    topic: Memory
    difficulty: 2
    q: "Which lesson came straight after the break, and who taught it?"
    options: ["Art, Ms Neha", "Art, Ms Simran", "English, Ms Neha", "Hindi, Mr Vikram"]
    answer: A
    explanation: "After the break came Period 4: **Art, with Ms Neha**. Hindi was the period just before the break."

  - id: mk4-stack-fill
    topic: Visual
    difficulty: 2
    q: "How many more cubes are needed to turn this pile into a solid block 4 cubes long, 2 cubes deep and 4 cubes high? (Some cubes in the pile are hidden under others.)"
    figure: { kind: stack, heights: [[4, 3, 2, 1], [3, 2, 1, 1]] }
    options: ["13", "15", "17", "32"]
    answer: B
    explanation: |
      The pile has back row 4 + 3 + 2 + 1 = 10 and front row 3 + 2 + 1 + 1 = 7, so 17 cubes.
      The full block needs 4 × 2 × 4 = 32 cubes, so **15** more are needed. 17 is the number already in the pile.

  - id: mk4-syllogism-kettles
    topic: Analytical
    difficulty: 2
    q: "Statements: All kettles are pots. All pots are heavy. Some heavy things are red.\nConclusions: I. All kettles are heavy. II. Some pots are red.\nWhich conclusions follow?"
    options: ["Only I follows", "Only II follows", "Both I and II follow", "Neither follows"]
    answer: A
    explanation: |
      Kettles sit inside pots, and pots sit inside heavy things, so every kettle is heavy: I follows.
      The red heavy things could all be outside the pots, so II need not be true. **Only I follows.**

  - id: mk4-coded-words
    topic: Verbal
    difficulty: 3
    q: "In a code language:\n'red rose blooms' is written 'pa ti no'\n'blooms in spring' is written 'zu no fe'\n'red ink spills' is written 'ti ga mo'\nThe codes in each group can be in any order. What is the code for 'rose'?"
    options: ["ti", "no", "pa", "zu"]
    answer: C
    explanation: |
      'red' is in the first and third groups, and the only code they share is ti, so red = ti.
      'blooms' is in the first and second groups, and they share only no, so blooms = no.
      That leaves **pa** for 'rose' in the first group.

  - id: mk4-die-two-rolls
    topic: Visual
    difficulty: 3
    q: "This is a standard die: opposite faces add up to 7. It is tipped over once to the right, and then tipped over once towards you. Which number is on top now?"
    figure: { kind: dice, pips: true, cubes: [{ top: 3, front: 1, right: 2 }] }
    options: ["1", "4", "5", "6"]
    answer: D
    explanation: |
      The hidden faces are bottom 7 − 3 = 4, back 7 − 1 = 6 and left 7 − 2 = 5.
      Tipping to the right brings the left face (5) to the top. The front and back don't move, so the back is still 6.
      Tipping towards you brings the back face to the top: **6**. 5 is the top after only the first tip.

  - id: mk4-slow-clock
    topic: Numerical
    difficulty: 3
    q: "A wall clock loses 3 minutes every hour. Vikram sets it to the correct time at 8:00 am. Later the same day, the clock shows 3:36 pm. What is the correct time then?"
    options: ["3:12 pm", "3:57 pm", "4:00 pm", "4:12 pm"]
    answer: C
    explanation: |
      In each real hour, the slow clock moves only 57 minutes. From 8:00 am to 3:36 pm the clock has moved 7 h 36 min = 456 minutes.
      456 ÷ 57 = 8, so 8 real hours have passed and the correct time is **4:00 pm**.
      3:57 pm adds 3 minutes for only 7 hours; 3:12 pm subtracts the lost time instead of adding it.

  - id: mk4-row-six
    topic: Analytical
    difficulty: 3
    q: "Six children sit in a row, all facing north.\nAnjali sits at the right end.\nExactly two children sit between Arjun and Joseph.\nNeha sits immediately to the left of Vikram.\nSimran sits next to Anjali.\nJoseph sits somewhere to the right of Vikram.\nWho sits third from the right end?"
    options: ["Neha", "Vikram", "Simran", "Joseph"]
    answer: D
    explanation: |
      Anjali is 6th from the left and Simran, next to her, is 5th. Arjun, Joseph, Neha and Vikram fill places 1 to 4.
      Arjun and Joseph have two children between them, so they are in places 1 and 4. Joseph is to the right of Vikram, so Joseph can't be 1st: Arjun 1st, Joseph 4th, and Neha and Vikram take 2nd and 3rd.
      Row: Arjun, Neha, Vikram, Joseph, Simran, Anjali. Third from the right is **Joseph**.
---

This is a timed mock test: **15 questions in 15 minutes**, three from each of the five areas (numbers, words, memory, shapes and cubes, and logic), mixed together.

## How it works

The timer starts when you press Start. When it runs out, the test is **submitted automatically** with the answers you have marked. There is **no negative marking**, so answer every question, even if it is a guess.

## Pacing

- Allow about **one minute per question**. Check the timer after question 5 and question 10: you should have about 10 and 5 minutes left.
- **Skip and come back.** If you can't see a way in after half a minute, move on. A fresh look later often helps.
- The memory block takes about a minute: 35 seconds to study a timetable, then three questions. Say each line to yourself as you read it, for example "two, Science, Anjali".
- For puzzles with many clues, start with the clue that fixes a position (such as "at the right end") and build out from there.
