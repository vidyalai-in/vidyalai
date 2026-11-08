---
title: "Mock test 1: mixed reasoning"
description: "A timed 15-minute test with three questions from each area: numbers, words, memory, shapes and cubes, and logic, all mixed together."
slug: mock-tests-1
family: mock-tests
part: 1
timeMinutes: 15
stems:
  - id: mk1-board
    title: "Memory: the departure board"
    text: "Trains leaving Pune Junction today. Read each row once, then say it to yourself."
    studySeconds: 40
    figure:
      kind: table
      caption: "Departures"
      header: ["To", "Leaves at", "Platform"]
      rows:
        - ["Mumbai", "06:15", 3]
        - ["Nashik", "07:40", 1]
        - ["Goa", "09:05", 4]
        - ["Hyderabad", "11:30", 2]
        - ["Nagpur", "13:50", 3]
questions:
  - id: mk1-div-4-and-6
    topic: Numerical
    difficulty: 1
    q: "How many whole numbers from 1 to 100 can be divided exactly by both 4 and 6?"
    options: ["4", "8", "16", "33"]
    answer: B
    explanation: |
      A number divisible by both 4 and 6 is a multiple of their LCM, 12: 12, 24, …, 96. Since 100 ÷ 12 = 8 remainder 4, there are **8**.
      4 comes from using 4 × 6 = 24 instead of the LCM. 33 counts numbers divisible by 4 *or* 6.

  - id: mk1-litre-volume
    topic: Verbal
    difficulty: 1
    q: "Kilogram : Weight :: Litre : ?"
    options: ["Volume", "Length", "Milk", "Time"]
    answer: A
    explanation: |
      A kilogram is a unit for measuring weight, and a litre is a unit for measuring **volume**.
      Milk is measured in litres, but milk is a thing, not the quantity being measured.

  - id: mk1-mirror-jaipur
    topic: Visual
    difficulty: 1
    q: "A mirror is placed to the right of the text below. Which option shows its mirror image?\n**JAIPUR 27**"
    optionFigures:
      - { kind: glyphs, text: "JAIPUR 27", transform: flipV }
      - { kind: glyphs, text: "72 RUPIAJ", transform: none }
      - { kind: glyphs, text: "JAIPUR 27", transform: rot180 }
      - { kind: glyphs, text: "JAIPUR 27", transform: flipH }
    answer: D
    explanation: |
      A mirror on the right swaps left and right: the order is reversed **and** every letter is flipped sideways, while top and bottom stay where they are. That is **D**.
      A is the water image (upside down). B only reverses the order and leaves the letters normal. C is the text turned upside down by a half turn.

  - id: mk1-aunt
    topic: Analytical
    difficulty: 1
    q: "Pointing to a woman, Kabir says, 'She is the wife of my mother's only brother.' How is the woman related to Kabir?"
    options: ["Cousin", "Sister", "Grandmother", "Aunt"]
    answer: D
    explanation: |
      Kabir's mother's brother is Kabir's uncle (his mama). The uncle's wife is Kabir's **aunt** (his mami).
      A cousin would be the uncle's child, not his wife.

  - id: mk1-code-plant
    topic: Verbal
    difficulty: 2
    q: "In a code language, PLANT is written as QKBMU. How is GREEN written in that code?"
    options: ["HQFDO", "HSFFO", "HQFDM", "FSDFM"]
    answer: A
    explanation: |
      P→Q (+1), L→K (−1), A→B (+1), N→M (−1), T→U (+1): the letters go +1, −1, +1, −1, +1.
      GREEN: G+1 = H, R−1 = Q, E+1 = F, E−1 = D, N+1 = O, giving **HQFDO**. HSFFO adds 1 to every letter.

  - id: mk1-two-discounts
    topic: Numerical
    difficulty: 2
    q: "A school bag is priced at ₹800. The shop gives 20% off, and then a further 10% off the reduced price. What does the bag cost now?"
    options: ["₹560", "₹576", "₹640", "₹720"]
    answer: B
    explanation: |
      After 20% off: 800 × 0.8 = ₹640. After a further 10% off: 640 × 0.9 = **₹576**.
      ₹560 wrongly adds the two discounts to make 30%. The second 10% is taken from ₹640, not from ₹800.

  - id: mk1-dice-letters
    topic: Visual
    difficulty: 2
    q: "Three views of the same cube are shown. Which letter is opposite K?"
    figure:
      kind: dice
      cubes:
        - { top: J, front: K, right: H }
        - { top: L, front: M, right: J }
        - { top: N, front: K, right: M }
      labels: ["(i)", "(ii)", "(iii)"]
    options: ["H", "J", "L", "M"]
    answer: C
    explanation: |
      Faces seen together are neighbours. K is seen with J and H in view (i), and with N and M in view (iii). That makes four neighbours: H, J, M, N.
      The only letter never seen with K is **L**, so L is opposite K. View (ii) does not show K, so it does not help here.

  - id: mk1-row-middle
    topic: Analytical
    difficulty: 2
    q: "In a row of children, Simran is 12th from the left end and 20th from the right end. Joseph stands exactly in the middle of the row. How many children stand between Simran and Joseph?"
    options: ["2", "3", "4", "8"]
    answer: B
    explanation: |
      Children in the row = 12 + 20 − 1 = 31, so the middle child is 16th from the left. That is Joseph.
      Between 12th and 16th are the 13th, 14th and 15th: **3** children. 4 is the gap 16 − 12, which counts Joseph too.

  - id: mk1-board-goa
    stem: mk1-board
    topic: Memory
    difficulty: 2
    q: "From which platform did the train to Goa leave?"
    options: ["1", "2", "3", "4"]
    answer: D
    explanation: "The Goa train left at 09:05 from platform **4**."

  - id: mk1-board-1130
    stem: mk1-board
    topic: Memory
    difficulty: 2
    q: "Where was the 11:30 train going?"
    options: ["Hyderabad", "Nashik", "Goa", "Nagpur"]
    answer: A
    explanation: "The 11:30 train was going to **Hyderabad**, from platform 2. The Nagpur train was the last one, at 13:50."

  - id: mk1-board-platform-3
    stem: mk1-board
    topic: Memory
    difficulty: 3
    q: "How many of the five trains left from platform 3?"
    options: ["1", "2", "3", "4"]
    answer: B
    explanation: "**2** trains used platform 3: the first one (Mumbai, 06:15) and the last one (Nagpur, 13:50). Noticing a repeat like this while you study makes it easy to recall."

  - id: mk1-calendar-nov
    topic: Numerical
    difficulty: 2
    q: "Independence Day, 15 August 2026, is a Saturday. On which day of the week is Children's Day, 14 November 2026?"
    options: ["Thursday", "Friday", "Saturday", "Sunday"]
    answer: C
    explanation: |
      Count the days after 15 August: 16 more in August, 30 in September, 31 in October and 14 in November. 16 + 30 + 31 + 14 = 91.
      91 = 13 × 7, which is exactly 13 weeks, so it is the same day again: **Saturday**. Sunday comes from counting 15 August as well (92 days).

  - id: mk1-cricket-sorted
    topic: Verbal
    difficulty: 3
    q: "The letters of the word CRICKET are rearranged in alphabetical order. How many letters are still in the same place as before?"
    options: ["1", "2", "3", "4"]
    answer: C
    explanation: |
      CRICKET has 7 letters. In alphabetical order they are C C E I K R T.
      Compare place by place with C R I C K E T: the 1st (C), the 5th (K) and the 7th (T) match, so **3** letters stay put. Don't forget the 1st letter: CRICKET starts with C, and so does the sorted list.

  - id: mk1-painted-125
    topic: Visual
    difficulty: 3
    q: "A cube is painted on all its faces and then cut into 125 equal small cubes (5 × 5 × 5). How many small cubes have paint on at most one face?"
    figure: { kind: cubegrid, n: 5 }
    options: ["27", "54", "81", "98"]
    answer: C
    explanation: |
      *At most one* means no paint or exactly one painted face.
      No paint: the inside block, 3 × 3 × 3 = 27. Exactly one face: 3 × 3 = 9 middle squares on each of the 6 faces, so 54.
      Total 27 + 54 = **81**. 98 counts the cubes with at least one painted face (125 − 27).

  - id: mk1-walk-back
    topic: Analytical
    difficulty: 3
    q: "Priya walks 5 km east. She turns left and walks 3 km, turns left again and walks 9 km, and then turns left once more and walks 6 km. Where is she now, compared with her starting point?"
    options: ["5 km south-west", "5 km north-east", "5 km north-west", "7 km south-west"]
    answer: A
    explanation: |
      Her turns take her east 5, north 3, west 9, south 6.
      East–west: 5 − 9 = 4 km west. North–south: 3 − 6 = 3 km south. A 3–4–5 triangle gives **5 km south-west**.
      7 km adds the 4 and the 3 instead of using the straight line.
---

This is a timed mock test: **15 questions in 15 minutes**, three from each of the five areas (numbers, words, memory, shapes and cubes, and logic), all mixed together.

## How it works

The clock starts when you press Start. When the time runs out, the test is **submitted automatically**, so answer every question you can. There is **no negative marking**: a wrong answer costs nothing, so never leave a question blank.

## Pacing

- Aim for about **one minute per question**. If a question is taking much longer, skip it and come back at the end.
- Easy marks later in the test count as much as hard ones now, so don't get stuck.
- The memory block takes about a minute: 40 seconds to study a departure board, then three quick questions. Look for patterns while you study, such as a number that appears twice.
- Keep the last minute or two for the questions you skipped. If you still can't solve one, make your best guess.
