---
title: "Mock test 5: mixed reasoning"
description: "A timed 15-minute test with 15 mixed questions: a bar chart, a magic square, codes, a memory card, counting triangles, paper folding, a talking parrot and library days."
slug: "mock-tests-5"
family: "mock-tests"
part: 5
timeMinutes: 15
stems:
  - id: mk5-trip
    title: "Memorize: the Class 7 trip to Mysuru"
    studySeconds: 40
    text: "Study the plan for the day. It will be hidden when the time is up."
    figure: {"kind": "table", "caption": "Trip plan", "header": ["Time", "Place", "Activity"], "rows": [["7:30 am", "School gate", "Board Bus 14"], ["10:15 am", "Zoo", "Bird walk"], ["12:45 pm", "Lakeside café", "Lunch"], ["2:00 pm", "Rail museum", "Toy-train ride"], ["4:30 pm", "Palace lawns", "Group photo"]]}
questions:
  - id: mk5-calendar-atlas
    topic: "Verbal"
    difficulty: 1
    q: "Calendar is to dates as atlas is to ___."
    options: ["Words", "Maps", "Recipes", "Prices"]
    answer: B
    explanation: |
      A calendar is a book or chart of dates, and an atlas is a book of **maps**.
      A dictionary, not an atlas, is a book of words.

  - id: mk5-paper-percent
    topic: "Numerical"
    difficulty: 2
    q: "The bar chart shows how many kilograms of waste paper five sections of Class 7 collected for recycling. What percentage of all the paper was collected by 7C?"
    figure: {"kind": "bars", "title": "Waste paper collected", "labels": ["7A", "7B", "7C", "7D", "7E"], "values": [45, 30, 60, 25, 40], "unit": "kg"}
    options: ["25%", "30%", "35%", "60%"]
    answer: B
    explanation: |
      Total = 45 + 30 + 60 + 25 + 40 = 200 kg. 7C collected 60 kg.
      60 ÷ 200 × 100 = **30%**. 60% reads the bar as a percentage without dividing by the total.

  - id: mk5-hari-kriti
    topic: "Analytical"
    difficulty: 1
    q: "Neel is the only son of Ramesh, and Kriti is Neel's sister. Hari is Ramesh's father. How is Hari related to Kriti?"
    options: ["Grandfather", "Father", "Uncle", "Brother"]
    answer: A
    explanation: |
      Kriti is Neel's sister, so Ramesh is her father too. Hari is her father's father: her **grandfather**.

  - id: mk5-water-house
    topic: "Visual"
    difficulty: 1
    q: "The water is below the picture. Which option is the **water image** of the picture?"
    figure: {"kind": "shapes", "frame": true, "transform": "none", "items": [{"t": "poly", "p": [[20, 50], [50, 24], [80, 50]], "fill": "none"}, {"t": "rect", "p": [62, 27, 8, 14], "fill": "grey"}, {"t": "rect", "p": [25, 50, 50, 38], "fill": "none"}, {"t": "rect", "p": [33, 66, 12, 22], "fill": "solid"}, {"t": "circle", "c": [14, 14], "r": 7, "fill": "solid"}], "alt": "A house with a sloping roof, a chimney, a door and a sun"}
    optionFigures:
      - {"kind": "shapes", "frame": true, "transform": "flipH", "items": [{"t": "poly", "p": [[20, 50], [50, 24], [80, 50]], "fill": "none"}, {"t": "rect", "p": [62, 27, 8, 14], "fill": "grey"}, {"t": "rect", "p": [25, 50, 50, 38], "fill": "none"}, {"t": "rect", "p": [33, 66, 12, 22], "fill": "solid"}, {"t": "circle", "c": [14, 14], "r": 7, "fill": "solid"}], "alt": "A house with a sloping roof, a chimney, a door and a sun"}
      - {"kind": "shapes", "frame": true, "transform": "rot180", "items": [{"t": "poly", "p": [[20, 50], [50, 24], [80, 50]], "fill": "none"}, {"t": "rect", "p": [62, 27, 8, 14], "fill": "grey"}, {"t": "rect", "p": [25, 50, 50, 38], "fill": "none"}, {"t": "rect", "p": [33, 66, 12, 22], "fill": "solid"}, {"t": "circle", "c": [14, 14], "r": 7, "fill": "solid"}], "alt": "A house with a sloping roof, a chimney, a door and a sun"}
      - {"kind": "shapes", "frame": true, "transform": "rot90", "items": [{"t": "poly", "p": [[20, 50], [50, 24], [80, 50]], "fill": "none"}, {"t": "rect", "p": [62, 27, 8, 14], "fill": "grey"}, {"t": "rect", "p": [25, 50, 50, 38], "fill": "none"}, {"t": "rect", "p": [33, 66, 12, 22], "fill": "solid"}, {"t": "circle", "c": [14, 14], "r": 7, "fill": "solid"}], "alt": "A house with a sloping roof, a chimney, a door and a sun"}
      - {"kind": "shapes", "frame": true, "transform": "flipV", "items": [{"t": "poly", "p": [[20, 50], [50, 24], [80, 50]], "fill": "none"}, {"t": "rect", "p": [62, 27, 8, 14], "fill": "grey"}, {"t": "rect", "p": [25, 50, 50, 38], "fill": "none"}, {"t": "rect", "p": [33, 66, 12, 22], "fill": "solid"}, {"t": "circle", "c": [14, 14], "r": 7, "fill": "solid"}], "alt": "A house with a sloping roof, a chimney, a door and a sun"}
    answer: D
    explanation: |
      A water image turns the picture **upside down** but keeps left and right as they are. The roof goes to the bottom, and the sun, the door and the chimney stay on the same sides as before: **D**.
      A is the mirror image (left and right swapped), B is the picture turned half a turn (which also swaps left and right) and C is a quarter turn.

  - id: mk5-magic-square
    topic: "Numerical"
    difficulty: 2
    q: "In this magic square, the three numbers in every row, every column and both diagonals add up to the same total. Which number replaces the question mark?"
    figure: {"kind": "grid", "cells": [[17, 6, 16], ["?", 13, ""], ["", 20, ""]], "shade": [[1, 0]]}
    options: ["10", "11", "12", "14"]
    answer: C
    explanation: |
      The top row gives the total: 17 + 6 + 16 = 39.
      Diagonal from the top right: 16 + 13 + (bottom left) = 39, so the bottom left is 10.
      Left column: 17 + ? + 10 = 39, so ? = **12**. 10 is the bottom-left number, not the one asked for.

  - id: mk5-trip-lunch
    stem: mk5-trip
    topic: "Memory"
    difficulty: 1
    q: "Where did the class have lunch?"
    options: ["Lakeside café", "Zoo", "Rail museum", "Palace lawns"]
    answer: A
    explanation: |
      The plan said 12:45 pm, **Lakeside café**, lunch.

  - id: mk5-trip-museum
    stem: mk5-trip
    topic: "Memory"
    difficulty: 2
    q: "At what time was the class at the rail museum?"
    options: ["12:45 pm", "2:00 pm", "2:15 pm", "4:30 pm"]
    answer: B
    explanation: |
      The rail museum (toy-train ride) was at **2:00 pm**. 12:45 pm was lunch and 4:30 pm was the group photo.

  - id: mk5-trip-gap
    stem: mk5-trip
    topic: "Memory"
    difficulty: 3
    q: "How long after boarding the bus did the bird walk start?"
    options: ["2 h 15 min", "2 h 45 min", "3 h 15 min", "3 h 45 min"]
    answer: B
    explanation: |
      The bus was at 7:30 am and the bird walk at the zoo was at 10:15 am.
      From 7:30 to 10:30 is 3 hours, so to 10:15 it is 15 minutes less: **2 h 45 min**.

  - id: mk5-parrot-day
    topic: "Analytical"
    difficulty: 2
    q: "A talking parrot, Mitthu, always lies on Mondays, Wednesdays and Fridays. On the other four days it always tells the truth. One day Mitthu says, \"Yesterday I told the truth.\" What day is it?"
    options: ["Tuesday", "Thursday", "Saturday", "Sunday"]
    answer: D
    explanation: |
      On a truthful day, yesterday must also be truthful: only Sunday works (Saturday was truthful). Tuesday, Thursday and Saturday each come after a lying day.
      On a lying day the sentence must be false, so yesterday was a lying day too. But Monday, Wednesday and Friday all come after truthful days. So it is **Sunday**.

  - id: mk5-code-glove
    topic: "Verbal"
    difficulty: 2
    q: "In a code, BRICK is written as MEKTD. How is GLOVE written in the same code?"
    options: ["GXQNI", "INQXG", "FWPMH", "GXQNJ"]
    answer: A
    explanation: |
      Each letter moves 2 places forward (B → D, R → T, I → K, C → E, K → M), giving DTKEM, and then the word is written backwards: MEKTD.
      GLOVE → INQXG → backwards **GXQNI**. INQXG forgets to reverse; FWPMH moves only 1 place.

  - id: mk5-count-triangles
    topic: "Visual"
    difficulty: 2
    q: "How many triangles are there in this figure?"
    figure: {"kind": "shapes", "items": [{"t": "line", "p": [10, 10, 90, 10]}, {"t": "line", "p": [90, 10, 90, 90]}, {"t": "line", "p": [90, 90, 10, 90]}, {"t": "line", "p": [10, 90, 10, 10]}, {"t": "line", "p": [10, 10, 90, 90]}, {"t": "line", "p": [90, 10, 10, 90]}, {"t": "line", "p": [50, 10, 50, 90]}], "alt": "A square with both diagonals and a vertical line through the middle"}
    options: ["8", "10", "12", "14"]
    answer: C
    explanation: |
      The diagonals cut the square into 4 quarter triangles, and the vertical line cuts the top and bottom quarters in half.
      Smallest triangles: 2 (top) + 2 (bottom) + 1 (left) + 1 (right) = 6. The whole top quarter and the whole bottom quarter: 2 more. Half of the square on one side of a diagonal: 4 more.
      6 + 2 + 4 = **12**. 8 is the count without the vertical line.

  - id: mk5-hash-operation
    topic: "Numerical"
    difficulty: 3
    q: "For any two numbers, a # b means a × b − a − b. For example, 5 # 3 = 15 − 5 − 3 = 7. If (n # 4) # 2 = 15, what is n?"
    options: ["5", "6", "7", "9"]
    answer: C
    explanation: |
      Work from the outside. Let m = n # 4. Then m # 2 = 2m − m − 2 = m − 2 = 15, so m = 17.
      Now n # 4 = 4n − n − 4 = 3n − 4 = 17, so 3n = 21 and n = **7**.
      Check: 7 # 4 = 28 − 11 = 17, and 17 # 2 = 34 − 19 = 15 ✓.

  - id: mk5-fewer-less
    topic: "Verbal"
    difficulty: 2
    q: "Choose the correct word for the blank.\nThere are ___ cars on our street on Sundays than on weekdays."
    options: ["less", "lesser", "little", "fewer"]
    answer: D
    explanation: |
      Cars can be counted (one car, two cars), so we use **fewer**. *Less* is for things we do not count one by one, such as less water or less noise.

  - id: mk5-fold-triangle
    topic: "Visual"
    difficulty: 3
    q: "A square sheet of paper is folded in half from right to left, and then from bottom to top. A triangle is then cut out of the folded paper, as shown in the last picture. Which option shows the sheet when it is opened out?"
    figure: {"kind": "shapes", "w": 328, "h": 100, "items": [{"t": "line", "p": [12, 12, 88, 12]}, {"t": "line", "p": [88, 12, 88, 88]}, {"t": "line", "p": [88, 88, 12, 88]}, {"t": "line", "p": [12, 88, 12, 12]}, {"t": "line", "p": [50.0, 12.0, 50.0, 88.0], "dash": true}, {"t": "poly", "p": [[65.2, 50.0], [61.4, 52.04], [57.6, 53.76], [53.8, 54.92], [50.0, 55.32], [46.2, 54.92], [42.4, 53.76], [38.6, 52.04]], "open": true}, {"t": "arrow", "p": [38.6, 52.04, 34.8, 50.0]}, {"t": "line", "p": [126, 12, 164.0, 12.0]}, {"t": "line", "p": [164.0, 12.0, 164.0, 88.0]}, {"t": "line", "p": [164.0, 88.0, 126, 88]}, {"t": "line", "p": [126, 88, 126, 12]}, {"t": "line", "p": [126.0, 50.0, 164.0, 50.0], "dash": true}, {"t": "poly", "p": [[145.0, 65.2], [147.04, 61.4], [148.76, 57.6], [149.92, 53.8], [150.32, 50.0], [149.92, 46.2], [148.76, 42.4], [147.04, 38.6]], "open": true}, {"t": "arrow", "p": [147.04, 38.6, 145.0, 34.8]}, {"t": "line", "p": [240, 12, 278.0, 12.0]}, {"t": "line", "p": [278.0, 12.0, 278.0, 50.0], "dash": true}, {"t": "line", "p": [278.0, 50.0, 240.0, 50.0], "dash": true}, {"t": "line", "p": [240.0, 50.0, 240, 12]}, {"t": "poly", "p": [[247.6, 21.12], [265.84, 21.12], [247.6, 31.76]], "fill": "solid"}]}
    optionFigures:
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [50.0, 1.0, 50.0, 99.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 50.0, 1.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[10.8, 12.76], [34.32, 12.76], [10.8, 26.48]], "fill": "solid"}, {"t": "poly", "p": [[59.8, 12.76], [83.32, 12.76], [59.8, 26.48]], "fill": "solid"}, {"t": "poly", "p": [[10.8, 61.76], [34.32, 61.76], [10.8, 75.48]], "fill": "solid"}, {"t": "poly", "p": [[59.8, 61.76], [83.32, 61.76], [59.8, 75.48]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [50.0, 1.0, 50.0, 99.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 50.0, 1.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[10.8, 12.76], [34.32, 12.76], [10.8, 26.48]], "fill": "solid"}, {"t": "poly", "p": [[10.8, 78.09], [34.32, 78.09], [10.8, 91.81]], "fill": "solid"}, {"t": "poly", "p": [[73.52, 12.76], [97.04, 12.76], [73.52, 26.48]], "fill": "solid"}, {"t": "poly", "p": [[73.52, 78.09], [97.04, 78.09], [73.52, 91.81]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [50.0, 1.0, 50.0, 99.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 50.0, 1.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[10.8, 12.76], [34.32, 12.76], [10.8, 26.48]], "fill": "solid"}, {"t": "poly", "p": [[10.8, 87.24], [34.32, 87.24], [10.8, 73.52]], "fill": "solid"}, {"t": "poly", "p": [[89.2, 12.76], [65.68, 12.76], [89.2, 26.48]], "fill": "solid"}, {"t": "poly", "p": [[89.2, 87.24], [65.68, 87.24], [89.2, 73.52]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [50.0, 1.0, 50.0, 99.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 50.0, 1.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[10.8, 12.76], [34.32, 12.76], [10.8, 26.48]], "fill": "solid"}, {"t": "poly", "p": [[10.8, 87.24], [34.32, 87.24], [10.8, 73.52]], "fill": "solid"}]}
    answer: C
    explanation: |
      Each fold works like a mirror. Undo the last fold: the triangle gets a mirror-image partner below the middle line. Undo the first fold: both get mirror-image partners on the right.
      So there are 4 triangles, one in each corner, each with its square corner pointing into a corner of the sheet: **C**.
      A slides the triangle without flipping it, B puts the holes in the right places but does not flip their shape, and D undoes only one fold.

  - id: mk5-library-days
    topic: "Analytical"
    difficulty: 3
    q: "Bhavna, Chirag, Dhruv, Esha and Faiz each visit the library on a different day from Monday to Friday. Esha goes the day after Chirag. Bhavna goes on an earlier day than Dhruv, but not on Monday. Dhruv does not go on Friday. Chirag does not go on Monday. Who goes on Wednesday?"
    options: ["Bhavna", "Chirag", "Dhruv", "Esha"]
    answer: C
    explanation: |
      Bhavna is not on Monday and is before Dhruv, who is not on Friday, so Bhavna and Dhruv are on two of Tuesday, Wednesday, Thursday.
      Chirag and Esha need two days in a row, not starting on Monday. Tuesday–Wednesday or Wednesday–Thursday would leave Bhavna and Dhruv only one of those three days, so Chirag and Esha take Thursday–Friday. That forces Bhavna on Tuesday and Dhruv on Wednesday.
      So Faiz goes on Monday, and **Dhruv** goes on Wednesday.
---

This mock test mixes 15 questions from all five areas: numbers, words, memory, pictures and logic.

## How the test works

- You have **15 minutes**. The clock starts when you press Start, and the test is **submitted automatically** when the time runs out.
- There is **no negative marking**, so answer every question. A careful guess is better than a blank.
- Questions 6 to 8 share one memory card. You get 40 seconds to study it, and then it is hidden.
- After you submit, each question shows the right answer and a short explanation.

## Pacing

Fifteen questions in fifteen minutes is about one minute each. Some take twenty seconds and some take two minutes.

- Do a quick first pass. If a question looks long, pick your best guess, note its number and move on.
- Leave about two minutes at the end to go back to the ones you marked.
- In picture questions, rule out the options that are clearly wrong first. It is often faster than working out the answer from scratch.
- Read the last line of each question twice. "Right" or "left", "opposite" or "next to": one word changes the answer.
