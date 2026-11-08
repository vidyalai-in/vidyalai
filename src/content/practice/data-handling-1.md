---
title: "Data handling 1: tables and bar charts"
description: "Read values from bar charts and tables, then find totals, differences, averages, percentages and how many items pass a mark."
slug: data-handling-1
family: data-handling
part: 1
timeMinutes: 12
stems:
  - id: dh1-library
    title: "Books borrowed from the school library"
    text: "The bar chart shows how many books the students of a school borrowed from the library in each month from January to June."
    figure: { kind: bars, title: "Books borrowed", labels: [Jan, Feb, Mar, Apr, May, Jun], values: [40, 64, 36, 80, 50, 30], unit: "Books" }
  - id: dh1-marks
    title: "Class test marks"
    text: "Five friends wrote tests in three subjects. Each test was out of 100."
    figure:
      kind: table
      caption: "Marks out of 100"
      header: [Name, Maths, Science, English]
      rows:
        - [Asha, 78, 85, 72]
        - [Ravi, 92, 70, 81]
        - [Meera, 85, 88, 79]
        - [Kabir, 66, 74, 90]
        - [Neha, 88, 91, 70]
  - id: dh1-runs
    title: "Arjun's cricket scores"
    text: "The bar chart shows the runs Arjun scored in six school cricket matches. He was out for 0 (a duck) in match 4."
    figure: { kind: bars, title: "Runs scored", labels: [M1, M2, M3, M4, M5, M6], values: [45, 12, 63, 0, 36, 54], unit: "Runs" }
questions:
  - id: dh1-lib-april-march
    stem: dh1-library
    topic: Reading a chart
    difficulty: 1
    q: "How many more books were borrowed in April than in March?"
    options: ["36", "44", "46", "116"]
    answer: B
    explanation: |
      April shows 80 and March shows 36. Difference = 80 − 36 = **44**.
      116 adds the two bars instead of subtracting.

  - id: dh1-lib-total
    stem: dh1-library
    topic: Totals and averages
    difficulty: 1
    q: "How many books were borrowed in all six months together?"
    options: ["240", "260", "270", "300"]
    answer: D
    explanation: |
      40 + 64 + 36 + 80 + 50 + 30 = **300**.
      260 leaves out January and 270 leaves out June.

  - id: dh1-lib-above-average
    stem: dh1-library
    topic: Totals and averages
    difficulty: 2
    q: "In how many months were MORE books borrowed than the average for the six months?"
    options: ["2", "3", "4", "5"]
    answer: A
    explanation: |
      Average = 300 ÷ 6 = 50 books a month. Only February (64) and April (80) are above 50, so the answer is **2**.
      May has exactly 50, which equals the average but is not more than it, so 3 is wrong.

  - id: dh1-lib-fall-percent
    stem: dh1-library
    topic: Percentages
    difficulty: 3
    q: "By what percentage did the number of books borrowed fall from May to June?"
    options: ["20%", "40%", "60%", "about 66.7%"]
    answer: B
    explanation: |
      The fall is 50 − 30 = 20 books. Percentage fall = 20 ÷ 50 × 100 = **40%**, since you always divide by the old value (May).
      About 66.7% divides by the new value (20 ÷ 30), and 60% is June as a percentage of May.

  - id: dh1-marks-top
    stem: dh1-marks
    topic: Reading a table
    difficulty: 1
    q: "Who has the highest total in the three subjects?"
    options: ["Asha", "Ravi", "Neha", "Meera"]
    answer: D
    explanation: |
      Totals: Asha 235, Ravi 243, Meera 85 + 88 + 79 = 252, Kabir 230, Neha 249. **Meera** is highest.
      Neha has the single highest Science mark (91), but her total is 3 marks less than Meera's.

  - id: dh1-marks-maths-average
    stem: dh1-marks
    topic: Totals and averages
    difficulty: 2
    q: "What is the average Maths mark of the five friends?"
    options: ["80.2", "81.6", "81.8", "85"]
    answer: C
    explanation: |
      Maths total = 78 + 92 + 85 + 66 + 88 = 409. Average = 409 ÷ 5 = **81.8**.
      81.6 is the Science average, and 85 is just the middle mark when they are put in order.

  - id: dh1-marks-two-subjects
    stem: dh1-marks
    topic: Reading a table
    difficulty: 2
    q: "How many of the friends scored more than 80 in at least two subjects?"
    options: ["2", "3", "4", "5"]
    answer: B
    explanation: |
      Check each row: Ravi (92, 81), Meera (85, 88) and Neha (88, 91) each have two marks above 80. Asha has only 85 and Kabir only 90.
      So the answer is **3**. Meera's 79 in English does not count, but she already has two.

  - id: dh1-marks-kabir-maths
    stem: dh1-marks
    topic: Totals and averages
    difficulty: 3
    q: "Kabir retakes only the Maths test. How many marks more than before must he score in Maths so that his average for the three subjects becomes 80?"
    options: ["10", "14", "20", "30"]
    answer: A
    explanation: |
      An average of 80 in 3 subjects needs a total of 80 × 3 = 240. Kabir's total now is 66 + 74 + 90 = 230.
      He needs **10** more marks, so a Maths score of 76. 14 is 80 − 66, which wrongly makes the Maths mark itself 80.

  - id: dh1-runs-over-40
    stem: dh1-runs
    topic: Reading a chart
    difficulty: 1
    q: "In how many matches did Arjun score more than 40 runs?"
    options: ["2", "3", "4", "5"]
    answer: B
    explanation: |
      Scores above 40: 45 (M1), 63 (M3) and 54 (M6). That is **3** matches. 36 in M5 is below 40.

  - id: dh1-runs-match-share
    stem: dh1-runs
    topic: Percentages
    difficulty: 2
    q: "What percentage of Arjun's total runs did he score in match 3?"
    options: ["6.3%", "21%", "30%", "63%"]
    answer: C
    explanation: |
      Total = 45 + 12 + 63 + 0 + 36 + 54 = 210. Share = 63 ÷ 210 × 100 = **30%**.
      63% treats the 63 runs as if the total were 100.

  - id: dh1-runs-without-lowest
    stem: dh1-runs
    topic: Totals and averages
    difficulty: 2
    q: "If Arjun's lowest score is left out, what is his average for the other five matches?"
    options: ["35", "39.6", "42", "52.5"]
    answer: C
    explanation: |
      His lowest score is 0 (match 4), so the total stays 210 but is shared by 5 matches: 210 ÷ 5 = **42**.
      35 still divides by 6. 39.6 leaves out the 12 instead of the 0.

  - id: dh1-runs-seventh
    stem: dh1-runs
    topic: Totals and averages
    difficulty: 3
    q: "Arjun plays a 7th match. After it, his average for all 7 matches is exactly 40. How many runs did he score in the 7th match?"
    options: ["35", "40", "45", "70"]
    answer: D
    explanation: |
      An average of 40 over 7 matches means a total of 40 × 7 = 280. Before the 7th match he had 210.
      So he scored 280 − 210 = **70**. 40 forgets that the new score must also lift the old average of 35.
---

This set tests how well you read a bar chart or a table and then work with the numbers you find.

## Read first, then calculate

- Read the **title, labels and unit** before anything else. Check whether a bar shows books, runs or marks.
- On these charts each bar has its value printed on top. Use the printed value, not a guess from the scale.
- In a table, go along a **row** for one person and down a **column** for one subject.

## The calculations you need

| You want | Do this |
|---|---|
| Total | add every value |
| Difference | larger − smaller |
| Average | total ÷ number of values |
| Percentage of a total | part ÷ total × 100 |
| Percentage rise or fall | change ÷ **old** value × 100 |

To find a missing value from an average, turn the average back into a total: **total = average × count**.

## Worked example

*Neha's scores in four quizzes are 6, 9, 7 and 8. What must she score in a fifth quiz to make her average 8?*

An average of 8 over 5 quizzes needs a total of 8 × 5 = 40. She has 6 + 9 + 7 + 8 = 30, so she needs **10**.

## Common traps

- "More than" the average does not include a value that is **equal** to it.
- A score of 0 is still a value. It counts when you find an average, and it can be the lowest score.
- For a percentage change, divide by where you **started**, not where you ended.
