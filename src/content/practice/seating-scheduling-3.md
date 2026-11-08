---
title: "Seating and scheduling 3: timetables and matching"
description: "Put six club meetings in order across a week, and match four friends to a subject and a colour using a grid of clues."
slug: seating-scheduling-3
family: seating-scheduling
part: 3
timeMinutes: 15
stems:
  - id: ss3-clubs
    title: "The club week"
    text: "Six school clubs meet on six days, from Monday to Saturday, one club each day: **Art, Chess, Dance, Music, Quiz** and **Science**. Fill in the timetable from the clues."
    figure: { kind: table, caption: "Club timetable", header: [Mon, Tue, Wed, Thu, Fri, Sat], rows: [["?", "?", "?", "?", "?", "?"]] }
    lines:
      - "Chess meets on the day after Science."
      - "Dance meets on a later day than Music."
      - "There are exactly two days between the Art day and the Music day."
      - "Neither Art nor Music meets on Monday."
      - "Art meets on an earlier day than Science."
  - id: ss3-friends
    title: "Subjects and colours"
    text: "Asha, Kabir, Neha and Farhan each have a different favourite subject (**Maths, Science, English, History**) and a different favourite colour (**red, blue, green, yellow**). Use the grid to keep track."
    figure: { kind: table, caption: "Fill in as you go", header: [Name, Subject, Colour], rows: [[Asha, "?", "?"], [Kabir, "?", "?"], [Neha, "?", "?"], [Farhan, "?", "?"]] }
    lines:
      - "Neha likes red."
      - "The person who likes History likes red."
      - "Neither Asha nor Farhan likes Maths."
      - "The person who likes Maths likes yellow."
      - "Asha does not like English."
      - "Farhan does not like green."
questions:
  - id: ss3-thursday
    stem: ss3-clubs
    topic: Weekly timetable
    difficulty: 1
    q: "Which club meets on Thursday?"
    options: ["Science", "Chess", "Music", "Art"]
    answer: B
    explanation: |
      Art and Music are 3 days apart and neither is on Monday, so they take Tue and Fri, or Wed and Sat. Art needs Science and Chess after it, and Music needs Dance after it, so Art is Tuesday and Music Friday.
      Science and Chess then fit only on Wed and Thu, Dance is Saturday and Quiz Monday. Thursday is **Chess**.

  - id: ss3-art-day
    stem: ss3-clubs
    topic: Weekly timetable
    difficulty: 1
    q: "On which day does the Art club meet?"
    options: ["Tuesday", "Wednesday", "Thursday", "Friday"]
    answer: A
    explanation: |
      Art and Music must take Tue and Fri, or Wed and Sat. Art needs two clubs after it (Science, then Chess), and Music needs Dance after it.
      The only way is Art on **Tuesday** and Music on Friday.

  - id: ss3-monday
    stem: ss3-clubs
    topic: Weekly timetable
    difficulty: 2
    q: "Which club meets on Monday?"
    options: ["Art", "Dance", "Science", "Quiz"]
    answer: D
    explanation: |
      Art (Tue), Science (Wed), Chess (Thu), Music (Fri) and Dance (Sat) are all placed. The only club left for Monday is **Quiz**.

  - id: ss3-after-music
    stem: ss3-clubs
    topic: Weekly timetable
    difficulty: 2
    q: "How many clubs meet on days after the Music club?"
    options: ["0", "1", "2", "3"]
    answer: B
    explanation: |
      Music meets on Friday. Only Saturday comes after it, so **1** club (Dance) meets later.

  - id: ss3-quiz-dance-gap
    stem: ss3-clubs
    topic: Weekly timetable
    difficulty: 2
    q: "How many days come in between the Quiz club's day and the Dance club's day (not counting those two days)?"
    options: ["3", "4", "5", "6"]
    answer: B
    explanation: |
      Quiz is Monday and Dance is Saturday. The days between are Tuesday to Friday: **4** days.
      5 is Saturday − Monday, which counts the gap, not the days in between.

  - id: ss3-true-week
    stem: ss3-clubs
    topic: Weekly timetable
    difficulty: 3
    q: "Which of these is true?"
    options: ["Science meets on the day before Art.", "Music meets on Thursday.", "Chess meets two days after Art.", "Dance meets on the day after Chess."]
    answer: C
    explanation: |
      Week: Quiz, Art, Science, Chess, Music, Dance (Mon to Sat). Art is Tuesday and Chess is Thursday, so **Chess meets two days after Art**.
      Dance is two days after Chess, not one.

  - id: ss3-science
    stem: ss3-friends
    topic: Matching grid
    difficulty: 1
    q: "Who likes Science?"
    options: ["Asha", "Kabir", "Neha", "Farhan"]
    answer: A
    explanation: |
      Neha likes red, so she likes History. Asha and Farhan do not like Maths, so Kabir does. Asha does not like English, so **Asha** likes Science and Farhan likes English.

  - id: ss3-kabir-subject
    stem: ss3-friends
    topic: Matching grid
    difficulty: 1
    q: "Which subject does Kabir like?"
    options: ["English", "History", "Maths", "Science"]
    answer: C
    explanation: |
      History goes with red, which is Neha. Of the other three, Asha and Farhan do not like Maths, so Kabir likes **Maths**.

  - id: ss3-farhan-colour
    stem: ss3-friends
    topic: Matching grid
    difficulty: 2
    q: "What is Farhan's favourite colour?"
    options: ["Green", "Red", "Yellow", "Blue"]
    answer: D
    explanation: |
      Red is Neha's, and yellow goes with Maths, which is Kabir. Farhan and Asha share blue and green. Farhan does not like green, so his colour is **blue**.

  - id: ss3-green-subject
    stem: ss3-friends
    topic: Matching grid
    difficulty: 2
    q: "The person whose favourite colour is green likes which subject?"
    options: ["English", "History", "Maths", "Science"]
    answer: D
    explanation: |
      Farhan likes blue, so Asha likes green. Asha's subject is **Science**.

  - id: ss3-english-pair
    stem: ss3-friends
    topic: Matching grid
    difficulty: 3
    q: "Who likes English, and what is that person's favourite colour?"
    options: ["Farhan, blue", "Farhan, green", "Neha, blue", "Asha, green"]
    answer: A
    explanation: |
      The grid ends as: Asha – Science – green, Kabir – Maths – yellow, Neha – History – red, Farhan – English – blue.
      So it is **Farhan, blue**.

  - id: ss3-not-true
    stem: ss3-friends
    topic: Matching grid
    difficulty: 3
    q: "Which of these is **not** true?"
    options: ["Asha likes green.", "Kabir likes yellow.", "Farhan likes Science.", "The person who likes History likes red."]
    answer: C
    explanation: |
      Farhan likes English, not Science, so **"Farhan likes Science"** is the false one. The other three match the grid.
---

This set tests timetable and matching puzzles: fitting people or events into days or categories so that every clue holds.

## Timetables

Write the days in a row and fill them in as you go.

| Clue | Meaning |
|---|---|
| B is on the day after A | B = A + 1 |
| two days between A and B | they are 3 days apart |
| A is earlier than B | A is somewhere before B, not always just before |

Start with clues that fix a gap, such as "two days between". List the few ways the gap can fit, then cross out the ones that break another clue.

## Matching grids

When people are matched to two things (a subject and a colour), make a table with one row per person.
- A direct clue ("Neha likes red") goes in at once.
- A linking clue ("whoever likes History likes red") joins the two lists: here it tells you Neha likes History.
- When a row has only one choice left, fill it in. When a choice is left for only one person, that person gets it.

## Worked example

*Tara, Dev and Ali like tea, milk and juice, one each. Dev does not like tea. Ali likes milk.*

Ali has milk, so Dev, who does not like tea, has juice. Tara has **tea**.

## Common trap

"Two days between Monday and Thursday" is true, since Tuesday and Wednesday lie between them. Count the days in between, not the difference of the day numbers.
