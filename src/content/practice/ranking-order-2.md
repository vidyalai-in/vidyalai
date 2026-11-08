---
title: "Ranking and order 2: comparisons and coded inequalities"
description: "Put people or things in order from clues like taller, heavier and older, decode comparison symbols, and decide what must be true when the order is not fully known."
slug: ranking-order-2
family: ranking-order
part: 2
timeMinutes: 12
stems:
  - id: ro2-heights
    title: "Five friends, five heights"
    text: "Diya, Esha, Farhan, Gaurav and Hina all have different heights."
    lines:
      - "Esha is taller than Diya but shorter than Hina."
      - "Farhan is shorter than Diya."
      - "Gaurav is taller than Esha but shorter than Hina."
  - id: ro2-code
    title: "A weighing code"
    text: "In this code, the symbols compare weights."
    lines:
      - "P @ Q means P is heavier than Q."
      - "P # Q means P is lighter than Q."
      - "P & Q means P and Q weigh the same."
  - id: ro2-ages
    title: "Six cousins"
    text: "Ira, Jai, Kiran, Lata, Mohit and Nisha are cousins. No two of them are the same age."
    lines:
      - "Ira is older than Jai but younger than Kiran."
      - "Lata is older than Kiran."
      - "Mohit is younger than Jai."
      - "Nisha is older than Jai."
questions:
  - id: ro2-shortest
    topic: Comparisons
    difficulty: 1
    q: "Asha is taller than Ravi. Ravi is taller than Meera. Who is the shortest?"
    options: ["Asha", "Ravi", "Meera", "Cannot be found"]
    answer: C
    explanation: |
      In order from tallest: Asha, Ravi, Meera. So **Meera** is the shortest.

  - id: ro2-oldest
    topic: Comparisons
    difficulty: 1
    q: "Kabir is older than Priya but younger than Arjun. Neha is older than Arjun. Who is the oldest?"
    options: ["Kabir", "Priya", "Arjun", "Neha"]
    answer: D
    explanation: |
      From the first sentence: Arjun > Kabir > Priya. Neha is older than Arjun, so she goes at the top.
      Order: Neha, Arjun, Kabir, Priya. **Neha** is the oldest.

  - id: ro2-lightest-box
    topic: Comparisons
    difficulty: 1
    q: "There are four boxes, A, B, C and D. A is heavier than B. C is lighter than B. D is heavier than A. Which box is the lightest?"
    options: ["A", "B", "C", "D"]
    answer: C
    explanation: |
      D > A > B, and C is lighter than B. So the order from heaviest is D, A, B, C, and **C** is the lightest.

  - id: ro2-heights-tallest
    stem: ro2-heights
    topic: Ordering puzzles
    difficulty: 1
    q: "Who is the tallest?"
    options: ["Esha", "Farhan", "Gaurav", "Hina"]
    answer: D
    explanation: |
      Hina is taller than both Esha and Gaurav, and they are taller than Diya and Farhan.
      So **Hina** is the tallest.

  - id: ro2-heights-middle
    stem: ro2-heights
    topic: Ordering puzzles
    difficulty: 2
    q: "Who is the third tallest?"
    options: ["Diya", "Esha", "Gaurav", "Hina"]
    answer: B
    explanation: |
      Join the clues into one line, tallest first: Hina, Gaurav, Esha, Diya, Farhan.
      The third tallest is **Esha**, who is also in the middle of the five.

  - id: ro2-scores
    topic: Comparisons
    difficulty: 2
    q: "In a quiz, Ravi scored more than Simran but less than Joseph. Anjali scored less than Simran. Vikram scored more than Joseph. Everyone got a different score. Who got the second highest score?"
    options: ["Joseph", "Ravi", "Simran", "Vikram"]
    answer: A
    explanation: |
      Joseph > Ravi > Simran > Anjali, and Vikram > Joseph.
      Order: Vikram, Joseph, Ravi, Simran, Anjali. **Joseph** is second. Vikram is the highest.

  - id: ro2-code-must
    stem: ro2-code
    topic: Coded inequalities
    difficulty: 2
    q: "If A @ B, B & C and C @ D, which of these must be true?"
    options: ["A @ D", "D @ A", "B # D", "A & C"]
    answer: A
    explanation: |
      A is heavier than B, B weighs the same as C, and C is heavier than D. So A > B = C > D.
      That means A is heavier than D: **A @ D**. A & C is false, because A is heavier than C.

  - id: ro2-code-lightest
    stem: ro2-code
    topic: Coded inequalities
    difficulty: 2
    q: "If M # N, N # O and P @ O, which one is the lightest?"
    options: ["M", "N", "O", "P"]
    answer: A
    explanation: |
      M is lighter than N, N is lighter than O, and P is heavier than O.
      So M < N < O < P, and **M** is the lightest. P is the heaviest.

  - id: ro2-code-chain
    stem: ro2-code
    topic: Coded inequalities
    difficulty: 3
    q: "If W @ X, Y # X and Z @ Y, which of these must be true?"
    options: ["Z @ X", "W @ Y", "W @ Z", "Z & X"]
    answer: B
    explanation: |
      W > X and Y < X, so W > X > Y. That gives **W @ Y** for certain.
      Z is only compared with Y (Z > Y). Z could be heavier, lighter or the same as X or W, so the other three do not have to be true.

  - id: ro2-ages-youngest
    stem: ro2-ages
    topic: Ordering puzzles
    difficulty: 2
    q: "Who is the youngest?"
    options: ["Ira", "Jai", "Mohit", "Nisha"]
    answer: C
    explanation: |
      Lata > Kiran > Ira > Jai > Mohit, and Nisha is older than Jai. So Ira, Kiran, Lata and Nisha are all older than Jai, and Mohit is younger than Jai.
      **Mohit** is the youngest. Jai is the second youngest.

  - id: ro2-ages-must
    stem: ro2-ages
    topic: Ordering puzzles
    difficulty: 3
    q: "Which of these must be true?"
    options: ["Lata is the oldest", "Nisha is older than Kiran", "Ira is the third oldest", "Exactly four cousins are older than Jai"]
    answer: D
    explanation: |
      We only know Nisha is older than Jai. She could be anywhere above Jai, even the oldest. So the first three are not certain.
      Ira, Kiran, Lata and Nisha are all older than Jai, and Mohit is younger. So **exactly four cousins are older than Jai** is always true.

  - id: ro2-buildings
    topic: Ordering puzzles
    difficulty: 3
    q: "Five buildings A, B, C, D and E all have different heights. A is taller than only two of the others. B is taller than C but shorter than D. E is not the tallest. C is not the shortest. Which building is the second tallest?"
    options: ["A", "B", "C", "D"]
    answer: B
    explanation: |
      A is taller than exactly two, so A is 3rd. D, B and C fill three of the places 1, 2, 4 and 5 in that order, and E takes the other.
      E cannot be 1st. If E were 2nd or 4th, C would be last, which is not allowed. So E is 5th: D, B, A, C, E, and **B** is the second tallest.
---

These questions give comparisons such as "taller than" or "older than" and ask you to put people or things in order.

## Make one line

Write the names in a single line, biggest on the left and smallest on the right. Add one clue at a time:

- "X is taller than Y" puts X to the left of Y.
- "X is taller than Y but shorter than Z" gives Z, X, Y.
- If a clue does not fit the line yet, write it beside the line and come back to it.

*Example:* "Om is heavier than Raj. Sam is lighter than Raj. Tia is heavier than Om." The line is Tia, Om, Raj, Sam.

## Coded comparisons

Some questions use symbols in place of words, for example "P @ Q means P is heavier than Q". Change every symbol back into words or into >, < and = before you start. Then join the pieces: if A > B and B = C, then A > C.

## When the order is not complete

Sometimes the clues do not fix every place. Then a question may ask what **must** be true. A statement must be true only if it holds in every order that fits the clues.

- If someone is compared with only one person, they could be anywhere on one side of that person.
- Try placing that person as high as possible, then as low as possible. If a statement fails in either case, it does not have to be true.
- Things like "who is youngest" or "how many are older than X" can still be fixed even when the full order is not.
