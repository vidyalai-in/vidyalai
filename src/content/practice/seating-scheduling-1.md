---
title: "Seating and scheduling 1: sitting in a row"
description: "Five or six friends sit in a row facing north. Place them from the clues, then answer questions about left, right, ends and who sits between."
slug: seating-scheduling-1
family: seating-scheduling
part: 1
timeMinutes: 15
stems:
  - id: ss1-bench
    title: "Five on a bench"
    text: "Asha, Ravi, Meera, Kabir and Neha sit in a row on a bench. They all **face north**, so each person's left hand points to the west end of the bench and their right hand to the east end."
    lines:
      - "Meera sits exactly in the middle."
      - "Ravi sits to the immediate right of Meera."
      - "Asha sits second to the left of Meera."
      - "Neha does not sit next to Ravi."
  - id: ss1-six
    title: "Six at the cinema"
    text: "Arjun, Priya, Farhan, Simran, Joseph and Anjali sit in six seats in a row at the cinema. They all **face north** (towards the screen), so 'left' and 'right' are the same for all of them."
    lines:
      - "Farhan sits at one end of the row."
      - "Anjali sits next to Farhan."
      - "Exactly one person sits between Joseph and Priya."
      - "Simran sits to the immediate right of Priya."
      - "Neither Arjun nor Joseph sits at an end."
      - "Arjun does not sit next to Anjali."
questions:
  - id: ss1-right-end
    stem: ss1-bench
    topic: Five in a row
    difficulty: 1
    q: "Who sits at the right end of the bench?"
    options: ["Asha", "Meera", "Ravi", "Kabir"]
    answer: D
    explanation: |
      Number the seats 1 to 5 from the left. Meera is 3, Ravi is 4 and Asha is 1. Neha cannot be 5 (next to Ravi), so she is 2, and **Kabir** is 5, the right end.
      The order is Asha, Neha, Meera, Ravi, Kabir.

  - id: ss1-left-of-meera
    stem: ss1-bench
    topic: Five in a row
    difficulty: 1
    q: "Who sits to the immediate left of Meera?"
    options: ["Asha", "Neha", "Ravi", "Kabir"]
    answer: B
    explanation: |
      Order from left: Asha, Neha, Meera, Ravi, Kabir. The seat just left of Meera is **Neha's**. Ravi is on her right.

  - id: ss1-between
    stem: ss1-bench
    topic: Five in a row
    difficulty: 1
    q: "Who sits between Meera and Kabir?"
    options: ["Asha", "Neha", "Ravi", "Nobody"]
    answer: C
    explanation: |
      Meera is in seat 3 and Kabir in seat 5. Seat 4 between them is **Ravi**.

  - id: ss1-second-left
    stem: ss1-bench
    topic: Five in a row
    difficulty: 1
    q: "Who sits second to the left of Kabir?"
    options: ["Meera", "Neha", "Ravi", "Asha"]
    answer: A
    explanation: |
      Kabir is in seat 5. Two seats to his left is seat 3: **Meera**. Ravi is only the first to his left.

  - id: ss1-how-many-between
    stem: ss1-bench
    topic: Five in a row
    difficulty: 2
    q: "How many people sit between Asha and Ravi?"
    options: ["1", "2", "3", "4"]
    answer: B
    explanation: |
      Asha is in seat 1 and Ravi in seat 4. Between them are Neha and Meera: **2** people.
      3 is the gap 4 − 1, which counts one seat too many.

  - id: ss1-ends
    stem: ss1-bench
    topic: Five in a row
    difficulty: 2
    q: "Which two people sit at the two ends of the bench?"
    options: ["Asha and Kabir", "Asha and Ravi", "Neha and Kabir", "Meera and Kabir"]
    answer: A
    explanation: |
      Order: Asha, Neha, Meera, Ravi, Kabir. The ends are **Asha and Kabir**.

  - id: ss1-six-right-end
    stem: ss1-six
    topic: Six in a row
    difficulty: 2
    q: "Who sits at the right end of the row?"
    options: ["Farhan", "Priya", "Simran", "Joseph"]
    answer: C
    explanation: |
      If Farhan were at the right end, Anjali would be 5th, Priya and Simran 1st and 2nd, Joseph 3rd and Arjun 4th, next to Anjali. That breaks a clue. So Farhan is 1st and Anjali 2nd.
      Arjun cannot be 3rd (next to Anjali), and Priya and Simran need two seats side by side, so the row is Farhan, Anjali, Joseph, Arjun, Priya, **Simran**.

  - id: ss1-third-right
    stem: ss1-six
    topic: Six in a row
    difficulty: 2
    q: "Who sits third to the right of Anjali?"
    options: ["Joseph", "Arjun", "Priya", "Simran"]
    answer: C
    explanation: |
      Order from the left: Farhan, Anjali, Joseph, Arjun, Priya, Simran. Anjali is in seat 2, and three seats to her right is seat 5: **Priya**.
      Arjun is only second to her right.

  - id: ss1-six-between
    stem: ss1-six
    topic: Six in a row
    difficulty: 2
    q: "How many people sit between Farhan and Priya?"
    options: ["2", "3", "4", "5"]
    answer: B
    explanation: |
      Farhan is in seat 1 and Priya in seat 5. Anjali, Joseph and Arjun sit between them: **3** people.

  - id: ss1-midway
    stem: ss1-six
    topic: Six in a row
    difficulty: 3
    q: "Who sits exactly halfway between Anjali and Simran, with the same number of people on each side?"
    options: ["Farhan", "Joseph", "Priya", "Arjun"]
    answer: D
    explanation: |
      Anjali is in seat 2 and Simran in seat 6. The middle seat is (2 + 6) ÷ 2 = 4: **Arjun**, with Joseph on one side and Priya on the other.

  - id: ss1-swap
    stem: ss1-six
    topic: Six in a row
    difficulty: 3
    q: "Farhan and Simran swap seats. Who now sits to the immediate right of Simran?"
    options: ["Farhan", "Anjali", "Joseph", "Priya"]
    answer: B
    explanation: |
      Before the swap: Farhan, Anjali, Joseph, Arjun, Priya, Simran. After it, Simran is in seat 1 and Farhan in seat 6.
      The seat to Simran's right is seat 2: **Anjali**. Priya would be the answer for Farhan's left, not Simran's right.

  - id: ss1-true-statement
    stem: ss1-six
    topic: Six in a row
    difficulty: 3
    q: "Which of these is true?"
    options: ["Arjun sits to the immediate left of Joseph.", "Anjali sits at one end.", "Priya sits between Arjun and Joseph.", "Joseph sits second to the left of Priya."]
    answer: D
    explanation: |
      Order: Farhan, Anjali, Joseph, Arjun, Priya, Simran. Joseph is seat 3 and Priya seat 5, so **Joseph sits second to the left of Priya**.
      Arjun is to the immediate **right** of Joseph, and it is Arjun who sits between Joseph and Priya.
---

This set tests row arrangements: placing people in a line from clues, then reading off positions.

## Set up the row

Draw the seats as numbered boxes, 1 on the left. When everyone faces **north**, their left is the west end, so their left and yours match when you draw north at the top.

| Clue | Meaning with seats |
|---|---|
| X is to the immediate right of Y | X = Y + 1 |
| X is second to the left of Y | X = Y − 2 |
| One person sits between X and Y | they are 2 seats apart |
| X is at an end | seat 1 or the last seat |
| X is next to Y | 1 seat apart, either side |

If people faced **south**, their left would be on *your* right. Always check which way they face.

## The method

1. Place the fixed clues first: middle seat, ends, exact distances from a placed person.
2. Then use "next to" and "not next to" clues. If there are two choices, write both rows and test each.
3. Cross out any row that breaks a clue. Exactly one should remain.

## Worked example

*Dev, Tara, Ali and Bina sit in a row facing north. Dev is at the left end. Tara is second to the right of Dev. Bina is not next to Dev.*

Dev is seat 1 and Tara is seat 3. Bina cannot be seat 2 (next to Dev), so she is seat 4 and Ali is seat 2. The row is **Dev, Ali, Tara, Bina**.

## Common trap

"How many people sit between seats 1 and 4?" is 2, not 3. Count the people strictly between, not the gap.
