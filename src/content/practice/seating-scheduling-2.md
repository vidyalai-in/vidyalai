---
title: "Seating and scheduling 2: around a round table"
description: "Six or eight friends sit around a round table facing the centre. Work out who sits opposite, beside, and to whose left or right."
slug: seating-scheduling-2
family: seating-scheduling
part: 2
timeMinutes: 15
stems:
  - id: ss2-six
    title: "Six at a round table"
    text: "Vikram, Neha, Joseph, Anjali, Ravi and Priya sit around a round table, evenly spaced. Everyone **faces the centre**. 'Left' and 'right' always mean the person's own left and right as they sit facing the centre."
    lines:
      - "Anjali sits opposite Vikram."
      - "Neha sits to the immediate left of Vikram."
      - "Priya sits next to Vikram."
      - "Joseph does not sit next to Priya."
  - id: ss2-eight
    title: "Eight at a round table"
    text: "Asha, Farhan, Meera, Joseph, Simran, Kabir, Neha and Arjun sit around a round table, evenly spaced. Everyone **faces the centre**. 'Left' and 'right' always mean the person's own left and right as they sit facing the centre."
    lines:
      - "Simran sits opposite Asha."
      - "Kabir sits to the immediate left of Simran."
      - "Farhan sits between Asha and Meera, next to both of them."
      - "Joseph sits opposite Arjun."
      - "Arjun sits next to Asha."
questions:
  - id: ss2-opp-neha
    stem: ss2-six
    topic: Six around a table
    difficulty: 1
    q: "Who sits opposite Neha?"
    options: ["Joseph", "Anjali", "Ravi", "Priya"]
    answer: C
    explanation: |
      Going clockwise (which is each person's left when they face the centre): Vikram, Neha, Joseph, Anjali, Ravi, Priya.
      Neha is three seats from **Ravi** either way, so Ravi is opposite her.

  - id: ss2-right-vikram
    stem: ss2-six
    topic: Six around a table
    difficulty: 1
    q: "Who sits to the immediate right of Vikram?"
    options: ["Neha", "Priya", "Ravi", "Anjali"]
    answer: B
    explanation: |
      Neha is on Vikram's left, and Priya is next to Vikram, so she must be on his right: **Priya**.

  - id: ss2-common-neighbour
    stem: ss2-six
    topic: Six around a table
    difficulty: 1
    q: "Who sits next to both Neha and Anjali?"
    options: ["Joseph", "Vikram", "Ravi", "Priya"]
    answer: A
    explanation: |
      Joseph cannot sit next to Priya, so he takes the seat after Neha, not the one beside Priya. That puts **Joseph** between Neha and Anjali.

  - id: ss2-second-right-anjali
    stem: ss2-six
    topic: Six around a table
    difficulty: 2
    q: "Who sits second to the right of Anjali?"
    options: ["Vikram", "Joseph", "Priya", "Neha"]
    answer: D
    explanation: |
      Clockwise order: Vikram, Neha, Joseph, Anjali, Ravi, Priya. Right means anticlockwise, so from Anjali go back two seats: Joseph, then **Neha**.
      Priya is second to Anjali's **left**.

  - id: ss2-neighbours-ravi
    stem: ss2-six
    topic: Six around a table
    difficulty: 2
    q: "Who are Ravi's two neighbours?"
    options: ["Anjali and Priya", "Joseph and Anjali", "Priya and Vikram", "Joseph and Priya"]
    answer: A
    explanation: |
      Clockwise order: Vikram, Neha, Joseph, Anjali, Ravi, Priya. Ravi sits between **Anjali and Priya**.

  - id: ss2-third-left-joseph
    stem: ss2-six
    topic: Six around a table
    difficulty: 2
    q: "Who sits third to the left of Joseph?"
    options: ["Vikram", "Anjali", "Ravi", "Priya"]
    answer: D
    explanation: |
      At a table of six, the third seat either way is the one directly opposite. Opposite Joseph is **Priya**.
      Count it: from Joseph go left (clockwise) to Anjali, Ravi, Priya.

  - id: ss2-opp-kabir
    stem: ss2-eight
    topic: Eight around a table
    difficulty: 1
    q: "Who sits opposite Kabir?"
    options: ["Asha", "Farhan", "Meera", "Neha"]
    answer: B
    explanation: |
      Number the seats clockwise with Asha at 0. Simran is 4 and Kabir 5. Farhan and Arjun take the two seats beside Asha. If Arjun took seat 1, Joseph would be opposite him at 5, which is Kabir's seat.
      So Arjun is 7, Joseph 3, Farhan 1, Meera 2 and Neha 6. Opposite Kabir is **Farhan**.

  - id: ss2-left-meera
    stem: ss2-eight
    topic: Eight around a table
    difficulty: 2
    q: "Who sits to the immediate left of Meera?"
    options: ["Farhan", "Simran", "Joseph", "Asha"]
    answer: C
    explanation: |
      Clockwise: Asha, Farhan, Meera, Joseph, Simran, Kabir, Neha, Arjun. Facing the centre, your left is the next seat clockwise, so Meera's left is **Joseph**.
      Farhan is on her right.

  - id: ss2-neighbours-neha
    stem: ss2-eight
    topic: Eight around a table
    difficulty: 2
    q: "Who are Neha's two neighbours?"
    options: ["Simran and Kabir", "Kabir and Asha", "Arjun and Asha", "Kabir and Arjun"]
    answer: D
    explanation: |
      Clockwise: Asha, Farhan, Meera, Joseph, Simran, Kabir, Neha, Arjun. Neha is the only one left for the seat between **Kabir and Arjun**.

  - id: ss2-third-right-kabir
    stem: ss2-eight
    topic: Eight around a table
    difficulty: 3
    q: "Who sits third to the right of Kabir?"
    options: ["Meera", "Joseph", "Farhan", "Arjun"]
    answer: A
    explanation: |
      Right means anticlockwise. From Kabir go back three seats: Simran, Joseph, **Meera**.
      Going the wrong way (Neha, Arjun, Asha) is the usual slip.

  - id: ss2-second-left-opposite
    stem: ss2-eight
    topic: Eight around a table
    difficulty: 3
    q: "Who sits second to the left of the person who sits opposite Meera?"
    options: ["Arjun", "Asha", "Farhan", "Kabir"]
    answer: B
    explanation: |
      Opposite Meera is Neha. From Neha go two seats clockwise (her left): Arjun, then **Asha**.

  - id: ss2-true-eight
    stem: ss2-eight
    topic: Eight around a table
    difficulty: 3
    q: "Which of these is true?"
    options: ["Joseph sits to the immediate right of Meera.", "Kabir sits second to the right of Joseph.", "Neha sits opposite Meera.", "Arjun sits next to both Neha and Farhan."]
    answer: C
    explanation: |
      Clockwise: Asha, Farhan, Meera, Joseph, Simran, Kabir, Neha, Arjun. Meera and Neha are four seats apart, so **Neha sits opposite Meera**.
      Joseph is on Meera's left, and Kabir is second to Joseph's left, not right.
---

This set tests circular seating: people around a round table, all facing the centre.

## Left and right at a round table

Draw the table as a circle seen from above. When a person faces the **centre**:
- their **left** is the next seat **clockwise**;
- their **right** is the next seat **anticlockwise**.

Check it by standing at the bottom of the circle and facing up to the centre: your left hand points to the seat on the clockwise side.

| Clue | What to do |
|---|---|
| X is to the immediate left of Y | X is one seat clockwise from Y |
| X is second to the right of Y | two seats anticlockwise from Y |
| X sits opposite Y | half the table away (3 seats at 6, 4 seats at 8) |
| X sits next to Y | either side |

## The method

1. A circle has no first seat, so put one person at the top and place others relative to them.
2. Use "opposite" and "immediate left/right" clues first, since they fix a seat exactly.
3. Use "next to" and "not next to" last. If there are two choices, try both.

## Worked example

*Four friends face the centre. Dev is opposite Tara. Ali is on Dev's immediate left.*

Put Dev at the bottom and Tara at the top. Dev's left is clockwise, which from the bottom is the left-hand seat on your drawing. So Ali sits on the left side, and the fourth friend on the right.

## Common trap

Using your own left instead of the sitter's. If everyone faced **outwards**, left and right would swap.
