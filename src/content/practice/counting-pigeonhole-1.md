---
title: "Counting and pigeonhole 1: systematic counting"
description: "Handshakes, choosing and arranging, routes on a grid, and counting numbers and digits without missing any or counting twice."
slug: counting-pigeonhole-1
family: counting-pigeonhole
part: 1
timeMinutes: 12
questions:
  - id: cp1-handshakes-eight
    topic: "Handshakes and choosing"
    difficulty: 1
    q: "Eight friends meet at a party. Each of them shakes hands with every other friend exactly once. How many handshakes are there in all?"
    options: ["28", "49", "56", "64"]
    answer: A
    explanation: |
      Each of the 8 shakes hands with 7 others: 8 × 7 = 56. But every handshake joins **two** people, so it has been counted twice.
      Handshakes = 56 ÷ 2 = **28**. 56 forgets to halve.

  - id: cp1-outfits
    topic: "Arrangements"
    difficulty: 1
    q: "Ravi has 3 shirts, 4 pairs of trousers and 2 caps. An outfit is one shirt, one pair of trousers and one cap. How many different outfits can he make?"
    options: ["9", "12", "24", "36"]
    answer: C
    explanation: |
      For each choice the next choices multiply: 3 × 4 × 2 = **24**.
      9 adds the numbers instead of multiplying, and 12 forgets the caps.

  - id: cp1-photo-line
    topic: "Arrangements"
    difficulty: 1
    q: "Asha, Kabir, Meera and Joseph stand in a line for a photo. In how many different orders can they stand?"
    options: ["4", "12", "16", "24"]
    answer: D
    explanation: |
      4 choices for the first place, then 3, then 2, then 1: 4 × 3 × 2 × 1 = **24**.
      12 stops after two places, and 16 lets the same person stand twice.

  - id: cp1-page-digits
    topic: "Digit counting"
    difficulty: 1
    q: "A notebook has its pages numbered from 1 to 60. How many digits are printed in all?"
    options: ["102", "111", "120", "180"]
    answer: B
    explanation: |
      Pages 1–9 use 1 digit each: 9 digits. Pages 10–60 are 51 pages with 2 digits each: 102 digits.
      Total = 9 + 102 = **111**. 120 gives every page two digits, and 102 leaves out pages 1–9.

  - id: cp1-grid-paths
    topic: "Grid paths"
    difficulty: 2
    q: "A counter starts on square A. Each move takes it one square to the right or one square down. In how many different ways can it reach square B?"
    figure: { kind: "grid", cells: [["A", "", "", ""], ["", "", "", ""], ["", "", "", "B"]] }
    options: ["7", "10", "12", "20"]
    answer: B
    explanation: |
      Every route has 3 moves right and 2 moves down, 5 moves in all. A route is fixed once you choose which 2 of the 5 moves are 'down'.
      List them (DDRRR, DRDRR, ...) or count square by square from A, adding the number above and the number to the left: **10**.
      12 counts the squares, not the routes.

  - id: cp1-digit-sum-five
    topic: "Digit counting"
    difficulty: 2
    q: "How many 3-digit numbers have digits that add up to 5? (For example, 203 is one of them.)"
    options: ["10", "12", "15", "21"]
    answer: C
    explanation: |
      Go by the first digit. First digit 1: the other two add to 4, giving 5 ways (04, 13, 22, 31, 40). First digit 2: 4 ways. 3: 3 ways. 4: 2 ways. 5: 1 way (500).
      Total = 5 + 4 + 3 + 2 + 1 = **15**. 21 also counts 'numbers' that start with 0, such as 041, which are not 3-digit numbers.

  - id: cp1-captain-vice
    topic: "Handshakes and choosing"
    difficulty: 2
    q: "A team of 6 players must pick a captain and a different vice-captain. In how many ways can this be done?"
    options: ["11", "12", "15", "30"]
    answer: D
    explanation: |
      6 choices for captain, then 5 left for vice-captain: 6 × 5 = **30**.
      15 is the number of ways to pick two players when the order does not matter; here it does, because the two jobs are different.

  - id: cp1-sevens-written
    topic: "Digit counting"
    difficulty: 2
    q: "Priya writes all the numbers from 1 to 100. How many times does she write the digit 7?"
    options: ["10", "11", "19", "20"]
    answer: D
    explanation: |
      In the units place: 7, 17, 27, ..., 97 is 10 times. In the tens place: 70 to 79 is 10 times.
      Total = **20**. 19 counts the *numbers* that contain a 7, so 77 is counted once instead of twice.

  - id: cp1-handshakes-reverse
    topic: "Handshakes and choosing"
    difficulty: 2
    q: "At a meeting, everyone shook hands with everyone else exactly once. There were 45 handshakes in all. How many people were at the meeting?"
    options: ["8", "9", "10", "11"]
    answer: C
    explanation: |
      With n people there are n × (n − 1) ÷ 2 handshakes. Try the options: 9 people give 9 × 8 ÷ 2 = 36, and 10 people give 10 × 9 ÷ 2 = 45 ✓.
      So there were **10** people. 11 people would give 55.

  - id: cp1-grid-blocked
    topic: "Grid paths"
    difficulty: 3
    q: "A counter moves from A to B. Each move takes it one square to the right or one square down. It may not land on the shaded square marked ×. How many different routes are there?"
    figure: { kind: "grid", cells: [["A", "", "", ""], ["", "", "×", ""], ["", "", "", ""], ["", "", "", "B"]], shade: [[1, 2]] }
    options: ["9", "11", "14", "20"]
    answer: B
    explanation: |
      Without the × square there are 20 routes. Count the routes that pass through ×: 3 ways to reach it from A, and 3 ways to go on from it to B, so 3 × 3 = 9.
      Routes that avoid it = 20 − 9 = **11**. 20 ignores the blocked square, and 9 counts the routes through it.

  - id: cp1-pages-from-digits
    topic: "Digit counting"
    difficulty: 3
    q: "The pages of a book are numbered 1, 2, 3, ... to the last page. Printing all the page numbers uses 492 digits. How many pages does the book have?"
    options: ["200", "201", "246", "303"]
    answer: A
    explanation: |
      Pages 1–9 use 9 digits and pages 10–99 use 90 × 2 = 180 digits, so 189 digits take us to page 99.
      That leaves 492 − 189 = 303 digits, which is 303 ÷ 3 = 101 three-digit pages: pages 100 to 200.
      So the book has **200** pages. 303 forgets to divide by 3, and 201 adds one page too many.

  - id: cp1-no-repeat-digits
    topic: "Arrangements"
    difficulty: 3
    q: "Using only the digits 0, 1, 2, 3 and 4, how many 3-digit numbers can be made if no digit is used twice in the same number?"
    options: ["48", "60", "64", "100"]
    answer: A
    explanation: |
      The first digit can't be 0, so it has 4 choices. The second digit can be anything except the first: 4 choices (0 is allowed now). The third: 3 choices.
      4 × 4 × 3 = **48**. 60 (= 5 × 4 × 3) lets a number start with 0, 64 forgets that the third digit has only 3 choices left, and 100 lets digits repeat.
---

This set tests careful counting: making a list in a fixed order so that nothing is missed and nothing is counted twice.

## Three rules that do most of the work

| Situation | Method | Example |
|---|---|---|
| One choice, then another | **multiply** the numbers of choices | 3 shirts and 2 caps: 3 × 2 = 6 outfits |
| Arranging n things in a line | n × (n − 1) × ... × 1 | 3 people: 3 × 2 × 1 = 6 orders |
| Every pair once (handshakes, matches) | n × (n − 1) ÷ 2 | 5 players: 5 × 4 ÷ 2 = 10 matches |

When the order of the two picks matters (captain and vice-captain, gold and silver), **don't** divide by 2. When it doesn't (a pair of friends, a handshake), divide by 2.

## Paths on a grid

If you may move only right or down, write in each square the number of ways to reach it: the number in the square above plus the number in the square to its left. Squares along the top row and left column get 1. A blocked square gets 0.

## Counting digits

Split the numbers by how many digits they have:
- 1 to 9: 9 numbers × 1 digit = 9 digits
- 10 to 99: 90 numbers × 2 digits = 180 digits
- 100 to 999: 900 numbers × 3 digits = 2700 digits

## Worked example

*How many 2-digit numbers have digits adding to 4?* Go by the tens digit: 13, 22, 31, 40. That is 4 numbers. Note that 04 is not a 2-digit number.

## Common traps

- A number that "starts with 0" is not a 3-digit number.
- Counting the **numbers that contain** a digit is different from counting **how many times** the digit is written (11 has two 1s).
- Count the routes, not the squares.
