---
title: "Counting and pigeonhole 2: worst cases and two sets"
description: "How many must you pick to be sure of a pair? Worst-case reasoning with socks, marbles and birthdays, and counting with two overlapping groups."
slug: counting-pigeonhole-2
family: counting-pigeonhole
part: 2
timeMinutes: 12
questions:
  - id: cp2-socks-pair
    topic: "Sure of a pair"
    difficulty: 1
    q: "A drawer has 10 red socks and 10 blue socks, all mixed up. The room is dark. What is the smallest number of socks Arjun must take out to be **sure** of getting two of the same colour?"
    options: ["2", "3", "11", "20"]
    answer: B
    explanation: |
      The unluckiest start is one red and one blue. The next sock must match one of them.
      So **3** socks are enough. 2 might work, but you can't be sure. 11 is how many you need to be sure of one *red* sock (all 10 blue could come first).

  - id: cp2-cricket-football
    topic: "Two sets"
    difficulty: 1
    q: "In a class of 40 students, 25 like cricket and 20 like football. Every student likes at least one of the two games. How many like both?"
    options: ["5", "15", "20", "45"]
    answer: A
    explanation: |
      25 + 20 = 45, but there are only 40 students. The extra 5 have been counted twice, once in each group.
      So **5** like both. 45 forgets that some students are in both groups.

  - id: cp2-birth-month
    topic: "Sure of a pair"
    difficulty: 1
    q: "What is the smallest number of people in a group that makes it **certain** that at least two of them were born in the same month?"
    options: ["11", "12", "13", "24"]
    answer: C
    explanation: |
      There are 12 months. In the worst case, the first 12 people are born in 12 different months. The 13th person must share a month with someone.
      So **13** people. With 12 it is possible that every month is different.

  - id: cp2-music-art
    topic: "Two sets"
    difficulty: 1
    q: "In a group of 30 children, 18 learn music, 12 learn art and 5 learn both. How many children learn neither music nor art?"
    options: ["5", "13", "25", "35"]
    answer: A
    explanation: |
      Children who learn at least one = 18 + 12 − 5 = 25 (take away the 5 who were counted twice).
      Neither = 30 − 25 = **5**. 25 is the number who learn at least one, not the answer asked for.

  - id: cp2-three-colours-pair
    topic: "Sure of a pair"
    difficulty: 2
    q: "A box has 5 red, 7 blue and 9 green socks. Without looking, what is the smallest number Meera must take out to be **sure** of two socks of the same colour?"
    options: ["4", "6", "10", "21"]
    answer: A
    explanation: |
      There are 3 colours. In the worst case the first 3 socks are one of each colour. The 4th must match one of them.
      So **4**. The numbers 5, 7 and 9 don't matter here, as long as each colour has at least two socks.

  - id: cp2-one-of-each
    topic: "Worst case"
    difficulty: 2
    q: "A bag has 4 red, 6 blue and 8 green marbles. What is the smallest number of marbles Kabir must take out, without looking, to be **sure** of getting at least one marble of every colour?"
    options: ["3", "9", "14", "15"]
    answer: D
    explanation: |
      Worst case: he takes all 8 green and all 6 blue first, which is 14 marbles and still no red. The next one must be red.
      So **15** marbles. 14 forgets the last marble, and 3 only works if he is lucky.

  - id: cp2-newspapers
    topic: "Two sets"
    difficulty: 2
    q: "In a housing society of 50 families, 32 families read a Hindi newspaper, 26 read an English newspaper, and 6 read neither. How many families read both?"
    options: ["6", "8", "14", "18"]
    answer: C
    explanation: |
      Families that read at least one paper = 50 − 6 = 44. Then 32 + 26 = 58 counts the 'both' families twice.
      Both = 58 − 44 = **14**. 8 (= 58 − 50) forgets the 6 families that read neither. 18 is the number reading *only* Hindi.

  - id: cp2-three-or-five
    topic: "Two sets"
    difficulty: 2
    q: "How many whole numbers from 1 to 100 are divisible by 3 or by 5 (or by both)?"
    options: ["6", "41", "47", "53"]
    answer: C
    explanation: |
      Multiples of 3: 33. Multiples of 5: 20. Multiples of both are multiples of 15: 6 (15, 30, ..., 90).
      33 + 20 − 6 = **47**. 53 forgets to take away the 6 numbers counted twice.

  - id: cp2-four-same-month
    topic: "Worst case"
    difficulty: 2
    q: "A school wants to be **certain** that at least 4 of the students in a hall were born in the same month. What is the smallest number of students needed in the hall?"
    options: ["4", "12", "36", "37"]
    answer: D
    explanation: |
      Worst case: every month has 3 students, which is 12 × 3 = 36 students with no month reaching 4. One more student makes some month reach 4.
      So **37**. 36 is the largest number that still allows *no* month to have 4.

  - id: cp2-tea-coffee
    topic: "Two sets"
    difficulty: 3
    q: "In an office of 60 people, 10 like both tea and coffee, and 5 like neither. The number who like *only* tea is twice the number who like *only* coffee. How many people like tea?"
    options: ["15", "25", "30", "40"]
    answer: D
    explanation: |
      People who like exactly one drink = 60 − 10 − 5 = 45. Only tea = 2 × only coffee, so only coffee = 45 ÷ 3 = 15 and only tea = 30.
      Everyone who likes tea = only tea + both = 30 + 10 = **40**. 30 counts only the 'only tea' group.

  - id: cp2-gloves
    topic: "Worst case"
    difficulty: 3
    q: "A box holds 5 pairs of gloves, each pair a different colour. All 10 gloves are mixed up. Simran takes gloves out without looking. What is the smallest number she must take to be **sure** of a left glove and a right glove of the same colour?"
    options: ["2", "6", "7", "11"]
    answer: B
    explanation: |
      Worst case: the first 5 gloves are all left gloves, one of each colour (or all right gloves). No pair yet.
      The 6th glove must be the other hand of one of those colours. So **6**. 2 only works if she is lucky.

  - id: cp2-three-red
    topic: "Worst case"
    difficulty: 3
    q: "A jar has 6 red, 8 blue and 10 green beads. What is the smallest number of beads Neha must take out, without looking, to be **sure** of getting 3 red beads?"
    options: ["7", "21", "22", "24"]
    answer: B
    explanation: |
      Worst case: all 8 blue and all 10 green come out first, 18 beads with no red. Then she needs 3 more, all of which must be red.
      18 + 3 = **21**. 7 is the answer to a different question: being sure of 3 beads of the *same* colour (2 + 2 + 2 + 1).
---

This set tests worst-case thinking: how many things you must pick to be **sure** of something, and how to count people who belong to two groups.

## The pigeonhole idea

If you put more pigeons than holes, some hole gets at least two pigeons. To be **sure**, always imagine the unluckiest picks first.

- **Sure of a pair (any colour):** with n colours, the worst case is one of each, so you need n + 1.
- **Sure of k of the same kind:** the worst case is k − 1 of every kind, so you need n × (k − 1) + 1.
- **Sure of one particular colour:** the worst case is *all* the other colours first.

## Two sets

For two overlapping groups A and B:

| Group | How to count |
|---|---|
| In A or B (or both) | A + B − both |
| Neither | total − (A + B − both) |
| Only A | A − both |

Adding A and B counts the people in both groups **twice**, so you take them away once. Drawing two circles and filling in the overlap first helps.

## Worked example

*A bowl has 3 mangoes and 5 bananas. How many fruits must you pick blindly to be sure of a mango?* Worst case: all 5 bananas first. The 6th fruit must be a mango. Answer: **6**.

## Common traps

- "Might happen" is not "sure to happen". Being lucky doesn't count.
- Read whether the question wants *two of the same colour* or *two of one particular colour*. The answers are very different.
- "Only coffee" and "coffee" are different groups.
