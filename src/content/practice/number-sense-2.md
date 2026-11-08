---
title: "Number sense 2: HCF, LCM and remainders"
description: "HCF and LCM in word problems, the HCF × LCM rule, remainder puzzles and number patterns."
slug: number-sense-2
family: number-sense
part: 2
timeMinutes: 15
questions:
  - id: hcf-24-36
    topic: HCF and LCM
    difficulty: 1
    q: "What is the HCF (highest common factor) of 24 and 36?"
    options: ["6", "12", "18", "72"]
    answer: B
    explanation: |
      24 = 2 × 2 × 2 × 3 and 36 = 2 × 2 × 3 × 3. The common part is 2 × 2 × 3 = **12**.
      72 is the LCM, not the HCF. The HCF can never be bigger than the smaller number.

  - id: lcm-6-8-12
    topic: HCF and LCM
    difficulty: 1
    q: "What is the smallest number that is divisible by 6, 8 and 12?"
    options: ["24", "48", "72", "96"]
    answer: A
    explanation: |
      Go through the multiples of the biggest number, 12: 12, 24, ...
      12 is not divisible by 8, but **24** is divisible by 6, 8 and 12. That is the LCM.

  - id: ribbons-hcf
    topic: HCF and LCM
    difficulty: 2
    q: "Two ribbons are 84 cm and 120 cm long. Both are cut into pieces of the same length, as long as possible, with nothing left over. How long is each piece?"
    options: ["6 cm", "12 cm", "24 cm", "42 cm"]
    answer: B
    explanation: |
      "Same length, as long as possible, nothing left over" means the HCF.
      84 = 2 × 2 × 3 × 7 and 120 = 2 × 2 × 2 × 3 × 5. Common part: 2 × 2 × 3 = **12 cm**.
      (That makes 7 + 10 = 17 pieces.) 24 does not divide 84; 42 does not divide 120.

  - id: bells-lcm
    topic: HCF and LCM
    difficulty: 2
    q: "Three school bells ring every 10, 15 and 20 minutes. They all ring together at 9:00 am. When do they next ring together?"
    options: ["9:30 am", "9:45 am", "10:00 am", "11:00 am"]
    answer: C
    explanation: |
      "Next time together" means the LCM. LCM of 10, 15, 20: multiples of 20 are 20, 40, 60; 60 is the first that 15 also divides.
      So they ring together every 60 minutes: next at **10:00 am**.

  - id: hcf-lcm-product
    topic: HCF and LCM
    difficulty: 2
    q: "The HCF of two numbers is 6 and their LCM is 180. One of the numbers is 36. What is the other?"
    options: ["30", "45", "60", "90"]
    answer: A
    explanation: |
      For two numbers, **HCF × LCM = product of the numbers**.
      6 × 180 = 1080, and 1080 ÷ 36 = **30**. Check: HCF(36, 30) = 6 and LCM(36, 30) = 180 ✓.

  - id: remainder-double
    topic: Remainders
    difficulty: 2
    q: "A number leaves a remainder of 4 when divided by 6. What remainder does twice that number leave when divided by 6?"
    options: ["0", "1", "2", "4"]
    answer: C
    explanation: |
      Try a real number: 10 leaves remainder 4 when divided by 6. Twice 10 is 20, and 20 = 6 × 3 + **2**.
      In general, double the remainder: 4 × 2 = 8, and 8 leaves remainder 2 when divided by 6.

  - id: sum-odd-numbers
    topic: Number patterns
    difficulty: 2
    q: "What is the sum of the first 10 odd numbers (1 + 3 + 5 + ... )?"
    options: ["55", "90", "100", "110"]
    answer: C
    explanation: |
      1 = 1, 1 + 3 = 4, 1 + 3 + 5 = 9, ... The sum of the first n odd numbers is n × n.
      So the first 10 odd numbers add up to 10 × 10 = **100**. (55 is 1 + 2 + ... + 10.)

  - id: two-digit-swap
    topic: Digit puzzles
    difficulty: 3
    q: "A two-digit number is 4 times the sum of its digits. If 27 is added to it, the digits swap places. What is the number?"
    options: ["24", "36", "48", "63"]
    answer: B
    explanation: |
      Test the choices; it is quicker than algebra.
      24: digit sum 6, and 4 × 6 = 24 ✓, but 24 + 27 = 51, not 42 ✗.
      **36**: 4 × 9 = 36 ✓ and 36 + 27 = 63 ✓.
      (Adding 27 swaps the digits when the units digit is 3 more than the tens digit.)

  - id: units-digit-7-35
    topic: Number patterns
    difficulty: 3
    q: "What is the units digit of 7³⁵ (35 sevens multiplied together)?"
    options: ["9", "7", "1", "3"]
    answer: D
    explanation: |
      Units digits of powers of 7 repeat in a cycle of 4: 7, 9, 3, 1, 7, 9, 3, 1, ...
      35 ÷ 4 = 8 remainder 3, so 7³⁵ ends like the 3rd item in the cycle: **3**.

  - id: remainder-1-div-7
    topic: Remainders
    difficulty: 3
    q: "Find the smallest number that leaves a remainder of 1 when divided by each of 2, 3, 4, 5 and 6, but is exactly divisible by 7."
    options: ["121", "241", "301", "361"]
    answer: C
    explanation: |
      Remainder 1 for 2, 3, 4, 5, 6 means the number is (a multiple of their LCM, 60) + 1: 61, 121, 181, 241, 301, ...
      Check each for 7: 61, 121, 181, 241 are not multiples of 7, but **301** = 7 × 43 ✓.

  - id: smallest-3-digit-rem-3
    topic: Remainders
    difficulty: 3
    q: "What is the smallest 3-digit number that leaves a remainder of 3 when divided by 5, by 6 and by 8?"
    options: ["103", "123", "163", "243"]
    answer: B
    explanation: |
      The number is (a multiple of the LCM of 5, 6, 8) + 3. LCM = 120.
      Multiples of 120 plus 3: 3, **123**, 243, ... The smallest with 3 digits is 123.
      103 fails: 103 ÷ 6 leaves 1.

  - id: pairs-hcf-4
    topic: HCF and LCM
    difficulty: 3
    q: "The product of two numbers is 192 and their HCF is 4. How many such pairs of numbers are there?"
    options: ["2", "3", "4", "6"]
    answer: A
    explanation: |
      Write the numbers as 4x and 4y, where x and y have no common factor (otherwise the HCF would be bigger than 4).
      4x × 4y = 192, so x × y = 12. Pairs that multiply to 12: (1, 12), (2, 6), (3, 4). Throw out (2, 6), because both are even.
      That leaves **2** pairs: (4, 48) and (12, 16).
---

These questions use HCF, LCM and remainders. Most can be done in your head if you spot which tool the question needs.

## HCF or LCM? Read the question

- **HCF** (highest common factor) for questions about cutting, sharing or grouping into the *largest equal parts*: ribbons, rows of students, packing sweets equally.
- **LCM** (lowest common multiple) for questions about things that *repeat and meet again*: bells, traffic lights, runners on a track, buses leaving a depot.
- For any two numbers: **HCF × LCM = the product of the two numbers.**

## Remainder questions

"Leaves remainder *r* when divided by *a*, *b* and *c*" means the number is (a multiple of LCM of *a*, *b*, *c*) + *r*. Write the list and test the choices.

## Worked example

*What is the smallest number of sweets that can be shared equally among 4, 6 or 9 children with none left over?*

The number must be a multiple of 4, 6 and 9, so find the LCM. 4 = 2 × 2, 6 = 2 × 3, 9 = 3 × 3. Take each prime as many times as it appears most: 2 × 2 × 3 × 3 = **36**.

## Tip: test the options

In an MCQ test you already have four possible answers. Plugging them into the question is often faster than solving from scratch. It also catches careless mistakes.
