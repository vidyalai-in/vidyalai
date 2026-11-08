---
title: "Number series 1: finding the rule"
description: "Series built from differences, ratios, squares, cubes and primes, two series mixed together, two-step rules, and spotting the wrong number."
slug: number-series-1
family: number-series
part: 1
timeMinutes: 12
questions:
  - id: ns1-growing-gaps
    topic: Differences
    difficulty: 1
    q: "What comes next? 3, 5, 9, 15, 23, ?"
    options: ["31", "33", "35", "46"]
    answer: B
    explanation: |
      The gaps are 2, 4, 6, 8, so the next gap is 10: 23 + 10 = **33**.
      31 repeats the last gap of 8 instead of growing it.

  - id: ns1-times-three
    topic: Ratios and steps
    difficulty: 1
    q: "What comes next? 2, 6, 18, 54, ?"
    options: ["90", "108", "162", "216"]
    answer: C
    explanation: |
      Each number is 3 times the one before: 54 × 3 = **162**.
      108 doubles 54 instead of tripling it.

  - id: ns1-next-prime
    topic: Squares, cubes, primes
    difficulty: 1
    q: "What comes next? 2, 3, 5, 7, 11, 13, ?"
    options: ["14", "15", "16", "17"]
    answer: D
    explanation: |
      These are the prime numbers in order. The next prime after 13 is **17**.
      15 = 3 × 5 is not prime, even though the last gap was 2.

  - id: ns1-cube-gap
    topic: Squares, cubes, primes
    difficulty: 1
    q: "Which number is missing? 1, 8, 27, ?, 125, 216"
    options: ["64", "72", "81", "100"]
    answer: A
    explanation: |
      These are cubes: 1³, 2³, 3³, ?, 5³, 6³. The missing one is 4³ = 4 × 4 × 4 = **64**.
      81 = 9 × 9 is a square, not a cube.

  - id: ns1-two-series
    topic: Alternating
    difficulty: 2
    q: "What comes next? 4, 20, 7, 18, 10, 16, 13, ?"
    options: ["11", "12", "14", "16"]
    answer: C
    explanation: |
      Two series take turns. Odd places: 4, 7, 10, 13 (+3 each time). Even places: 20, 18, 16, ... (−2 each time).
      The next number is in the even places: 16 − 2 = **14**.
      16 would come from adding 3 to 13, but the next place belongs to the other series.

  - id: ns1-double-plus-one
    topic: Ratios and steps
    difficulty: 2
    q: "What comes next? 3, 7, 15, 31, 63, ?"
    options: ["126", "127", "189", "191"]
    answer: B
    explanation: |
      Each number is double the one before, plus 1: 3 × 2 + 1 = 7, 7 × 2 + 1 = 15, and so on.
      63 × 2 + 1 = **127**. (The gaps 4, 8, 16, 32, 64 give the same answer.) 126 forgets the + 1.

  - id: ns1-add-squares
    topic: Differences
    difficulty: 2
    q: "What comes next? 1, 2, 6, 15, 31, ?"
    options: ["47", "56", "62", "64"]
    answer: B
    explanation: |
      The gaps are 1, 4, 9, 16: the squares 1², 2², 3², 4². The next gap is 5² = 25.
      31 + 25 = **56**. 47 adds 16 again; 62 just doubles 31.

  - id: ns1-times-plus
    topic: Alternating
    difficulty: 2
    q: "What comes next? 2, 4, 7, 14, 17, 34, ?"
    options: ["37", "51", "68", "71"]
    answer: A
    explanation: |
      Two steps take turns: × 2, then + 3. 2 × 2 = 4, 4 + 3 = 7, 7 × 2 = 14, 14 + 3 = 17, 17 × 2 = 34.
      The next step is + 3: 34 + 3 = **37**. 68 doubles twice in a row.

  - id: ns1-wrong-square-plus
    topic: Wrong number
    difficulty: 2
    q: "One number in this series is wrong. Which one? 5, 10, 17, 26, 37, 52, 65"
    options: ["10", "26", "37", "52"]
    answer: D
    explanation: |
      Each number is one more than a square: 2² + 1 = 5, 3² + 1 = 10, 4² + 1 = 17, 5² + 1 = 26, 6² + 1 = 37, 7² + 1 = 50, 8² + 1 = 65.
      The sixth number should be 50, so **52** is wrong. The gaps 5, 7, 9, 11, 13, 15 show the same thing.

  - id: ns1-prime-gaps
    topic: Differences
    difficulty: 3
    q: "What comes next? 10, 12, 15, 20, 27, 38, ?"
    options: ["49", "50", "51", "53"]
    answer: C
    explanation: |
      The gaps are 2, 3, 5, 7, 11: the prime numbers in order. The next prime is 13.
      38 + 13 = **51**. 49 adds 11 again; 50 adds 12, which is not prime.

  - id: ns1-wrong-doubling
    topic: Wrong number
    difficulty: 3
    q: "One number in this series is wrong. Which one? 3, 5, 9, 17, 33, 66, 129"
    options: ["9", "17", "33", "66"]
    answer: D
    explanation: |
      The rule is double and subtract 1: 3 × 2 − 1 = 5, 5 × 2 − 1 = 9, 9 × 2 − 1 = 17, 17 × 2 − 1 = 33, 33 × 2 − 1 = 65, 65 × 2 − 1 = 129.
      So **66** should be 65. The trap: 33 × 2 = 66 looks right, but then 66 × 2 − 1 = 131, not 129.

  - id: ns1-cube-minus-n
    topic: Squares, cubes, primes
    difficulty: 3
    q: "What comes next? 0, 6, 24, 60, 120, ?"
    options: ["180", "200", "210", "216"]
    answer: C
    explanation: |
      Each number is n³ − n: 1 − 1 = 0, 8 − 2 = 6, 27 − 3 = 24, 64 − 4 = 60, 125 − 5 = 120. Next: 216 − 6 = **210**.
      Check with gaps: 6, 18, 36, 60 grow by 12, 18, 24, so the next gap is 90 and 120 + 90 = 210. 180 just repeats the gap of 60.
---

A number series follows one hidden rule. Your job is to find the rule, check it on **every** number, and then use it.

## Try these rules in order

| Look for | Example |
|---|---|
| Same gap | 4, 9, 14, 19 (+5) |
| Gaps that grow | 1, 2, 4, 7, 11 (gaps 1, 2, 3, 4) |
| Same multiplier | 5, 10, 20, 40 (× 2) |
| Special numbers | squares 1, 4, 9, 16; cubes 1, 8, 27; primes 2, 3, 5, 7 |
| Two steps | × 2, then + 1, again and again |
| Two series mixed | 1, 10, 2, 20, 3, 30 (odd places and even places) |

If the gaps don't make sense, write the gaps of the gaps. If the numbers jump fast, try dividing. If they go up and down, split the odd and even places.

## Worked example

Series: 2, 6, 12, 20, 30, ?

Gaps: 4, 6, 8, 10. The gaps grow by 2, so the next gap is 12 and the answer is 30 + 12 = 42. Another way to see it: 1 × 2, 2 × 3, 3 × 4, 4 × 5, 5 × 6, so next is 6 × 7 = 42. Two methods giving the same answer is a good sign.

## Finding the wrong number

Find the rule using most of the numbers. Then work out what each number *should* be. The wrong one is the number that doesn't match. Be careful: one wrong number spoils **two** gaps, the one before it and the one after it. So when two gaps next to each other look odd, the number between them is usually the culprit.

## Common traps

- Checking the rule on only the first two or three numbers.
- Squares and cubes look alike: 64 is both 8² and 4³, but 81 is only a square.
- In a mixed series, the next number may belong to the other series.
