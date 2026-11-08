---
title: "Number sense 1: factors, primes and divisibility"
description: "Counting factors, spotting primes, divisibility rules for 3, 4, 9 and 11, place value and perfect squares."
slug: number-sense-1
family: number-sense
part: 1
timeMinutes: 15
questions:
  - id: factors-36
    topic: Factors
    difficulty: 1
    q: "How many factors does 36 have?"
    options: ["4", "6", "8", "9"]
    answer: D
    explanation: |
      List factors in pairs that multiply to 36: 1 × 36, 2 × 18, 3 × 12, 4 × 9, 6 × 6.
      That gives 1, 2, 3, 4, 6, 9, 12, 18, 36. The pair 6 × 6 gives only one factor, so the count is **9**.
      A number has an odd number of factors only when it is a perfect square.

  - id: div-3-and-4
    topic: Divisibility
    difficulty: 1
    q: "Which number is divisible by both 3 and 4?"
    options: ["522", "612", "314", "734"]
    answer: B
    explanation: |
      Divisible by 3: the digit sum is a multiple of 3. Divisible by 4: the last two digits form a multiple of 4.
      612: 6 + 1 + 2 = 9 ✓ and 12 is a multiple of 4 ✓. So **612** works.
      522 fails the 4 test (22), 314 and 734 fail the 3 test.

  - id: which-prime
    topic: Primes
    difficulty: 1
    q: "Which of these is a prime number?"
    options: ["51", "57", "87", "89"]
    answer: D
    explanation: |
      Check the digit sums: 51 (6), 57 (12) and 87 (15) are all multiples of 3, so each is divisible by 3.
      51 = 3 × 17, 57 = 3 × 19, 87 = 3 × 29. **89** has no factor up to 9 (since 9 × 9 = 81 and 10 × 10 = 100), so it is prime.

  - id: place-value-7
    topic: Place value
    difficulty: 1
    q: "What is the sum of the place values of the two 7s in 47,372?"
    options: ["7", "70", "707", "7070"]
    answer: D
    explanation: |
      In 47,372 the first 7 is in the thousands place (7000) and the second is in the tens place (70).
      7000 + 70 = **7070**. (The *face value* of each 7 is just 7.)

  - id: blank-div-9
    topic: Divisibility
    difficulty: 1
    q: "Which digit in the blank makes 5 _ 2 divisible by 9?"
    options: ["1", "2", "4", "7"]
    answer: B
    explanation: |
      A number is divisible by 9 when its digit sum is a multiple of 9.
      5 + _ + 2 = 7 + _. The next multiple of 9 after 7 is 9, so the blank is **2**: 522 = 9 × 58.

  - id: multiples-of-7
    topic: Multiples
    difficulty: 1
    q: "How many multiples of 7 are there from 1 to 100?"
    options: ["14", "15", "16", "17"]
    answer: A
    explanation: |
      Divide: 100 ÷ 7 = 14 remainder 2. The multiples are 7, 14, ..., 98 (= 7 × 14), so there are **14**.

  - id: six-factors
    topic: Factors
    difficulty: 2
    q: "What is the smallest number that has exactly 6 factors?"
    options: ["8", "10", "16", "12"]
    answer: D
    explanation: |
      12 has factors 1, 2, 3, 4, 6, 12, which is six of them. Check the smaller choices: 8 has 4 (1, 2, 4, 8), 10 has 4, and 16 has 5.
      Trying 1 to 11 shows none has six factors, so **12** is the smallest.

  - id: digits-3a6b
    topic: Divisibility
    difficulty: 2
    q: "The 4-digit number 3 a 6 b is divisible by both 5 and 9, and b is not 0. What is a?"
    options: ["2", "4", "6", "8"]
    answer: B
    explanation: |
      Divisible by 5 means the last digit is 0 or 5. b is not 0, so b = 5.
      Divisible by 9: 3 + a + 6 + 5 = 14 + a must be a multiple of 9, so 14 + a = 18 and a = **4**. The number is 3465 = 9 × 385.

  - id: primes-50-70
    topic: Primes
    difficulty: 2
    q: "How many prime numbers are there between 50 and 70?"
    options: ["4", "5", "6", "7"]
    answer: A
    explanation: |
      Skip the even numbers and the multiples of 5. Check the rest for 3 and 7:
      51 (3 × 17), **53**, 57 (3 × 19), **59**, **61**, 63 (7 × 9), **67**, 69 (3 × 23). That is **4** primes.

  - id: make-square-72
    topic: Squares
    difficulty: 2
    q: "What is the smallest number you must multiply 72 by to get a perfect square?"
    options: ["2", "3", "4", "6"]
    answer: A
    explanation: |
      72 = 2 × 2 × 2 × 3 × 3. In a perfect square every prime comes in pairs.
      The 3s are paired, but there are three 2s, so one 2 is alone. Multiply by **2**: 72 × 2 = 144 = 12 × 12.

  - id: odd-number-of-factors
    topic: Factors
    difficulty: 3
    q: "How many numbers from 1 to 100 have an odd number of factors?"
    options: ["10", "12", "25", "50"]
    answer: A
    explanation: |
      Factors come in pairs (like 3 × 12 for 36). A pair breaks only when both parts are the same, as in 6 × 6.
      So only perfect squares have an odd number of factors: 1, 4, 9, ..., 100. That is **10** numbers.

  - id: div-by-11
    topic: Divisibility
    difficulty: 3
    q: "Which number is divisible by 11?"
    options: ["41029", "41109", "52316", "61247"]
    answer: C
    explanation: |
      Rule for 11: add the alternate digits, then subtract the other alternate digits. If the result is 0 or a multiple of 11, the number is divisible by 11.
      52316: (5 + 3 + 6) − (2 + 1) = 14 − 3 = 11 ✓. So **52316** = 11 × 4756.
      41029: (4 + 0 + 9) − (1 + 2) = 10 ✗. 41109: (4 + 1 + 9) − (1 + 0) = 13 ✗. 61247: (6 + 2 + 7) − (1 + 4) = 10 ✗.

  - id: zeros-50-factorial
    topic: Factors
    difficulty: 3
    q: "How many zeros are there at the end of 1 × 2 × 3 × ... × 50?"
    options: ["5", "10", "11", "12"]
    answer: D
    explanation: |
      Each zero at the end needs a 10 = 2 × 5. There are many more 2s than 5s, so count the 5s.
      Multiples of 5 up to 50: 10 numbers give one 5 each. 25 and 50 are 5 × 5 times something, so each gives one extra 5.
      10 + 2 = **12** zeros.
---

These questions test how well you know your numbers, not long calculations. Most can be done in your head in under a minute if you use the right rule.

## Rules worth knowing by heart

| Divisible by | Check |
|---|---|
| 2 | last digit is even |
| 3 | digit sum is a multiple of 3 |
| 4 | last two digits form a multiple of 4 |
| 5 | last digit is 0 or 5 |
| 6 | divisible by both 2 and 3 |
| 8 | last three digits form a multiple of 8 |
| 9 | digit sum is a multiple of 9 |
| 11 | (sum of alternate digits) − (sum of the others) is 0 or a multiple of 11 |

## Factors come in pairs

To count the factors of a number, list them in pairs that multiply to it: for 36 that is 1 × 36, 2 × 18, 3 × 12, 4 × 9 and 6 × 6. Stop when the pair meets in the middle. A perfect square has one pair with both numbers the same, so it has an odd number of factors.

## Is it prime?

Check divisibility only by primes up to the square root. For a number below 100, testing 2, 3, 5 and 7 is enough.
