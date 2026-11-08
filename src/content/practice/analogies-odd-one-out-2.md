---
title: "Analogies and odd one out 2: number and letter analogies"
description: "Find the rule that links two numbers or two letter groups, check it on every given pair, and apply it to the last one."
slug: analogies-odd-one-out-2
family: analogies-odd-one-out
part: 2
timeMinutes: 12
questions:
  - id: an2-times-three
    topic: Number analogies
    difficulty: 1
    q: "2 : 6 :: 5 : 15 :: 8 : ?"
    options: ["16", "21", "24", "32"]
    answer: C
    explanation: |
      Rule: multiply by 3. 2 × 3 = 6 and 5 × 3 = 15, so 8 × 3 = **24**.

  - id: an2-ab-cd
    topic: Letter analogies
    difficulty: 1
    q: "AB : CD :: EF : ?"
    options: ["FG", "GH", "HI", "EG"]
    answer: B
    explanation: |
      Each letter moves 2 places forward: A → C, B → D. So E → G and F → H, giving **GH**.

  - id: an2-cat-dog
    topic: Letter analogies
    difficulty: 1
    q: "CAT : DBU :: DOG : ?"
    options: ["EPH", "CNF", "EPG", "FQI"]
    answer: A
    explanation: |
      Each letter moves 1 place forward: C → D, A → B, T → U.
      DOG becomes D → E, O → P, G → H: **EPH**. CNF moves each letter back instead.

  - id: an2-squares
    topic: Number analogies
    difficulty: 1
    q: "6 : 36 :: 9 : 81 :: 11 : ?"
    options: ["99", "110", "121", "132"]
    answer: C
    explanation: |
      Rule: square the number. 6 × 6 = 36 and 9 × 9 = 81, so 11 × 11 = **121**.
      The two given pairs rule out "multiply by 6", since 9 × 6 is not 81.

  - id: an2-cube-plus-one
    topic: Number analogies
    difficulty: 2
    q: "2 : 9 :: 3 : 28 :: 4 : ?"
    options: ["17", "63", "64", "65"]
    answer: D
    explanation: |
      Rule: cube the number, then add 1. 2³ + 1 = 9 and 3³ + 1 = 28, so 4³ + 1 = 64 + 1 = **65**.
      64 forgets the +1.

  - id: an2-opposite-letters
    topic: Letter analogies
    difficulty: 2
    q: "ACE : ZXV :: BDF : ?"
    options: ["XVT", "YWU", "YXW", "UWY"]
    answer: B
    explanation: |
      Each letter is replaced by its opposite letter (A ↔ Z, B ↔ Y, C ↔ X, ...). The positions of a letter and its opposite add up to 27.
      B → Y, D → W, F → U, giving **YWU**. UWY has the right letters in the wrong order.

  - id: an2-plus-three
    topic: Letter analogies
    difficulty: 2
    q: "BAD : EDG :: CAB : ?"
    options: ["DBC", "EDF", "FCE", "FDE"]
    answer: D
    explanation: |
      Each letter moves 3 places forward: B → E, A → D, D → G.
      CAB becomes C → F, A → D, B → E: **FDE**.

  - id: an2-digit-product
    topic: Number analogies
    difficulty: 2
    q: "23 : 6 :: 45 : 20 :: 67 : ?"
    options: ["13", "36", "42", "48"]
    answer: C
    explanation: |
      Rule: multiply the two digits. 2 × 3 = 6 and 4 × 5 = 20, so 6 × 7 = **42**.
      13 adds the digits, which does not fit 45 : 20.

  - id: an2-letter-position-square
    topic: Letter and number
    difficulty: 2
    q: "D : 16 :: F : 36 :: H : ?"
    options: ["64", "72", "81", "100"]
    answer: A
    explanation: |
      D is the 4th letter and 4² = 16. F is the 6th letter and 6² = 36.
      H is the 8th letter, so 8² = **64**. 81 uses I (the 9th letter) by mistake.

  - id: an2-n-times-next
    topic: Number analogies
    difficulty: 3
    q: "3 : 12 :: 5 : 30 :: 8 : ?"
    options: ["63", "64", "70", "72"]
    answer: D
    explanation: |
      Rule: multiply the number by the next number. 3 × 4 = 12 and 5 × 6 = 30, so 8 × 9 = **72**.
      64 squares 8 and misses the pattern; 3² = 9, not 12.

  - id: an2-plus-one-two-three
    topic: Letter analogies
    difficulty: 3
    q: "CAT : DCW :: PEN : ?"
    options: ["QFO", "QGQ", "QGP", "RGQ"]
    answer: B
    explanation: |
      The letters move by different amounts: the 1st moves +1 (C → D), the 2nd +2 (A → C) and the 3rd +3 (T → W).
      PEN: P + 1 = Q, E + 2 = G, N + 3 = Q, giving **QGQ**. QFO moves every letter by just 1.

  - id: an2-letter-pair-product
    topic: Letter and number
    difficulty: 3
    q: "AZ : 26 :: BY : 50 :: CX : ?"
    options: ["27", "69", "72", "75"]
    answer: C
    explanation: |
      Multiply the positions of the two letters. A × Z = 1 × 26 = 26 and B × Y = 2 × 25 = 50.
      C × X = 3 × 24 = **72**. 27 adds the positions instead, but that gives 27 for every pair, so it cannot explain 26 and 50.
---

In a number or letter analogy, the first pair (or two pairs) follows a rule. You find the rule and apply it to the last item.

## Method

1. Guess a simple rule from the first pair: add, multiply, square, cube, move letters forward or back.
2. **Check the rule on every pair given.** Many rules fit one pair; usually only one fits all of them.
3. Apply the rule to the last item and compute carefully.

## Rules to try first

| For numbers | For letters |
|---|---|
| × or + a fixed number | move each letter forward or back by a fixed number |
| square (n²) or cube (n³), maybe ± 1 | move letters by +1, +2, +3, ... in turn |
| n × (n + 1) | opposite letter: positions add up to 27 (A ↔ Z) |
| sum or product of digits | position number of a letter, its square, or a sum of positions |

Learn the positions: A = 1, E = 5, J = 10, O = 15, T = 20, Z = 26.

## Worked example

*4 : 17 :: 6 : 37 :: 7 : ?*

Try 4² = 16, plus 1 = 17. Check on the second pair: 6² + 1 = 37 ✓. Apply it: 7² + 1 = 49 + 1 = **50**.

"Multiply by 4, then add 1" also fits the first pair, but 6 × 4 + 1 = 25, not 37. That is why you check every pair.

## Trap

Off-by-one mistakes are the most common. Count letter positions carefully, and remember that "+1 after squaring" and "square of one more" are different: 4² + 1 = 17, but (4 + 1)² = 25.
