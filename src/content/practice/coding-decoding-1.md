---
title: "Coding and decoding 1: letter shifts and number codes"
description: "Codes that move letters forward or back, reverse the word or the alphabet, and turn letters into numbers by position or by sum."
slug: coding-decoding-1
family: coding-decoding
part: 1
timeMinutes: 12
questions:
  - id: cd1-shift-lemon
    topic: Letter shifts
    difficulty: 1
    q: "In a code, MANGO is written as NBOHP. How is LEMON written in the same code?"
    options: ["KDLNM", "MFNPO", "NGOQP", "NOMEL"]
    answer: B
    explanation: |
      Each letter moves **one** place forward: M→N, A→B, N→O, G→H, O→P.
      So L→M, E→F, M→N, O→P, N→O gives **MFNPO**. KDLNM moves each letter back instead of forward.

  - id: cd1-number-train
    topic: Number codes
    difficulty: 1
    q: "In a code, HOME is written as 8-15-13-5. How is TRAIN written in the same code?"
    options: ["7-9-26-18-13", "14-9-1-18-20", "20-18-1-9-14", "21-19-2-10-15"]
    answer: C
    explanation: |
      Each letter is replaced by its place in the alphabet: H = 8, O = 15, M = 13, E = 5.
      T = 20, R = 18, A = 1, I = 9, N = 14, so TRAIN is **20-18-1-9-14**. 14-9-1-18-20 is the same numbers in reverse order.

  - id: cd1-opposite-gold
    topic: Reverse codes
    difficulty: 1
    q: "In a code, A is written as Z, B as Y, C as X, and so on. How is GOLD written in this code?"
    options: ["DLOG", "HPME", "SKNV", "TLOW"]
    answer: D
    explanation: |
      Each letter is swapped with its partner from the other end. A letter and its partner add up to 27: G (7) ↔ T (20), O (15) ↔ L (12), L ↔ O, D (4) ↔ W (23).
      So GOLD is **TLOW**. SKNV uses 26 instead of 27, so every letter is one place off.

  - id: cd1-shift-horse
    topic: Letter shifts
    difficulty: 1
    q: "In a code, TIGER is written as RGECP. How is HORSE written in the same code?"
    options: ["FMPQC", "GNQRD", "ESROH", "JQTUG"]
    answer: A
    explanation: |
      Each letter moves **two** places back: T→R, I→G, G→E, E→C, R→P.
      H→F, O→M, R→P, S→Q, E→C gives **FMPQC**. JQTUG moves two places forward, the wrong way.

  - id: cd1-reverse-shift-brave
    topic: Reverse codes
    difficulty: 2
    q: "In a code, SMART is written as USBNT. How is BRAVE written in the same code?"
    options: ["CSBWF", "DUZQA", "EVARB", "FWBSC"]
    answer: D
    explanation: |
      Write SMART backwards (TRAMS), then move each letter one place forward: T→U, R→S, A→B, M→N, S→T = USBNT.
      BRAVE backwards is EVARB, and one place forward is **FWBSC**. CSBWF only shifts the letters, and EVARB only reverses them.

  - id: cd1-alternate-flower
    topic: Letter shifts
    difficulty: 2
    q: "In a code, GARDEN is written as HZSCFM. How is FLOWER written in the same code?"
    options: ["EKNVDQ", "EMNXDS", "GKPVFQ", "GMPXFS"]
    answer: C
    explanation: |
      The shifts take turns: +1, −1, +1, −1, ... (G→H, A→Z, R→S, D→C, E→F, N→M).
      F+1 = G, L−1 = K, O+1 = P, W−1 = V, E+1 = F, R−1 = Q gives **GKPVFQ**. EMNXDS starts the pattern with −1 instead of +1.

  - id: cd1-sum-cow
    topic: Number codes
    difficulty: 2
    q: "If CAT = 24 and DOG = 26, what is COW in the same code?"
    options: ["40", "41", "42", "43"]
    answer: B
    explanation: |
      Each word is the sum of its letter places: C + A + T = 3 + 1 + 20 = 24 and D + O + G = 4 + 15 + 7 = 26.
      COW = 3 + 15 + 23 = **41**. 40 comes from taking O as 14.

  - id: cd1-reverse-sum-rain
    topic: Number codes
    difficulty: 2
    q: "In a code, letters are numbered from the end of the alphabet: Z = 1, Y = 2, X = 3, ..., A = 26. A word's code is the sum of its letter numbers. What is the code for RAIN?"
    options: ["42", "62", "66", "68"]
    answer: C
    explanation: |
      A letter's number from the end is 27 minus its usual place. R = 27 − 18 = 9, A = 26, I = 27 − 9 = 18, N = 27 − 14 = 13.
      9 + 26 + 18 + 13 = **66**. 42 uses the usual A = 1 order, and 62 subtracts from 26 instead of 27.

  - id: cd1-double-bead
    topic: Number codes
    difficulty: 2
    q: "In a code, FACE is written as 12-2-6-10. How is BEAD written in the same code?"
    options: ["4-10-2-8", "2-5-1-4", "3-6-2-5", "4-10-8-2"]
    answer: A
    explanation: |
      F = 6, A = 1, C = 3, E = 5, and each place is **doubled**: 12-2-6-10.
      B = 2, E = 5, A = 1, D = 4, doubled: **4-10-2-8**. 2-5-1-4 forgets to double.

  - id: cd1-growing-stone
    topic: Letter shifts
    difficulty: 3
    q: "In a code, WATER is written as XCWIW. How is STONE written in the same code?"
    options: ["TUPOF", "TVRRJ", "RRLJZ", "XXRPF"]
    answer: B
    explanation: |
      The shift grows by one each time: W+1 = X, A+2 = C, T+3 = W, E+4 = I, R+5 = W.
      S+1 = T, T+2 = V, O+3 = R, N+4 = R, E+5 = J gives **TVRRJ**. XXRPF starts with +5 and counts down, which is the pattern backwards.

  - id: cd1-opposite-reverse-mango
    topic: Reverse codes
    difficulty: 3
    q: "In a code, PLANT is written as GMZOK. How is MANGO written in the same code?"
    options: ["LTMZN", "NZMTL", "OGNAM", "KSLYM"]
    answer: A
    explanation: |
      Two steps. Swap each letter with its partner from the other end (P↔K, L↔O, A↔Z, N↔M, T↔G) to get KOZMG, then write it backwards: GMZOK.
      MANGO → NZMTL → backwards **LTMZN**. NZMTL forgets to reverse; KSLYM uses 26 instead of 27 for the partner letters.

  - id: cd1-sum-length-fish
    topic: Number codes
    difficulty: 3
    q: "In a code, GO = 44 and BAD = 21. What is FISH in the same code?"
    options: ["42", "84", "126", "168"]
    answer: D
    explanation: |
      The code is (sum of letter places) × (number of letters). GO: (7 + 15) × 2 = 44. BAD: (2 + 1 + 4) × 3 = 21.
      FISH: (6 + 9 + 19 + 8) × 4 = 42 × 4 = **168**. 42 is only the sum, and 126 multiplies by 3 as if every word had three letters.
---

This set tests codes that change each letter by a fixed rule, and codes that turn letters into numbers.

## Know the alphabet by number

| A | E | J | O | T | Y |
|---|---|---|---|---|---|
| 1 | 5 | 10 | 15 | 20 | 25 |

Remember **EJOTY** (5, 10, 15, 20, 25) and count a step or two from the nearest one. R is two before T, so R = 18.

## Common rules

- **Shift:** every letter moves the same number of places, forward or back. After Z, go round to A.
- **Changing shift:** the shift takes turns (+1, −1, +1, ...) or grows (+1, +2, +3, ...).
- **Reverse the word:** write it backwards, often together with a shift.
- **Opposite letter:** A↔Z, B↔Y, C↔X. A letter and its opposite always add up to **27**, so the opposite of G (7) is the 20th letter, T.
- **Number codes:** a letter's place (A = 1), its place from the end (Z = 1), double the place, or the sum of all the places in the word.

## Worked example

*In a code, COLD is written as DPME. How is WARM written?*

Compare letter by letter: C→D, O→P, L→M, D→E. Each letter moves one place forward. So W→X, A→B, R→S, M→N, and WARM is written **XBSN**.

## Tip

Always test your rule on **every** letter of the example, not just the first two. A rule that works for the first letter may fail on the third. When a code has two steps (reverse and shift), it does not matter which you do first, but you must do both.
