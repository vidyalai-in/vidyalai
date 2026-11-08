---
title: "Attention to detail 1: spot the difference"
description: "Find the pair that is exactly the same, count a digit or letter in a grid, and compare two lists for small differences."
slug: attention-detail-1
family: attention-detail
part: 1
timeMinutes: 12
questions:
  - id: ad1-same-code
    topic: "Same or different"
    difficulty: 1
    q: "Which pair of codes is exactly the same?"
    options: ["`PT4M-8XW1` and `PT4M-8XW1`", "`KL7B-9R2Q` and `KL7B-9RQ2`", "`HD3S-6NV0` and `HD3S-6NVO`", "`ZR8E-2LK5` and `ZR8F-2LK5`"]
    answer: A
    explanation: |
      Check one character at a time, left to right. Only **PT4M-8XW1 and PT4M-8XW1** match.
      The others differ: 2Q / Q2 are swapped, 0 (zero) / O (letter), and E / F.

  - id: ad1-same-name
    topic: "Same or different"
    difficulty: 1
    q: "Which of these is spelt exactly like **Priyanka Raghunathan**?"
    options: ["Priyanka Ragunathan", "Priyanaka Raghunathan", "Priyanka Raghunathan", "Priyanka Raghunthan"]
    answer: C
    explanation: |
      Only **Priyanka Raghunathan** matches. *Ragunathan* drops the h, *Priyanaka* adds an extra a, and *Raghunthan* drops the a in -nath-.
      Long names hide small slips in the middle, so check the middle letters carefully.

  - id: ad1-count-sevens
    topic: "Counting in a grid"
    difficulty: 1
    q: "How many times does the digit 7 appear in this grid?"
    figure: { kind: "grid", cells: [[6, 3, 7, 1, 2, 8], [2, 6, 9, 1, 8, 4], [1, 2, 7, 7, 2, 4], [2, 8, 7, 1, 9, 2], [4, 9, 1, 9, 9, 7]] }
    options: ["5", "6", "7", "8"]
    answer: A
    explanation: |
      Go row by row and keep a running total: 1 + 0 + 2 + 1 + 1 = **5**.
      Counting row by row (instead of jumping around) stops you missing one or counting one twice.

  - id: ad1-vehicle
    topic: "Same or different"
    difficulty: 1
    q: "Which pair of vehicle numbers is exactly the same?"
    options: ["`MH 12 AB 4507` and `MH 12 AB 4570`", "`KA 05 MN 2318` and `KA 05 NM 2318`", "`TN 09 RK 6642` and `TN 06 RK 6642`", "`DL 3C BT 9061` and `DL 3C BT 9061`"]
    answer: D
    explanation: |
      Only **DL 3C BT 9061 and DL 3C BT 9061** match. The others have 07 / 70 swapped, MN / NM swapped, and 09 / 06.
      Swaps of two neighbouring characters are the easiest slips to miss.

  - id: ad1-code-places
    topic: "Same or different"
    difficulty: 2
    q: "Compare `M4QX-82LT-0P9` with `M4OX-82TL-0P6`. In how many places (character positions) are they different?"
    options: ["3", "4", "5", "6"]
    answer: B
    explanation: |
      Place by place: Q / O differ, L / T and T / L are two places, and 9 / 6 differ. That is **4** places.
      It is tempting to count the LT / TL swap as one difference, but it changes two positions.

  - id: ad1-matching-rows
    topic: "Comparing lists"
    difficulty: 2
    q: "In how many rows is the entry in List A exactly the same as the entry in List B?"
    figure: { kind: "table", header: ["Row", "List A", "List B"], rows: [[1, "TN-58213", "TN-58213"], [2, "KA-30947", "KA-30497"], [3, "GJ-71605", "GJ-71605"], [4, "WB-26638", "WB-26683"], [5, "RJ-49120", "RJ-49120"], [6, "AP-83574", "AP-83574"], [7, "UP-15092", "UP-15029"]] }
    options: ["1", "2", "3", "4"]
    answer: D
    explanation: |
      Rows 1, 3, 5 and 6 match, so **4** rows. Rows 2, 4 and 7 each have two digits swapped (947 / 497, 38 / 83, 92 / 29).

  - id: ad1-count-b
    topic: "Counting in a grid"
    difficulty: 2
    q: "How many times does the letter **b** appear in this grid?"
    figure: { kind: "grid", cells: [["b", "q", "b", "q", "d", "b"], ["d", "b", "d", "d", "d", "b"], ["b", "d", "d", "q", "q", "d"], ["b", "b", "b", "q", "p", "d"], ["d", "p", "q", "p", "d", "d"]] }
    options: ["9", "10", "11", "12"]
    answer: A
    explanation: |
      Row by row: 3 + 2 + 1 + 3 + 0 = **9**. The letters b, d, p and q are the same shape turned or flipped, so look at where the stick is: **b** has its stick up on the left.

  - id: ad1-train-lists
    topic: "Comparing lists"
    difficulty: 2
    q: "List A and List B are meant to hold the same six train numbers in a different order. Which of these numbers is in List A but not in List B?"
    figure: { kind: "table", header: ["#", "List A", "List B"], rows: [[1, "12951", "16526"], [2, "16526", "12627"], [3, "22691", "22961"], [4, "12627", "17230"], [5, "11013", "12951"], [6, "17230", "11013"]] }
    options: ["12951", "22691", "11013", "17230"]
    answer: B
    explanation: |
      Tick each List A number when you find it in List B. **22691** has no match: List B has 22961, with two digits swapped. The other three appear in both lists.

  - id: ad1-sentences
    topic: "Comparing lists"
    difficulty: 2
    q: "How many words or numbers are written differently in these two sentences?\n**1:** The blue bus to Mysuru leaves from gate 6 at half past nine.\n**2:** The blue bus to Mysore leaves from gate 9 at half past six."
    options: ["1", "2", "3", "4"]
    answer: C
    explanation: |
      Mysuru / Mysore, 6 / 9 and nine / six: **3** words. The gate number and the time have been swapped between them, so each counts once.

  - id: ad1-seven-then-four
    topic: "Counting in a grid"
    difficulty: 3
    q: "Read each row from left to right. How many times is a 7 followed straight away by a 4 in the same row?"
    figure: { kind: "grid", cells: [[4, 4, 7, 7, 7, 4, 2], [4, 1, 4, 4, 4, 7, 4], [7, 7, 2, 7, 1, 7, 7], [2, 4, 4, 2, 4, 2, 1], [7, 2, 1, 7, 7, 4, 7]] }
    options: ["2", "3", "4", "5"]
    answer: B
    explanation: |
      Move along each row looking only for a 7, then check the digit just to its right. Row by row: 1 + 1 + 0 + 0 + 1 = **3**.
      Don't count 4 followed by 7 (there are 3 of those), and don't join the end of one row to the start of the next.

  - id: ad1-common-codes
    topic: "Comparing lists"
    difficulty: 3
    q: "How many codes appear in **both** lists exactly? (The order of the lists is different.)"
    figure: { kind: "table", header: ["#", "List A", "List B"], rows: [[1, "R7K2", "H2N5"], [2, "M3P8", "Q9T4"], [3, "Q9T4", "M3B8"], [4, "B6W1", "L4C7"], [5, "H2N5", "B6W7"], [6, "X8D3", "X8D3"], [7, "L4C7", "R7K2"]] }
    options: ["2", "3", "4", "5"]
    answer: D
    explanation: |
      Take each code in List A and look for it in List B: R7K2, Q9T4, H2N5, X8D3, L4C7 are in both, so **5**.
      M3P8 / M3B8 and B6W1 / B6W7 look alike but are not the same.

  - id: ad1-count-ones
    topic: "Counting in a grid"
    difficulty: 3
    q: "How many times is the **digit** 1 written in this grid?"
    figure: { kind: "grid", cells: [[11, 47, 101, 36], [71, 19, 8, 110], [52, 111, 14, 60], [17, 93, 41, 21]], size: 52 }
    options: ["10", "14", "15", "16"]
    answer: C
    explanation: |
      Count digits, not numbers: 11, 101 and 110 have two 1s each and 111 has three. Row by row: 4 + 4 + 4 + 3 = **15**.
      10 is the number of *boxes* that contain a 1.
---

This set tests how carefully you compare and count: spotting tiny differences between codes, names and lists, and counting a symbol in a grid without missing one.

## Comparing two things

- **One character at a time.** Put your finger (or your eyes) on the first character of each, then move both one step. Don't read the whole word: your brain fills in what it expects.
- **Know the usual tricks:** two neighbours swapped (45 / 54), look-alikes (0 and O, 1 and I, 6 and 9, b and d), a letter dropped or added in the middle of a long name.
- A swap of two neighbours changes **two** positions.

## Counting in a grid

- Go **row by row** and write (or say) the count for each row, then add. Jumping around the grid is how symbols get missed or counted twice.
- If the question asks for a *pattern* (a 7 followed by a 4), look for the first symbol, then check its neighbour. "Followed by" means the next one in reading order, inside the same row.
- Counting **digits** is different from counting **numbers**: 111 has three 1s.

## Comparing two lists

Take each item from the first list and find it in the second list, ticking as you go. An item that looks almost right (12627 and 12672) is **not** a match.

## Worked example

*How many places differ between `AB12-XY` and `AB21-YX`?* Go position by position: A and A match, B and B match, then 1/2 and 2/1 differ, the dashes match, and X/Y and Y/X differ. That is **4** places.

## Tip

Accuracy first, speed second. In a real test, a slow correct answer beats a fast wrong one.
