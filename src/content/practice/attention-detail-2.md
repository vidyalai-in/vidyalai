---
title: "Attention to detail 2: following instructions and checking"
description: "Follow step-by-step instructions on letters, numbers and grids, then find the wrong number in a bill or the record that doesn't match."
slug: attention-detail-2
family: attention-detail
part: 2
timeMinutes: 12
questions:
  - id: ad2-planet
    topic: "Following instructions"
    difficulty: 1
    q: "Write the word **PLANET**. Cross out every letter that comes straight after a vowel. Which letters are left, in order?"
    options: ["PAET", "PLAE", "PLAET", "PLNT"]
    answer: B
    explanation: |
      The vowels are A and E. The letter after A is N and the letter after E is T, so cross out N and T.
      That leaves **PLAE**. PAET crosses out the letters *before* the vowels, and PLNT crosses out the vowels themselves.

  - id: ad2-fourth-letter
    topic: "Following instructions"
    difficulty: 1
    q: "Cross out every vowel in this row of letters. Which letter is then 4th from the left?"
    figure: { kind: "grid", cells: [["R", "T", "A", "M", "E", "K", "O", "S", "U", "B"]], size: 38 }
    options: ["E", "M", "S", "K"]
    answer: D
    explanation: |
      After crossing out A, E, O and U the row reads R T M K S B. The 4th letter is **K**.
      M is the 4th letter *before* crossing anything out.

  - id: ad2-bill-line
    topic: "Checking bills"
    difficulty: 1
    q: "Each amount should be quantity × rate. Which line of this bill has the wrong amount?"
    figure: { kind: "table", header: ["Item", "Quantity", "Rate (₹)", "Amount (₹)"], rows: [["Pencils", 4, 6, 24], ["Erasers", 3, 5, 15], ["Notebooks", 2, 35, 60], ["Sharpener", 1, 12, 12]] }
    options: ["Pencils", "Erasers", "Notebooks", "Sharpener"]
    answer: C
    explanation: |
      Check each line: 4 × 6 = 24 ✓, 3 × 5 = 15 ✓, 2 × 35 = 70, not 60 ✗, 1 × 12 = 12 ✓. So **Notebooks** is wrong.

  - id: ad2-parcel
    topic: "Matching records"
    difficulty: 1
    q: "A courier checks the address on each parcel against the address in the register. For which parcel do the two addresses NOT match?"
    figure: { kind: "table", header: ["Parcel", "On the parcel", "In the register"], rows: [["1", "14/B, Gandhi Nagar, Nashik", "14/8, Gandhi Nagar, Nashik"], ["2", "7, Lake Road, Bhopal", "7, Lake Road, Bhopal"], ["3", "22, MG Road, Kochi", "22, MG Road, Kochi"], ["4", "9/A, Park Street, Kolkata", "9/A, Park Street, Kolkata"]] }
    options: ["Parcel 1", "Parcel 2", "Parcel 3", "Parcel 4"]
    answer: A
    explanation: |
      In **Parcel 1** the parcel says 14/B (the letter B) but the register says 14/8 (the digit 8). The other three match exactly.

  - id: ad2-mathematics
    topic: "Following instructions"
    difficulty: 2
    q: "In the word **MATHEMATICS**, cross out every letter that comes straight after a vowel. How many letters are left?"
    options: ["5", "6", "7", "8"]
    answer: C
    explanation: |
      The vowels are A, E, A and I. The letters straight after them are T, M, T and C: 4 letters crossed out.
      11 − 4 = **7** letters are left (MAHEAIS). 4 counts the crossed-out letters instead.

  - id: ad2-bill-totals
    topic: "Checking bills"
    difficulty: 2
    q: "Each row shows the prices on a bill and the total printed on it. Which bill has the wrong total?"
    figure: { kind: "table", header: ["Bill", "Prices (₹)", "Printed total (₹)"], rows: [["P", "45 + 120 + 35", 200], ["Q", "68 + 74 + 19", 161], ["R", "250 + 85 + 60", 385], ["S", "132 + 47 + 21", 200]] }
    options: ["Bill P", "Bill Q", "Bill R", "Bill S"]
    answer: C
    explanation: |
      Add each row: P: 45 + 120 + 35 = 200 ✓. Q: 68 + 74 + 19 = 161 ✓. R: 250 + 85 + 60 = 395, not 385 ✗. S: 132 + 47 + 21 = 200 ✓.
      So **Bill R** is wrong: its printed total is 10 short.

  - id: ad2-roll-numbers
    topic: "Matching records"
    difficulty: 2
    q: "The class and roll number in the register should match each student's ID card. Whose ID card does NOT match?"
    figure: { kind: "table", header: ["Student", "Register", "ID card"], rows: [["Asha", "7B-14", "7B-14"], ["Farhan", "8A-09", "8A-09"], ["Neha", "7C-21", "7C-21"], ["Vikram", "8B-16", "8B-61"], ["Simran", "8A-30", "8A-30"]] }
    options: ["Vikram", "Asha", "Neha", "Simran"]
    answer: A
    explanation: |
      **Vikram**: the register says 8B-16 but his ID card says 8B-61, the two digits swapped. Every other row matches.

  - id: ad2-bigger-than-before
    topic: "Following instructions"
    difficulty: 2
    q: "Circle every number in this row that is bigger than the number just before it. How many numbers do you circle?"
    figure: { kind: "grid", cells: [[3, 8, 5, 2, 9, 6, 4, 7, 1, 5]], size: 38 }
    options: ["1", "2", "3", "4"]
    answer: D
    explanation: |
      Compare each number with its left neighbour: 8 > 3 ✓, 9 > 2 ✓, 7 > 4 ✓, 5 > 1 ✓. The others are smaller than the number before them.
      So **4** circles. The first number has nothing before it, so it is never circled.

  - id: ad2-grid-moves
    topic: "Following instructions"
    difficulty: 2
    q: "Start on the shaded square (H). Move 3 squares right, then 2 squares down, then 1 square left. Which letter are you on now?"
    figure: { kind: "grid", cells: [["H", "R", "W", "K", "P"], ["T", "C", "M", "B", "Z"], ["F", "N", "G", "L", "S"], ["D", "Y", "Q", "V", "J"]], shade: [[0, 0]] }
    options: ["G", "L", "M", "N"]
    answer: A
    explanation: |
      3 right from H takes you to K. 2 down from K takes you to B, then L. 1 left from L is **G**.
      L is where you are if you forget the last move.

  - id: ad2-cricket
    topic: "Following instructions"
    difficulty: 3
    q: "Write **CRICKET**. Step 1: swap the first and last letters. Step 2: change every vowel to the letter that comes just after it in the alphabet. What do you get?"
    options: ["CRJCKFT", "TRHCKDC", "TRJCKEC", "TRJCKFC"]
    answer: D
    explanation: |
      Step 1: CRICKET → TRICKEC. Step 2: I becomes J and E becomes F: **TRJCKFC**.
      TRHCKDC uses the letter *before* each vowel, CRJCKFT skips step 1, and TRJCKEC forgets to change the E.

  - id: ad2-one-wrong
    topic: "Checking bills"
    difficulty: 3
    q: "Exactly **one** number in this bill is wrong. Which one?"
    figure: { kind: "table", header: ["Item", "Quantity", "Rate (₹)", "Amount (₹)"], rows: [["Rice (kg)", 5, 64, 320], ["Dal (kg)", 2, 118, 236], ["Oil (litre)", 3, 145, 425], ["Sugar (kg)", 4, 46, 184], ["Total", "", "", 1175]] }
    options: ["The amount for dal", "The amount for oil", "The amount for sugar", "The total"]
    answer: B
    explanation: |
      Check every line first: 5 × 64 = 320 ✓, 2 × 118 = 236 ✓, 3 × 145 = 435, not 425 ✗, 4 × 46 = 184 ✓.
      With oil corrected, 320 + 236 + 435 + 184 = 1175, which matches the total. So **the amount for oil** is the one wrong number.
      The shown amounts add to 1165, which tempts you to blame the total, but then the oil line would still be wrong, making two mistakes.

  - id: ad2-phone-lists
    topic: "Matching records"
    difficulty: 3
    q: "A shop's customer list and a courier's list should show the same phone numbers. In how many rows do the two numbers NOT match exactly?"
    figure: { kind: "table", header: ["Name", "Shop's list", "Courier's list"], rows: [["Ravi", "98201 45763", "98201 45763"], ["Meera", "99674 20318", "99674 23018"], ["Joseph", "90045 61287", "90045 61287"], ["Priya", "87543 09126", "87543 09162"], ["Kabir", "93218 77450", "93218 77450"], ["Anjali", "80916 34502", "80961 34502"]] }
    options: ["2", "3", "4", "5"]
    answer: B
    explanation: |
      Compare digit by digit, in chunks of five. Meera (20318 / 23018), Priya (09126 / 09162) and Anjali (80916 / 80961) differ: **3** rows.
      Ravi, Joseph and Kabir match exactly.
---

This set tests whether you can follow instructions exactly and check numbers quickly: crossing out letters by a rule, moving on a grid, and finding the one wrong number in a bill or a list.

## Following instructions

- **Read the whole instruction before you start.** "After a vowel" and "before a vowel" give different answers.
- **Apply the rule to the original.** When you cross out letters, decide which ones to cross using the word as it was, then cross them all at once.
- **Do the steps in the order given**, and write down the result of each step.
- On a grid, move one square at a time and touch each square as you count.

## Checking bills and records

- Check **every line** (quantity × rate), then the total. A total that doesn't match can be caused by a wrong line.
- For totals, a quick test with the units digits catches many errors: 68 + 74 + 19 must end in 8 + 4 + 9 = 21, so in 1.
- Compare codes, phone numbers and addresses in small chunks: 98201 45763 is two chunks of five.

## Worked example

*Cross out every letter that comes straight after a consonant in TEACHER.* The consonants are T, C, H and R. The letters after them are E (after T), H (after C) and E (after H); R is last, so nothing comes after it. What is left: **T A C R**.

## Common traps

- Counting the crossed-out letters when the question asks how many are **left**.
- The 4th letter "after crossing out" is not the 4th letter of the original row.
- Swapped neighbours (16 and 61) are the most common copying mistake.
