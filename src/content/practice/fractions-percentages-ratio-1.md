---
title: "Fractions, percentages and ratio 1: fractions and decimals"
description: "Comparing and ordering fractions and decimals, adding, multiplying and dividing them, finding a fraction of a quantity, and word problems."
slug: fractions-percentages-ratio-1
family: fractions-percentages-ratio
part: 1
timeMinutes: 12
questions:
  - id: fpr1-largest-fraction
    topic: Comparing
    difficulty: 1
    q: "Which of these fractions is the largest?"
    options: ["5/8", "3/5", "7/12", "4/7"]
    answer: A
    explanation: |
      Turn each into a decimal: 5/8 = 0.625, 3/5 = 0.6, 7/12 ≈ 0.583, 4/7 ≈ 0.571.
      The largest is **5/8**. A bigger denominator does not mean a bigger fraction.

  - id: fpr1-add-fractions
    topic: Operations
    difficulty: 1
    q: "What is 2/3 + 3/4?"
    options: ["5/12", "5/7", "1 1/12", "1 5/12"]
    answer: D
    explanation: |
      Use the common denominator 12: 2/3 = 8/12 and 3/4 = 9/12. So 8/12 + 9/12 = 17/12 = **1 5/12**.
      5/7 comes from adding the tops and the bottoms, which is never allowed.

  - id: fpr1-fraction-of-amount
    topic: Fraction of a quantity
    difficulty: 1
    q: "Priya has ₹240. She spends 3/8 of it on a gift. How much does the gift cost?"
    options: ["₹30", "₹80", "₹90", "₹640"]
    answer: C
    explanation: |
      1/8 of ₹240 is 240 ÷ 8 = ₹30, so 3/8 is 3 × 30 = **₹90**.
      ₹30 is only 1/8. ₹640 comes from dividing by 3 and multiplying by 8, the wrong way round.

  - id: fpr1-decimal-product
    topic: Operations
    difficulty: 1
    q: "What is 0.4 × 0.03?"
    options: ["0.0012", "0.012", "0.12", "1.2"]
    answer: B
    explanation: |
      Multiply without the points: 4 × 3 = 12. The two numbers have 1 + 2 = 3 decimal places, so the answer has 3 too: **0.012**.

  - id: fpr1-smallest-value
    topic: Comparing
    difficulty: 2
    q: "Which of these is the smallest?"
    options: ["0.45", "4/9", "0.449", "7/15"]
    answer: B
    explanation: |
      Change the fractions to decimals: 4/9 = 0.444... and 7/15 = 0.466...
      Now compare to three places: 0.450, 0.444, 0.449, 0.467. The smallest is **4/9**.
      0.449 is smaller than 0.45, but not smaller than 4/9.

  - id: fpr1-divide-mixed
    topic: Operations
    difficulty: 2
    q: "What is 1 1/2 ÷ 3/4?"
    options: ["1/2", "8/9", "1 1/8", "2"]
    answer: D
    explanation: |
      Write 1 1/2 as 3/2. To divide by 3/4, multiply by its upside-down 4/3: 3/2 × 4/3 = 12/6 = **2**.
      1 1/8 is what you get by multiplying 3/2 × 3/4 instead of dividing.

  - id: fpr1-glasses
    topic: Fraction of a quantity
    difficulty: 2
    q: "In a class, 3/5 of the students are girls and the rest are boys. 1/4 of the girls and 1/3 of the boys wear glasses. What fraction of the whole class wears glasses?"
    options: ["1/12", "17/60", "7/24", "7/12"]
    answer: B
    explanation: |
      Girls with glasses: 1/4 of 3/5 = 3/20. Boys are 2/5 of the class, so boys with glasses: 1/3 of 2/5 = 2/15.
      3/20 + 2/15 = 9/60 + 8/60 = **17/60**.
      7/12 just adds 1/4 + 1/3, but those are fractions of different groups.

  - id: fpr1-tank
    topic: Word problems
    difficulty: 2
    q: "A water tank is 3/5 full. After 24 litres are used, it is 1/3 full. How many litres does the full tank hold?"
    options: ["40 litres", "72 litres", "90 litres", "120 litres"]
    answer: C
    explanation: |
      The 24 litres used are the drop from 3/5 to 1/3: 3/5 − 1/3 = 9/15 − 5/15 = 4/15 of the tank.
      If 4/15 is 24 litres, then 1/15 is 6 litres and the full tank is 15 × 6 = **90 litres**.

  - id: fpr1-money-left
    topic: Word problems
    difficulty: 2
    q: "Ravi spends 1/3 of his money on books. Then he spends 1/4 of the money that is left on a bag. He now has ₹360. How much did he have at the start?"
    options: ["₹720", "₹864", "₹1080", "₹1440"]
    answer: A
    explanation: |
      After the books, 2/3 is left. The bag takes 1/4 of that, so 3/4 of 2/3 = 1/2 of his money is left.
      Half his money is ₹360, so he started with **₹720**.
      ₹864 comes from taking 1/4 of the *whole* amount for the bag, not 1/4 of what was left.

  - id: fpr1-find-fraction
    topic: Word problems
    difficulty: 3
    q: "If 1 is added to the numerator of a fraction, it becomes 1/2. If instead 3 is added to its denominator, it becomes 1/3. What is the fraction?"
    options: ["1/4", "3/8", "5/12", "7/16"]
    answer: C
    explanation: |
      All four options pass the first test (2/4, 4/8, 6/12, 8/16 are all 1/2), so check the second one.
      1/7, 3/11 and 7/19 are not 1/3, but 5/(12 + 3) = 5/15 = 1/3 ✓. The fraction is **5/12**.

  - id: fpr1-product-pattern
    topic: Operations
    difficulty: 3
    q: "What is (1 − 1/2) × (1 − 1/3) × (1 − 1/4) × ... × (1 − 1/20)?"
    options: ["1/20", "1/19", "1/10", "19/20"]
    answer: A
    explanation: |
      Each bracket is a simple fraction: 1/2 × 2/3 × 3/4 × ... × 19/20.
      Every top cancels the bottom just before it, leaving only the first top (1) and the last bottom (20): **1/20**.

  - id: fpr1-bouncing-ball
    topic: Word problems
    difficulty: 3
    q: "A ball is dropped from a height of 64 m. Each time it hits the ground, it bounces back up to 3/4 of the height it fell from. How far has it travelled in all when it hits the ground for the third time?"
    options: ["148 m", "196 m", "232 m", "268 m"]
    answer: C
    explanation: |
      Down 64 (1st hit). Up 48 and down 48 (2nd hit). Up 36 and down 36 (3rd hit).
      Total: 64 + 96 + 72 = **232 m**.
      148 m counts each height only once, but after every bounce the ball goes up *and* comes down.
---

This set tests how comfortably you handle fractions and decimals: comparing them, working with them and using them in short stories.

## The rules

| To | Do this |
|---|---|
| compare fractions | turn them into decimals, or give them the same denominator |
| add or subtract | make the denominators the same first |
| multiply | top × top and bottom × bottom |
| divide | flip the second fraction, then multiply |
| multiply decimals | multiply as whole numbers, then count the decimal places in both |
| find 3/8 of a number | divide by 8, then multiply by 3 |

## "Of the rest" questions

Read carefully what each fraction is a fraction *of*. "1/4 of the rest" is 1/4 of what is left, not 1/4 of the whole. A safe method is to track what is left after each step as a fraction of the whole.

## Worked example

*Meera reads 2/5 of a book on Monday and 1/2 of the remaining pages on Tuesday. 45 pages are left. How many pages does the book have?*

After Monday, 3/5 is left. Tuesday takes half of that, so 3/10 of the book is left. If 3/10 is 45 pages, then 1/10 is 15 pages, and the whole book is **150 pages**.

## Tip: let the options help

When a question gives two conditions, check the options against the harder one first. Often several options pass the easy test and only one passes both.
