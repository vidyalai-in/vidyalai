---
title: "Ranking and order 1: positions in a row or class"
description: "Positions from the left and the right, or from the top and the bottom: totals, people in between, and what changes when two people swap places."
slug: ranking-order-1
family: ranking-order
part: 1
timeMinutes: 12
questions:
  - id: ro1-asha-row
    topic: Totals
    difficulty: 1
    q: "In a row of students, Asha is 8th from the left end and 12th from the right end. How many students are in the row?"
    options: ["18", "19", "20", "21"]
    answer: B
    explanation: |
      Total = left position + right position − 1 = 8 + 12 − 1 = **19**.
      Subtract 1 because Asha is counted from both ends. 20 forgets this.

  - id: ro1-ravi-rank
    topic: Positions
    difficulty: 1
    q: "In a class of 40 students, Ravi's rank is 15th from the top. What is his rank from the bottom?"
    options: ["24", "25", "26", "27"]
    answer: C
    explanation: |
      Rank from the bottom = total − rank from the top + 1 = 40 − 15 + 1 = **26**.
      25 forgets to add 1: there are 25 students below Ravi, so he is the 26th from the bottom.

  - id: ro1-meera-queue
    topic: Totals
    difficulty: 1
    q: "Meera is 6th in a queue at a ticket counter, counting from the front. There are 9 people behind her. How many people are in the queue?"
    options: ["14", "15", "16", "17"]
    answer: B
    explanation: |
      Meera and the 5 people in front of her make 6. Add the 9 behind her: 6 + 9 = **15**.
      Do not subtract 1 here: the 9 people behind her do not include Meera.

  - id: ro1-between-left
    topic: Between
    difficulty: 1
    q: "In a row, Kabir is 7th from the left end and Priya is 12th from the left end. How many children are there between Kabir and Priya?"
    options: ["4", "5", "6", "19"]
    answer: A
    explanation: |
      The children between them are in places 8, 9, 10 and 11. That is 12 − 7 − 1 = **4**.
      5 is the difference in places; it counts Priya as well.

  - id: ro1-arjun-neha
    topic: Between
    difficulty: 2
    q: "There are 30 children in a row. Arjun is 10th from the left end and Neha is 8th from the right end. How many children are there between Arjun and Neha?"
    options: ["11", "12", "13", "14"]
    answer: B
    explanation: |
      First change Neha's place to 'from the left': 30 − 8 + 1 = 23rd from the left.
      The children between them are in places 11 to 22: 23 − 10 − 1 = **12**.

  - id: ro1-swap-total
    topic: Interchanging
    difficulty: 2
    q: "In a row, Joseph is 9th from the left end and Farhan is 7th from the right end. They swap places. Now Joseph is 15th from the left end. How many children are in the row?"
    options: ["15", "20", "21", "22"]
    answer: C
    explanation: |
      After the swap, Joseph stands in Farhan's old place. So that place is 15th from the left and 7th from the right.
      Total = 15 + 7 − 1 = **21**. Joseph's old place (9th from the left) is not needed.

  - id: ro1-diya-esha
    topic: Totals
    difficulty: 2
    q: "In a row of girls, Diya is 11th from the left end. Esha stands 4 places to the right of Diya, nearer the right end. Esha is 9th from the right end. How many girls are in the row?"
    options: ["19", "20", "22", "23"]
    answer: D
    explanation: |
      Esha is 11 + 4 = 15th from the left and 9th from the right.
      Total = 15 + 9 − 1 = **23**. 19 uses Diya's place instead of Esha's.

  - id: ro1-middle
    topic: Positions
    difficulty: 2
    q: "31 children stand in a row. Vikram is exactly in the middle. Anjali stands 5 places to the left of Vikram, nearer the left end. What is Anjali's position from the right end?"
    options: ["11", "16", "20", "21"]
    answer: D
    explanation: |
      The middle of 31 is the 16th place (15 children on each side). Anjali is 16 − 5 = 11th from the left.
      From the right: 31 − 11 + 1 = **21**. 11 is her place from the left end.

  - id: ro1-rahul-rank
    topic: Positions
    difficulty: 2
    q: "In a class of 45 students, Neha's rank is 14th from the top. Rahul's rank is 7 places below Neha's. What is Rahul's rank from the bottom?"
    options: ["25", "31", "32", "39"]
    answer: A
    explanation: |
      'Below' means a lower rank, so a bigger number: Rahul is 14 + 7 = 21st from the top.
      From the bottom: 45 − 21 + 1 = **25**. 39 is what you get by moving 7 places up instead of down.

  - id: ro1-passed-class
    topic: Totals
    difficulty: 3
    q: "Among the students who passed a test, Simran is 9th from the top and 28th from the bottom. 4 students in the class failed and 2 were absent. How many students are in the class?"
    options: ["36", "40", "42", "43"]
    answer: C
    explanation: |
      Students who passed = 9 + 28 − 1 = 36. Add those who failed and those who were absent: 36 + 4 + 2 = **42**.
      40 leaves out the absent students. 43 forgets the − 1.

  - id: ro1-smallest-row
    topic: Between
    difficulty: 3
    q: "Priya is 12th from the left end of a row, and Kabir is 17th from the right end. There are exactly 5 children between them. What is the smallest possible number of children in the row?"
    options: ["22", "28", "34", "35"]
    answer: A
    explanation: |
      Case 1, Kabir to Priya's right: Kabir is 12 + 6 = 18th from the left, so the total is 18 + 17 − 1 = 34.
      Case 2, Kabir to Priya's left: Kabir is 12 − 6 = 6th from the left, so the total is 6 + 17 − 1 = 22. The smallest is **22**.

  - id: ro1-swap-new-place
    topic: Interchanging
    difficulty: 3
    q: "In a row, Isha is 8th from the left end and Tara is 15th from the right end. They swap places, and Isha is now 20th from the left end. What is Tara's position from the right end now?"
    options: ["8", "15", "26", "27"]
    answer: D
    explanation: |
      Isha is now in Tara's old place: 20th from the left and 15th from the right. Total = 20 + 15 − 1 = 34.
      Tara is now in Isha's old place, 8th from the left, so she is 34 − 8 + 1 = **27th** from the right.
---

These questions are about places in a row, a queue or a class ranking. Each one needs only a little arithmetic, but it is easy to be off by one.

## Three rules

| You know | Use |
|---|---|
| Place from the left (L) and from the right (R) | Total = L + R − 1 |
| Total (N) and place from one end (L) | Place from the other end = N − L + 1 |
| Two places from the same end, a and b | Number between them = b − a − 1 |

The − 1 in the first rule is there because the person is counted from both ends. Ranks work the same way: "from the top" is like "from the left" and "from the bottom" is like "from the right".

## Change to one end first

If one place is given from the left and the other from the right, change one of them so both are from the same end. Then count.

*Example:* In a row of 20, Ali is 4th from the left and Sara is 5th from the right. Sara is 20 − 5 + 1 = 16th from the left. Between them: 16 − 4 − 1 = 11 children.

## Swapping places

When two people swap, each one takes the other's old place. Find the place that is described from both ends after the swap, and use it for the total.

## Traps

- "There are 9 people behind her" does not include her, so you simply add.
- If you do not know who is to the left, try both orders. A question may ask for the smallest or the largest total.
- Also count anyone not in the ranking, such as students who were absent.
