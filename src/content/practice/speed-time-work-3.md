---
title: "Speed, time and work 3: time and work, and ages"
description: "People working together, pipes filling and leaks emptying a tank, and age puzzles about the past and the future."
slug: speed-time-work-3
family: speed-time-work
part: 3
timeMinutes: 12
questions:
  - id: stw3-together
    topic: Working together
    difficulty: 1
    q: "Asha can paint a room in 10 days. Ravi can paint the same room in 15 days. How long will they take if they work together?"
    options: ["5 days", "6 days", "12.5 days", "25 days"]
    answer: B
    explanation: |
      In one day Asha does 1/10 and Ravi 1/15 of the room. Together: 1/10 + 1/15 = 3/30 + 2/30 = 1/6.
      So they finish in **6 days**. 12.5 days averages the two times, but two people are faster than either one alone.

  - id: stw3-mother-age
    topic: Ages
    difficulty: 1
    q: "Priya is 12 years old. Her mother is 3 times as old as Priya. How old will her mother be when Priya is 20?"
    options: ["36", "40", "44", "60"]
    answer: C
    explanation: |
      Mother's age now = 3 × 12 = 36. Priya turns 20 in 8 years, and her mother also gets 8 years older: 36 + 8 = **44**.
      60 assumes the mother stays 3 times as old, but the age *difference* stays the same, not the ratio.

  - id: stw3-sum-difference
    topic: Ages
    difficulty: 1
    q: "The ages of Meera and her brother Arjun add up to 30 years. Arjun is 4 years older than Meera. How old is Meera?"
    options: ["13 years", "15 years", "17 years", "26 years"]
    answer: A
    explanation: |
      Take away the difference and share the rest equally: (30 − 4) ÷ 2 = **13 years**. Arjun is 17.
      Check: 13 + 17 = 30 ✓ and 17 − 13 = 4 ✓.

  - id: stw3-part-of-job
    topic: Working together
    difficulty: 1
    q: "Kabir does 1/5 of a job in 2 days. Working at the same rate, how many days will he take for the whole job?"
    options: ["2.5 days", "7 days", "10 days", "12 days"]
    answer: C
    explanation: |
      The whole job is 5 times 1/5 of it, so it takes 5 × 2 = **10 days**.
      2.5 days divides 2 by 4/5, and 7 days adds 2 and 5.

  - id: stw3-tap-and-leak
    topic: Pipes and tanks
    difficulty: 2
    q: "A tap fills a tank in 8 hours. A leak at the bottom can empty the full tank in 12 hours. If the tank is empty and both are working, how long will it take to fill?"
    options: ["4 hours", "10 hours", "20 hours", "24 hours"]
    answer: D
    explanation: |
      Each hour the tap fills 1/8 and the leak empties 1/12. Net filling per hour: 1/8 − 1/12 = 3/24 − 2/24 = 1/24.
      So the tank fills in **24 hours**. 4 hours is 12 − 8, which mixes up hours and work per hour.

  - id: stw3-find-b
    topic: Working together
    difficulty: 2
    q: "Neha and Vikram together can finish a project in 12 days. Neha alone takes 20 days. How long would Vikram take alone?"
    options: ["8 days", "16 days", "30 days", "32 days"]
    answer: C
    explanation: |
      Vikram's work per day = together − Neha = 1/12 − 1/20 = 5/60 − 3/60 = 2/60 = 1/30.
      So Vikram alone takes **30 days**. 8 days (20 − 12) would make him faster than both together, which is impossible.

  - id: stw3-three-pipes
    topic: Pipes and tanks
    difficulty: 2
    q: "Pipe P fills a tank in 10 hours and pipe Q fills it in 15 hours. Pipe R empties the full tank in 30 hours. If all three are opened together on an empty tank, how long will it take to fill?"
    options: ["6 hours", "7.5 hours", "8 hours", "12.5 hours"]
    answer: B
    explanation: |
      Per hour: 1/10 + 1/15 − 1/30 = 3/30 + 2/30 − 1/30 = 4/30 = 2/15 of the tank.
      Time = 15/2 = **7.5 hours**. 6 hours forgets that R is taking water out.

  - id: stw3-father-son
    topic: Ages
    difficulty: 2
    q: "A father is 4 times as old as his son. In 5 years, the father will be 3 times as old as the son. How old is the son now?"
    options: ["10 years", "15 years", "20 years", "40 years"]
    answer: A
    explanation: |
      Test the options. Son 10, father 40. In 5 years: 15 and 45, and 45 = 3 × 15 ✓. So the son is **10 years** old.
      15 is the son's age in 5 years, and 40 is the father's age now.

  - id: stw3-age-ratio
    topic: Ages
    difficulty: 2
    q: "The ages of Anjali and her cousin Vikram are in the ratio 3 : 5. After 6 years, the ratio will be 3 : 4. How old is Anjali now?"
    options: ["6 years", "9 years", "10 years", "12 years"]
    answer: A
    explanation: |
      Try ages in the ratio 3 : 5: (3, 5), (6, 10), (9, 15), ... Add 6 years to each.
      (6, 10) becomes (12, 16) = 3 : 4 ✓. So Anjali is **6 years** old. 10 is Vikram's age and 12 is Anjali's age after 6 years.

  - id: stw3-a-leaves
    topic: Working together
    difficulty: 3
    q: "Farhan can build a model in 12 days and Joseph in 18 days. They work together for 4 days, then Farhan leaves. How many more days does Joseph need to finish it?"
    options: ["6 days", "8 days", "10 days", "14 days"]
    answer: B
    explanation: |
      Together per day: 1/12 + 1/18 = 3/36 + 2/36 = 5/36. In 4 days: 20/36 = 5/9 of the model.
      Left: 4/9. Joseph does 1/18 per day, so he needs 4/9 × 18 = **8 days**.
      14 days (18 − 4) ignores Farhan's share of the first 4 days.

  - id: stw3-leak
    topic: Pipes and tanks
    difficulty: 3
    q: "A pipe can fill a tank in 5 hours. Because of a leak, it actually takes 6 hours to fill. If the tank is full and the pipe is closed, how long will the leak take to empty it?"
    options: ["1 hour", "11 hours", "30 hours", "60 hours"]
    answer: C
    explanation: |
      Per hour: pipe alone 1/5, pipe with leak 1/6. The leak takes away 1/5 − 1/6 = 1/30 every hour.
      So the leak empties the full tank in **30 hours**. 1 hour is just 6 − 5: the leak is slow, not fast.

  - id: stw3-mother-daughter
    topic: Ages
    difficulty: 3
    q: "Five years ago, Simran's mother was 3 times as old as Simran. Five years from now, she will be twice as old as Simran. How old is Simran's mother now?"
    options: ["15 years", "25 years", "30 years", "35 years"]
    answer: D
    explanation: |
      Let Simran's age five years ago be x, so her mother's was 3x. Ten years after that they are x + 10 and 3x + 10.
      3x + 10 = 2(x + 10) gives x = 10. Five years ago they were 10 and 30, so the mother is now **35 years** old.
      Check: in five years, 40 = 2 × 20 ✓. 15 years is Simran's age now, not her mother's.
---

This set tests time-and-work problems, pipes and tanks, and age puzzles. One idea runs through the first two: count the **work done in one unit of time**.

## Work per day

If someone finishes a job in 10 days, they do **1/10** of it each day. To combine people, **add** their daily work, then flip the answer to get the days.

| Situation | Work per day (or hour) |
|---|---|
| two people together | 1/a + 1/b |
| a filling pipe and a leak | 1/fill − 1/empty |
| one person, once you know the pair | 1/together − 1/other |

Never add or average the *days*. Two people are always faster than either one alone.

## Ages

- Everyone gets older by the **same** number of years, so the **age difference never changes**.
- "Five years ago" means take 5 from everyone's age; "in five years" means add 5.
- In an MCQ, testing the options is often the quickest method.

## Worked example

*A can do a job in 6 days and B in 3 days. How long together?*

Per day: 1/6 + 1/3 = 1/6 + 2/6 = 3/6 = 1/2. Half the job each day, so **2 days**.

## Common trap

When part of a job is already done, first work out the fraction that is **left**, then divide it by the work per day of whoever is still working.
