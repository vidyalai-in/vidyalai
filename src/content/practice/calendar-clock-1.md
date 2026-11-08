---
title: "Calendar and clock 1: days, dates and leap years"
description: "Finding the day of the week after or before some days, odd days, leap years, years with the same calendar, and birthdays."
slug: calendar-clock-1
family: calendar-clock
part: 1
timeMinutes: 12
questions:
  - id: cc1-after-30-days
    topic: Counting days
    difficulty: 1
    q: "Today is Wednesday. What day of the week will it be 30 days from today?"
    options: ["Thursday", "Friday", "Saturday", "Sunday"]
    answer: B
    explanation: |
      Every 7 days the day of the week repeats. 30 = 4 × 7 + 2, so only the 2 extra days matter.
      Wednesday + 2 days = **Friday**.

  - id: cc1-which-leap
    topic: Leap years
    difficulty: 1
    q: "Which of these years is a leap year?"
    options: ["1900", "2018", "2024", "2100"]
    answer: C
    explanation: |
      **2024** is divisible by 4 and is not a century year, so it is a leap year.
      1900 and 2100 are century years, and a century year is a leap year only if it is divisible by 400. 2018 is not divisible by 4.

  - id: cc1-odd-days-year
    topic: Odd days
    difficulty: 1
    q: "How many odd days are there in an ordinary year of 365 days?"
    options: ["0", "1", "2", "5"]
    answer: B
    explanation: |
      Odd days are the days left over after taking out whole weeks. 365 = 52 × 7 + 1, so there is **1** odd day.
      A leap year (366 days) has 2 odd days.

  - id: cc1-new-year-2028
    topic: Odd days
    difficulty: 1
    q: "1 January 2027 is a Friday. What day of the week is 1 January 2028?"
    options: ["Saturday", "Sunday", "Monday", "Tuesday"]
    answer: A
    explanation: |
      From 1 January 2027 to 1 January 2028 is the whole of 2027. 2027 is not a leap year, so it has 1 odd day.
      Friday + 1 = **Saturday**. Sunday would be right only if 2027 had 29 February.

  - id: cc1-before-45-days
    topic: Counting days
    difficulty: 2
    q: "Arjun's cricket match is on a Sunday. What day of the week was it 45 days before the match?"
    options: ["Monday", "Tuesday", "Wednesday", "Thursday"]
    answer: D
    explanation: |
      45 = 6 × 7 + 3, so there are 3 odd days. Going **back** 3 days from Sunday: Saturday, Friday, **Thursday**.
      Wednesday is what you get by going forward 3 days by mistake.

  - id: cc1-third-thursday
    topic: Dates in a month
    difficulty: 2
    q: "In a certain month, the first Monday falls on the 5th. What is the date of the third Thursday of that month?"
    options: ["15th", "18th", "19th", "22nd"]
    answer: A
    explanation: |
      The 5th is a Monday, so count back: the 4th is Sunday, the 3rd Saturday, the 2nd Friday and the 1st Thursday.
      Thursdays are the 1st, 8th and 15th, so the third Thursday is the **15th**.
      22nd comes from taking the 8th as the first Thursday.

  - id: cc1-days-inclusive
    topic: Counting days
    difficulty: 2
    q: "Priya's school drama practice runs every day from 10 March to 25 May, both days included. On how many days is there practice?"
    options: ["75", "76", "77", "78"]
    answer: C
    explanation: |
      March: 10th to 31st is 22 days (31 − 10 + 1). April: 30 days. May: 1st to 25th is 25 days.
      22 + 30 + 25 = **77**. 76 forgets to count the first day, 10 March.

  - id: cc1-republic-day
    topic: Odd days
    difficulty: 2
    q: "Republic Day, 26 January 2027, is a Tuesday. What day of the week is 26 January 2029?"
    options: ["Wednesday", "Thursday", "Friday", "Saturday"]
    answer: C
    explanation: |
      26 Jan 2027 to 26 Jan 2028: the 365 days of an ordinary year, 1 odd day.
      26 Jan 2028 to 26 Jan 2029 includes 29 February 2028: 366 days, 2 odd days.
      Total 3 odd days: Tuesday + 3 = **Friday**. Thursday misses the leap day.

  - id: cc1-same-as-2026
    topic: Same calendar
    difficulty: 2
    q: "Which is the next year after 2026 that has exactly the same calendar as 2026?"
    options: ["2032", "2034", "2036", "2037"]
    answer: D
    explanation: |
      The years must start on the same day and both be ordinary (not leap) years. Add up odd days from 2026: 2026 (1), 2027 (1), 2028 (2), 2029 (1), 2030 (1), 2031 (1), 2032 (2), 2033 (1), 2034 (1), 2035 (1), 2036 (2).
      The total is 14, a multiple of 7, so **2037** starts on the same day as 2026 and is also ordinary.
      2032 is a leap year, so its calendar can never match.

  - id: cc1-same-as-2028
    topic: Same calendar
    difficulty: 3
    q: "2028 is a leap year. Which is the next year after it with exactly the same calendar as 2028?"
    options: ["2034", "2039", "2052", "2056"]
    answer: D
    explanation: |
      The year must also be a leap year, so try 2032, 2036, ... Each 4-year block has 1 + 1 + 1 + 2 = 5 odd days.
      We need the odd days to add up to a multiple of 7: 5 × 7 = 35, which takes 7 blocks of 4 years, that is 28 years.
      2028 + 28 = **2056**. 2034 and 2039 are not even leap years.

  - id: cc1-birthday-born
    topic: Birthdays
    difficulty: 3
    q: "Meera's 13th birthday is on Saturday, 14 November 2026. On which day of the week was she born?"
    options: ["Wednesday", "Thursday", "Saturday", "Sunday"]
    answer: B
    explanation: |
      She was born on 14 November 2013. The 13 years between include three 29 Februaries (2016, 2020 and 2024).
      Odd days = 13 + 3 = 16 = 2 × 7 + 2, so go back 2 days from Saturday: **Thursday**.
      Sunday forgets the leap days; Wednesday also counts 2012, which is before her birth.

  - id: cc1-year-2100
    topic: Leap years
    difficulty: 3
    q: "1 January 2000 was a Saturday. What day of the week will 1 January 2100 be?"
    options: ["Thursday", "Friday", "Saturday", "Sunday"]
    answer: B
    explanation: |
      The years 2000 to 2099 come in between. Leap years among them: 2000, 2004, ..., 2096, which is 25 years. 2000 counts, because it is divisible by 400.
      Odd days = 100 + 25 = 125 = 17 × 7 + 6. Saturday + 6 = **Friday**.
      Thursday comes from leaving out 2000 as a leap year.
---

Calendar questions are about counting days, but you never need to count all of them. You only need the days left over after whole weeks.

## Odd days

Take any number of days and remove complete weeks of 7. What is left over is the number of **odd days**, and it tells you how far the day of the week moves.

| Period | Days | Odd days |
|---|---|---|
| Ordinary year | 365 | 1 |
| Leap year | 366 | 2 |
| 30-day month | 30 | 2 |
| 31-day month | 31 | 3 |

Count forward to go into the future and backward to go into the past.

## Leap years

A year is a leap year if it is divisible by 4. **But** a century year (1900, 2000, 2100) is a leap year only if it is divisible by 400. So 2000 was a leap year and 2100 will not be.

When you jump from a date in one year to the same date in the next, check whether a 29 February falls **between** the two dates. That decides 1 or 2 odd days.

## Worked example

15 August 2026 is a Saturday. What day is 15 August 2027? The year in between contains no 29 February (2027 is ordinary), so 1 odd day: Sunday.

## Same calendar

Two years have the same calendar when they start on the same day **and** are both leap or both ordinary. Add the odd days year by year until the total is a multiple of 7.

## Common traps

- Inclusive counting: "from the 3rd to the 10th, both included" is 8 days, not 7.
- Forgetting that the leap day matters only if 29 February lies inside the period.
