---
title: "Calendar and clock 2: clock hands, faulty clocks and mirrors"
description: "Angles between the hands, when the hands meet, point opposite or make a right angle, clocks that gain or lose time, and mirror images of a clock."
slug: calendar-clock-2
family: calendar-clock
part: 2
timeMinutes: 12
questions:
  - id: cc2-angle-four
    topic: Angles
    difficulty: 1
    q: "What is the smaller angle between the hands of this clock?"
    figure:
      kind: shapes
      alt: "A clock face showing 4 o'clock"
      items:
        - { t: circle, c: [50, 50], r: 46, fill: none }
        - { t: line, p: [50, 15, 50, 5], w: 3 }
        - { t: line, p: [70, 15.4, 72.5, 11], w: 1.5 }
        - { t: line, p: [84.6, 30, 89, 27.5], w: 1.5 }
        - { t: line, p: [85, 50, 95, 50], w: 3 }
        - { t: line, p: [84.6, 70, 89, 72.5], w: 1.5 }
        - { t: line, p: [70, 84.6, 72.5, 89], w: 1.5 }
        - { t: line, p: [50, 85, 50, 95], w: 3 }
        - { t: line, p: [30, 84.6, 27.5, 89], w: 1.5 }
        - { t: line, p: [15.4, 70, 11, 72.5], w: 1.5 }
        - { t: line, p: [15, 50, 5, 50], w: 3 }
        - { t: line, p: [15.4, 30, 11, 27.5], w: 1.5 }
        - { t: line, p: [30, 15.4, 27.5, 11], w: 1.5 }
        - { t: line, p: [50, 50, 69.1, 61], w: 5 }
        - { t: line, p: [50, 50, 50, 16], w: 2.5 }
        - { t: circle, c: [50, 50], r: 3, fill: solid }
    options: ["60°", "90°", "120°", "150°"]
    answer: C
    explanation: |
      The clock shows 4:00. The 12 numbers are 360 ÷ 12 = 30° apart.
      The hands are 4 numbers apart, so the angle is 4 × 30° = **120°**.

  - id: cc2-minute-hand-25
    topic: Angles
    difficulty: 1
    q: "Through how many degrees does the minute hand turn in 25 minutes?"
    options: ["12.5°", "25°", "125°", "150°"]
    answer: D
    explanation: |
      The minute hand turns 360° in 60 minutes, which is 6° every minute. 25 × 6° = **150°**.
      12.5° is how far the *hour* hand turns in 25 minutes (½° a minute).

  - id: cc2-hour-hand-40
    topic: Angles
    difficulty: 1
    q: "Through how many degrees does the hour hand turn from 2:00 to 2:40?"
    options: ["20°", "40°", "80°", "240°"]
    answer: A
    explanation: |
      The hour hand turns 30° in 60 minutes, which is ½° every minute. In 40 minutes it turns 40 × ½° = **20°**.
      240° is how far the minute hand turns in the same time.

  - id: cc2-gains-two
    topic: Faulty clocks
    difficulty: 1
    q: "A clock gains 2 minutes every hour. Vikram sets it to the correct time at 9:00 am. What will it show when the correct time is 3:00 pm the same day?"
    options: ["2:48 pm", "3:02 pm", "3:10 pm", "3:12 pm"]
    answer: D
    explanation: |
      From 9:00 am to 3:00 pm is 6 hours, so the clock gains 6 × 2 = 12 minutes. It is ahead, so it shows **3:12 pm**.
      2:48 pm would be a clock that *loses* 12 minutes.

  - id: cc2-angle-three-thirty
    topic: Angles
    difficulty: 2
    q: "What is the smaller angle between the hands of this clock?"
    figure:
      kind: shapes
      alt: "A clock face showing half past three"
      items:
        - { t: circle, c: [50, 50], r: 46, fill: none }
        - { t: line, p: [50, 15, 50, 5], w: 3 }
        - { t: line, p: [70, 15.4, 72.5, 11], w: 1.5 }
        - { t: line, p: [84.6, 30, 89, 27.5], w: 1.5 }
        - { t: line, p: [85, 50, 95, 50], w: 3 }
        - { t: line, p: [84.6, 70, 89, 72.5], w: 1.5 }
        - { t: line, p: [70, 84.6, 72.5, 89], w: 1.5 }
        - { t: line, p: [50, 85, 50, 95], w: 3 }
        - { t: line, p: [30, 84.6, 27.5, 89], w: 1.5 }
        - { t: line, p: [15.4, 70, 11, 72.5], w: 1.5 }
        - { t: line, p: [15, 50, 5, 50], w: 3 }
        - { t: line, p: [15.4, 30, 11, 27.5], w: 1.5 }
        - { t: line, p: [30, 15.4, 27.5, 11], w: 1.5 }
        - { t: line, p: [50, 50, 71.3, 55.7], w: 5 }
        - { t: line, p: [50, 50, 50, 84], w: 2.5 }
        - { t: circle, c: [50, 50], r: 3, fill: solid }
    options: ["60°", "75°", "90°", "105°"]
    answer: B
    explanation: |
      The clock shows 3:30. Minute hand: 30 × 6° = 180°. Hour hand: 3 × 30° + 30 × ½° = 105°.
      Angle = 180° − 105° = **75°**. 90° assumes the hour hand is still on the 3, but it has moved halfway to the 4.

  - id: cc2-angle-seven-twenty
    topic: Angles
    difficulty: 2
    q: "What is the angle between the hands of a clock at 7:20?"
    options: ["80°", "90°", "100°", "120°"]
    answer: C
    explanation: |
      Use angle = 30 × H − 5.5 × M (take the positive value): 30 × 7 − 5.5 × 20 = 210 − 110 = **100°**.
      90° assumes the hour hand stays on the 7; in 20 minutes it moves on by 10°.

  - id: cc2-together-12-hours
    topic: Hands together or apart
    difficulty: 2
    q: "How many times do the hour hand and the minute hand point in exactly the same direction in 12 hours (from noon to midnight, counting noon but not midnight)?"
    options: ["11", "12", "22", "24"]
    answer: A
    explanation: |
      The minute hand catches up with the hour hand about once an hour, but between 11 and 1 they meet only once, at 12.
      So in 12 hours they are together **11** times. 12 counts an extra meeting between 11 and 12 that never happens; 22 is how often they make a right angle.

  - id: cc2-mirror-clock
    topic: Mirror images
    difficulty: 2
    q: "Kabir sees this clock in a mirror. What is the real time?"
    figure:
      kind: shapes
      transform: flipH
      alt: "The mirror image of a clock face. The hands look like they show 9:10."
      items:
        - { t: circle, c: [50, 50], r: 46, fill: none }
        - { t: line, p: [50, 15, 50, 5], w: 3 }
        - { t: line, p: [70, 15.4, 72.5, 11], w: 1.5 }
        - { t: line, p: [84.6, 30, 89, 27.5], w: 1.5 }
        - { t: line, p: [85, 50, 95, 50], w: 3 }
        - { t: line, p: [84.6, 70, 89, 72.5], w: 1.5 }
        - { t: line, p: [70, 84.6, 72.5, 89], w: 1.5 }
        - { t: line, p: [50, 85, 50, 95], w: 3 }
        - { t: line, p: [30, 84.6, 27.5, 89], w: 1.5 }
        - { t: line, p: [15.4, 70, 11, 72.5], w: 1.5 }
        - { t: line, p: [15, 50, 5, 50], w: 3 }
        - { t: line, p: [15.4, 30, 11, 27.5], w: 1.5 }
        - { t: line, p: [30, 15.4, 27.5, 11], w: 1.5 }
        - { t: line, p: [50, 50, 71.9, 48.1], w: 5 }
        - { t: line, p: [50, 50, 20.6, 33], w: 2.5 }
        - { t: circle, c: [50, 50], r: 3, fill: solid }
    options: ["2:10", "2:50", "3:10", "9:10"]
    answer: B
    explanation: |
      The mirror image seems to show 9:10. For a mirror image, real time = 11:60 − the time you see.
      11:60 − 9:10 = **2:50**. 9:10 is just reading the mirror image as if it were real.

  - id: cc2-loses-four
    topic: Faulty clocks
    difficulty: 2
    q: "A watch loses 4 minutes every day. Neha sets it right at 12 noon on Monday. What will it show at 12 noon on the coming Saturday of the same week?"
    options: ["11:40 am", "11:44 am", "12:16 pm", "12:20 pm"]
    answer: A
    explanation: |
      From Monday noon to Saturday noon is 5 days (Tue, Wed, Thu, Fri, Sat). It loses 5 × 4 = 20 minutes.
      It is behind, so it shows **11:40 am**. 11:44 am counts only 4 days.

  - id: cc2-together-four-five
    topic: Hands together or apart
    difficulty: 3
    q: "At what time between 4 and 5 o'clock are the two hands of a clock together?"
    options: ["18 1/3 minutes past 4", "20 minutes past 4", "21 9/11 minutes past 4", "22 minutes past 4"]
    answer: C
    explanation: |
      At 4:00 the minute hand is 120° behind the hour hand. Each minute it gains 6° − ½° = 5.5° on the hour hand.
      Time to catch up = 120 ÷ 5.5 = 240/11 = **21 9/11 minutes past 4**.
      20 minutes forgets that the hour hand keeps moving.

  - id: cc2-opposite-two-three
    topic: Hands together or apart
    difficulty: 3
    q: "At what time between 2 and 3 o'clock do the hands of a clock point in exactly opposite directions?"
    options: ["21 9/11 minutes past 2", "40 minutes past 2", "42 minutes past 2", "43 7/11 minutes past 2"]
    answer: D
    explanation: |
      At 2:00 the minute hand is 60° behind. To be opposite, it must get 180° ahead, so it gains 60 + 180 = 240°.
      At 5.5° a minute: 240 ÷ 5.5 = 480/11 = **43 7/11 minutes past 2**.
      21 9/11 is when the hands are *together*; 40 forgets that the hour hand moves.

  - id: cc2-right-angle-five
    topic: Hands together or apart
    difficulty: 3
    q: "After 5 o'clock, when are the hands of a clock at right angles for the first time?"
    options: ["10 minutes past 5", "10 10/11 minutes past 5", "15 minutes past 5", "16 4/11 minutes past 5"]
    answer: B
    explanation: |
      At 5:00 the hands are 150° apart. The gap must shrink to 90°, so the minute hand gains 150 − 90 = 60°.
      60 ÷ 5.5 = 120/11 = **10 10/11 minutes past 5**. 10 minutes assumes the hour hand stays on the 5.
---

Clock questions turn time into angles. A full turn is 360°, and everything follows from how fast each hand moves.

## How fast the hands move

| Hand | Full turn | Per hour | Per minute |
|---|---|---|---|
| Minute hand | 60 minutes | 360° | 6° |
| Hour hand | 12 hours | 30° | ½° |

So every minute the minute hand gains **5.5°** on the hour hand.

## Angle between the hands

At H hours and M minutes: angle = 30 × H − 5.5 × M. If the answer is negative, drop the minus sign. If it is more than 180°, subtract it from 360° to get the smaller angle.

Worked example: at 2:20, 30 × 2 − 5.5 × 20 = 60 − 110 = −50, so the angle is 50°.

## Hands together, opposite, at right angles

Find how many degrees the minute hand must **gain** on the hour hand, then divide by 5.5. The answers are often fractions with 11 at the bottom, like 5 5/11 minutes. In 12 hours, the hands are together 11 times, opposite 11 times and at right angles 22 times.

## Faulty clocks

A clock that gains is ahead of the real time, so add. A clock that loses is behind, so subtract. Multiply the gain or loss per hour (or per day) by the number of hours (or days).

## Mirror images

For a mirror placed beside the clock: real time = 11:60 − time seen in the mirror. Example: a mirror image showing 3:15 means the real time is 8:45.

## Common traps

- Forgetting that the hour hand moves between the numbers.
- Counting days wrongly from one weekday to another.
