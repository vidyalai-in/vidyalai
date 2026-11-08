---
title: "Speed, time and work 1: speed, distance and time"
description: "Changing between km/h and m/s, finding speed, distance and time, average speed, and two people meeting or one catching up with the other."
slug: speed-time-work-1
family: speed-time-work
part: 1
timeMinutes: 12
questions:
  - id: stw1-kmh-to-ms
    topic: Units
    difficulty: 1
    q: "A car is moving at 72 km/h. What is its speed in metres per second?"
    options: ["7.2 m/s", "12 m/s", "20 m/s", "259.2 m/s"]
    answer: C
    explanation: |
      To change km/h to m/s, multiply by 5/18 (1 km/h = 1000 m ÷ 3600 s).
      72 × 5 ÷ 18 = **20 m/s**. 259.2 multiplies by 18/5, which is the rule for going the other way.

  - id: stw1-cycle-speed
    topic: Speed and time
    difficulty: 1
    q: "Meera cycles 18 km in 1 hour 20 minutes. What is her speed?"
    options: ["13.5 km/h", "15 km/h", "18 km/h", "24 km/h"]
    answer: A
    explanation: |
      1 hour 20 minutes = 1 1/3 hours = 4/3 hours. Speed = 18 ÷ 4/3 = 18 × 3/4 = **13.5 km/h**.
      15 km/h treats 1 hour 20 minutes as 1.2 hours. Minutes are sixtieths of an hour, not tenths.

  - id: stw1-distance-in-minutes
    topic: Speed and time
    difficulty: 1
    q: "A bus travels at 48 km/h. How far does it go in 25 minutes?"
    options: ["12 km", "20 km", "24 km", "1200 km"]
    answer: B
    explanation: |
      48 km/h is 48 km in 60 minutes, so 4 km every 5 minutes.
      25 minutes = 5 × 5 minutes, so the bus goes 5 × 4 = **20 km**. (1200 km forgets to change minutes to hours.)

  - id: stw1-ms-to-kmh
    topic: Units
    difficulty: 1
    q: "A cheetah can run at 30 m/s for a short time. What is this speed in km/h?"
    options: ["8.3 km/h", "18 km/h", "50 km/h", "108 km/h"]
    answer: D
    explanation: |
      To change m/s to km/h, multiply by 18/5. 30 × 18 ÷ 5 = **108 km/h**.
      8.3 km/h uses 5/18, the rule for the other direction. A cheetah is faster than a car in town, so a small answer cannot be right.

  - id: stw1-round-trip
    topic: Average speed
    difficulty: 2
    q: "Farhan drives 60 km to his grandmother's house at 30 km/h and comes back the same way at 60 km/h. What is his average speed for the whole trip?"
    options: ["40 km/h", "45 km/h", "48 km/h", "90 km/h"]
    answer: A
    explanation: |
      Average speed = total distance ÷ total time. Going takes 60 ÷ 30 = 2 hours, coming back 60 ÷ 60 = 1 hour.
      120 km ÷ 3 hours = **40 km/h**.
      45 km/h averages the two speeds, but he spends longer at the slower speed, so the average is lower.

  - id: stw1-two-stretches
    topic: Average speed
    difficulty: 2
    q: "Arjun travels for 2 hours at 50 km/h and then for 3 hours at 40 km/h. What is his average speed?"
    options: ["44 km/h", "45 km/h", "46 km/h", "90 km/h"]
    answer: A
    explanation: |
      Distance = 2 × 50 + 3 × 40 = 100 + 120 = 220 km, in 5 hours. Average = 220 ÷ 5 = **44 km/h**.
      45 km/h ignores that he spends more time at 40 km/h. 46 km/h mixes up which speed goes with which time.

  - id: stw1-meeting-point
    topic: Meeting and overtaking
    difficulty: 2
    q: "Towns P and Q are 180 km apart. A car leaves P for Q at 40 km/h. At the same time a van leaves Q for P at 50 km/h. How far from Q do they meet?"
    options: ["80 km", "90 km", "100 km", "120 km"]
    answer: C
    explanation: |
      Moving towards each other, they close the gap at 40 + 50 = 90 km/h, so they meet after 180 ÷ 90 = 2 hours.
      In 2 hours the van covers 2 × 50 = **100 km** from Q. 80 km is the distance from P.

  - id: stw1-catch-up
    topic: Meeting and overtaking
    difficulty: 2
    q: "Ravi starts walking from home at 8:00 am at 5 km/h. Kabir sets off from the same home at 8:30 am on the same road, cycling at 15 km/h. At what time does Kabir catch up with Ravi?"
    options: ["8:40 am", "8:45 am", "9:00 am", "9:15 am"]
    answer: B
    explanation: |
      By 8:30, Ravi is 5 × 1/2 = 2.5 km ahead. Kabir gains 15 − 5 = 10 km every hour.
      Time to gain 2.5 km = 2.5 ÷ 10 = 1/4 hour = 15 minutes, so he catches up at **8:45 am**.
      8:40 divides 2.5 by 15, forgetting that Ravi keeps walking.

  - id: stw1-speed-ratio
    topic: Speed and time
    difficulty: 2
    q: "Simran and Joseph walk the same distance to a park. Their speeds are in the ratio 3 : 4. Simran takes 20 minutes longer than Joseph. How long does Joseph take?"
    options: ["45 minutes", "60 minutes", "80 minutes", "100 minutes"]
    answer: B
    explanation: |
      For the same distance, time is in the inverse ratio of speed, so Simran : Joseph times = 4 : 3.
      The difference of 1 part is 20 minutes, so Joseph takes 3 × 20 = **60 minutes**. (80 minutes is Simran's time.)

  - id: stw1-second-half
    topic: Average speed
    difficulty: 3
    q: "Neha drives the first half of the distance of a journey at 40 km/h. How fast must she drive the second half of the distance so that her average speed for the whole journey is 48 km/h?"
    options: ["52 km/h", "56 km/h", "60 km/h", "64 km/h"]
    answer: C
    explanation: |
      Pick a distance: 240 km, so each half is 120 km. At 48 km/h the whole trip must take 240 ÷ 48 = 5 hours.
      The first half takes 120 ÷ 40 = 3 hours, leaving 2 hours for 120 km: **60 km/h**.
      56 km/h assumes the two speeds simply average to 48, but the halves do not take equal times.

  - id: stw1-circular-track
    topic: Meeting and overtaking
    difficulty: 3
    q: "Priya and Anjali start together from the same point on a 600 m circular track and run in opposite directions. Priya runs at 5 m/s and Anjali at 3 m/s. After how many seconds do they first meet?"
    options: ["75 seconds", "120 seconds", "200 seconds", "300 seconds"]
    answer: A
    explanation: |
      Running in opposite directions, together they cover 5 + 3 = 8 m every second. They meet when they have covered one full lap between them.
      600 ÷ 8 = **75 seconds**. 300 seconds uses 5 − 3, the rule for running in the same direction.

  - id: stw1-late-early
    topic: Speed and time
    difficulty: 3
    q: "If Vikram walks to school at 4 km/h, he is 5 minutes late. If he walks at 5 km/h, he is 10 minutes early. How far is his school?"
    options: ["2 km", "3 km", "4 km", "5 km"]
    answer: D
    explanation: |
      The two times differ by 5 + 10 = 15 minutes. Test the options: for 5 km, 5 ÷ 4 = 75 minutes and 5 ÷ 5 = 60 minutes, a difference of 15 ✓.
      So the school is **5 km** away. For 4 km the times are 60 and 48 minutes, only 12 apart.
---

This set tests speed, distance and time: changing units, average speed, and two moving people.

## The rules

**Distance = speed × time.** Cover the one you want in the triangle: speed = distance ÷ time, time = distance ÷ speed.

| Change | Multiply by |
|---|---|
| km/h to m/s | 5/18 |
| m/s to km/h | 18/5 |

Change minutes to hours by dividing by 60: 20 minutes = 1/3 hour, 45 minutes = 3/4 hour.

## Average speed

Average speed = **total distance ÷ total time**. Do not average the speeds unless the times are equal. For the same distance at two speeds, the average is always below the plain average of the speeds.

## Two moving people

- **Towards each other:** the gap closes at the **sum** of their speeds.
- **Same direction:** the faster one gains at the **difference** of their speeds.

## Worked example

*Asha and Kabir are 21 km apart and walk towards each other at 3 km/h and 4 km/h. When do they meet?*

The gap closes at 3 + 4 = 7 km/h, so they meet after 21 ÷ 7 = **3 hours**. By then Asha has walked 9 km and Kabir 12 km, and 9 + 12 = 21 ✓.

## Tip

When a question gives no distance, pick a handy one, such as the LCM of the speeds. The answer comes out the same, and the sums get easier.
