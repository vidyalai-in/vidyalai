---
title: "Number and word memory 1: numbers, timetables and prices"
description: "Study a phone number, a list of numbers, a bus timetable and a shopping list for under a minute each, then answer from memory."
slug: number-word-memory-1
family: number-word-memory
part: 1
timeMinutes: 12
stems:
  - id: contact
    title: "Memory card 1: Kabir's contact details"
    studySeconds: 25
    lines:
      - "Name: Kabir Mehta"
      - "Mobile: 97431 08251"
      - "PIN code: 411038"
  - id: numbers
    title: "Memory card 2: Eight numbers"
    studySeconds: 30
    lines:
      - "47,  82,  19,  65,  30,  74,  58,  23"
  - id: buses
    title: "Memory card 3: Morning buses from Kothrud depot"
    studySeconds: 40
    figure: { kind: "table", header: ["Bus", "Going to", "Leaves at"], rows: [["21", "Railway Station", "7:15"], ["34", "Airport", "7:40"], ["7", "City Market", "8:05"], ["52", "University", "8:30"]] }
  - id: shopping
    title: "Memory card 4: Anjali's shopping list"
    studySeconds: 35
    lines:
      - "Rice, 5 kg: ₹320"
      - "Milk, 2 litres: ₹108"
      - "Apples, 1 kg: ₹180"
      - "Soap, 3 bars: ₹135"
      - "Bread, 1 loaf: ₹45"
questions:
  - id: nw1-mobile
    stem: contact
    topic: "Phone numbers and codes"
    difficulty: 1
    q: "What was Kabir's mobile number?"
    options: ["97431 08521", "97413 08251", "97431 08215", "97431 08251"]
    answer: D
    explanation: |
      It was **97431 08251**. The other options use the same digits with two of them swapped, the most common memory slip.
      Reading a long number in two chunks (97431 and 08251) makes it easier to hold.

  - id: nw1-pin
    stem: contact
    topic: "Phone numbers and codes"
    difficulty: 2
    q: "What was the PIN code?"
    options: ["410038", "411038", "411083", "413108"]
    answer: B
    explanation: |
      The PIN code was **411038**. 411083 swaps the last two digits. A trick: 411 is the start, then 0 3 8.

  - id: nw1-repeat-digit
    stem: contact
    topic: "Phone numbers and codes"
    difficulty: 3
    q: "Which digit appeared twice in Kabir's mobile number?"
    options: ["1", "4", "7", "9"]
    answer: A
    explanation: |
      In 97431 08251 the digit **1** comes twice: 9743**1** 0825**1**. Each of the others appears only once.
      Noticing repeats while you study is a good way to fix a number in your mind.

  - id: nw1-in-list
    stem: numbers
    topic: "Number lists"
    difficulty: 1
    q: "Which of these numbers was in the list?"
    options: ["28", "32", "56", "82"]
    answer: D
    explanation: |
      **82** was the 2nd number. 28, 32 and 56 are 82, 23 and 65 with their digits reversed, a common trap.

  - id: nw1-after-19
    stem: numbers
    topic: "Number lists"
    difficulty: 2
    q: "Which number came just after 19 in the list?"
    options: ["30", "47", "65", "82"]
    answer: C
    explanation: |
      The list went 47, 82, 19, **65**, 30, ... 82 came just *before* 19, not after.

  - id: nw1-even-count
    stem: numbers
    topic: "Number lists"
    difficulty: 2
    q: "How many of the eight numbers were even?"
    options: ["3", "4", "5", "6"]
    answer: B
    explanation: |
      The even numbers were 82, 30, 74, 58: **4** of them. The other four (47, 19, 65, 23) were odd.

  - id: nw1-airport-time
    stem: buses
    topic: "Timetables"
    difficulty: 1
    q: "At what time did the bus to the Airport leave?"
    options: ["7:40", "8:05", "8:30", "8:40"]
    answer: A
    explanation: |
      Bus 34 to the Airport left at **7:40**. 8:05 and 8:30 were the times of other buses on the same card.

  - id: nw1-bus-7
    stem: buses
    topic: "Timetables"
    difficulty: 2
    q: "Where did bus number 7 go?"
    options: ["Railway Station", "University", "Airport", "City Market"]
    answer: D
    explanation: |
      Bus 7 went to **City Market**, leaving at 8:05. Small bus numbers are easy to mix up with the times (7:15, 7:40).

  - id: nw1-gap
    stem: buses
    topic: "Timetables"
    difficulty: 3
    q: "How many minutes after bus 21 did bus 52 leave?"
    options: ["45", "75", "85", "115"]
    answer: B
    explanation: |
      Bus 21 left at 7:15 and bus 52 at 8:30. From 7:15 to 8:15 is 60 minutes, and 15 more is **75 minutes**.
      115 treats 8:30 − 7:15 as 830 − 715.

  - id: nw1-rice
    stem: shopping
    topic: "Prices"
    difficulty: 1
    q: "How many kilograms of rice were on the list?"
    options: ["1 kg", "3 kg", "5 kg", "10 kg"]
    answer: C
    explanation: |
      The list said **5 kg** of rice. 1 kg was the apples, and 3 was the number of soap bars.

  - id: nw1-apples
    stem: shopping
    topic: "Prices"
    difficulty: 2
    q: "How much did the apples cost?"
    options: ["₹108", "₹135", "₹180", "₹320"]
    answer: C
    explanation: |
      Apples cost **₹180**. ₹108 was the milk, ₹135 the soap and ₹320 the rice.

  - id: nw1-milk-bread
    stem: shopping
    topic: "Prices"
    difficulty: 3
    q: "What was the total cost of the milk and the bread together?"
    options: ["₹153", "₹163", "₹225", "₹288"]
    answer: A
    explanation: |
      Milk ₹108 + bread ₹45 = **₹153**. ₹225 uses the apples (₹180) in place of the milk.
---

This set tests short-term memory for numbers: phone numbers, lists, timetables and prices. Each card is shown for a short time, then hidden while you answer.

## How to hold numbers

1. **Chunk them.** Nobody remembers 9743108251 as ten digits. Read it as 97431 08251, or as 97-43-10-82-51. Five small chunks are easier than ten digits.
2. **Say them in your head.** "Forty-seven, eighty-two, nineteen" sticks better than looking.
3. **Notice patterns.** Repeated digits, numbers that go up, even and odd, a digit that is missing. A pattern you noticed is a fact you can recall.
4. **Link numbers to their labels.** In a timetable, join each bus to its place and its time: "34 flies to the Airport at 7:40". Linking is what stops you mixing up rows.
5. **Read every row once, then test yourself.** Close your eyes for two seconds and recall the card, then look again at what you missed.

## Worked example

*Card: Bus 18 to Dadar at 6:50.* A link: "18-year-old catches the 6:50 to Dadar." If asked "Where did bus 18 go?" you recall the whole sentence, not three loose facts.

## Common traps

- **Swapped digits**: 411038 and 411083, 65 and 56. The options will include them on purpose.
- **Wrong row**: the right time, but for a different bus.
- Questions like "how many were even" or "what is the total" need you to *use* the numbers, so remember the exact values, not just roughly.

## How this set works

Each card has a **Start studying** button. When time runs out (or you press "I'm ready"), the card hides and its questions appear.
