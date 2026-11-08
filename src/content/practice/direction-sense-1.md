---
title: "Direction sense 1: walking a path"
description: "Follow a path of walks and turns, then find where the walker ends up: the direction from the start and the straight-line distance."
slug: direction-sense-1
family: direction-sense
part: 1
timeMinutes: 12
questions:
  - id: ds1-asha
    topic: Distance
    difficulty: 1
    q: "Asha walks 4 km north and then 3 km east. How far is she from her starting point in a straight line?"
    options: ["1 km", "5 km", "7 km", "12 km"]
    answer: B
    explanation: |
      The two walks make the two short sides of a right-angled triangle. Straight-line distance = √(4² + 3²) = √25 = **5 km**.
      7 km is the distance she walked, not how far she is from the start.

  - id: ds1-ravi
    topic: Direction
    difficulty: 1
    q: "Ravi walks 10 m east. He turns to his left and walks 6 m. Then he turns to his left again and walks 10 m. Where is he now, compared with his starting point?"
    options: ["6 m north", "6 m south", "10 m west", "26 m north"]
    answer: A
    explanation: |
      Facing east, his left is north, so he walks 6 m north. Facing north, his left is west, so the last 10 m cancel the first 10 m east.
      He ends **6 m north** of the start.

  - id: ds1-meera
    topic: Direction
    difficulty: 1
    q: "Meera walks 6 km south and then 8 km west. Where is she now, compared with her starting point?"
    options: ["10 km north-east", "10 km south-east", "10 km south-west", "14 km south-west"]
    answer: C
    explanation: |
      She is south and west of the start, so she is to the south-west. Distance = √(6² + 8²) = √100 = 10 km.
      So she is **10 km south-west**. 14 km is the total she walked.

  - id: ds1-kabir
    topic: Direction
    difficulty: 1
    q: "Kabir walks 5 km north. He turns to his right and walks 3 km. He turns to his right again and walks 5 km. Where is he now, compared with his starting point?"
    options: ["3 km west", "5 km north", "13 km east", "3 km east"]
    answer: D
    explanation: |
      Facing north, his right is east: 3 km east. Facing east, his right is south: the 5 km south cancels the 5 km north.
      He ends **3 km east** of the start.

  - id: ds1-priya
    topic: Distance
    difficulty: 2
    q: "Priya walks 4 km east, then 6 km north, then 5 km east, then 6 km north. How far is she from her starting point in a straight line?"
    options: ["11 km", "13 km", "15 km", "21 km"]
    answer: C
    explanation: |
      Add the walks in each direction: east 4 + 5 = 9 km, north 6 + 6 = 12 km.
      Distance = √(9² + 12²) = √225 = **15 km**. 21 km is the total distance she walked.

  - id: ds1-arjun
    topic: Distance
    difficulty: 2
    q: "Arjun walks 9 m west. He turns to his left and walks 5 m, turns to his left again and walks 3 m, and then turns to his right and walks 3 m. How far is he from his starting point in a straight line?"
    options: ["8 m", "10 m", "14 m", "20 m"]
    answer: B
    explanation: |
      Facing west, left is south (5 m south). Facing south, left is east (3 m east). Facing east, right is south (3 m south).
      Net: 9 − 3 = 6 m west and 5 + 3 = 8 m south. Distance = √(6² + 8²) = **10 m**.

  - id: ds1-simran
    topic: Direction
    difficulty: 2
    q: "Simran walks 20 m north. She turns to her right and walks 10 m, turns to her right again and walks 20 m, and then turns to her left and walks 5 m. Which direction is she facing now?"
    options: ["North", "East", "South", "West"]
    answer: B
    explanation: |
      Track only the facing: north → right → east → right → south → left → **east**.
      A right turn moves you clockwise round N, E, S, W; a left turn moves you anticlockwise.

  - id: ds1-bus
    topic: Direction
    difficulty: 2
    q: "A bus leaves the depot and travels 5 km south, then 12 km east. Where is the bus now, compared with the depot?"
    options: ["13 km south-east", "13 km north-east", "17 km south-east", "7 km south-west"]
    answer: A
    explanation: |
      The bus is south and east of the depot, so it is to the south-east.
      Distance = √(5² + 12²) = √169 = 13 km. So it is **13 km south-east**.

  - id: ds1-joseph
    topic: Distance
    difficulty: 2
    q: "Joseph cycles 20 km north, then 12 km east, then 4 km south. How far is he from his starting point in a straight line?"
    options: ["20 km", "28 km", "32 km", "36 km"]
    answer: A
    explanation: |
      North 20 − south 4 = 16 km north, and 12 km east.
      Distance = √(12² + 16²) = √400 = **20 km**. 36 km is the total he cycled.

  - id: ds1-vikram
    topic: Direction
    difficulty: 3
    q: "Vikram starts from his house and cycles 5 km south. He turns to his left and cycles 10 km, turns to his left and cycles 20 km, and then turns to his left and cycles 2 km. Where is he now, compared with his house?"
    options: ["13 km north-east", "17 km north-west", "17 km north-east", "23 km north-east"]
    answer: C
    explanation: |
      Facing south, left is east (10 km east). Then left is north (20 km north). Then left is west (2 km west).
      Net: 10 − 2 = 8 km east and 20 − 5 = 15 km north. Distance = √(8² + 15²) = √289 = 17 km, so **17 km north-east**.

  - id: ds1-two-friends
    topic: Distance
    difficulty: 3
    q: "Neha and Farhan start from the same gate. Neha walks 6 km north and then 4 km east. Farhan walks 2 km south and then 2 km west. How far apart are they now in a straight line?"
    options: ["4 km", "6 km", "8 km", "10 km"]
    answer: D
    explanation: |
      Neha is 4 km east and 6 km north of the gate. Farhan is 2 km west and 2 km south.
      The gap is 4 + 2 = 6 km east–west and 6 + 2 = 8 km north–south. Distance = √(6² + 8²) = **10 km**.

  - id: ds1-diya
    topic: Direction
    difficulty: 3
    q: "Diya walks 24 m west, turns to her right and walks 7 m. She now wants to walk straight back to her starting point. How far, and in which direction, must she walk?"
    options: ["25 m north-west", "25 m north-east", "31 m south-east", "25 m south-east"]
    answer: D
    explanation: |
      Facing west, her right is north, so she ends 24 m west and 7 m north of the start (to the north-west).
      To go back she must walk the opposite way, south-east, for √(24² + 7²) = √625 = 25 m: **25 m south-east**.
---

These questions give a path of walks and turns. Your job is to find where the walker ends up compared with the start.

## Draw it on a grid

Draw the start as a dot, with **north up** and **east to the right**. Then draw each walk as an arrow, one at a time.

"Turns to her left" or "turns to his right" means the walker's own left or right, so it depends on which way they are facing:

| Facing | Their right | Their left |
|---|---|---|
| North | East | West |
| East | South | North |
| South | West | East |
| West | North | South |

## Net movement and the straight line

Add up all the north–south walks (north minus south) and all the east–west walks (east minus west). These two totals are the short sides of a right-angled triangle, and the straight-line distance is the long side:

distance = √(east–west total² + north–south total²)

*Example:* 9 m east, 4 m north, 3 m west. Net: 6 m east and 4 m north. That is about √52 m away, to the north-east.

Learn these triples to save time: 3, 4, 5; 6, 8, 10; 5, 12, 13; 8, 15, 17; 7, 24, 25.

## Naming the direction

If the end point is north and east of the start, we call it north-east, even when it is not exactly halfway between. "Due east" means exactly east.

## Common traps

- The straight-line distance is not the total distance walked.
- To go back to the start, walk the opposite direction: north-east becomes south-west.
