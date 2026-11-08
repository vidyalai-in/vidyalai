---
title: "Direction sense 2: turns, shadows and positions"
description: "Find which way someone faces after turns, use shadows at sunrise and sunset, and work out where places are compared with each other."
slug: direction-sense-2
family: direction-sense
part: 2
timeMinutes: 12
questions:
  - id: ds2-ravi-turn
    topic: Turns
    difficulty: 1
    q: "Ravi is facing east. He turns 90° clockwise. Which direction is he facing now?"
    options: ["North", "East", "South", "West"]
    answer: C
    explanation: |
      Clockwise goes N → E → S → W. One quarter turn clockwise from east is **south**.

  - id: ds2-asha-turns
    topic: Turns
    difficulty: 1
    q: "Asha is facing west. She turns to her left, and then turns to her left again. Which direction is she facing now?"
    options: ["North", "East", "South", "West"]
    answer: B
    explanation: |
      Facing west, her left is south. Facing south, her left is east. So she faces **east**.
      Two left turns (or two right turns) always make you face the opposite way.

  - id: ds2-sunrise-shadow
    topic: Shadows
    difficulty: 1
    q: "Just after sunrise, Kabir stands facing the sun. In which direction does his shadow fall?"
    options: ["North", "East", "South", "West"]
    answer: D
    explanation: |
      The sun rises in the east, so Kabir faces east. A shadow always falls away from the sun, behind him.
      So the shadow falls to the **west**.

  - id: ds2-temple
    topic: Positions
    difficulty: 1
    q: "The bank is to the south of the post office. The post office is to the south of the temple. In which direction is the temple from the bank?"
    options: ["North", "East", "South", "West"]
    answer: A
    explanation: |
      Going north from the bank you reach the post office, then the temple. So the temple is to the **north** of the bank.
      'South' is the direction of the bank from the temple, the question turned round.

  - id: ds2-meera-135
    topic: Turns
    difficulty: 2
    q: "Meera is facing east. She turns 135° anticlockwise. Which direction is she facing now?"
    options: ["North-east", "South-east", "South-west", "North-west"]
    answer: D
    explanation: |
      Anticlockwise from east: 90° brings her to north, and 45° more brings her to **north-west**.
      South-east is what you get by turning 135° the wrong way (clockwise).

  - id: ds2-morning-shadow
    topic: Shadows
    difficulty: 2
    q: "At 7 o'clock in the morning, Neha stands in an open field. Her shadow falls exactly to her right. Which direction is she facing?"
    options: ["North", "East", "South", "West"]
    answer: C
    explanation: |
      In the morning the sun is in the east, so shadows fall to the west. Her right side is west.
      The person whose right is west is facing **south**.

  - id: ds2-villages
    topic: Positions
    difficulty: 2
    q: "Village B is 4 km east of village A. Village C is 3 km north of B. Village D is 4 km west of C. Where is village D, compared with village A?"
    options: ["3 km north", "3 km south", "4 km west", "5 km north-west"]
    answer: A
    explanation: |
      East 4 km, then west 4 km cancel out. Only the 3 km north is left.
      So D is **3 km north** of A. 5 km would be the distance from A to C.

  - id: ds2-vikram-turns
    topic: Turns
    difficulty: 2
    q: "Vikram is facing south-west. He turns 90° clockwise, and then 180° anticlockwise. Which direction is he facing now?"
    options: ["North-east", "South-east", "South-west", "North-west"]
    answer: B
    explanation: |
      Combine the turns: 90° clockwise and 180° anticlockwise make 90° anticlockwise.
      90° anticlockwise from south-west is **south-east**.

  - id: ds2-library
    topic: Positions
    difficulty: 2
    q: "The school is 3 km north of the park. The library is 3 km east of the park. In which direction is the library from the school?"
    options: ["North-east", "South-east", "South-west", "North-west"]
    answer: B
    explanation: |
      Stand at the school. The park is south of you, and the library is east of the park.
      So the library is to the **south-east** of the school. North-west is the school seen from the library.

  - id: ds2-face-to-face
    topic: Shadows
    difficulty: 3
    q: "Just before sunset, Arjun and Priya stand facing each other. Arjun's shadow falls exactly to his right. Which direction is Priya facing?"
    options: ["North", "East", "South", "West"]
    answer: C
    explanation: |
      At sunset the sun is in the west, so shadows fall to the east. Arjun's right is east, so Arjun faces north.
      Priya faces Arjun, so she faces the opposite way: **south**. 'North' is Arjun's direction, not Priya's.

  - id: ds2-lamp-well
    topic: Positions
    difficulty: 3
    q: "In a garden, a tree is 4 m north of a well. A bench is 6 m west of the tree. A lamp post is 12 m south of the bench. How far is the lamp post from the well, and in which direction?"
    options: ["8 m south-west", "14 m south-west", "10 m north-west", "10 m south-west"]
    answer: D
    explanation: |
      Put the well at the centre. The bench is 6 m west and 4 m north. The lamp is 12 m further south: 6 m west and 8 m south of the well.
      Distance = √(6² + 8²) = 10 m, so the lamp post is **10 m south-west** of the well.

  - id: ds2-game-names
    topic: Turns
    difficulty: 3
    q: "In a game, the directions have new names: north is called 'east', east is called 'south', south is called 'west' and west is called 'north'. Kavya faces the direction the game calls 'north', and then turns 90° to her right. Which real direction is she facing now?"
    options: ["North", "East", "South", "West"]
    answer: A
    explanation: |
      The game calls west 'north', so Kavya first faces real west. Facing west, her right is real **north**.
      'East' is the game's name for that direction, not the real one.
---

These questions are about which way people face and where places are compared with each other. A quick sketch with north at the top solves almost all of them.

## Turning

Write the directions round a circle: N, NE, E, SE, S, SW, W, NW. Each step is 45°.

- **Clockwise** (like clock hands) goes N → E → S → W. A right turn is 90° clockwise.
- **Anticlockwise** goes N → W → S → E. A left turn is 90° anticlockwise.
- "Her left" or "his right" always means the person's own left or right, from the way they face.

To combine several turns, add the clockwise turns and subtract the anticlockwise ones. *Example:* facing north, turn 45° clockwise and then 90° anticlockwise. Net: 45° anticlockwise, so she faces north-west.

## Shadows

In these puzzles the sun rises exactly in the east and sets exactly in the west. A shadow points away from the sun.

| Time | Sun in the | Shadows fall to the |
|---|---|---|
| Morning, sunrise | East | West |
| Evening, sunset | West | East |

Shadow questions often say the shadow falls to the person's left or right. First find the shadow's direction, then ask: "Which way must I face for that side to be my left (or right)?"

## Positions of places

Put the first place at the centre and mark each new place from the one before it. Read the question carefully: "the direction of X from Y" means you stand at Y and look towards X.
