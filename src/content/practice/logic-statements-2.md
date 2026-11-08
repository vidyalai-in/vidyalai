---
title: "Logic and statements 2: truth-tellers and liars"
description: "Knights who always tell the truth and knaves who always lie, and 'who did it' puzzles where some of the people are lying."
slug: logic-statements-2
family: logic-statements
part: 2
timeMinutes: 12
stems:
  - id: ls2-island
    title: "The island of knights and knaves"
    text: "Everyone on this island is either a **knight** or a **knave**. Knights always tell the truth. Knaves always lie. You cannot tell them apart by looking."
questions:
  - id: ls2-never-say
    stem: ls2-island
    topic: Knights and knaves
    difficulty: 1
    q: "Which sentence can nobody on the island ever say?"
    options: ["I am a knight.", "I am a knave.", "My brother is a knave.", "The sea is salty."]
    answer: B
    explanation: |
      A knight saying "I am a knave" would be lying, and a knave saying it would be telling the truth. Neither can happen, so **"I am a knave"** is never said.
      Both kinds can say "I am a knight": the knight truthfully, the knave as a lie.

  - id: ls2-both-knaves
    stem: ls2-island
    topic: Knights and knaves
    difficulty: 1
    q: "Asha says, \"Ravi and I are both knaves.\" What are Asha and Ravi?"
    options: ["Both are knights", "Both are knaves", "Asha is a knight and Ravi is a knave", "Asha is a knave and Ravi is a knight"]
    answer: D
    explanation: |
      A knight would never call herself a knave, so Asha is a knave and her sentence is false.
      It is false only if they are **not** both knaves, so Ravi is a knight. **Asha is a knave and Ravi is a knight.**

  - id: ls2-ravi-meera
    stem: ls2-island
    topic: Knights and knaves
    difficulty: 1
    q: "Ravi says, \"Meera is a knave.\" Meera says, \"Ravi and I are both knights.\" What are Ravi and Meera?"
    options: ["Both are knights", "Both are knaves", "Ravi is a knight and Meera is a knave", "Ravi is a knave and Meera is a knight"]
    answer: C
    explanation: |
      If Meera were a knight, her sentence would make Ravi a knight, but then his words "Meera is a knave" would be true. That clashes.
      So Meera is a knave, Ravi's sentence is true, and **Ravi is a knight and Meera is a knave**.

  - id: ls2-kabir-neha
    stem: ls2-island
    topic: Knights and knaves
    difficulty: 2
    q: "Kabir says, \"Neha is a knight.\" Neha says, \"Kabir and I are of different kinds.\" What are Kabir and Neha?"
    options: ["Both are knights", "Both are knaves", "Kabir is a knight and Neha is a knave", "Kabir is a knave and Neha is a knight"]
    answer: B
    explanation: |
      If Kabir is a knight, Neha is a knight too, but then her words "different kinds" are false. That clashes.
      So Kabir is a knave and his sentence is a lie: Neha is a knave. Check: they are the same kind, so Neha's sentence is false, as a knave's must be. **Both are knaves.**

  - id: ls2-three-who
    stem: ls2-island
    topic: Knights and knaves
    difficulty: 2
    q: "Anjali says, \"Farhan is a knave.\" Farhan says, \"Simran is a knave.\" Simran says, \"Anjali and Farhan are both knaves.\" Who is a knight?"
    options: ["Only Anjali", "Only Simran", "Only Farhan", "Anjali and Simran"]
    answer: C
    explanation: |
      If Simran were a knight, Anjali and Farhan would both be knaves. Then Anjali's "Farhan is a knave" would be true, which a knave cannot say. So Simran is a knave.
      Then Farhan's "Simran is a knave" is true, so he is a knight, and Anjali's "Farhan is a knave" is a lie. **Only Farhan** is a knight.

  - id: ls2-how-many
    stem: ls2-island
    topic: Knights and knaves
    difficulty: 2
    q: "Arjun, Priya and Joseph are islanders. Arjun says, \"Exactly one of us three is a knight.\" Priya says, \"Exactly two of us are knights.\" Joseph says, \"All three of us are knaves.\" How many of them are knights?"
    options: ["0", "1", "2", "3"]
    answer: B
    explanation: |
      Joseph cannot be a knight, or he would be calling himself a knave. So his sentence is false, and at least one of them is a knight.
      Arjun and Priya cannot both be right. If only Arjun is a knight, everything fits: Priya and Joseph lie. So **1** is a knight.

  - id: ls2-vikram-three
    stem: ls2-island
    topic: Knights and knaves
    difficulty: 3
    q: "Vikram says, \"Neha is a knave.\" Neha says, \"Vikram and Joseph are of the same kind.\" Joseph says, \"Vikram is a knave.\" Which is true?"
    options: ["Vikram is a knight, Neha and Joseph are knaves", "Neha is the only knight", "Vikram and Joseph are both knights", "All three are knaves"]
    answer: A
    explanation: |
      If Vikram is a knave, Neha is a knight, so Vikram and Joseph are the same kind: Joseph is a knave. But then Joseph's "Vikram is a knave" is true. That clashes.
      So Vikram is a knight, Neha is a knave, and Vikram and Joseph are different kinds, so Joseph is a knave. **Vikram is a knight, Neha and Joseph are knaves.**

  - id: ls2-vase
    topic: Who did it
    difficulty: 1
    q: "One of four children broke a flower vase. Exactly **one** of them is lying.\nNeha: \"I did not break it.\" Kabir: \"Neha broke it.\" Meera: \"Kabir is telling the truth.\" Ravi: \"I did not break it.\"\nWho broke the vase?"
    options: ["Neha", "Kabir", "Meera", "Ravi"]
    answer: A
    explanation: |
      Kabir and Meera say the same thing, so they are both truthful or both lying. Only one person lies, so both are truthful: **Neha** broke it.
      Then Neha is the one liar, and Ravi's sentence is true. Everything fits.

  - id: ls2-laddoo
    topic: Who did it
    difficulty: 2
    q: "One of four friends ate the last laddoo. Exactly **one** of them is telling the truth.\nFarhan: \"Simran ate it.\" Simran: \"Joseph ate it.\" Joseph: \"Simran is lying.\" Anjali: \"I did not eat it.\"\nWho ate the laddoo?"
    options: ["Farhan", "Simran", "Joseph", "Anjali"]
    answer: D
    explanation: |
      Simran and Joseph cannot both lie or both tell the truth, since Joseph says Simran is lying. So the one truthful person is one of them, and Farhan and Anjali both lie.
      Anjali lying about "I did not eat it" means **Anjali** ate it. Check: Simran's sentence is false, Joseph's is true. Only Joseph is truthful.

  - id: ls2-remote
    topic: Who did it
    difficulty: 2
    q: "Someone hid the TV remote. Exactly **one** of these four is lying.\nAsha: \"Priya hid it.\" Priya: \"Meera hid it.\" Kabir: \"I did not hide it.\" Meera: \"Priya is lying.\"\nWho hid the remote?"
    options: ["Priya", "Asha", "Meera", "Kabir"]
    answer: A
    explanation: |
      Asha and Priya name different people, so at least one of them lies. If Priya told the truth, Asha and Meera would both be lying, which is two liars.
      So Priya is the liar and Asha is right: **Priya** hid it.

  - id: ls2-race
    topic: Who did it
    difficulty: 3
    q: "Simran, Arjun, Farhan and Joseph ran a race. There were no ties. Exactly **one** of them is lying.\nSimran: \"I finished before Arjun.\" Arjun: \"I won the race.\" Farhan: \"I came second.\" Joseph: \"I finished just after Simran.\"\nWho finished last?"
    options: ["Simran", "Arjun", "Farhan", "Joseph"]
    answer: D
    explanation: |
      Simran and Arjun cannot both be right: if Arjun won, nobody finished before him. So one of them is the liar, and Farhan and Joseph tell the truth.
      Farhan is 2nd, so Simran and Joseph (one just after the other) are 3rd and 4th. That leaves 1st for Arjun, so Simran is the liar and **Joseph** finished last.

  - id: ls2-count-liars
    topic: Who did it
    difficulty: 3
    q: "Four friends each say one sentence. Asha: \"Exactly one of us four is lying.\" Kabir: \"Exactly two of us are lying.\" Vikram: \"Exactly three of us are lying.\" Neha: \"All four of us are lying.\"\nWho is telling the truth?"
    options: ["Asha", "Kabir", "Vikram", "Neha"]
    answer: C
    explanation: |
      The four sentences give different numbers, so at most one can be true. If none were true, all four would be lying and Neha would be right, which clashes.
      So exactly one is true and three are lying. That is what **Vikram** says.
---

This set tests truth-and-lie puzzles: you find out who is lying by testing each possibility until only one fits.

## Knights and knaves

A knight's sentence is always true and a knave's is always false. So for each person, the sentence is true **exactly when** the speaker is a knight.

- Nobody can say "I am a knave".
- If A says "B is a knight", then A and B are the same kind.
- If A says "B is a knave", then A and B are different kinds.

## The method: suppose and check

1. Suppose the first person is a knight. Follow what that forces, one sentence at a time.
2. If you reach a clash, the guess was wrong, so the person is a knave.
3. Always check the final answer against **every** sentence.

## Worked example

*Tara says, "Dev is a knave." Dev says, "Tara is a knave."*

Each calls the other a knave, so they are different kinds. But either way round fits! This puzzle has two answers, which is why real puzzles add one more clue.

## "Who did it" with liars

When you are told how many people lie, try each suspect as the culprit. Count how many sentences become false. Keep the suspect for whom the count matches.

Look first for two sentences that **clash** (they cannot both be true) or that **agree** (they stand or fall together). That often cuts the work in half.

## Common trap

Stopping at the first answer that fits most clues. The right answer fits every clue.
