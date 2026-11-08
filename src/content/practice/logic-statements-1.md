---
title: "Logic and statements 1: if-then and conclusions"
description: "What must be true after an if-then rule, what only could be true, rules with 'only if' and 'unless', and statements with conclusions."
slug: logic-statements-1
family: logic-statements
part: 1
timeMinutes: 12
questions:
  - id: ls1-rain-match
    topic: If-then
    difficulty: 1
    q: "Rule: If it rains on Sunday, the cricket match is cancelled. It rained on Sunday. What must be true?"
    options: ["The match was played", "It will rain next Sunday too", "The match may or may not have been cancelled", "The match was cancelled"]
    answer: D
    explanation: |
      The "if" part happened (it rained), so the "then" part must happen too. **The match was cancelled.**

  - id: ls1-homework
    topic: If-then
    difficulty: 1
    q: "Rule: If Ravi finishes his homework, he goes out to play. Today Ravi did not go out to play. What must be true?"
    options: ["Ravi did not finish his homework", "Ravi finished his homework", "Ravi was not feeling well", "Nothing can be said about his homework"]
    answer: A
    explanation: |
      Had he finished his homework, he would have gone out. He did not go out, so **he did not finish his homework**.
      When the "then" part is false, the "if" part must be false too.

  - id: ls1-same-meaning
    topic: If-then
    difficulty: 1
    q: "Which sentence means exactly the same as: 'If a number ends in 0, it is divisible by 5'?"
    options: ["If a number is divisible by 5, it ends in 0.", "If a number does not end in 0, it is not divisible by 5.", "If a number is not divisible by 5, it does not end in 0.", "Some numbers that end in 0 are not divisible by 5."]
    answer: C
    explanation: |
      Swap the two parts and say "not" to both: **if a number is not divisible by 5, it does not end in 0**.
      The first two options are wrong: 15 is divisible by 5 but does not end in 0.

  - id: ls1-library-card
    topic: Statements and conclusions
    difficulty: 1
    q: "Statement: Every student of Class 8B has a library card. Priya is a student of Class 8B.\nConclusions: I. Everyone who has a library card is in Class 8B. II. Priya has a library card.\nWhich conclusions follow?"
    options: ["Only I follows", "Only II follows", "Both follow", "Neither follows"]
    answer: B
    explanation: |
      Priya is in 8B and every 8B student has a card, so II is true.
      Students of other classes may have cards too, so I does not follow. **Only II follows.**

  - id: ls1-medal
    topic: If-then
    difficulty: 2
    q: "Rule: If Meera wins the quiz, she gets a medal. Meera got a medal. What can you say?"
    options: ["Meera won the quiz", "Meera did not win the quiz", "Meera may or may not have won the quiz", "Meera did not take part in the quiz"]
    answer: C
    explanation: |
      The rule says what happens when she wins. It does not say a win is the only way to get a medal (she might have won it in a race).
      So **she may or may not have won the quiz**. "She won" is the most common mistake.

  - id: ls1-admit-card
    topic: Only if and unless
    difficulty: 2
    q: "Rule: You can enter the exam hall only if you have an admit card. Which of these must be true?"
    options: ["Everyone who has an admit card entered the hall.", "Everyone who entered the hall has an admit card.", "Someone without an admit card may be in the hall.", "Nobody in the hall has an admit card."]
    answer: B
    explanation: |
      "Only if" puts the condition after it: being in the hall means you have a card. So **everyone who entered the hall has an admit card**.
      Having a card does not force you to go in, so the first option need not be true.

  - id: ls1-rocket
    topic: Only if and unless
    difficulty: 2
    q: "Rule: The rocket will not be launched unless the sky is clear. On Monday the rocket was launched. What must be true?"
    options: ["The sky was clear on Monday", "The sky was cloudy on Monday", "The rocket is launched on every clear day", "Nothing can be said about the sky"]
    answer: A
    explanation: |
      "Not launched unless clear" means: if the sky is not clear, no launch. There was a launch, so **the sky was clear on Monday**.
      The rule does not say every clear day has a launch.

  - id: ls1-bus-chain
    topic: Chains and rules
    difficulty: 2
    q: "Rules: If Kabir wakes up by 6, he catches the 7 o'clock bus. If he catches the 7 o'clock bus, he reaches school on time. Today Kabir reached school late. What must be true?"
    options: ["Kabir woke up by 6", "Kabir caught the 7 o'clock bus", "Kabir did not catch the 7 o'clock bus", "The bus was late today"]
    answer: C
    explanation: |
      Catching the bus would have got him there on time. He was late, so **he did not catch the 7 o'clock bus**.
      Going one more step back, he also did not wake up by 6, so the first option is false.

  - id: ls1-museum
    topic: Statements and conclusions
    difficulty: 2
    q: "Statement: The museum is closed on Mondays. Today the museum is open.\nConclusions: I. Today is not Monday. II. The museum will be open tomorrow.\nWhich conclusions follow?"
    options: ["Only I follows", "Only II follows", "Both follow", "Neither follows"]
    answer: A
    explanation: |
      If today were Monday, the museum would be closed. It is open, so I is true.
      Today could be Sunday, and then tomorrow is Monday and the museum is closed. So II does not follow. **Only I follows.**

  - id: ls1-holiday-rain
    topic: Chains and rules
    difficulty: 3
    q: "Rule: If it is a holiday or it is raining, Arjun stays at home. Today Arjun went out. What must be true?"
    options: ["It is a holiday, but it is not raining", "It is raining, but it is not a holiday", "It is not a holiday, and it is not raining", "It may be a holiday"]
    answer: C
    explanation: |
      Either a holiday **or** rain would be enough to keep Arjun at home. He went out, so neither happened: **it is not a holiday, and it is not raining**.

  - id: ls1-team-could
    topic: Chains and rules
    difficulty: 3
    q: "The coach has two rules for the team. If Vikram plays, Anjali also plays. If Anjali plays, Joseph does not play. Which of these **could** be true?"
    options: ["Vikram and Joseph both play.", "Vikram plays but Anjali does not.", "Anjali and Joseph both play.", "Joseph plays but Vikram does not."]
    answer: D
    explanation: |
      If Vikram plays, then Anjali plays, and then Joseph cannot. So Vikram and Joseph never play together, and the first three options all break a rule.
      **Joseph plays but Vikram does not** works: Joseph plays, Anjali and Vikram sit out.

  - id: ls1-ripe-mangoes
    topic: Statements and conclusions
    difficulty: 3
    q: "Statement: Some of the mangoes in this basket are ripe.\nConclusions: I. All the mangoes in the basket are ripe. II. Some of the mangoes in the basket are not ripe.\nWhich conclusions follow? (Remember: *some* means at least one, and possibly all.)"
    options: ["Only I follows", "Only II follows", "Both follow", "Neither follows"]
    answer: D
    explanation: |
      "Some" means at least one, and it could be all. So the basket may be all ripe or only partly ripe.
      I and II are both only possible. **Neither follows.** Picking II reads "some" as "not all".
---

This set tests if-then rules: what **must** be true, what only **could** be true, and what does not follow at all.

## The one sure step

A rule "If P, then Q" gives you exactly two sure steps.

| You know | You can say |
|---|---|
| P happened | Q happened |
| Q did not happen | P did not happen |
| Q happened | nothing sure about P |
| P did not happen | nothing sure about Q |

So "If P, then Q" means the same as "If not Q, then not P". It does **not** mean "If Q, then P".

## Only if and unless

- "A only if B" means: whenever A happens, B is true. So it is "If A, then B".
- "Not A unless B" means: without B, no A. This is also "If A, then B".

## Worked example

*Rule: If the shop is open, its light is on. The light is off.* The light would be on if the shop were open, so the shop is **closed**. But if the light were on, you could not say the shop is open: someone may be cleaning inside.

## Statements and conclusions

Use only what the statement says. A conclusion follows only if it must be true, not if it is just likely. "Some" means at least one and maybe all.

## Common trap

Turning the rule around. "If you win, you get a medal" does not mean every medal comes from a win.
