---
title: "Syllogisms and Venn diagrams 1: all, some and no"
description: "Two or three statements with all, some and no, and which conclusions must follow from them. Includes the either-or case."
slug: syllogisms-venn-1
family: syllogisms-venn
part: 1
timeMinutes: 12
questions:
  - id: sv1-roses-plants
    topic: Two statements
    difficulty: 1
    q: "Statements: All roses are flowers. All flowers are plants.\nConclusions: I. All roses are plants. II. All plants are roses.\nWhich conclusions follow?"
    options: ["Only I follows", "Only II follows", "Both follow", "Neither follows"]
    answer: A
    explanation: |
      Draw roses inside flowers, and flowers inside plants. Then roses are inside plants, so **only I follows**.
      II turns the statement around. A tree is a plant but not a rose.

  - id: sv1-cats-pets
    topic: Two statements
    difficulty: 1
    q: "Statements: All cats are animals. Some animals are pets.\nConclusions: I. Some cats are pets. II. Some animals are cats.\nWhich conclusions follow?"
    options: ["Only I follows", "Only II follows", "Both follow", "Neither follows"]
    answer: B
    explanation: |
      The cats circle is inside animals, and pets overlap animals somewhere. The pets may sit in the part with no cats, so I need not be true.
      There is at least one cat and every cat is an animal, so some animals are cats. **Only II follows.**

  - id: sv1-fish-parrots
    topic: Statements with no
    difficulty: 1
    q: "Statements: No fish is a bird. All parrots are birds.\nConclusions: I. No parrot is a fish. II. Some birds are not fish.\nWhich conclusions follow?"
    options: ["Only I follows", "Only II follows", "Both follow", "Neither follows"]
    answer: C
    explanation: |
      Parrots are inside birds, and birds are fully apart from fish. So no parrot is a fish (I).
      Any parrot is a bird that is not a fish, so II is true too. **Both follow.**

  - id: sv1-boys-dancers
    topic: Two statements
    difficulty: 1
    q: "Statements: Some boys are singers. Some singers are dancers.\nConclusions: I. Some boys are dancers. II. All singers are boys.\nWhich conclusions follow?"
    options: ["Only I follows", "Only II follows", "Both follow", "Neither follows"]
    answer: D
    explanation: |
      The singers who dance may be different singers from the ones who are boys, so I need not be true.
      "Some singers are boys" does not mean all of them are. **Neither follows.**

  - id: sv1-books-rulers
    topic: Two statements
    difficulty: 2
    q: "Statements: Some books are pens. All pens are rulers.\nConclusions: I. Some books are rulers. II. All rulers are pens.\nWhich conclusions follow?"
    options: ["Only I follows", "Only II follows", "Both follow", "Neither follows"]
    answer: A
    explanation: |
      The books that are pens are also rulers, because every pen is a ruler. So I is true.
      II turns "all pens are rulers" around, which is not allowed. **Only I follows.**

  - id: sv1-apples-cherries
    topic: Statements with no
    difficulty: 2
    q: "Statements: No apple is a banana. Some bananas are cherries.\nConclusions: I. No cherry is an apple. II. Some cherries are not apples.\nWhich conclusions follow?"
    options: ["Only I follows", "Only II follows", "Both follow", "Neither follows"]
    answer: B
    explanation: |
      The cherries that are bananas cannot be apples, so some cherries are not apples (II).
      Other cherries, outside bananas, may still be apples, so I need not be true. **Only II follows.**

  - id: sv1-doctors-graduates
    topic: Statements with no
    difficulty: 2
    q: "Statements: All doctors are graduates. No graduate is a child.\nConclusions: I. Some graduates are not doctors. II. Some doctors are children.\nWhich conclusions follow?"
    options: ["Only I follows", "Only II follows", "Both follow", "Neither follows"]
    answer: D
    explanation: |
      Doctors are inside graduates, and graduates are apart from children, so no doctor is a child. II is false.
      The doctors circle might fill the whole graduates circle, so I need not be true. **Neither follows.**

  - id: sv1-kites-toys
    topic: Two statements
    difficulty: 2
    q: "Statements: All kites are toys. Some toys are red.\nWhich conclusion definitely follows?"
    options: ["All toys are kites", "Some kites are red", "Some toys are kites", "No kite is red"]
    answer: C
    explanation: |
      There is at least one kite, and it is a toy, so **some toys are kites**.
      The red toys may or may not include kites, so "Some kites are red" and "No kite is red" are both only possible.

  - id: sv1-chairs-beds
    topic: Statements with no
    difficulty: 2
    q: "Statements: Some chairs are tables. No table is a bed.\nConclusions: I. Some beds are not chairs. II. Some chairs are not beds.\nWhich conclusions follow?"
    options: ["Only I follows", "Only II follows", "Both follow", "Neither follows"]
    answer: B
    explanation: |
      The chairs that are tables cannot be beds, so II is true.
      Every bed might also be a chair (just not one of the table-chairs), so I need not be true. **Only II follows.**

  - id: sv1-pilots-dancers
    topic: Three statements
    difficulty: 3
    q: "Statements: All pilots are swimmers. Some swimmers are singers. No singer is a dancer.\nConclusions: I. Some swimmers are not dancers. II. Some pilots are not dancers.\nWhich conclusions follow?"
    options: ["Only I follows", "Only II follows", "Both follow", "Neither follows"]
    answer: A
    explanation: |
      The swimmers who sing can never be dancers, so I is true.
      The pilots may all be swimmers who do not sing, and they could all be dancers. So II need not be true. **Only I follows.**

  - id: sv1-coins-gold
    topic: Either-or
    difficulty: 3
    q: "Statements: All coins are metals. Some metals are gold.\nConclusions: I. Some coins are gold. II. No coin is gold.\nWhich is correct?"
    options: ["Only I follows", "Only II follows", "Either I or II follows", "Neither follows"]
    answer: C
    explanation: |
      Neither conclusion is certain on its own. But I and II talk about the same two groups, and exactly one of them must be true: either some coin is gold or none is.
      So the answer is **either I or II follows**. "Neither follows" misses this pair.

  - id: sv1-mangoes-bitter
    topic: Three statements
    difficulty: 3
    q: "Statements: All mangoes are sweet. All sweet things are tasty. No tasty thing is bitter.\nConclusions: I. No mango is bitter. II. Some bitter things are not sweet.\nWhich conclusions follow?"
    options: ["Only I follows", "Only II follows", "Both follow", "Neither follows"]
    answer: C
    explanation: |
      Mangoes are inside sweet things, which are inside tasty things, and bitter things are outside tasty. So no mango is bitter (I).
      There is at least one bitter thing. It is not tasty, so it cannot be sweet. II is true too. **Both follow.**
---

This set tests whether you can tell what **must** be true from a few statements, and what is only **possible**.

## The rules

- Believe the statements, even if they sound silly. Use only what they say.
- Every group named has at least one member.
- **Some** means "at least one, maybe all". "Some A are B" does not tell you that some A are *not* B.
- A conclusion follows only if it is true in **every** diagram you can draw that fits the statements.

| Statement | Diagram |
|---|---|
| All A are B | A inside B |
| No A is B | A and B apart |
| Some A are B | A and B share at least one member |

## How to check

Draw one diagram that fits. Then try to draw a different diagram that also fits but makes the conclusion false. If you can, the conclusion does not follow.

## Worked example

*Statements: All pens are blue. Some blue things are cups. Does "Some pens are cups" follow?*

Draw pens inside blue. Now put the cups so they overlap blue only where there are no pens. Both statements still hold, but no pen is a cup. So the conclusion **does not follow**.

## Either-or

If neither conclusion is certain, but the two talk about the same pair of groups and one of them **has** to be true (such as "Some A are B" and "No A is B"), the answer is "either I or II follows".

## Common trap

"All A are B" cannot be turned around into "All B are A". It does give "Some B are A", because A has at least one member.
