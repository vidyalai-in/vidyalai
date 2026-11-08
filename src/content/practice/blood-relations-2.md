---
title: "Blood relations 2: coded relations and family trees"
description: "Decode relations written with symbols like A + B, and build small family trees from a list of clues."
slug: blood-relations-2
family: blood-relations
part: 2
timeMinutes: 12
stems:
  - id: br2-code
    title: "A relation code"
    text: "In this code, each symbol stands for a relation. A chain such as **P × Q − R** means P × Q and also Q − R."
    lines:
      - "A + B means A is the mother of B."
      - "A − B means A is the brother of B."
      - "A × B means A is the father of B."
      - "A ÷ B means A is the sister of B."
  - id: br2-tree
    title: "A family of six"
    text: "Read all the clues, draw the family tree, then answer."
    lines:
      - "P, Q, R, S, T and U are six members of a family. There are exactly two married couples."
      - "P is a man, and Q is his wife."
      - "T is the only daughter of P and Q, and R is T's brother."
      - "R is married to U, and U is the mother of S."
      - "S, a boy, is the only grandchild of P."
questions:
  - id: br2-gauri
    topic: Family trees
    difficulty: 1
    q: "Nikhil and Rhea are the children of Anand and Gauri. Rhea's daughter is Ira. How is Gauri related to Ira?"
    options: ["Grandmother", "Mother", "Aunt", "Sister"]
    answer: A
    explanation: |
      Gauri is Rhea's mother, and Rhea is Ira's mother. Your mother's mother is your **grandmother**.

  - id: br2-code-uncle
    stem: br2-code
    topic: Coded relations
    difficulty: 1
    q: "If K − L + M, how is K related to M?"
    options: ["Father", "Uncle", "Brother", "Grandfather"]
    answer: B
    explanation: |
      K − L: K is L's brother. L + M: L is M's mother.
      So K is the brother of M's mother, which makes K M's **uncle**. 'Brother' only describes K and L.

  - id: br2-code-father
    stem: br2-code
    topic: Coded relations
    difficulty: 1
    q: "If P × Q − R, how is P related to R?"
    options: ["Brother", "Uncle", "Father", "Grandfather"]
    answer: C
    explanation: |
      P × Q: P is Q's father. Q − R: Q is R's brother, so Q and R have the same father.
      So P is R's **father** too.

  - id: br2-code-nephew
    stem: br2-code
    topic: Coded relations
    difficulty: 2
    q: "If L ÷ M + N − O, how is N related to L?"
    options: ["Son", "Brother", "Cousin", "Nephew"]
    answer: D
    explanation: |
      L ÷ M: L is M's sister. M + N: M is N's mother. N − O: N is a boy (O's brother).
      So N is the son of L's sister, which makes him L's **nephew**. He is M's son, not L's.

  - id: br2-code-grandfather
    stem: br2-code
    topic: Coded relations
    difficulty: 2
    q: "Which of these means that M is the grandfather of N?"
    options: ["M × P − N", "M × P + N", "M + P × N", "M − P + N"]
    answer: B
    explanation: |
      **M × P + N**: M is P's father and P is N's mother, so M is the father of N's mother, N's grandfather.
      M × P − N makes M the father of N. M + P × N makes M a grandmother. M − P + N makes M an uncle.

  - id: br2-code-granddaughter
    stem: br2-code
    topic: Coded relations
    difficulty: 3
    q: "If D + E × F − G ÷ H, how is G related to D?"
    options: ["Daughter", "Grandson", "Granddaughter", "Niece"]
    answer: C
    explanation: |
      D is E's mother. E is F's father. F is G's brother, so G is also E's child. G ÷ H says G is someone's sister, so G is a girl.
      G is a girl whose father is D's son, so G is D's **granddaughter**. Without the last link, G's gender would be unknown.

  - id: br2-tree-father
    stem: br2-tree
    topic: Family trees
    difficulty: 1
    q: "Who is the father of S?"
    options: ["P", "Q", "R", "U"]
    answer: C
    explanation: |
      U is S's mother, and U is married to R. So **R** is S's father. P is S's grandfather.

  - id: br2-tree-aunt
    stem: br2-tree
    topic: Family trees
    difficulty: 2
    q: "How is T related to S?"
    options: ["Aunt", "Mother", "Sister", "Grandmother"]
    answer: A
    explanation: |
      T is R's sister, and R is S's father. Your father's sister is your **aunt**.
      T cannot be S's sister: S's parents are R and U, and T is a daughter of P and Q.

  - id: br2-tree-in-law
    stem: br2-tree
    topic: Family trees
    difficulty: 2
    q: "How is U related to Q?"
    options: ["Daughter", "Sister", "Niece", "Daughter-in-law"]
    answer: D
    explanation: |
      R is T's brother, so R is a son of P and Q. U is R's wife.
      The wife of Q's son is Q's **daughter-in-law**. U is not Q's daughter, because T is the only daughter.

  - id: br2-tree-father-in-law
    stem: br2-tree
    topic: Family trees
    difficulty: 3
    q: "How is P related to U?"
    options: ["Father", "Father-in-law", "Grandfather", "Brother"]
    answer: B
    explanation: |
      U is married to R, and R is P's son. The father of your husband is your **father-in-law**.
      P is S's grandfather, but U is S's mother, so P is one generation above U, not two.

  - id: br2-kaveri
    topic: Family trees
    difficulty: 2
    q: "Aditi's husband Varun has only one sibling, his sister Ritu. Ritu's daughter is Kaveri. How is Kaveri related to Aditi?"
    options: ["Niece", "Daughter", "Cousin", "Sister"]
    answer: A
    explanation: |
      Kaveri is the daughter of Varun's sister, so she is Varun's niece.
      Aditi is Varun's wife, so Kaveri is Aditi's **niece** too (her husband's sister's daughter).

  - id: br2-raj
    topic: Family trees
    difficulty: 3
    q: "Sunita is Raj's mother-in-law. Raj's wife Pallavi has only one brother, Tushar. Tushar's daughter is Mira. How is Raj related to Mira?"
    options: ["Father", "Brother", "Cousin", "Uncle"]
    answer: D
    explanation: |
      Pallavi is Tushar's sister, so she is Mira's aunt (her father's sister).
      Raj is married to Mira's aunt, so Raj is Mira's **uncle**. Sunita is Mira's grandmother, but you do not need her for this question.
---

This set tests whether you can turn relation clues, written in words or in a symbol code, into a family tree and read answers from it.

## Coded relations

A code gives each symbol a relation, for example "A + B means A is the mother of B". A chain like A + B × C is read in pairs: first A + B, then B × C. Write each pair as a plain sentence, then join them.

Notice what each symbol tells you about gender. "A is the mother of B" tells you A is female, but it says nothing about B. If the question asks "How is B related to A?", you may need another link to know whether B is a son or a daughter.

## Draw the tree

Use a simple drawing for clues:

- write males and females differently, for example a square for a male and a circle for a female;
- join a married couple with a double line (=);
- draw a line down from parents to their children, and put brothers and sisters side by side.

*Example:* "Sonal is Amit's wife. Amit's son is Yash." Put Sonal = Amit on the top row and Yash below them. Yash is Sonal's son too.

## Common traps

- Brothers and sisters have the same parents, so a clue about one gives you the other's parents too.
- "Only daughter" or "only brother" rules out everyone else, so use it to fix who is who.
- In-laws join through a marriage, not by birth. Your husband's sister is not your sister.
- Do not answer the question backwards. "How is P related to U?" means "P is U's what?"
