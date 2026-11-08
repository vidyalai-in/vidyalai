---
title: "Coding and decoding 2: word swaps and code tables"
description: "Codes where one word is called by another's name, code sentences where you match each word to its code, and codes that mix letters and numbers."
slug: coding-decoding-2
family: coding-decoding
part: 2
timeMinutes: 12
questions:
  - id: cd2-sub-bag
    topic: Substitution
    difficulty: 1
    q: "In a code language, *pen* is called *book*, *book* is called *bag*, *bag* is called *desk*, and *desk* is called *chair*. In this language, what do you carry your books to school in?"
    options: ["bag", "book", "desk", "chair"]
    answer: C
    explanation: |
      In real life you carry books in a **bag**. In this language a bag is called **desk**.
      "bag" is the real name, and "chair" goes one step too far along the list.

  - id: cd2-sub-camel
    topic: Substitution
    difficulty: 1
    q: "In a code language, *cow* is called *goat*, *goat* is called *horse*, *horse* is called *camel*, and *camel* is called *elephant*. In this language, what is the name of the animal known as the ship of the desert?"
    options: ["camel", "elephant", "goat", "horse"]
    answer: B
    explanation: |
      The ship of the desert is the camel. In this language a camel is called **elephant**.
      "horse" reads the list the wrong way round: it is the horse that is called camel.

  - id: cd2-table-milk
    topic: Code tables
    difficulty: 1
    q: "In a code language, \"ba lo\" means \"hot tea\" and \"lo mi\" means \"hot milk\". What is the code for *milk*?"
    options: ["mi", "lo", "ba", "It cannot be decided"]
    answer: A
    explanation: |
      *hot* is in both sentences, and so is the code *lo*. So hot = lo.
      In the second sentence, the other word, milk, must be **mi**. (ba is tea.)

  - id: cd2-mixed-mole
    topic: Mixed codes
    difficulty: 1
    q: "In a code, MANGO is written as 47193 and LEMON is written as 52431. How is MOLE written in the same code?"
    options: ["3452", "4325", "4352", "4532"]
    answer: C
    explanation: |
      Each letter always gets the same digit. From MANGO: M = 4, A = 7, N = 1, G = 9, O = 3. From LEMON: L = 5, E = 2 (and M, O, N match).
      MOLE = 4, 3, 5, 2 = **4352**.

  - id: cd2-sub-field
    topic: Substitution
    difficulty: 2
    q: "In a code language, *doctor* is called *teacher*, *teacher* is called *farmer*, *farmer* is called *pilot*, *pilot* is called *tailor*, and *tailor* is called *driver*. If someone in this language says \"The pilot is working in the field\", who is really in the field?"
    options: ["doctor", "farmer", "pilot", "tailor"]
    answer: B
    explanation: |
      Read the list backwards here: which person is *called* pilot? "farmer is called pilot", so the word pilot means a real **farmer**.
      This fits, since farmers work in fields. "tailor" goes the wrong way: tailor is what a real pilot is called.

  - id: cd2-sub-road-river
    topic: Substitution
    difficulty: 2
    q: "In a code language, *road* is called *river*, *river* is called *sky*, *sky* is called *field*, *field* is called *forest*, and *forest* is called *road*. In this language, where do cars run, and where do boats sail?"
    options: ["road and river", "sky and field", "forest and road", "river and sky"]
    answer: D
    explanation: |
      Cars run on a road, which is called **river**. Boats sail on a river, which is called **sky**.
      "road and river" gives the real names, not the code names.

  - id: cd2-table-kite
    topic: Code tables
    difficulty: 2
    q: "In a code language:\n\"pa lu ge\" means \"kite flies high\"\n\"ge ro mak\" means \"high blue sky\"\n\"tor lu ro\" means \"my blue kite\"\nThe codes are not always in the same order as the words. What does *pa* mean?"
    options: ["flies", "high", "kite", "sky"]
    answer: A
    explanation: |
      Find the shared words. *high* is in sentences 1 and 2, and so is *ge*. *kite* is in sentences 1 and 3, and so is *lu*.
      In sentence 1, high = ge and kite = lu, so the last code, pa, is **flies**.

  - id: cd2-table-tea
    topic: Code tables
    difficulty: 2
    q: "In a code language:\n\"ra mi so\" means \"tea is hot\"\n\"so ve du\" means \"coffee is cold\"\n\"du ka mi\" means \"hot and cold\"\nThe codes are not always in the same order as the words. What is the code for *tea*?"
    options: ["mi", "ra", "so", "du"]
    answer: B
    explanation: |
      *is* is in sentences 1 and 2: the shared code is so. *hot* is in sentences 1 and 3: the shared code is mi.
      In sentence 1, that leaves **ra** for tea. "mi" is hot, the first word that is easy to find.

  - id: cd2-mixed-vowels
    topic: Mixed codes
    difficulty: 2
    q: "In a code, each vowel is replaced by a number (A = 1, E = 2, I = 3, O = 4, U = 5) and each consonant is replaced by the next letter of the alphabet. PLATE is written as QM1U2. How is MOUSE written?"
    options: ["L45R2", "NPVTF", "N54T2", "N45T2"]
    answer: D
    explanation: |
      M is a consonant → N. O → 4. U → 5. S is a consonant → T. E → 2.
      MOUSE = **N45T2**. L45R2 uses the letter before instead of the next one, and N54T2 mixes up the numbers for O and U.

  - id: cd2-table-bus
    topic: Code tables
    difficulty: 3
    q: "In a code language:\n\"ta mo ki\" means \"big red bus\"\n\"ki zu ne\" means \"red car stops\"\n\"mo ne fa\" means \"car is big\"\nThe codes are not always in the same order as the words. How would \"bus stops\" be written?"
    options: ["ki zu", "ne zu", "ta fa", "ta zu"]
    answer: D
    explanation: |
      red (sentences 1, 2) = ki, big (1, 3) = mo, car (2, 3) = ne.
      Then sentence 1 leaves bus = ta, and sentence 2 leaves stops = zu. So "bus stops" is **ta zu**. ne zu would mean "car stops".

  - id: cd2-table-mud
    topic: Code tables
    difficulty: 3
    q: "In a code language:\n\"si fo ka\" means \"rain makes mud\"\n\"ka ple ru\" means \"mud on shoes\"\n\"ru dim yo\" means \"shoes get wet\"\nThe codes are not always in the same order as the words. For which word can the code be found for certain?"
    options: ["rain", "makes", "on", "wet"]
    answer: C
    explanation: |
      mud (sentences 1, 2) = ka and shoes (2, 3) = ru. In sentence 2 that leaves **on** = ple, for certain.
      rain and makes share si and fo, but nothing says which is which. The same is true of get and wet with dim and yo.

  - id: cd2-mixed-swap
    topic: Mixed codes
    difficulty: 3
    q: "In a code, PLANET is written as TMBOFP and MONKEY is written as YPOLFM. How is CHAIR written in the same code?"
    options: ["RIBJC", "DIBJS", "RHAIC", "RIBJD"]
    answer: A
    explanation: |
      The first and last letters swap places (PLANET → T....P), and each letter in the middle moves one place forward (LANE → MBOF).
      CHAIR: swap C and R, and HAI → IBJ, giving **RIBJC**. RHAIC forgets to shift the middle, and RIBJD also shifts the last letter.
---

This set tests three kinds of code: word swaps, code sentences and codes that mix letters with numbers.

## Word swaps (substitution)

"If *pen* is called *book*, ..." means: in this language, whenever you mean a real pen, you say *book*. To answer, first find the **real** thing the question is about, then look up what it is called.

Read carefully in which direction the question goes. "What is a camel called?" goes forward along the list. "Someone says *pilot*: who is it really?" goes backward, so look for the word that is *called* pilot.

## Code sentences

Each word has its own code, but the codes may be in any order.

1. Find a word that appears in two sentences.
2. The code that appears in the same two sentences is its code.
3. Remove the codes you know. What is left belongs to the remaining words.

If two words appear only together in one sentence, you cannot tell their codes apart.

## Worked example

*"ko ra" means "ripe mango" and "ra ti" means "mango juice". What is the code for juice?*

*mango* is in both sentences, and *ra* is the shared code. So mango = ra, and in the second sentence juice = **ti**.

## Mixed codes

Some codes treat letters differently: vowels become numbers and consonants shift, or the first and last letters swap. Check your rule on every letter of every example given. With two examples, a rule that fits only one of them is the wrong rule.
