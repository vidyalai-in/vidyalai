---
title: "Data handling 2: pie charts and mixed data"
description: "Turn pie-chart percentages into rupees, students and central angles, and combine a pie chart or table with other data."
slug: data-handling-2
family: data-handling
part: 2
timeMinutes: 12
stems:
  - id: dh2-budget
    title: "Meera's family budget"
    text: "Meera's family spends **₹24,000** in a month. The pie chart shows how the money is shared out."
    figure: { kind: pie, title: "Monthly spending", labels: [Food, Rent, Education, Travel, Electricity, Others], values: [35, 25, 15, 12, 5, 8] }
  - id: dh2-sports
    title: "Favourite sports"
    text: "Each of the **800** students in a school chose one favourite sport. The pie chart shows the percentage choosing each sport. Exactly half of the 800 students are girls, and the list shows how many girls chose each sport."
    figure: { kind: pie, title: "Favourite sport (all students)", labels: [Cricket, Football, Badminton, Kabaddi, Chess], values: [35, 25, 20, 12, 8] }
    lines:
      - "Girls who chose Cricket: 140"
      - "Girls who chose Football: 90"
      - "Girls who chose Badminton: 96"
      - "Girls who chose Kabaddi: 48"
      - "Girls who chose Chess: 26"
  - id: dh2-shop
    title: "A shop over two years"
    text: "Simran's shop had total sales of **₹2,00,000** in 2024 and **₹2,50,000** in 2025. The table shows what share of each year's sales came from each item."
    figure:
      kind: table
      caption: "Share of total sales"
      header: [Item, "2024", "2025"]
      rows:
        - [Books, "30%", "28%"]
        - [Toys, "20%", "18%"]
        - [Stationery, "35%", "40%"]
        - [Gifts, "15%", "14%"]
questions:
  - id: dh2-angle-to-percent
    topic: Central angles
    difficulty: 1
    q: "In a pie chart, one sector has a central angle of 72°. What percentage of the whole does it show?"
    options: ["20%", "28%", "36%", "72%"]
    answer: A
    explanation: |
      The whole circle is 360°, which stands for 100%. So 72° is 72 ÷ 360 × 100 = **20%**.
      72% forgets that the angles add up to 360, not 100.

  - id: dh2-budget-food
    stem: dh2-budget
    topic: Pie charts
    difficulty: 1
    q: "How much does the family spend on food in a month?"
    options: ["₹840", "₹6,000", "₹8,400", "₹9,600"]
    answer: C
    explanation: |
      Food is 35% of ₹24,000 = 24,000 × 35 ÷ 100 = **₹8,400**.
      ₹840 slips a zero, and ₹6,000 is the rent (25%).

  - id: dh2-budget-education-angle
    stem: dh2-budget
    topic: Central angles
    difficulty: 1
    q: "What is the central angle of the Education sector?"
    options: ["15°", "36°", "54°", "60°"]
    answer: C
    explanation: |
      Each 1% of a pie is 360° ÷ 100 = 3.6°. Education is 15%, so its angle is 15 × 3.6 = **54°**.
      15° forgets to change the percentage into degrees.

  - id: dh2-budget-rent-travel
    stem: dh2-budget
    topic: Pie charts
    difficulty: 2
    q: "How much more does the family spend on rent than on travel each month?"
    options: ["₹1,300", "₹3,120", "₹3,600", "₹6,000"]
    answer: B
    explanation: |
      Rent − Travel = 25% − 12% = 13% of ₹24,000 = **₹3,120**.
      (Check: rent ₹6,000 − travel ₹2,880 = ₹3,120.) ₹1,300 treats 13% as ₹100 per 1%.

  - id: dh2-budget-food-rent-angle
    stem: dh2-budget
    topic: Central angles
    difficulty: 2
    q: "What is the total central angle of the Food and Rent sectors together?"
    options: ["60°", "108°", "180°", "216°"]
    answer: D
    explanation: |
      Food + Rent = 35% + 25% = 60%. Angle = 60 × 3.6 = **216°**.
      60° forgets to multiply by 3.6. 180° would be exactly half the pie, which is 50%, not 60%.

  - id: dh2-sports-badminton
    stem: dh2-sports
    topic: Pie charts
    difficulty: 1
    q: "How many students in all chose Badminton?"
    options: ["20", "120", "160", "200"]
    answer: C
    explanation: |
      Badminton is 20% of 800 = 800 × 20 ÷ 100 = **160** students.
      200 is the Football figure (25% of 800).

  - id: dh2-sports-cricket-boys
    stem: dh2-sports
    topic: Comparing data
    difficulty: 2
    q: "How many boys chose Cricket?"
    options: ["110", "140", "170", "280"]
    answer: B
    explanation: |
      Cricket is 35% of 800 = 280 students. 140 of them are girls, so 280 − 140 = **140** are boys.
      280 is all the students who chose Cricket, boys and girls together.

  - id: dh2-sports-more-girls
    stem: dh2-sports
    topic: Comparing data
    difficulty: 2
    q: "Which sport was chosen by MORE girls than boys?"
    options: ["Badminton", "Cricket", "Kabaddi", "Chess"]
    answer: A
    explanation: |
      Boys = all students − girls. **Badminton**: 160 students, 96 girls, 64 boys, so girls are more.
      Cricket (140 and 140) and Kabaddi (48 and 48) are equal, and Chess has 26 girls but 38 boys.

  - id: dh2-sports-football-girls
    stem: dh2-sports
    topic: Comparing data
    difficulty: 3
    q: "What percentage of all the GIRLS chose Football?"
    options: ["11.25%", "22.5%", "25%", "45%"]
    answer: B
    explanation: |
      There are 400 girls, and 90 of them chose Football: 90 ÷ 400 × 100 = **22.5%**.
      25% is Football's share of all 800 students. 11.25% divides by 800, and 45% divides by the 200 Football fans.

  - id: dh2-votes-total
    topic: Central angles
    difficulty: 3
    q: "In a class election, the votes are shown in a pie chart. Asha's sector has a central angle of 126° and Ravi's has 90°. Asha got 140 more votes than Ravi. How many votes were cast in all?"
    options: ["504", "1,260", "1,400", "3,600"]
    answer: C
    explanation: |
      The difference in angle is 126° − 90° = 36°, and that stands for 140 votes. 36° is one tenth of 360°.
      So the whole circle stands for 140 × 10 = **1,400** votes. (Asha 490, Ravi 350.)
      504 is 140 × 3.6, which mixes up degrees and votes.

  - id: dh2-shop-stationery
    stem: dh2-shop
    topic: Comparing data
    difficulty: 2
    q: "By how many rupees did the stationery sales go up from 2024 to 2025?"
    options: ["₹10,000", "₹12,500", "₹25,000", "₹30,000"]
    answer: D
    explanation: |
      2024: 35% of ₹2,00,000 = ₹70,000. 2025: 40% of ₹2,50,000 = ₹1,00,000. Rise = **₹30,000**.
      ₹10,000 and ₹12,500 take the 5% rise in share of only one year's total. The totals are different, so work out each year in rupees.

  - id: dh2-shop-fell
    stem: dh2-shop
    topic: Comparing data
    difficulty: 3
    q: "Which item's sales in rupees went DOWN from 2024 to 2025?"
    options: ["Books", "Toys", "Gifts", "None of them"]
    answer: D
    explanation: |
      Books ₹60,000 → ₹70,000, Toys ₹40,000 → ₹45,000, Gifts ₹30,000 → ₹35,000, Stationery ₹70,000 → ₹1,00,000. Every item went up, so the answer is **none of them**.
      The shares of books, toys and gifts fell, but they are shares of a bigger total.
---

This set tests pie charts: turning percentages into amounts and angles, and putting a pie chart together with other data.

## The two conversions

A pie chart is a circle of **360°** that stands for **100%** of the total.

| You know | You want | Do this |
|---|---|---|
| a percentage | an amount | percentage × total ÷ 100 |
| a percentage | the central angle | percentage × 3.6° |
| a central angle | the percentage | angle ÷ 3.6 (or angle ÷ 360 × 100) |
| a central angle | an amount | angle ÷ 360 × total |

The number 3.6 comes from 360° ÷ 100. So 50% is 180°, 25% is 90° and 10% is 36°.

## Worked example

*A school spends ₹60,000 on a fair. The pie sector for food is 108°. How much is spent on food?*

108° ÷ 3.6 = 30%, and 30% of ₹60,000 = **₹18,000**. Or directly: 108 ÷ 360 × 60,000 = ₹18,000.

## Mixing a chart with other data

- A pie chart gives **shares**. To compare two different groups or two different years, change each share into a real amount first.
- If you know what one part is worth, you can find the whole. If 18° stands for 50 people, then 360° (twenty times as much) stands for 1,000 people.
- When a question asks about only one group, such as only the girls, divide by **that group's** total, not the whole school's.

## Common trap

A smaller share of a bigger total can still be a bigger amount. 20% of ₹1,00,000 is less than 18% of ₹1,50,000.
