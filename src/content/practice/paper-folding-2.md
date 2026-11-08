---
title: "Paper folding 2: fold, cut and count"
description: "Cut shapes from folded paper and picture the opened sheet, count the pieces after cutting, and find the shape of a folded sheet."
slug: "paper-folding-2"
family: "paper-folding"
part: 2
timeMinutes: 12
questions:
  - id: pf2-notch-diamond
    topic: "Cut and open"
    difficulty: 1
    q: "A square sheet of paper is folded in half from left to right. The black part is then cut away from the folded paper, as shown in the last picture. Which option shows the sheet when it is opened out? (Black shows paper that has been cut away.)"
    figure: {"kind": "shapes", "w": 214, "h": 100, "frame": true, "items": [{"t": "line", "p": [12.0, 12.0, 88.0, 12.0]}, {"t": "line", "p": [88.0, 12.0, 88.0, 88.0]}, {"t": "line", "p": [88.0, 88.0, 12.0, 88.0]}, {"t": "line", "p": [12.0, 88.0, 12.0, 12.0]}, {"t": "line", "p": [50.0, 88.0, 50.0, 12.0], "dash": true}, {"t": "poly", "p": [[31.0, 50.0], [35.75, 46.67], [40.5, 44.3], [45.25, 42.88], [50.0, 42.4], [54.75, 42.88], [59.5, 44.3], [64.25, 46.67]], "open": true}, {"t": "arrow", "p": [64.25, 46.67, 69.0, 50.0]}, {"t": "line", "p": [164.0, 12.0, 202.0, 12.0]}, {"t": "line", "p": [202.0, 12.0, 202.0, 88.0]}, {"t": "line", "p": [202.0, 88.0, 164.0, 88.0]}, {"t": "line", "p": [164.0, 88.0, 164.0, 12.0], "dash": true}, {"t": "poly", "p": [[164.0, 40.88], [176.16, 50.0], [164.0, 59.12]], "fill": "solid"}]}
    optionFigures:
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [50.0, 99.0, 50.0, 1.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[50.0, 38.24], [65.68, 50.0], [50.0, 61.76]], "fill": "solid"}, {"t": "poly", "p": [[50.0, 38.24], [34.32, 50.0], [50.0, 61.76]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [50.0, 99.0, 50.0, 1.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[50.0, 38.24], [65.68, 50.0], [50.0, 61.76]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [50.0, 99.0, 50.0, 1.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[99.0, 38.24], [83.32, 50.0], [99.0, 61.76]], "fill": "solid"}, {"t": "poly", "p": [[1.0, 38.24], [16.68, 50.0], [1.0, 61.76]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [50.0, 99.0, 50.0, 1.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[50.0, 38.24], [65.68, 50.0], [50.0, 61.76]], "fill": "solid"}, {"t": "poly", "p": [[34.32, 38.24], [50.0, 50.0], [34.32, 61.76]], "fill": "solid"}]}
    answer: A
    explanation: |
      The triangle is cut from the **folded edge**, so it goes through both layers there. When you open the sheet, the two triangles meet at the fold and make one diamond-shaped hole in the middle: **A**.
      B forgets the second layer. C would come from cutting the open edge on the right, not the fold. D slides the second triangle instead of flipping it.

  - id: pf2-count-one-fold
    topic: "Counting pieces"
    difficulty: 1
    q: "A square sheet of paper is folded in half from top to bottom. The folded paper is then cut once along the thick line, straight across from edge to edge. How many pieces of paper are there?"
    figure: {"kind": "shapes", "w": 214, "h": 100, "frame": true, "items": [{"t": "line", "p": [12.0, 12.0, 88.0, 12.0]}, {"t": "line", "p": [88.0, 12.0, 88.0, 88.0]}, {"t": "line", "p": [88.0, 88.0, 12.0, 88.0]}, {"t": "line", "p": [12.0, 88.0, 12.0, 12.0]}, {"t": "line", "p": [12.0, 50.0, 88.0, 50.0], "dash": true}, {"t": "poly", "p": [[50.0, 31.0], [53.32, 35.75], [55.7, 40.5], [57.12, 45.25], [57.6, 50.0], [57.12, 54.75], [55.7, 59.5], [53.32, 64.25]], "open": true}, {"t": "arrow", "p": [53.32, 64.25, 50.0, 69.0]}, {"t": "line", "p": [202.0, 50.0, 202.0, 88.0]}, {"t": "line", "p": [202.0, 88.0, 126.0, 88.0]}, {"t": "line", "p": [126.0, 88.0, 126.0, 50.0]}, {"t": "line", "p": [126.0, 50.0, 202.0, 50.0], "dash": true}, {"t": "line", "p": [122.96, 69.0, 205.04, 69.0], "w": 4}]}
    options: ["2", "3", "4", "6"]
    answer: B
    explanation: |
      The cut goes through both layers, so the opened sheet has two cuts, one in each half, both parallel to the fold.
      Two straight cuts across a sheet make **3** pieces: a top strip, a bottom strip and a middle band that contains the fold.
      2 forgets that the paper had two layers.

  - id: pf2-two-diagonal-shape
    topic: "Folded shapes"
    difficulty: 1
    q: "A square sheet of paper is folded in half along a diagonal, so that the top right corner meets the bottom left corner. The triangle is then folded in half again, so that its two sharp corners meet, as shown. What shape is the folded paper now?"
    figure: {"kind": "shapes", "w": 214, "h": 100, "frame": true, "items": [{"t": "line", "p": [12.0, 12.0, 88.0, 12.0]}, {"t": "line", "p": [88.0, 12.0, 88.0, 88.0]}, {"t": "line", "p": [88.0, 88.0, 12.0, 88.0]}, {"t": "line", "p": [12.0, 88.0, 12.0, 12.0]}, {"t": "line", "p": [12.0, 12.0, 88.0, 88.0], "dash": true}, {"t": "poly", "p": [[62.67, 37.33], [62.82, 43.82], [62.03, 49.37], [60.29, 53.96], [57.6, 57.6], [53.96, 60.29], [49.37, 62.03], [43.83, 62.82]], "open": true}, {"t": "arrow", "p": [43.83, 62.82, 37.33, 62.67]}, {"t": "line", "p": [126.0, 12.0, 202.0, 88.0], "dash": true}, {"t": "line", "p": [202.0, 88.0, 126.0, 88.0]}, {"t": "line", "p": [126.0, 88.0, 126.0, 12.0]}, {"t": "line", "p": [126.0, 88.0, 164.0, 50.0], "dash": true}, {"t": "poly", "p": [[138.67, 50.0], [143.5, 51.5], [147.85, 53.48], [151.73, 55.94], [155.13, 58.87], [158.06, 62.27], [160.52, 66.15], [162.5, 70.5]], "open": true}, {"t": "arrow", "p": [162.5, 70.5, 164.0, 75.33]}]}
    options: ["A square, 1/4 of the sheet", "A right-angled triangle, 1/4 of the sheet", "A right-angled triangle, 1/2 of the sheet", "A rectangle, 1/4 of the sheet"]
    answer: B
    explanation: |
      The first fold makes a right-angled triangle that is half the sheet. Folding it so its two sharp corners meet cuts it in half along the line from the right angle to the middle of the long side.
      Each half is again a right-angled triangle, so the result is a **right-angled triangle, 1/4 of the sheet**.
      1/2 of the sheet is the size after only the first fold.

  - id: pf2-semicircle-circle
    topic: "Cut and open"
    difficulty: 1
    q: "A square sheet of paper is folded in half from top to bottom. The black part is then cut away from the folded paper, as shown in the last picture. Which option shows the sheet when it is opened out? (Black shows paper that has been cut away.)"
    figure: {"kind": "shapes", "w": 214, "h": 100, "frame": true, "items": [{"t": "line", "p": [12.0, 12.0, 88.0, 12.0]}, {"t": "line", "p": [88.0, 12.0, 88.0, 88.0]}, {"t": "line", "p": [88.0, 88.0, 12.0, 88.0]}, {"t": "line", "p": [12.0, 88.0, 12.0, 12.0]}, {"t": "line", "p": [12.0, 50.0, 88.0, 50.0], "dash": true}, {"t": "poly", "p": [[50.0, 31.0], [53.32, 35.75], [55.7, 40.5], [57.12, 45.25], [57.6, 50.0], [57.12, 54.75], [55.7, 59.5], [53.32, 64.25]], "open": true}, {"t": "arrow", "p": [53.32, 64.25, 50.0, 69.0]}, {"t": "line", "p": [202.0, 50.0, 202.0, 88.0]}, {"t": "line", "p": [202.0, 88.0, 126.0, 88.0]}, {"t": "line", "p": [126.0, 88.0, 126.0, 50.0]}, {"t": "line", "p": [126.0, 50.0, 202.0, 50.0], "dash": true}, {"t": "poly", "p": [[156.4, 50.0], [156.18, 51.82], [155.53, 53.53], [154.49, 55.04], [153.12, 56.25], [151.49, 57.11], [149.72, 57.54], [147.88, 57.54], [146.11, 57.11], [144.48, 56.25], [143.11, 55.04], [142.07, 53.53], [141.42, 51.82], [141.2, 50.0]], "fill": "solid"}]}
    optionFigures:
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[40.2, 50.0], [39.92, 52.35], [39.08, 54.55], [37.74, 56.5], [35.97, 58.07], [33.88, 59.16], [31.58, 59.73], [29.22, 59.73], [26.92, 59.16], [24.83, 58.07], [23.06, 56.5], [21.72, 54.55], [20.88, 52.35], [20.6, 50.0]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[79.4, 50.0], [79.12, 52.35], [78.28, 54.55], [76.94, 56.5], [75.17, 58.07], [73.08, 59.16], [70.78, 59.73], [68.42, 59.73], [66.12, 59.16], [64.03, 58.07], [62.26, 56.5], [60.92, 54.55], [60.08, 52.35], [59.8, 50.0]], "fill": "solid"}, {"t": "poly", "p": [[79.4, 50.0], [79.12, 47.65], [78.28, 45.45], [76.94, 43.5], [75.17, 41.93], [73.08, 40.84], [70.78, 40.27], [68.42, 40.27], [66.12, 40.84], [64.03, 41.93], [62.26, 43.5], [60.92, 45.45], [60.08, 47.65], [59.8, 50.0]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[40.2, 1.0], [39.92, 3.35], [39.08, 5.55], [37.74, 7.5], [35.97, 9.07], [33.88, 10.16], [31.58, 10.73], [29.22, 10.73], [26.92, 10.16], [24.83, 9.07], [23.06, 7.5], [21.72, 5.55], [20.88, 3.35], [20.6, 1.0]], "fill": "solid"}, {"t": "poly", "p": [[40.2, 99.0], [39.92, 96.65], [39.08, 94.45], [37.74, 92.5], [35.97, 90.93], [33.88, 89.84], [31.58, 89.27], [29.22, 89.27], [26.92, 89.84], [24.83, 90.93], [23.06, 92.5], [21.72, 94.45], [20.88, 96.65], [20.6, 99.0]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[40.2, 50.0], [39.92, 52.35], [39.08, 54.55], [37.74, 56.5], [35.97, 58.07], [33.88, 59.16], [31.58, 59.73], [29.22, 59.73], [26.92, 59.16], [24.83, 58.07], [23.06, 56.5], [21.72, 54.55], [20.88, 52.35], [20.6, 50.0]], "fill": "solid"}, {"t": "poly", "p": [[40.2, 50.0], [39.92, 47.65], [39.08, 45.45], [37.74, 43.5], [35.97, 41.93], [33.88, 40.84], [31.58, 40.27], [29.22, 40.27], [26.92, 40.84], [24.83, 41.93], [23.06, 43.5], [21.72, 45.45], [20.88, 47.65], [20.6, 50.0]], "fill": "solid"}]}
    answer: D
    explanation: |
      A half circle cut from the **fold** opens into a full circle, centred on the fold line, at the same distance from the left edge: **D**.
      A forgets the top layer. B puts the circle on the wrong side, but opening a top-to-bottom fold never moves anything left or right. C would come from cutting the open edges, not the fold.

  - id: pf2-centre-corner
    topic: "Cut and open"
    difficulty: 2
    q: "A square sheet of paper is folded in half from top to bottom, and then from left to right. The black part is then cut away from the folded paper, as shown in the last picture. Which option shows the sheet when it is opened out? (Black shows paper that has been cut away.)"
    figure: {"kind": "shapes", "w": 328, "h": 100, "frame": true, "items": [{"t": "line", "p": [12.0, 12.0, 88.0, 12.0]}, {"t": "line", "p": [88.0, 12.0, 88.0, 88.0]}, {"t": "line", "p": [88.0, 88.0, 12.0, 88.0]}, {"t": "line", "p": [12.0, 88.0, 12.0, 12.0]}, {"t": "line", "p": [12.0, 50.0, 88.0, 50.0], "dash": true}, {"t": "poly", "p": [[50.0, 31.0], [53.32, 35.75], [55.7, 40.5], [57.12, 45.25], [57.6, 50.0], [57.12, 54.75], [55.7, 59.5], [53.32, 64.25]], "open": true}, {"t": "arrow", "p": [53.32, 64.25, 50.0, 69.0]}, {"t": "line", "p": [202.0, 50.0, 202.0, 88.0]}, {"t": "line", "p": [202.0, 88.0, 126.0, 88.0]}, {"t": "line", "p": [126.0, 88.0, 126.0, 50.0]}, {"t": "line", "p": [126.0, 50.0, 202.0, 50.0], "dash": true}, {"t": "line", "p": [164.0, 88.0, 164.0, 50.0], "dash": true}, {"t": "poly", "p": [[145.0, 69.0], [149.75, 67.34], [154.5, 66.15], [159.25, 65.44], [164.0, 65.2], [168.75, 65.44], [173.5, 66.15], [178.25, 67.34]], "open": true}, {"t": "arrow", "p": [178.25, 67.34, 183.0, 69.0]}, {"t": "line", "p": [316.0, 50.0, 316.0, 88.0]}, {"t": "line", "p": [316.0, 88.0, 278.0, 88.0]}, {"t": "line", "p": [278.0, 88.0, 278.0, 50.0], "dash": true}, {"t": "line", "p": [278.0, 50.0, 316.0, 50.0], "dash": true}, {"t": "poly", "p": [[278.0, 50.0], [289.4, 50.0], [278.0, 61.4]], "fill": "solid"}]}
    optionFigures:
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[50.0, 50.0], [64.7, 50.0], [50.0, 64.7]], "fill": "solid"}, {"t": "poly", "p": [[50.0, 50.0], [35.3, 50.0], [50.0, 64.7]], "fill": "solid"}, {"t": "poly", "p": [[50.0, 50.0], [64.7, 50.0], [50.0, 35.3]], "fill": "solid"}, {"t": "poly", "p": [[50.0, 50.0], [35.3, 50.0], [50.0, 35.3]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[50.0, 50.0], [64.7, 50.0], [50.0, 64.7]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[50.0, 50.0], [64.7, 50.0], [50.0, 64.7]], "fill": "solid"}, {"t": "poly", "p": [[50.0, 50.0], [35.3, 50.0], [50.0, 64.7]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[99.0, 99.0], [84.3, 99.0], [99.0, 84.3]], "fill": "solid"}, {"t": "poly", "p": [[1.0, 99.0], [15.7, 99.0], [1.0, 84.3]], "fill": "solid"}, {"t": "poly", "p": [[99.0, 1.0], [84.3, 1.0], [99.0, 15.7]], "fill": "solid"}, {"t": "poly", "p": [[1.0, 1.0], [15.7, 1.0], [1.0, 15.7]], "fill": "solid"}]}
    answer: A
    explanation: |
      The cut corner is where both folds meet, which is the **centre** of the sheet. Opening the second fold gives a triangle pointing down; opening the first fold adds its reflection above.
      Four small triangles meet at the centre and make a square hole standing on its corner: **A**.
      B forgets that the paper had four layers, and C opens only one fold. D would come from cutting the opposite corner, which is a corner of the sheet.

  - id: pf2-outer-corner
    topic: "Cut and open"
    difficulty: 2
    q: "A square sheet of paper is folded in half from top to bottom, and then from left to right. The black part is then cut away from the folded paper, as shown in the last picture. Which option shows the sheet when it is opened out? (Black shows paper that has been cut away.)"
    figure: {"kind": "shapes", "w": 328, "h": 100, "frame": true, "items": [{"t": "line", "p": [12.0, 12.0, 88.0, 12.0]}, {"t": "line", "p": [88.0, 12.0, 88.0, 88.0]}, {"t": "line", "p": [88.0, 88.0, 12.0, 88.0]}, {"t": "line", "p": [12.0, 88.0, 12.0, 12.0]}, {"t": "line", "p": [12.0, 50.0, 88.0, 50.0], "dash": true}, {"t": "poly", "p": [[50.0, 31.0], [53.32, 35.75], [55.7, 40.5], [57.12, 45.25], [57.6, 50.0], [57.12, 54.75], [55.7, 59.5], [53.32, 64.25]], "open": true}, {"t": "arrow", "p": [53.32, 64.25, 50.0, 69.0]}, {"t": "line", "p": [202.0, 50.0, 202.0, 88.0]}, {"t": "line", "p": [202.0, 88.0, 126.0, 88.0]}, {"t": "line", "p": [126.0, 88.0, 126.0, 50.0]}, {"t": "line", "p": [126.0, 50.0, 202.0, 50.0], "dash": true}, {"t": "line", "p": [164.0, 88.0, 164.0, 50.0], "dash": true}, {"t": "poly", "p": [[145.0, 69.0], [149.75, 67.34], [154.5, 66.15], [159.25, 65.44], [164.0, 65.2], [168.75, 65.44], [173.5, 66.15], [178.25, 67.34]], "open": true}, {"t": "arrow", "p": [178.25, 67.34, 183.0, 69.0]}, {"t": "line", "p": [316.0, 50.0, 316.0, 88.0]}, {"t": "line", "p": [316.0, 88.0, 278.0, 88.0]}, {"t": "line", "p": [278.0, 88.0, 278.0, 50.0], "dash": true}, {"t": "line", "p": [278.0, 50.0, 316.0, 50.0], "dash": true}, {"t": "poly", "p": [[316.0, 88.0], [303.84, 88.0], [316.0, 75.84]], "fill": "solid"}]}
    optionFigures:
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[99.0, 99.0], [83.32, 99.0], [99.0, 83.32]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[99.0, 99.0], [83.32, 99.0], [99.0, 83.32]], "fill": "solid"}, {"t": "poly", "p": [[1.0, 99.0], [16.68, 99.0], [1.0, 83.32]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[50.0, 50.0], [65.68, 50.0], [50.0, 65.68]], "fill": "solid"}, {"t": "poly", "p": [[50.0, 50.0], [34.32, 50.0], [50.0, 65.68]], "fill": "solid"}, {"t": "poly", "p": [[50.0, 50.0], [65.68, 50.0], [50.0, 34.32]], "fill": "solid"}, {"t": "poly", "p": [[50.0, 50.0], [34.32, 50.0], [50.0, 34.32]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[99.0, 99.0], [83.32, 99.0], [99.0, 83.32]], "fill": "solid"}, {"t": "poly", "p": [[1.0, 99.0], [16.68, 99.0], [1.0, 83.32]], "fill": "solid"}, {"t": "poly", "p": [[99.0, 1.0], [83.32, 1.0], [99.0, 16.68]], "fill": "solid"}, {"t": "poly", "p": [[1.0, 1.0], [16.68, 1.0], [1.0, 16.68]], "fill": "solid"}]}
    answer: D
    explanation: |
      The cut corner of the small square is a corner of the **original sheet** (it has no folds through it). All four corners of the sheet are stacked there, so all four are cut off: **D**.
      A and B forget some of the layers. C would come from cutting the corner where the folds meet.

  - id: pf2-count-diagonal-cut
    topic: "Counting pieces"
    difficulty: 2
    q: "A square sheet of paper is folded in half from top to bottom, and then from left to right. The small square is then cut along the thick line, from the corner where the folds meet to the opposite corner. How many pieces are there when it is opened?"
    figure: {"kind": "shapes", "w": 328, "h": 100, "frame": true, "items": [{"t": "line", "p": [12.0, 12.0, 88.0, 12.0]}, {"t": "line", "p": [88.0, 12.0, 88.0, 88.0]}, {"t": "line", "p": [88.0, 88.0, 12.0, 88.0]}, {"t": "line", "p": [12.0, 88.0, 12.0, 12.0]}, {"t": "line", "p": [12.0, 50.0, 88.0, 50.0], "dash": true}, {"t": "poly", "p": [[50.0, 31.0], [53.32, 35.75], [55.7, 40.5], [57.12, 45.25], [57.6, 50.0], [57.12, 54.75], [55.7, 59.5], [53.32, 64.25]], "open": true}, {"t": "arrow", "p": [53.32, 64.25, 50.0, 69.0]}, {"t": "line", "p": [202.0, 50.0, 202.0, 88.0]}, {"t": "line", "p": [202.0, 88.0, 126.0, 88.0]}, {"t": "line", "p": [126.0, 88.0, 126.0, 50.0]}, {"t": "line", "p": [126.0, 50.0, 202.0, 50.0], "dash": true}, {"t": "line", "p": [164.0, 88.0, 164.0, 50.0], "dash": true}, {"t": "poly", "p": [[145.0, 69.0], [149.75, 67.34], [154.5, 66.15], [159.25, 65.44], [164.0, 65.2], [168.75, 65.44], [173.5, 66.15], [178.25, 67.34]], "open": true}, {"t": "arrow", "p": [178.25, 67.34, 183.0, 69.0]}, {"t": "line", "p": [316.0, 50.0, 316.0, 88.0]}, {"t": "line", "p": [316.0, 88.0, 278.0, 88.0]}, {"t": "line", "p": [278.0, 88.0, 278.0, 50.0], "dash": true}, {"t": "line", "p": [278.0, 50.0, 316.0, 50.0], "dash": true}, {"t": "line", "p": [275.85, 47.85, 318.15, 90.15], "w": 4}]}
    options: ["2", "3", "4", "8"]
    answer: C
    explanation: |
      Open the folds one at a time. The cut runs from the centre of the sheet to its bottom right corner. Opening the left-right fold adds a cut from the centre to the bottom left corner, and opening the top-bottom fold adds cuts from the centre to the two top corners.
      So the sheet is cut along **both diagonals**, which makes **4** triangles.
      8 counts the layers, not the pieces.

  - id: pf2-corners-to-centre
    topic: "Folded shapes"
    difficulty: 2
    q: "Each corner of a square sheet of paper is folded in to the centre, along the dashed lines. What shape is the folded paper, and how big is it?"
    figure: {"kind": "shapes", "frame": true, "items": [{"t": "rect", "p": [12, 12, 76, 76], "fill": "none"}, {"t": "line", "p": [50.0, 12.0, 88.0, 50.0], "dash": true}, {"t": "line", "p": [88.0, 50.0, 50.0, 88.0], "dash": true}, {"t": "line", "p": [50.0, 88.0, 12.0, 50.0], "dash": true}, {"t": "line", "p": [12.0, 50.0, 50.0, 12.0], "dash": true}, {"t": "arrow", "p": [17.32, 17.32, 39.36, 39.36], "w": 2}, {"t": "arrow", "p": [82.68, 17.32, 60.64, 39.36], "w": 2}, {"t": "arrow", "p": [82.68, 82.68, 60.64, 60.64], "w": 2}, {"t": "arrow", "p": [17.32, 82.68, 39.36, 60.64], "w": 2}]}
    options: ["A square, 1/4 of the sheet", "An octagon, 3/4 of the sheet", "A triangle, 1/2 of the sheet", "A square, 1/2 of the sheet"]
    answer: D
    explanation: |
      The dashed lines join the middles of the four sides, so they make a square standing on its corner. Each corner flap folds over exactly onto one quarter of that square.
      The four flaps cover the inner square, so the result is a **square, 1/2 of the sheet** (the four corner triangles together equal the inner square).
      An octagon would appear only if the corners did not reach the centre.

  - id: pf2-open-edge-notch
    topic: "Cut and open"
    difficulty: 2
    q: "A square sheet of paper is folded in half from left to right. The black part is then cut away from the folded paper, as shown in the last picture. Which option shows the sheet when it is opened out? (Black shows paper that has been cut away.)"
    figure: {"kind": "shapes", "w": 214, "h": 100, "frame": true, "items": [{"t": "line", "p": [12.0, 12.0, 88.0, 12.0]}, {"t": "line", "p": [88.0, 12.0, 88.0, 88.0]}, {"t": "line", "p": [88.0, 88.0, 12.0, 88.0]}, {"t": "line", "p": [12.0, 88.0, 12.0, 12.0]}, {"t": "line", "p": [50.0, 88.0, 50.0, 12.0], "dash": true}, {"t": "poly", "p": [[31.0, 50.0], [35.75, 46.67], [40.5, 44.3], [45.25, 42.88], [50.0, 42.4], [54.75, 42.88], [59.5, 44.3], [64.25, 46.67]], "open": true}, {"t": "arrow", "p": [64.25, 46.67, 69.0, 50.0]}, {"t": "line", "p": [164.0, 12.0, 202.0, 12.0]}, {"t": "line", "p": [202.0, 12.0, 202.0, 88.0]}, {"t": "line", "p": [202.0, 88.0, 164.0, 88.0]}, {"t": "line", "p": [164.0, 88.0, 164.0, 12.0], "dash": true}, {"t": "poly", "p": [[202.0, 27.2], [190.6, 34.8], [202.0, 42.4]], "fill": "solid"}]}
    optionFigures:
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [50.0, 99.0, 50.0, 1.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[99.0, 20.6], [84.3, 30.4], [99.0, 40.2]], "fill": "solid"}, {"t": "poly", "p": [[1.0, 20.6], [15.7, 30.4], [1.0, 40.2]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [50.0, 99.0, 50.0, 1.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[99.0, 20.6], [84.3, 30.4], [99.0, 40.2]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [50.0, 99.0, 50.0, 1.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[50.0, 20.6], [64.7, 30.4], [50.0, 40.2]], "fill": "solid"}, {"t": "poly", "p": [[50.0, 20.6], [35.3, 30.4], [50.0, 40.2]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [50.0, 99.0, 50.0, 1.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[99.0, 20.6], [84.3, 30.4], [99.0, 40.2]], "fill": "solid"}, {"t": "poly", "p": [[1.0, 59.8], [15.7, 69.6], [1.0, 79.4]], "fill": "solid"}]}
    answer: A
    explanation: |
      The notch is cut from the right edge, which is an **open** edge: the left and right edges of the sheet lie on top of each other there. So both edges get a notch, at the same height near the top: **A**.
      B forgets the bottom layer. C would come from cutting the fold. D moves the second notch down, but opening a left-to-right fold never moves anything up or down.

  - id: pf2-notch-two-folds
    topic: "Cut and open"
    difficulty: 3
    q: "A square sheet of paper is folded in half from top to bottom, and then from left to right. The black part is then cut away from the folded paper, as shown in the last picture. Which option shows the sheet when it is opened out? (Black shows paper that has been cut away.)"
    figure: {"kind": "shapes", "w": 328, "h": 100, "frame": true, "items": [{"t": "line", "p": [12.0, 12.0, 88.0, 12.0]}, {"t": "line", "p": [88.0, 12.0, 88.0, 88.0]}, {"t": "line", "p": [88.0, 88.0, 12.0, 88.0]}, {"t": "line", "p": [12.0, 88.0, 12.0, 12.0]}, {"t": "line", "p": [12.0, 50.0, 88.0, 50.0], "dash": true}, {"t": "poly", "p": [[50.0, 31.0], [53.32, 35.75], [55.7, 40.5], [57.12, 45.25], [57.6, 50.0], [57.12, 54.75], [55.7, 59.5], [53.32, 64.25]], "open": true}, {"t": "arrow", "p": [53.32, 64.25, 50.0, 69.0]}, {"t": "line", "p": [202.0, 50.0, 202.0, 88.0]}, {"t": "line", "p": [202.0, 88.0, 126.0, 88.0]}, {"t": "line", "p": [126.0, 88.0, 126.0, 50.0]}, {"t": "line", "p": [126.0, 50.0, 202.0, 50.0], "dash": true}, {"t": "line", "p": [164.0, 88.0, 164.0, 50.0], "dash": true}, {"t": "poly", "p": [[145.0, 69.0], [149.75, 67.34], [154.5, 66.15], [159.25, 65.44], [164.0, 65.2], [168.75, 65.44], [173.5, 66.15], [178.25, 67.34]], "open": true}, {"t": "arrow", "p": [178.25, 67.34, 183.0, 69.0]}, {"t": "line", "p": [316.0, 50.0, 316.0, 88.0]}, {"t": "line", "p": [316.0, 88.0, 278.0, 88.0]}, {"t": "line", "p": [278.0, 88.0, 278.0, 50.0], "dash": true}, {"t": "line", "p": [278.0, 50.0, 316.0, 50.0], "dash": true}, {"t": "poly", "p": [[278.0, 62.92], [288.64, 69.0], [278.0, 75.08]], "fill": "solid"}]}
    optionFigures:
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[50.0, 66.66], [63.72, 74.5], [50.0, 82.34]], "fill": "solid"}, {"t": "poly", "p": [[50.0, 66.66], [36.28, 74.5], [50.0, 82.34]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[50.0, 66.66], [63.72, 74.5], [50.0, 82.34]], "fill": "solid"}, {"t": "poly", "p": [[50.0, 66.66], [36.28, 74.5], [50.0, 82.34]], "fill": "solid"}, {"t": "poly", "p": [[50.0, 33.34], [63.72, 25.5], [50.0, 17.66]], "fill": "solid"}, {"t": "poly", "p": [[50.0, 33.34], [36.28, 25.5], [50.0, 17.66]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[66.66, 50.0], [74.5, 63.72], [82.34, 50.0]], "fill": "solid"}, {"t": "poly", "p": [[33.34, 50.0], [25.5, 63.72], [17.66, 50.0]], "fill": "solid"}, {"t": "poly", "p": [[66.66, 50.0], [74.5, 36.28], [82.34, 50.0]], "fill": "solid"}, {"t": "poly", "p": [[33.34, 50.0], [25.5, 36.28], [17.66, 50.0]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [1.0, 50.0, 99.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 99.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "line", "p": [50.0, 1.0, 50.0, 50.0], "dash": true, "w": 1.5}, {"t": "poly", "p": [[50.0, 66.66], [63.72, 74.5], [50.0, 82.34]], "fill": "solid"}, {"t": "poly", "p": [[50.0, 66.66], [36.28, 74.5], [50.0, 82.34]], "fill": "solid"}, {"t": "poly", "p": [[50.0, 33.34], [63.72, 25.5], [50.0, 17.66]], "fill": "solid"}, {"t": "poly", "p": [[50.0, 33.34], [36.28, 25.5], [50.0, 17.66]], "fill": "solid"}, {"t": "poly", "p": [[66.66, 50.0], [74.5, 63.72], [82.34, 50.0]], "fill": "solid"}, {"t": "poly", "p": [[33.34, 50.0], [25.5, 63.72], [17.66, 50.0]], "fill": "solid"}, {"t": "poly", "p": [[66.66, 50.0], [74.5, 36.28], [82.34, 50.0]], "fill": "solid"}, {"t": "poly", "p": [[33.34, 50.0], [25.5, 36.28], [17.66, 50.0]], "fill": "solid"}]}
    answer: B
    explanation: |
      The notch is on the **second** fold (the up-down one), in the bottom half. Opening that fold turns it into a diamond on the up-down centre line, near the bottom.
      Opening the first fold copies it into the top half. So there are **two diamonds on the up-down centre line**: **B**.
      A forgets the first fold. C puts the diamonds on the wrong centre line, and D has two extra diamonds.

  - id: pf2-count-diamond-cut
    topic: "Counting pieces"
    difficulty: 3
    q: "A square sheet of paper is folded in half from top to bottom, and then from left to right. The small square is then cut along the thick line, which joins two of its corners but does not pass through the corner where the folds meet. How many pieces are there when it is opened?"
    figure: {"kind": "shapes", "w": 328, "h": 100, "frame": true, "items": [{"t": "line", "p": [12.0, 12.0, 88.0, 12.0]}, {"t": "line", "p": [88.0, 12.0, 88.0, 88.0]}, {"t": "line", "p": [88.0, 88.0, 12.0, 88.0]}, {"t": "line", "p": [12.0, 88.0, 12.0, 12.0]}, {"t": "line", "p": [12.0, 50.0, 88.0, 50.0], "dash": true}, {"t": "poly", "p": [[50.0, 31.0], [53.32, 35.75], [55.7, 40.5], [57.12, 45.25], [57.6, 50.0], [57.12, 54.75], [55.7, 59.5], [53.32, 64.25]], "open": true}, {"t": "arrow", "p": [53.32, 64.25, 50.0, 69.0]}, {"t": "line", "p": [202.0, 50.0, 202.0, 88.0]}, {"t": "line", "p": [202.0, 88.0, 126.0, 88.0]}, {"t": "line", "p": [126.0, 88.0, 126.0, 50.0]}, {"t": "line", "p": [126.0, 50.0, 202.0, 50.0], "dash": true}, {"t": "line", "p": [164.0, 88.0, 164.0, 50.0], "dash": true}, {"t": "poly", "p": [[145.0, 69.0], [149.75, 67.34], [154.5, 66.15], [159.25, 65.44], [164.0, 65.2], [168.75, 65.44], [173.5, 66.15], [178.25, 67.34]], "open": true}, {"t": "arrow", "p": [178.25, 67.34, 183.0, 69.0]}, {"t": "line", "p": [316.0, 50.0, 316.0, 88.0]}, {"t": "line", "p": [316.0, 88.0, 278.0, 88.0]}, {"t": "line", "p": [278.0, 88.0, 278.0, 50.0], "dash": true}, {"t": "line", "p": [278.0, 50.0, 316.0, 50.0], "dash": true}, {"t": "line", "p": [318.15, 47.85, 275.85, 90.15], "w": 4}]}
    options: ["2", "4", "5", "8"]
    answer: C
    explanation: |
      The cut joins the middle of the right side of the sheet to the middle of the bottom side. Opening the folds reflects it into all four quarters, so the cuts join the middles of all four sides and make a diamond.
      The pieces are the diamond in the middle and the **4** corner triangles: **5** pieces in all.
      4 forgets the middle piece, and 8 counts the layers.

  - id: pf2-count-cross-cut
    topic: "Counting pieces"
    difficulty: 3
    q: "A square sheet of paper is folded in half from top to bottom, and then from left to right. The small square is then cut along the two thick lines: once straight across its middle and once straight down its middle. How many pieces are there when it is opened?"
    figure: {"kind": "shapes", "w": 328, "h": 100, "frame": true, "items": [{"t": "line", "p": [12.0, 12.0, 88.0, 12.0]}, {"t": "line", "p": [88.0, 12.0, 88.0, 88.0]}, {"t": "line", "p": [88.0, 88.0, 12.0, 88.0]}, {"t": "line", "p": [12.0, 88.0, 12.0, 12.0]}, {"t": "line", "p": [12.0, 50.0, 88.0, 50.0], "dash": true}, {"t": "poly", "p": [[50.0, 31.0], [53.32, 35.75], [55.7, 40.5], [57.12, 45.25], [57.6, 50.0], [57.12, 54.75], [55.7, 59.5], [53.32, 64.25]], "open": true}, {"t": "arrow", "p": [53.32, 64.25, 50.0, 69.0]}, {"t": "line", "p": [202.0, 50.0, 202.0, 88.0]}, {"t": "line", "p": [202.0, 88.0, 126.0, 88.0]}, {"t": "line", "p": [126.0, 88.0, 126.0, 50.0]}, {"t": "line", "p": [126.0, 50.0, 202.0, 50.0], "dash": true}, {"t": "line", "p": [164.0, 88.0, 164.0, 50.0], "dash": true}, {"t": "poly", "p": [[145.0, 69.0], [149.75, 67.34], [154.5, 66.15], [159.25, 65.44], [164.0, 65.2], [168.75, 65.44], [173.5, 66.15], [178.25, 67.34]], "open": true}, {"t": "arrow", "p": [178.25, 67.34, 183.0, 69.0]}, {"t": "line", "p": [316.0, 50.0, 316.0, 88.0]}, {"t": "line", "p": [316.0, 88.0, 278.0, 88.0]}, {"t": "line", "p": [278.0, 88.0, 278.0, 50.0], "dash": true}, {"t": "line", "p": [278.0, 50.0, 316.0, 50.0], "dash": true}, {"t": "line", "p": [274.96, 69.0, 319.04, 69.0], "w": 4}, {"t": "line", "p": [297.0, 46.96, 297.0, 91.04], "w": 4}]}
    options: ["4", "8", "9", "16"]
    answer: C
    explanation: |
      Open the folds and draw every cut on the full sheet. The cut across becomes two cuts across the whole sheet, 1/4 and 3/4 of the way down. The cut down becomes two cuts down the whole sheet, 1/4 and 3/4 of the way across.
      Two cuts each way make a 3 × 3 grid, so there are **9** pieces.
      16 multiplies the 4 pieces of the small square by its 4 layers, but pieces that meet at a fold stay joined.

---

This set tests what happens when folded paper is cut, and what shape a sheet makes when it is folded.

## Cut and open

Every cut goes through **all the layers**. When you open a fold, each cut gets a mirror partner on the other side of the fold line.

- A cut **on a fold** opens into one hole that is symmetric about the fold. Half a circle on a fold becomes a full circle; a triangle becomes a diamond.
- A cut on an **open edge** (where separate edges lie on top of each other) appears on each of those edges.
- A cut at the corner where **two folds meet** opens into one hole at the centre of the sheet.

Open the folds in **reverse order**, last fold first.

## Counting pieces

Open the folds in your head and draw every cut line on the full sheet. Then count the regions. Two useful facts:

| Cuts on the opened sheet | Pieces |
|---|---|
| n straight cuts side by side, edge to edge | n + 1 |
| Both diagonals | 4 |

## Worked example

A sheet folded in half from left to right has a small square cut from the middle of its folded edge. When opened, the two squares join across the fold into one rectangle, twice as wide as it is tall, in the middle of the sheet.

## Folded shapes

Follow the outline one fold at a time. Each fold in half halves the area, and folding along a diagonal turns a square into a right-angled triangle.

## Common trap

Counting the **layers** instead of the **pieces**. With 8 layers and one cut across, there are 9 pieces, not 8.

In the pictures, dashed lines are folds and black areas are paper that has been cut away.
