---
title: "Figure patterns 3: matrices, completion and hidden figures"
description: "3 by 3 figure matrices with one box missing, the piece that completes a pattern, and the shape hidden inside a bigger figure."
slug: "figure-patterns-3"
family: "figure-patterns"
part: 3
timeMinutes: 12
questions:
  - id: fp3-matrix-sides
    topic: "Matrices"
    difficulty: 1
    q: "Which figure goes in the empty box?"
    figure: {"kind": "shapes", "w": 300, "h": 300, "items": [{"t": "line", "p": [1, 1, 99, 1], "w": 1.5}, {"t": "line", "p": [99, 1, 99, 99], "w": 1.5}, {"t": "line", "p": [99, 99, 1, 99], "w": 1.5}, {"t": "line", "p": [1, 99, 1, 1], "w": 1.5}, {"t": "poly", "p": [[50, 22], [75.98, 67], [24.02, 67]], "fill": "none"}, {"t": "line", "p": [101, 1, 199, 1], "w": 1.5}, {"t": "line", "p": [199, 1, 199, 99], "w": 1.5}, {"t": "line", "p": [199, 99, 101, 99], "w": 1.5}, {"t": "line", "p": [101, 99, 101, 1], "w": 1.5}, {"t": "poly", "p": [[171.21, 30.79], [171.21, 73.21], [128.79, 73.21], [128.79, 30.79]], "fill": "none"}, {"t": "line", "p": [201, 1, 299, 1], "w": 1.5}, {"t": "line", "p": [299, 1, 299, 99], "w": 1.5}, {"t": "line", "p": [299, 99, 201, 99], "w": 1.5}, {"t": "line", "p": [201, 99, 201, 1], "w": 1.5}, {"t": "poly", "p": [[250, 22], [278.53, 42.73], [267.63, 76.27], [232.37, 76.27], [221.47, 42.73]], "fill": "none"}, {"t": "line", "p": [1, 101, 99, 101], "w": 1.5}, {"t": "line", "p": [99, 101, 99, 199], "w": 1.5}, {"t": "line", "p": [99, 199, 1, 199], "w": 1.5}, {"t": "line", "p": [1, 199, 1, 101], "w": 1.5}, {"t": "poly", "p": [[71.21, 130.79], [71.21, 173.21], [28.79, 173.21], [28.79, 130.79]], "fill": "none"}, {"t": "line", "p": [101, 101, 199, 101], "w": 1.5}, {"t": "line", "p": [199, 101, 199, 199], "w": 1.5}, {"t": "line", "p": [199, 199, 101, 199], "w": 1.5}, {"t": "line", "p": [101, 199, 101, 101], "w": 1.5}, {"t": "poly", "p": [[150, 122], [178.53, 142.73], [167.63, 176.27], [132.37, 176.27], [121.47, 142.73]], "fill": "none"}, {"t": "line", "p": [201, 101, 299, 101], "w": 1.5}, {"t": "line", "p": [299, 101, 299, 199], "w": 1.5}, {"t": "line", "p": [299, 199, 201, 199], "w": 1.5}, {"t": "line", "p": [201, 199, 201, 101], "w": 1.5}, {"t": "poly", "p": [[250, 122], [275.98, 167], [224.02, 167]], "fill": "none"}, {"t": "line", "p": [1, 201, 99, 201], "w": 1.5}, {"t": "line", "p": [99, 201, 99, 299], "w": 1.5}, {"t": "line", "p": [99, 299, 1, 299], "w": 1.5}, {"t": "line", "p": [1, 299, 1, 201], "w": 1.5}, {"t": "poly", "p": [[50, 222], [78.53, 242.73], [67.63, 276.27], [32.37, 276.27], [21.47, 242.73]], "fill": "none"}, {"t": "line", "p": [101, 201, 199, 201], "w": 1.5}, {"t": "line", "p": [199, 201, 199, 299], "w": 1.5}, {"t": "line", "p": [199, 299, 101, 299], "w": 1.5}, {"t": "line", "p": [101, 299, 101, 201], "w": 1.5}, {"t": "poly", "p": [[150, 222], [175.98, 267], [124.02, 267]], "fill": "none"}, {"t": "line", "p": [201, 201, 299, 201], "w": 1.5}, {"t": "line", "p": [299, 201, 299, 299], "w": 1.5}, {"t": "line", "p": [299, 299, 201, 299], "w": 1.5}, {"t": "line", "p": [201, 299, 201, 201], "w": 1.5}, {"t": "text", "at": [250, 252], "s": "?", "size": 44}], "alt": "A 3 by 3 grid of figures with the bottom right box empty"}
    optionFigures:
      - {"kind": "shapes", "frame": true, "items": [{"t": "poly", "p": [[71.21, 30.79], [71.21, 73.21], [28.79, 73.21], [28.79, 30.79]], "fill": "none"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "poly", "p": [[50, 22], [75.98, 67], [24.02, 67]], "fill": "none"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "poly", "p": [[50, 22], [78.53, 42.73], [67.63, 76.27], [32.37, 76.27], [21.47, 42.73]], "fill": "none"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "poly", "p": [[65, 26.02], [80, 52], [65, 77.98], [35, 77.98], [20, 52], [35, 26.02]], "fill": "none"}]}
    answer: A
    explanation: |
      Each row has one triangle, one square and one pentagon. The bottom row already has a pentagon and a triangle, so the missing figure is the **square**: **A**.
      The columns follow the same rule, which confirms it.

  - id: fp3-matrix-dots
    topic: "Matrices"
    difficulty: 1
    q: "Which figure goes in the empty box?"
    figure: {"kind": "shapes", "w": 300, "h": 300, "items": [{"t": "line", "p": [1, 1, 99, 1], "w": 1.5}, {"t": "line", "p": [99, 1, 99, 99], "w": 1.5}, {"t": "line", "p": [99, 99, 1, 99], "w": 1.5}, {"t": "line", "p": [1, 99, 1, 1], "w": 1.5}, {"t": "circle", "c": [50, 50], "r": 5.5, "fill": "solid"}, {"t": "line", "p": [101, 1, 199, 1], "w": 1.5}, {"t": "line", "p": [199, 1, 199, 99], "w": 1.5}, {"t": "line", "p": [199, 99, 101, 99], "w": 1.5}, {"t": "line", "p": [101, 99, 101, 1], "w": 1.5}, {"t": "circle", "c": [141.5, 50], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [158.5, 50], "r": 5.5, "fill": "solid"}, {"t": "line", "p": [201, 1, 299, 1], "w": 1.5}, {"t": "line", "p": [299, 1, 299, 99], "w": 1.5}, {"t": "line", "p": [299, 99, 201, 99], "w": 1.5}, {"t": "line", "p": [201, 99, 201, 1], "w": 1.5}, {"t": "circle", "c": [233, 50], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [250, 50], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [267, 50], "r": 5.5, "fill": "solid"}, {"t": "line", "p": [1, 101, 99, 101], "w": 1.5}, {"t": "line", "p": [99, 101, 99, 199], "w": 1.5}, {"t": "line", "p": [99, 199, 1, 199], "w": 1.5}, {"t": "line", "p": [1, 199, 1, 101], "w": 1.5}, {"t": "circle", "c": [41.5, 150], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [58.5, 150], "r": 5.5, "fill": "solid"}, {"t": "line", "p": [101, 101, 199, 101], "w": 1.5}, {"t": "line", "p": [199, 101, 199, 199], "w": 1.5}, {"t": "line", "p": [199, 199, 101, 199], "w": 1.5}, {"t": "line", "p": [101, 199, 101, 101], "w": 1.5}, {"t": "circle", "c": [133, 150], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [150, 150], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [167, 150], "r": 5.5, "fill": "solid"}, {"t": "line", "p": [201, 101, 299, 101], "w": 1.5}, {"t": "line", "p": [299, 101, 299, 199], "w": 1.5}, {"t": "line", "p": [299, 199, 201, 199], "w": 1.5}, {"t": "line", "p": [201, 199, 201, 101], "w": 1.5}, {"t": "circle", "c": [241.5, 141.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [258.5, 141.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [241.5, 158.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [258.5, 158.5], "r": 5.5, "fill": "solid"}, {"t": "line", "p": [1, 201, 99, 201], "w": 1.5}, {"t": "line", "p": [99, 201, 99, 299], "w": 1.5}, {"t": "line", "p": [99, 299, 1, 299], "w": 1.5}, {"t": "line", "p": [1, 299, 1, 201], "w": 1.5}, {"t": "circle", "c": [33, 250], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [50, 250], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [67, 250], "r": 5.5, "fill": "solid"}, {"t": "line", "p": [101, 201, 199, 201], "w": 1.5}, {"t": "line", "p": [199, 201, 199, 299], "w": 1.5}, {"t": "line", "p": [199, 299, 101, 299], "w": 1.5}, {"t": "line", "p": [101, 299, 101, 201], "w": 1.5}, {"t": "circle", "c": [141.5, 241.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [158.5, 241.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [141.5, 258.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [158.5, 258.5], "r": 5.5, "fill": "solid"}, {"t": "line", "p": [201, 201, 299, 201], "w": 1.5}, {"t": "line", "p": [299, 201, 299, 299], "w": 1.5}, {"t": "line", "p": [299, 299, 201, 299], "w": 1.5}, {"t": "line", "p": [201, 299, 201, 201], "w": 1.5}, {"t": "text", "at": [250, 252], "s": "?", "size": 44}], "alt": "A 3 by 3 grid of figures with the bottom right box empty"}
    optionFigures:
      - {"kind": "shapes", "frame": true, "items": [{"t": "circle", "c": [41.5, 41.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [58.5, 41.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [41.5, 58.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [58.5, 58.5], "r": 5.5, "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "circle", "c": [33, 41.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [50, 41.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [67, 41.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [41.5, 58.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [58.5, 58.5], "r": 5.5, "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "circle", "c": [33, 41.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [50, 41.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [67, 41.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [33, 58.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [50, 58.5], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [67, 58.5], "r": 5.5, "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "circle", "c": [33, 33], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [50, 33], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [67, 33], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [33, 50], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [50, 50], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [67, 50], "r": 5.5, "fill": "solid"}, {"t": "circle", "c": [50, 67], "r": 5.5, "fill": "solid"}]}
    answer: B
    explanation: |
      Along each row the number of dots goes up by 1: 1, 2, 3, then 2, 3, 4. The bottom row has 3 and 4, so the empty box has **5 dots**: **B**.
      Going down each column also adds 1 dot, which gives 5 again.

  - id: fp3-complete-mirror
    topic: "Pattern completion"
    difficulty: 1
    q: "Which piece completes the pattern?"
    figure: {"kind": "shapes", "w": 200, "h": 200, "items": [{"t": "line", "p": [1, 1, 199, 1], "w": 1.5}, {"t": "line", "p": [199, 1, 199, 199], "w": 1.5}, {"t": "line", "p": [199, 199, 1, 199], "w": 1.5}, {"t": "line", "p": [1, 199, 1, 1], "w": 1.5}, {"t": "poly", "p": [[100, 100], [100, 52], [52, 100]], "fill": "grey"}, {"t": "circle", "c": [30, 30], "r": 11, "fill": "solid"}, {"t": "line", "p": [14, 74, 44, 44]}, {"t": "poly", "p": [[58, 12], [88, 12], [88, 34]], "fill": "none"}, {"t": "poly", "p": [[100, 100], [100, 52], [148, 100]], "fill": "grey"}, {"t": "circle", "c": [170, 30], "r": 11, "fill": "solid"}, {"t": "line", "p": [186, 74, 156, 44]}, {"t": "poly", "p": [[142, 12], [112, 12], [112, 34]], "fill": "none"}, {"t": "poly", "p": [[100, 100], [100, 148], [52, 100]], "fill": "grey"}, {"t": "circle", "c": [30, 170], "r": 11, "fill": "solid"}, {"t": "line", "p": [14, 126, 44, 156]}, {"t": "poly", "p": [[58, 188], [88, 188], [88, 166]], "fill": "none"}, {"t": "line", "p": [104, 104, 196, 104], "dash": true}, {"t": "line", "p": [196, 104, 196, 196], "dash": true}, {"t": "line", "p": [196, 196, 104, 196], "dash": true}, {"t": "line", "p": [104, 196, 104, 104], "dash": true}, {"t": "text", "at": [150, 152], "s": "?", "size": 44}], "alt": "A square pattern made of four pieces; the bottom right piece is missing"}
    optionFigures:
      - {"kind": "shapes", "items": [{"t": "line", "p": [0, 0, 100, 0], "w": 1.5}, {"t": "line", "p": [100, 0, 100, 100], "w": 1.5}, {"t": "line", "p": [100, 100, 0, 100], "w": 1.5}, {"t": "line", "p": [0, 100, 0, 0], "w": 1.5}, {"t": "poly", "p": [[0, 100], [0, 52], [48, 100]], "fill": "grey"}, {"t": "circle", "c": [70, 30], "r": 11, "fill": "solid"}, {"t": "line", "p": [86, 74, 56, 44]}, {"t": "poly", "p": [[42, 12], [12, 12], [12, 34]], "fill": "none"}]}
      - {"kind": "shapes", "items": [{"t": "line", "p": [0, 0, 100, 0], "w": 1.5}, {"t": "line", "p": [100, 0, 100, 100], "w": 1.5}, {"t": "line", "p": [100, 100, 0, 100], "w": 1.5}, {"t": "line", "p": [0, 100, 0, 0], "w": 1.5}, {"t": "poly", "p": [[100, 0], [100, 48], [52, 0]], "fill": "grey"}, {"t": "circle", "c": [30, 70], "r": 11, "fill": "solid"}, {"t": "line", "p": [14, 26, 44, 56]}, {"t": "poly", "p": [[58, 88], [88, 88], [88, 66]], "fill": "none"}]}
      - {"kind": "shapes", "items": [{"t": "line", "p": [0, 0, 100, 0], "w": 1.5}, {"t": "line", "p": [100, 0, 100, 100], "w": 1.5}, {"t": "line", "p": [100, 100, 0, 100], "w": 1.5}, {"t": "line", "p": [0, 100, 0, 0], "w": 1.5}, {"t": "poly", "p": [[0, 0], [0, 48], [48, 0]], "fill": "grey"}, {"t": "circle", "c": [70, 70], "r": 11, "fill": "solid"}, {"t": "line", "p": [86, 26, 56, 56]}, {"t": "poly", "p": [[42, 88], [12, 88], [12, 66]], "fill": "none"}]}
      - {"kind": "shapes", "items": [{"t": "line", "p": [0, 0, 100, 0], "w": 1.5}, {"t": "line", "p": [100, 0, 100, 100], "w": 1.5}, {"t": "line", "p": [100, 100, 0, 100], "w": 1.5}, {"t": "line", "p": [0, 100, 0, 0], "w": 1.5}, {"t": "poly", "p": [[100, 100], [100, 52], [52, 100]], "fill": "grey"}, {"t": "circle", "c": [30, 30], "r": 11, "fill": "solid"}, {"t": "line", "p": [14, 74, 44, 44]}, {"t": "poly", "p": [[58, 12], [88, 12], [88, 34]], "fill": "none"}]}
    answer: C
    explanation: |
      The pattern is the same on the left and the right (a mirror line down the middle) and the same at the top and the bottom (a mirror line across the middle).
      So the missing bottom right piece is the bottom left piece mirrored: the grey triangle in its top left corner, the black dot near its bottom right corner: **C**.
      A is the top right piece, which is mirrored the wrong way.

  - id: fp3-hidden-triangle
    topic: "Hidden figures"
    difficulty: 1
    q: "Which shape is hidden in this figure, the same size and the same way up (not turned)?"
    figure: {"kind": "shapes", "w": 150, "h": 150, "items": [{"t": "line", "p": [15, 15, 135, 15]}, {"t": "line", "p": [135, 15, 135, 135]}, {"t": "line", "p": [135, 135, 15, 135]}, {"t": "line", "p": [15, 135, 15, 15]}, {"t": "line", "p": [15, 135, 135, 15]}, {"t": "line", "p": [15, 75, 135, 75]}], "alt": "A figure made of straight lines"}
    optionFigures:
      - {"kind": "shapes", "w": 100, "h": 100, "items": [{"t": "poly", "p": [[20, 20], [80, 20], [80, 80]], "fill": "none"}]}
      - {"kind": "shapes", "w": 100, "h": 100, "items": [{"t": "poly", "p": [[20, 20], [80, 20], [80, 80], [20, 80]], "fill": "none"}]}
      - {"kind": "shapes", "w": 100, "h": 100, "items": [{"t": "poly", "p": [[8, 8], [92, 8], [8, 92]], "fill": "none"}]}
      - {"kind": "shapes", "w": 100, "h": 100, "items": [{"t": "poly", "p": [[20, 20], [80, 20], [20, 80]], "fill": "none"}]}
    answer: D
    explanation: |
      Look below the middle line on the left. The left side, the middle line and the slanting line make a small triangle with its square corner at the **top left**: **D**.
      A has its square corner at the top right; that triangle is only in the figure if you turn it. B needs an upright line in the middle, which is not drawn, and C is too big.

  - id: fp3-matrix-shading
    topic: "Matrices"
    difficulty: 2
    q: "Which figure goes in the empty box?"
    figure: {"kind": "shapes", "w": 300, "h": 300, "items": [{"t": "line", "p": [1, 1, 99, 1], "w": 1.5}, {"t": "line", "p": [99, 1, 99, 99], "w": 1.5}, {"t": "line", "p": [99, 99, 1, 99], "w": 1.5}, {"t": "line", "p": [1, 99, 1, 1], "w": 1.5}, {"t": "circle", "c": [50, 50], "r": 22, "fill": "solid"}, {"t": "line", "p": [101, 1, 199, 1], "w": 1.5}, {"t": "line", "p": [199, 1, 199, 99], "w": 1.5}, {"t": "line", "p": [199, 99, 101, 99], "w": 1.5}, {"t": "line", "p": [101, 99, 101, 1], "w": 1.5}, {"t": "poly", "p": [[150, 26], [175.11, 69.5], [124.89, 69.5]], "fill": "grey"}, {"t": "line", "p": [201, 1, 299, 1], "w": 1.5}, {"t": "line", "p": [299, 1, 299, 99], "w": 1.5}, {"t": "line", "p": [299, 99, 201, 99], "w": 1.5}, {"t": "line", "p": [201, 99, 201, 1], "w": 1.5}, {"t": "poly", "p": [[228, 28], [272, 28], [272, 72], [228, 72]], "fill": "none"}, {"t": "line", "p": [1, 101, 99, 101], "w": 1.5}, {"t": "line", "p": [99, 101, 99, 199], "w": 1.5}, {"t": "line", "p": [99, 199, 1, 199], "w": 1.5}, {"t": "line", "p": [1, 199, 1, 101], "w": 1.5}, {"t": "poly", "p": [[50, 126], [75.11, 169.5], [24.89, 169.5]], "fill": "none"}, {"t": "line", "p": [101, 101, 199, 101], "w": 1.5}, {"t": "line", "p": [199, 101, 199, 199], "w": 1.5}, {"t": "line", "p": [199, 199, 101, 199], "w": 1.5}, {"t": "line", "p": [101, 199, 101, 101], "w": 1.5}, {"t": "poly", "p": [[128, 128], [172, 128], [172, 172], [128, 172]], "fill": "solid"}, {"t": "line", "p": [201, 101, 299, 101], "w": 1.5}, {"t": "line", "p": [299, 101, 299, 199], "w": 1.5}, {"t": "line", "p": [299, 199, 201, 199], "w": 1.5}, {"t": "line", "p": [201, 199, 201, 101], "w": 1.5}, {"t": "circle", "c": [250, 150], "r": 22, "fill": "grey"}, {"t": "line", "p": [1, 201, 99, 201], "w": 1.5}, {"t": "line", "p": [99, 201, 99, 299], "w": 1.5}, {"t": "line", "p": [99, 299, 1, 299], "w": 1.5}, {"t": "line", "p": [1, 299, 1, 201], "w": 1.5}, {"t": "poly", "p": [[28, 228], [72, 228], [72, 272], [28, 272]], "fill": "grey"}, {"t": "line", "p": [101, 201, 199, 201], "w": 1.5}, {"t": "line", "p": [199, 201, 199, 299], "w": 1.5}, {"t": "line", "p": [199, 299, 101, 299], "w": 1.5}, {"t": "line", "p": [101, 299, 101, 201], "w": 1.5}, {"t": "circle", "c": [150, 250], "r": 22, "fill": "none"}, {"t": "line", "p": [201, 201, 299, 201], "w": 1.5}, {"t": "line", "p": [299, 201, 299, 299], "w": 1.5}, {"t": "line", "p": [299, 299, 201, 299], "w": 1.5}, {"t": "line", "p": [201, 299, 201, 201], "w": 1.5}, {"t": "text", "at": [250, 252], "s": "?", "size": 44}], "alt": "A 3 by 3 grid of figures with the bottom right box empty"}
    optionFigures:
      - {"kind": "shapes", "frame": true, "items": [{"t": "poly", "p": [[50, 26], [75.11, 69.5], [24.89, 69.5]], "fill": "grey"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "poly", "p": [[50, 26], [75.11, 69.5], [24.89, 69.5]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "poly", "p": [[28, 28], [72, 28], [72, 72], [28, 72]], "fill": "solid"}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "poly", "p": [[50, 26], [75.11, 69.5], [24.89, 69.5]], "fill": "none"}]}
    answer: B
    explanation: |
      Each row has one circle, one triangle and one square, and also one black, one grey and one white figure.
      The bottom row has a grey square and a white circle, so the missing figure is a **black triangle**: **B**.
      A and D have the right shape but the wrong shading, and C has the right shading but the wrong shape.

  - id: fp3-matrix-overlay
    topic: "Matrices"
    difficulty: 2
    q: "Which figure goes in the empty box?"
    figure: {"kind": "shapes", "w": 300, "h": 300, "items": [{"t": "line", "p": [1, 1, 99, 1], "w": 1.5}, {"t": "line", "p": [99, 1, 99, 99], "w": 1.5}, {"t": "line", "p": [99, 99, 1, 99], "w": 1.5}, {"t": "line", "p": [1, 99, 1, 1], "w": 1.5}, {"t": "line", "p": [50, 25, 50, 75]}, {"t": "line", "p": [101, 1, 199, 1], "w": 1.5}, {"t": "line", "p": [199, 1, 199, 99], "w": 1.5}, {"t": "line", "p": [199, 99, 101, 99], "w": 1.5}, {"t": "line", "p": [101, 99, 101, 1], "w": 1.5}, {"t": "line", "p": [125, 50, 175, 50]}, {"t": "line", "p": [201, 1, 299, 1], "w": 1.5}, {"t": "line", "p": [299, 1, 299, 99], "w": 1.5}, {"t": "line", "p": [299, 99, 201, 99], "w": 1.5}, {"t": "line", "p": [201, 99, 201, 1], "w": 1.5}, {"t": "line", "p": [225, 50, 275, 50]}, {"t": "line", "p": [250, 25, 250, 75]}, {"t": "line", "p": [1, 101, 99, 101], "w": 1.5}, {"t": "line", "p": [99, 101, 99, 199], "w": 1.5}, {"t": "line", "p": [99, 199, 1, 199], "w": 1.5}, {"t": "line", "p": [1, 199, 1, 101], "w": 1.5}, {"t": "line", "p": [25, 175, 75, 175]}, {"t": "line", "p": [25, 125, 75, 125]}, {"t": "line", "p": [101, 101, 199, 101], "w": 1.5}, {"t": "line", "p": [199, 101, 199, 199], "w": 1.5}, {"t": "line", "p": [199, 199, 101, 199], "w": 1.5}, {"t": "line", "p": [101, 199, 101, 101], "w": 1.5}, {"t": "line", "p": [125, 125, 175, 175]}, {"t": "line", "p": [201, 101, 299, 101], "w": 1.5}, {"t": "line", "p": [299, 101, 299, 199], "w": 1.5}, {"t": "line", "p": [299, 199, 201, 199], "w": 1.5}, {"t": "line", "p": [201, 199, 201, 101], "w": 1.5}, {"t": "line", "p": [225, 175, 275, 175]}, {"t": "line", "p": [225, 125, 275, 175]}, {"t": "line", "p": [225, 125, 275, 125]}, {"t": "line", "p": [1, 201, 99, 201], "w": 1.5}, {"t": "line", "p": [99, 201, 99, 299], "w": 1.5}, {"t": "line", "p": [99, 299, 1, 299], "w": 1.5}, {"t": "line", "p": [1, 299, 1, 201], "w": 1.5}, {"t": "line", "p": [25, 225, 25, 275]}, {"t": "line", "p": [75, 225, 75, 275]}, {"t": "line", "p": [101, 201, 199, 201], "w": 1.5}, {"t": "line", "p": [199, 201, 199, 299], "w": 1.5}, {"t": "line", "p": [199, 299, 101, 299], "w": 1.5}, {"t": "line", "p": [101, 299, 101, 201], "w": 1.5}, {"t": "line", "p": [175, 225, 125, 275]}, {"t": "line", "p": [125, 250, 175, 250]}, {"t": "line", "p": [201, 201, 299, 201], "w": 1.5}, {"t": "line", "p": [299, 201, 299, 299], "w": 1.5}, {"t": "line", "p": [299, 299, 201, 299], "w": 1.5}, {"t": "line", "p": [201, 299, 201, 201], "w": 1.5}, {"t": "text", "at": [250, 252], "s": "?", "size": 44}], "alt": "A 3 by 3 grid of figures with the bottom right box empty"}
    optionFigures:
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [25, 25, 75, 75]}, {"t": "line", "p": [25, 50, 75, 50]}, {"t": "line", "p": [25, 25, 25, 75]}, {"t": "line", "p": [75, 25, 75, 75]}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [25, 50, 75, 50]}, {"t": "line", "p": [25, 25, 25, 75]}, {"t": "line", "p": [75, 25, 75, 75]}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [75, 25, 25, 75]}, {"t": "line", "p": [25, 50, 75, 50]}, {"t": "line", "p": [25, 25, 25, 75]}, {"t": "line", "p": [75, 25, 75, 75]}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [75, 25, 25, 75]}, {"t": "line", "p": [25, 50, 75, 50]}]}
    answer: C
    explanation: |
      In each row, the third figure is the first two **drawn on top of each other**: an upright line and a line across make a plus; two lines across and a slanting line make a Z.
      So the two upright sides and the figure with a slanting line and a line across together give **C**.
      A slants the wrong way, B forgets the slanting line, and D forgets the upright sides.

  - id: fp3-complete-turn
    topic: "Pattern completion"
    difficulty: 2
    q: "Which piece completes the pattern?"
    figure: {"kind": "shapes", "w": 200, "h": 200, "items": [{"t": "line", "p": [1, 1, 199, 1], "w": 1.5}, {"t": "line", "p": [199, 1, 199, 199], "w": 1.5}, {"t": "line", "p": [199, 199, 1, 199], "w": 1.5}, {"t": "line", "p": [1, 199, 1, 1], "w": 1.5}, {"t": "poly", "p": [[100, 100], [54, 100], [100, 68]], "fill": "grey"}, {"t": "line", "p": [18, 20, 60, 20]}, {"t": "line", "p": [60, 20, 60, 46]}, {"t": "circle", "c": [28, 72], "r": 9, "fill": "solid"}, {"t": "poly", "p": [[100, 100], [146, 100], [100, 132]], "fill": "grey"}, {"t": "line", "p": [182, 180, 140, 180]}, {"t": "line", "p": [140, 180, 140, 154]}, {"t": "circle", "c": [172, 128], "r": 9, "fill": "solid"}, {"t": "poly", "p": [[100, 100], [100, 146], [68, 100]], "fill": "grey"}, {"t": "line", "p": [20, 182, 20, 140]}, {"t": "line", "p": [20, 140, 46, 140]}, {"t": "circle", "c": [72, 172], "r": 9, "fill": "solid"}, {"t": "line", "p": [104, 4, 196, 4], "dash": true}, {"t": "line", "p": [196, 4, 196, 96], "dash": true}, {"t": "line", "p": [196, 96, 104, 96], "dash": true}, {"t": "line", "p": [104, 96, 104, 4], "dash": true}, {"t": "text", "at": [150, 52], "s": "?", "size": 44}], "alt": "A square pattern made of four pieces; the top right piece is missing"}
    optionFigures:
      - {"kind": "shapes", "items": [{"t": "line", "p": [0, 0, 100, 0], "w": 1.5}, {"t": "line", "p": [100, 0, 100, 100], "w": 1.5}, {"t": "line", "p": [100, 100, 0, 100], "w": 1.5}, {"t": "line", "p": [0, 100, 0, 0], "w": 1.5}, {"t": "poly", "p": [[0, 100], [0, 54], [32, 100]], "fill": "grey"}, {"t": "line", "p": [80, 18, 80, 60]}, {"t": "line", "p": [80, 60, 54, 60]}, {"t": "circle", "c": [28, 28], "r": 9, "fill": "solid"}]}
      - {"kind": "shapes", "items": [{"t": "line", "p": [0, 0, 100, 0], "w": 1.5}, {"t": "line", "p": [100, 0, 100, 100], "w": 1.5}, {"t": "line", "p": [100, 100, 0, 100], "w": 1.5}, {"t": "line", "p": [0, 100, 0, 0], "w": 1.5}, {"t": "poly", "p": [[0, 100], [46, 100], [0, 68]], "fill": "grey"}, {"t": "line", "p": [82, 20, 40, 20]}, {"t": "line", "p": [40, 20, 40, 46]}, {"t": "circle", "c": [72, 72], "r": 9, "fill": "solid"}]}
      - {"kind": "shapes", "items": [{"t": "line", "p": [0, 0, 100, 0], "w": 1.5}, {"t": "line", "p": [100, 0, 100, 100], "w": 1.5}, {"t": "line", "p": [100, 100, 0, 100], "w": 1.5}, {"t": "line", "p": [0, 100, 0, 0], "w": 1.5}, {"t": "poly", "p": [[100, 0], [100, 46], [68, 0]], "fill": "grey"}, {"t": "line", "p": [20, 82, 20, 40]}, {"t": "line", "p": [20, 40, 46, 40]}, {"t": "circle", "c": [72, 72], "r": 9, "fill": "solid"}]}
      - {"kind": "shapes", "items": [{"t": "line", "p": [0, 0, 100, 0], "w": 1.5}, {"t": "line", "p": [100, 0, 100, 100], "w": 1.5}, {"t": "line", "p": [100, 100, 0, 100], "w": 1.5}, {"t": "line", "p": [0, 100, 0, 0], "w": 1.5}, {"t": "poly", "p": [[0, 0], [46, 0], [0, 32]], "fill": "grey"}, {"t": "line", "p": [82, 80, 40, 80]}, {"t": "line", "p": [40, 80, 40, 54]}, {"t": "circle", "c": [72, 28], "r": 9, "fill": "solid"}]}
    answer: A
    explanation: |
      This pattern is not a mirror pattern. Going clockwise, each piece is the one before it **turned a quarter turn clockwise**: compare the top left piece with the bottom right one (a half turn) and the bottom left one (three quarter turns).
      So the top right piece is the top left piece turned a quarter turn clockwise: the grey triangle goes to its bottom left corner and the black dot to its top left: **A**.
      B is the mirror image of the top left piece, the most tempting wrong answer. C and D are turned the wrong amount.

  - id: fp3-hidden-kite
    topic: "Hidden figures"
    difficulty: 2
    q: "Which shape is hidden in this figure, the same size and the same way up (not turned)?"
    figure: {"kind": "shapes", "w": 120, "h": 120, "items": [{"t": "line", "p": [10, 110, 110, 110]}, {"t": "line", "p": [110, 110, 60, 10]}, {"t": "line", "p": [60, 10, 10, 110]}, {"t": "line", "p": [35, 60, 85, 60]}, {"t": "line", "p": [60, 10, 60, 110]}, {"t": "line", "p": [35, 60, 60, 110]}, {"t": "line", "p": [85, 60, 60, 110]}], "alt": "A figure made of straight lines"}
    optionFigures:
      - {"kind": "shapes", "w": 112, "h": 112, "items": [{"t": "poly", "p": [[31, 31], [56, 31], [81, 81], [56, 81]], "fill": "none"}]}
      - {"kind": "shapes", "w": 112, "h": 112, "items": [{"t": "poly", "p": [[56, 6], [91, 56], [56, 106], [21, 56]], "fill": "none"}]}
      - {"kind": "shapes", "w": 112, "h": 112, "items": [{"t": "poly", "p": [[56, 21], [81, 56], [56, 91], [31, 56]], "fill": "none"}]}
      - {"kind": "shapes", "w": 112, "h": 112, "items": [{"t": "poly", "p": [[56, 6], [81, 56], [56, 106], [31, 56]], "fill": "none"}]}
    answer: D
    explanation: |
      Start at the top corner of the big triangle. Go down both sides to the line across the middle, then follow the two slanting lines down to the middle of the base. That outline is the tall kite: **D**.
      The other shapes need lines that are not drawn: A slants the wrong way at the bottom, B is too wide for the line across, and C has its top and bottom corners where no lines meet.

  - id: fp3-matrix-arrows
    topic: "Matrices"
    difficulty: 2
    q: "Which figure goes in the empty box?"
    figure: {"kind": "shapes", "w": 300, "h": 300, "items": [{"t": "line", "p": [1, 1, 99, 1], "w": 1.5}, {"t": "line", "p": [99, 1, 99, 99], "w": 1.5}, {"t": "line", "p": [99, 99, 1, 99], "w": 1.5}, {"t": "line", "p": [1, 99, 1, 1], "w": 1.5}, {"t": "arrow", "p": [50, 78, 50, 22]}, {"t": "line", "p": [101, 1, 199, 1], "w": 1.5}, {"t": "line", "p": [199, 1, 199, 99], "w": 1.5}, {"t": "line", "p": [199, 99, 101, 99], "w": 1.5}, {"t": "line", "p": [101, 99, 101, 1], "w": 1.5}, {"t": "arrow", "p": [130.2, 69.8, 169.8, 30.2]}, {"t": "line", "p": [201, 1, 299, 1], "w": 1.5}, {"t": "line", "p": [299, 1, 299, 99], "w": 1.5}, {"t": "line", "p": [299, 99, 201, 99], "w": 1.5}, {"t": "line", "p": [201, 99, 201, 1], "w": 1.5}, {"t": "arrow", "p": [222, 50, 278, 50]}, {"t": "line", "p": [1, 101, 99, 101], "w": 1.5}, {"t": "line", "p": [99, 101, 99, 199], "w": 1.5}, {"t": "line", "p": [99, 199, 1, 199], "w": 1.5}, {"t": "line", "p": [1, 199, 1, 101], "w": 1.5}, {"t": "arrow", "p": [22, 150, 78, 150]}, {"t": "line", "p": [101, 101, 199, 101], "w": 1.5}, {"t": "line", "p": [199, 101, 199, 199], "w": 1.5}, {"t": "line", "p": [199, 199, 101, 199], "w": 1.5}, {"t": "line", "p": [101, 199, 101, 101], "w": 1.5}, {"t": "arrow", "p": [130.2, 130.2, 169.8, 169.8]}, {"t": "line", "p": [201, 101, 299, 101], "w": 1.5}, {"t": "line", "p": [299, 101, 299, 199], "w": 1.5}, {"t": "line", "p": [299, 199, 201, 199], "w": 1.5}, {"t": "line", "p": [201, 199, 201, 101], "w": 1.5}, {"t": "arrow", "p": [250, 122, 250, 178]}, {"t": "line", "p": [1, 201, 99, 201], "w": 1.5}, {"t": "line", "p": [99, 201, 99, 299], "w": 1.5}, {"t": "line", "p": [99, 299, 1, 299], "w": 1.5}, {"t": "line", "p": [1, 299, 1, 201], "w": 1.5}, {"t": "arrow", "p": [50, 222, 50, 278]}, {"t": "line", "p": [101, 201, 199, 201], "w": 1.5}, {"t": "line", "p": [199, 201, 199, 299], "w": 1.5}, {"t": "line", "p": [199, 299, 101, 299], "w": 1.5}, {"t": "line", "p": [101, 299, 101, 201], "w": 1.5}, {"t": "arrow", "p": [169.8, 230.2, 130.2, 269.8]}, {"t": "line", "p": [201, 201, 299, 201], "w": 1.5}, {"t": "line", "p": [299, 201, 299, 299], "w": 1.5}, {"t": "line", "p": [299, 299, 201, 299], "w": 1.5}, {"t": "line", "p": [201, 299, 201, 201], "w": 1.5}, {"t": "text", "at": [250, 252], "s": "?", "size": 44}], "alt": "A 3 by 3 grid of figures with the bottom right box empty"}
    optionFigures:
      - {"kind": "shapes", "frame": true, "items": [{"t": "arrow", "p": [69.8, 30.2, 30.2, 69.8]}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "arrow", "p": [69.8, 69.8, 30.2, 30.2]}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "arrow", "p": [78, 50, 22, 50]}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "arrow", "p": [22, 50, 78, 50]}]}
    answer: C
    explanation: |
      Along each row the arrow turns **45° clockwise** from one box to the next. Each row also starts a quarter turn further on than the row above.
      The bottom row starts pointing down, then down-left, so the next arrow points **left**: **C**.
      A copies the box before it, B turns too far, and D points the opposite way.

  - id: fp3-matrix-cancel
    topic: "Matrices"
    difficulty: 3
    q: "Which figure goes in the empty box?"
    figure: {"kind": "shapes", "w": 300, "h": 300, "items": [{"t": "line", "p": [1, 1, 99, 1], "w": 1.5}, {"t": "line", "p": [99, 1, 99, 99], "w": 1.5}, {"t": "line", "p": [99, 99, 1, 99], "w": 1.5}, {"t": "line", "p": [1, 99, 1, 1], "w": 1.5}, {"t": "line", "p": [25, 25, 75, 75]}, {"t": "line", "p": [25, 25, 75, 25]}, {"t": "line", "p": [50, 25, 50, 75]}, {"t": "line", "p": [101, 1, 199, 1], "w": 1.5}, {"t": "line", "p": [199, 1, 199, 99], "w": 1.5}, {"t": "line", "p": [199, 99, 101, 99], "w": 1.5}, {"t": "line", "p": [101, 99, 101, 1], "w": 1.5}, {"t": "line", "p": [125, 25, 175, 75]}, {"t": "line", "p": [125, 50, 175, 50]}, {"t": "line", "p": [125, 25, 175, 25]}, {"t": "line", "p": [201, 1, 299, 1], "w": 1.5}, {"t": "line", "p": [299, 1, 299, 99], "w": 1.5}, {"t": "line", "p": [299, 99, 201, 99], "w": 1.5}, {"t": "line", "p": [201, 99, 201, 1], "w": 1.5}, {"t": "line", "p": [225, 50, 275, 50]}, {"t": "line", "p": [250, 25, 250, 75]}, {"t": "line", "p": [1, 101, 99, 101], "w": 1.5}, {"t": "line", "p": [99, 101, 99, 199], "w": 1.5}, {"t": "line", "p": [99, 199, 1, 199], "w": 1.5}, {"t": "line", "p": [1, 199, 1, 101], "w": 1.5}, {"t": "line", "p": [75, 125, 25, 175]}, {"t": "line", "p": [25, 125, 25, 175]}, {"t": "line", "p": [75, 125, 75, 175]}, {"t": "line", "p": [101, 101, 199, 101], "w": 1.5}, {"t": "line", "p": [199, 101, 199, 199], "w": 1.5}, {"t": "line", "p": [199, 199, 101, 199], "w": 1.5}, {"t": "line", "p": [101, 199, 101, 101], "w": 1.5}, {"t": "line", "p": [125, 175, 175, 175]}, {"t": "line", "p": [175, 125, 125, 175]}, {"t": "line", "p": [175, 125, 175, 175]}, {"t": "line", "p": [201, 101, 299, 101], "w": 1.5}, {"t": "line", "p": [299, 101, 299, 199], "w": 1.5}, {"t": "line", "p": [299, 199, 201, 199], "w": 1.5}, {"t": "line", "p": [201, 199, 201, 101], "w": 1.5}, {"t": "line", "p": [225, 175, 275, 175]}, {"t": "line", "p": [225, 125, 225, 175]}, {"t": "line", "p": [1, 201, 99, 201], "w": 1.5}, {"t": "line", "p": [99, 201, 99, 299], "w": 1.5}, {"t": "line", "p": [99, 299, 1, 299], "w": 1.5}, {"t": "line", "p": [1, 299, 1, 201], "w": 1.5}, {"t": "line", "p": [25, 275, 75, 275]}, {"t": "line", "p": [25, 225, 75, 275]}, {"t": "line", "p": [25, 225, 75, 225]}, {"t": "line", "p": [50, 225, 50, 275]}, {"t": "line", "p": [101, 201, 199, 201], "w": 1.5}, {"t": "line", "p": [199, 201, 199, 299], "w": 1.5}, {"t": "line", "p": [199, 299, 101, 299], "w": 1.5}, {"t": "line", "p": [101, 299, 101, 201], "w": 1.5}, {"t": "line", "p": [125, 275, 175, 275]}, {"t": "line", "p": [175, 225, 125, 275]}, {"t": "line", "p": [150, 225, 150, 275]}, {"t": "line", "p": [201, 201, 299, 201], "w": 1.5}, {"t": "line", "p": [299, 201, 299, 299], "w": 1.5}, {"t": "line", "p": [299, 299, 201, 299], "w": 1.5}, {"t": "line", "p": [201, 299, 201, 201], "w": 1.5}, {"t": "text", "at": [250, 252], "s": "?", "size": 44}], "alt": "A 3 by 3 grid of figures with the bottom right box empty"}
    optionFigures:
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [25, 25, 75, 75]}, {"t": "line", "p": [75, 25, 25, 75]}, {"t": "line", "p": [25, 25, 75, 25]}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [25, 75, 75, 75]}, {"t": "line", "p": [25, 25, 75, 75]}, {"t": "line", "p": [75, 25, 25, 75]}, {"t": "line", "p": [25, 25, 75, 25]}, {"t": "line", "p": [50, 25, 50, 75]}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [25, 75, 75, 75]}, {"t": "line", "p": [50, 25, 50, 75]}]}
      - {"kind": "shapes", "frame": true, "items": [{"t": "line", "p": [25, 75, 75, 75]}, {"t": "line", "p": [25, 25, 75, 75]}, {"t": "line", "p": [75, 25, 25, 75]}, {"t": "line", "p": [25, 25, 75, 25]}]}
    answer: A
    explanation: |
      Put the first two figures of a row on top of each other. Lines that are in **both** figures disappear, and lines that are in only one of them stay. In the top row, the top side and the slanting line are in both, so only the plus is left.
      In the bottom row, the bottom side and the upright line are in both, so they go. What stays is the top side and the two slanting lines: **A**.
      B keeps every line (that rule does not fit the top row), and C keeps only the shared lines.

  - id: fp3-hidden-house
    topic: "Hidden figures"
    difficulty: 3
    q: "Which shape is hidden in this figure, the same size and the same way up (not turned)? Only lines that are drawn can be used."
    figure: {"kind": "shapes", "w": 125, "h": 125, "items": [{"t": "line", "p": [12.5, 12.5, 112.5, 12.5]}, {"t": "line", "p": [112.5, 12.5, 112.5, 112.5]}, {"t": "line", "p": [112.5, 112.5, 12.5, 112.5]}, {"t": "line", "p": [12.5, 112.5, 12.5, 12.5]}, {"t": "line", "p": [12.5, 12.5, 112.5, 112.5]}, {"t": "line", "p": [112.5, 12.5, 12.5, 112.5]}, {"t": "line", "p": [62.5, 12.5, 112.5, 62.5]}, {"t": "line", "p": [112.5, 62.5, 62.5, 112.5]}, {"t": "line", "p": [62.5, 112.5, 12.5, 62.5]}, {"t": "line", "p": [12.5, 62.5, 62.5, 12.5]}, {"t": "line", "p": [62.5, 12.5, 62.5, 112.5]}], "alt": "A figure made of straight lines"}
    optionFigures:
      - {"kind": "shapes", "w": 108, "h": 108, "items": [{"t": "poly", "p": [[29, 41.5], [79, 41.5], [104, 66.5], [4, 66.5]], "fill": "none"}]}
      - {"kind": "shapes", "w": 108, "h": 108, "items": [{"t": "poly", "p": [[4, 54], [54, 4], [104, 54], [104, 104], [4, 104]], "fill": "none"}]}
      - {"kind": "shapes", "w": 108, "h": 108, "items": [{"t": "poly", "p": [[4, 54], [29, 29], [79, 29], [104, 54], [79, 79], [29, 79]], "fill": "none"}]}
      - {"kind": "shapes", "w": 108, "h": 108, "items": [{"t": "poly", "p": [[41.5, 4], [91.5, 54], [41.5, 104], [16.5, 79], [16.5, 29]], "fill": "none"}]}
    answer: B
    explanation: |
      The roof is the top half of the tilted square: the two slanting lines from the left middle point up to the top middle and down to the right middle. The walls are the lower halves of the left and right sides, and the floor is the bottom side: **B**.
      A and C need a line across between the points where the slanting lines cross, and no such line is drawn. D needs an upright line on the left between the crossing points, which is also missing.

  - id: fp3-complete-cut-piece
    topic: "Pattern completion"
    difficulty: 3
    q: "A square piece has been cut out of this drawing. Which piece fits the hole exactly?"
    figure: {"kind": "shapes", "w": 200, "h": 200, "alt": "A square drawing made of straight lines with a square piece cut out", "items": [{"t": "line", "p": [0, 0, 200, 0]}, {"t": "line", "p": [200, 0, 200, 200]}, {"t": "line", "p": [0, 200, 200, 200]}, {"t": "line", "p": [0, 0, 0, 200]}, {"t": "line", "p": [0, 0, 80, 80]}, {"t": "line", "p": [140, 140, 200, 200]}, {"t": "line", "p": [0, 140, 80, 60]}, {"t": "line", "p": [100, 40, 140, 0]}, {"t": "line", "p": [130, 0, 130, 40]}, {"t": "line", "p": [130, 140, 130, 200]}, {"t": "line", "p": [0, 200, 100, 140]}, {"t": "line", "p": [180, 92, 200, 80]}, {"t": "line", "p": [80, 40, 180, 40], "dash": true}, {"t": "line", "p": [180, 40, 180, 140], "dash": true}, {"t": "line", "p": [180, 140, 80, 140], "dash": true}, {"t": "line", "p": [80, 140, 80, 40], "dash": true}, {"t": "text", "at": [130, 92], "s": "?", "size": 44}]}
    optionFigures:
      - {"kind": "shapes", "items": [{"t": "line", "p": [0, 0, 100, 0], "w": 1.5}, {"t": "line", "p": [100, 0, 100, 100], "w": 1.5}, {"t": "line", "p": [100, 100, 0, 100], "w": 1.5}, {"t": "line", "p": [0, 100, 0, 0], "w": 1.5}, {"t": "line", "p": [100, 40, 40, 100]}, {"t": "line", "p": [100, 20, 80, 0]}, {"t": "line", "p": [50, 0, 50, 100]}, {"t": "line", "p": [80, 100, 0, 52]}]}
      - {"kind": "shapes", "items": [{"t": "line", "p": [0, 0, 100, 0], "w": 1.5}, {"t": "line", "p": [100, 0, 100, 100], "w": 1.5}, {"t": "line", "p": [100, 100, 0, 100], "w": 1.5}, {"t": "line", "p": [0, 100, 0, 0], "w": 1.5}, {"t": "line", "p": [20, 0, 100, 80]}, {"t": "line", "p": [0, 80, 80, 0]}, {"t": "line", "p": [80, 100, 100, 88]}]}
      - {"kind": "shapes", "items": [{"t": "line", "p": [0, 0, 100, 0], "w": 1.5}, {"t": "line", "p": [100, 0, 100, 100], "w": 1.5}, {"t": "line", "p": [100, 100, 0, 100], "w": 1.5}, {"t": "line", "p": [0, 100, 0, 0], "w": 1.5}, {"t": "line", "p": [0, 40, 60, 100]}, {"t": "line", "p": [50, 0, 50, 100]}, {"t": "line", "p": [20, 100, 100, 52]}]}
      - {"kind": "shapes", "items": [{"t": "line", "p": [0, 0, 100, 0], "w": 1.5}, {"t": "line", "p": [100, 0, 100, 100], "w": 1.5}, {"t": "line", "p": [100, 100, 0, 100], "w": 1.5}, {"t": "line", "p": [0, 100, 0, 0], "w": 1.5}, {"t": "line", "p": [0, 40, 60, 100]}, {"t": "line", "p": [0, 20, 20, 0]}, {"t": "line", "p": [50, 0, 50, 100]}, {"t": "line", "p": [20, 100, 100, 52]}]}
    answer: D
    explanation: |
      Look at where each line meets the edge of the hole, and follow it across. The long slanting line from the top left corner of the drawing comes in on the left edge of the hole and leaves through the bottom. The upright line crosses from top to bottom through the middle. A short slanting line cuts off the top left corner of the hole, and another slanting line rises from the bottom to the right edge.
      Only **D** has every line entering and leaving at the right places.
      A is the right piece flipped over, B is cut from a different place, and C has lost one of the slanting lines.

---

This set tests three kinds of figure puzzles: 3 by 3 matrices with one box missing, patterns with a missing piece, and shapes hidden inside a bigger figure.

## Matrices

Read each row from left to right and look for what changes. Check your rule on **every** complete row before you use it on the last one.

| Rule | What to look for |
|---|---|
| One of each | Each row has one triangle, one square, one circle (or one black, one grey, one white) |
| Counting | The number of dots or sides goes up by the same amount |
| Turning | A shape turns by the same angle from box to box |
| Adding | The third figure is the first two drawn on top of each other |
| Cancelling | Lines in both of the first two figures disappear; the rest stay |

**Worked example.** A row shows a circle, then a dot, then a circle with a dot in it. The third box is the first two put together. If the next row shows a square and then a line across, the third box is a square with a line across it.

## Pattern completion

Find the symmetry first. Is the pattern a **mirror image** left to right and top to bottom, or does each piece **turn** a quarter turn as you go round? A mirror pattern and a turning pattern need different pieces. When a piece is cut out of a line drawing, follow each line that runs into the hole and see where it must come out.

## Hidden figures

The shape must be found **the same size and the same way up**, and every side must lie on a drawn line. Check the corners: each corner of the shape must be a point where drawn lines meet.

## Common trap

A piece that is the mirror image of the right one looks almost the same. Check one corner at a time.
