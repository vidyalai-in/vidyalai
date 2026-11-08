---
title: "Mirror and water images 2: figures and turns"
description: "Mirror and water images of figures, quarter and half turns, telling a turned figure from a reflected one, and combined moves."
slug: "mirror-water-images-2"
family: "mirror-water-images"
part: 2
timeMinutes: 12
questions:
  - id: mw2-flag-mirror
    topic: "Mirror images"
    difficulty: 1
    q: "A mirror is placed to the right of this flag. Which option is its mirror image?"
    figure: {"kind": "shapes", "items": [{"t": "line", "p": [30, 12, 30, 88], "w": 3}, {"t": "poly", "p": [[30, 12], [78, 25], [30, 38]], "fill": "solid"}, {"t": "line", "p": [16, 88, 50, 88], "w": 3}], "alt": "The given figure"}
    optionFigures:
      - {"kind": "shapes", "transform": "flipV", "items": [{"t": "line", "p": [30, 12, 30, 88], "w": 3}, {"t": "poly", "p": [[30, 12], [78, 25], [30, 38]], "fill": "solid"}, {"t": "line", "p": [16, 88, 50, 88], "w": 3}]}
      - {"kind": "shapes", "transform": "rot90", "items": [{"t": "line", "p": [30, 12, 30, 88], "w": 3}, {"t": "poly", "p": [[30, 12], [78, 25], [30, 38]], "fill": "solid"}, {"t": "line", "p": [16, 88, 50, 88], "w": 3}]}
      - {"kind": "shapes", "transform": "flipH", "items": [{"t": "line", "p": [30, 12, 30, 88], "w": 3}, {"t": "poly", "p": [[30, 12], [78, 25], [30, 38]], "fill": "solid"}, {"t": "line", "p": [16, 88, 50, 88], "w": 3}]}
      - {"kind": "shapes", "transform": "rot180", "items": [{"t": "line", "p": [30, 12, 30, 88], "w": 3}, {"t": "poly", "p": [[30, 12], [78, 25], [30, 38]], "fill": "solid"}, {"t": "line", "p": [16, 88, 50, 88], "w": 3}]}
    answer: C
    explanation: |
      A mirror on the right swaps left and right but keeps top and bottom. The flag stays at the top of the pole but now points to the left, and the pole is on the right: **C**.
      A is the water image (flag at the bottom). B is turned a quarter turn, and D a half turn.

  - id: mw2-l-water
    topic: "Water images"
    difficulty: 1
    q: "Which option is the water image of this figure?"
    figure: {"kind": "shapes", "items": [{"t": "poly", "p": [[20, 12], [42, 12], [42, 62], [84, 62], [84, 86], [20, 86]], "fill": "grey"}, {"t": "circle", "c": [68, 30], "r": 9, "fill": "solid"}], "alt": "The given figure"}
    optionFigures:
      - {"kind": "shapes", "transform": "flipV", "items": [{"t": "poly", "p": [[20, 12], [42, 12], [42, 62], [84, 62], [84, 86], [20, 86]], "fill": "grey"}, {"t": "circle", "c": [68, 30], "r": 9, "fill": "solid"}]}
      - {"kind": "shapes", "transform": "flipH", "items": [{"t": "poly", "p": [[20, 12], [42, 12], [42, 62], [84, 62], [84, 86], [20, 86]], "fill": "grey"}, {"t": "circle", "c": [68, 30], "r": 9, "fill": "solid"}]}
      - {"kind": "shapes", "transform": "rot180", "items": [{"t": "poly", "p": [[20, 12], [42, 12], [42, 62], [84, 62], [84, 86], [20, 86]], "fill": "grey"}, {"t": "circle", "c": [68, 30], "r": 9, "fill": "solid"}]}
      - {"kind": "shapes", "transform": "rot270", "items": [{"t": "poly", "p": [[20, 12], [42, 12], [42, 62], [84, 62], [84, 86], [20, 86]], "fill": "grey"}, {"t": "circle", "c": [68, 30], "r": 9, "fill": "solid"}]}
    answer: A
    explanation: |
      In a water image, top and bottom swap but left and right stay. The tall arm stays on the left, the foot of the L moves from the bottom to the top, and the dot moves from the top right to the bottom right: **A**.
      B is the mirror image. C is a half turn, which also swaps left and right.

  - id: mw2-arrow-quarter-turn
    topic: "Rotations"
    difficulty: 1
    q: "Which option shows this figure turned a quarter turn (90°) clockwise?"
    figure: {"kind": "shapes", "items": [{"t": "arrow", "p": [18, 84, 58, 30]}, {"t": "rect", "p": [62, 62, 24, 24], "fill": "grey"}, {"t": "circle", "c": [26, 24], "r": 8, "fill": "none"}], "alt": "The given figure"}
    optionFigures:
      - {"kind": "shapes", "transform": "rot270", "items": [{"t": "arrow", "p": [18, 84, 58, 30]}, {"t": "rect", "p": [62, 62, 24, 24], "fill": "grey"}, {"t": "circle", "c": [26, 24], "r": 8, "fill": "none"}]}
      - {"kind": "shapes", "transform": "flipH", "items": [{"t": "arrow", "p": [18, 84, 58, 30]}, {"t": "rect", "p": [62, 62, 24, 24], "fill": "grey"}, {"t": "circle", "c": [26, 24], "r": 8, "fill": "none"}]}
      - {"kind": "shapes", "transform": "rot180", "items": [{"t": "arrow", "p": [18, 84, 58, 30]}, {"t": "rect", "p": [62, 62, 24, 24], "fill": "grey"}, {"t": "circle", "c": [26, 24], "r": 8, "fill": "none"}]}
      - {"kind": "shapes", "transform": "rot90", "items": [{"t": "arrow", "p": [18, 84, 58, 30]}, {"t": "rect", "p": [62, 62, 24, 24], "fill": "grey"}, {"t": "circle", "c": [26, 24], "r": 8, "fill": "none"}]}
    answer: D
    explanation: |
      In a quarter turn clockwise, whatever was at the top moves to the right, and what was on the left moves to the top.
      The ring (top left) goes to the top right, the grey square (bottom right) goes to the bottom left, and the arrow, which pointed up and to the right, now points down and to the right: **D**.
      A is turned anticlockwise. B is a mirror image, not a turn.

  - id: mw2-framed-half-turn
    topic: "Rotations"
    difficulty: 1
    q: "Which option shows this card turned upside down by a half turn (180°)?"
    figure: {"kind": "shapes", "frame": true, "items": [{"t": "poly", "p": [[8, 8], [44, 8], [8, 30]], "fill": "solid"}, {"t": "circle", "c": [70, 30], "r": 11, "fill": "none"}, {"t": "rect", "p": [18, 58, 18, 32], "fill": "grey"}, {"t": "line", "p": [56, 88, 90, 62]}], "alt": "The given figure"}
    optionFigures:
      - {"kind": "shapes", "frame": true, "transform": "flipV", "items": [{"t": "poly", "p": [[8, 8], [44, 8], [8, 30]], "fill": "solid"}, {"t": "circle", "c": [70, 30], "r": 11, "fill": "none"}, {"t": "rect", "p": [18, 58, 18, 32], "fill": "grey"}, {"t": "line", "p": [56, 88, 90, 62]}]}
      - {"kind": "shapes", "frame": true, "transform": "rot180", "items": [{"t": "poly", "p": [[8, 8], [44, 8], [8, 30]], "fill": "solid"}, {"t": "circle", "c": [70, 30], "r": 11, "fill": "none"}, {"t": "rect", "p": [18, 58, 18, 32], "fill": "grey"}, {"t": "line", "p": [56, 88, 90, 62]}]}
      - {"kind": "shapes", "frame": true, "transform": "flipH", "items": [{"t": "poly", "p": [[8, 8], [44, 8], [8, 30]], "fill": "solid"}, {"t": "circle", "c": [70, 30], "r": 11, "fill": "none"}, {"t": "rect", "p": [18, 58, 18, 32], "fill": "grey"}, {"t": "line", "p": [56, 88, 90, 62]}]}
      - {"kind": "shapes", "frame": true, "transform": "rot90", "items": [{"t": "poly", "p": [[8, 8], [44, 8], [8, 30]], "fill": "solid"}, {"t": "circle", "c": [70, 30], "r": 11, "fill": "none"}, {"t": "rect", "p": [18, 58, 18, 32], "fill": "grey"}, {"t": "line", "p": [56, 88, 90, 62]}]}
    answer: B
    explanation: |
      A half turn moves every part to the opposite corner: top left goes to bottom right, and so on.
      The black triangle goes from the top left corner to the bottom right corner, and the grey bar from the bottom left to the top right: **B**.
      A is the water image: the triangle went to the bottom but stayed on the left.

  - id: mw2-shaded-mirror
    topic: "Mirror images"
    difficulty: 2
    q: "Which option is the mirror image of this card, with the mirror on its right?"
    figure: {"kind": "shapes", "frame": true, "items": [{"t": "poly", "p": [[8, 92], [52, 92], [8, 48]], "fill": "grey"}, {"t": "arrow", "p": [38, 42, 86, 14]}, {"t": "circle", "c": [76, 72], "r": 7, "fill": "solid"}, {"t": "line", "p": [14, 14, 34, 14]}], "alt": "The given figure"}
    optionFigures:
      - {"kind": "shapes", "frame": true, "transform": "rot180", "items": [{"t": "poly", "p": [[8, 92], [52, 92], [8, 48]], "fill": "grey"}, {"t": "arrow", "p": [38, 42, 86, 14]}, {"t": "circle", "c": [76, 72], "r": 7, "fill": "solid"}, {"t": "line", "p": [14, 14, 34, 14]}]}
      - {"kind": "shapes", "frame": true, "transform": "flipH", "items": [{"t": "poly", "p": [[8, 92], [52, 92], [8, 48]], "fill": "grey"}, {"t": "arrow", "p": [38, 42, 86, 14]}, {"t": "circle", "c": [76, 72], "r": 7, "fill": "solid"}, {"t": "line", "p": [14, 14, 34, 14]}]}
      - {"kind": "shapes", "frame": true, "transform": "flipV", "items": [{"t": "poly", "p": [[8, 92], [52, 92], [8, 48]], "fill": "grey"}, {"t": "arrow", "p": [38, 42, 86, 14]}, {"t": "circle", "c": [76, 72], "r": 7, "fill": "solid"}, {"t": "line", "p": [14, 14, 34, 14]}]}
      - {"kind": "shapes", "frame": true, "transform": "rot90", "items": [{"t": "poly", "p": [[8, 92], [52, 92], [8, 48]], "fill": "grey"}, {"t": "arrow", "p": [38, 42, 86, 14]}, {"t": "circle", "c": [76, 72], "r": 7, "fill": "solid"}, {"t": "line", "p": [14, 14, 34, 14]}]}
    answer: B
    explanation: |
      In a mirror image only left and right swap. The grey triangle stays at the bottom but moves to the right corner, the short line stays at the top but moves to the right, and the arrow now points up and to the **left**: **B**.
      A is a half turn (the triangle went to the top). C is the water image.

  - id: mw2-hook-anticlockwise
    topic: "Rotations"
    difficulty: 2
    q: "Which option shows this hook turned a quarter turn (90°) **anticlockwise**?"
    figure: {"kind": "shapes", "items": [{"t": "poly", "p": [[30, 88], [30, 18], [74, 18], [74, 50], [52, 50]], "open": true}, {"t": "circle", "c": [30, 88], "r": 6, "fill": "solid"}], "alt": "The given figure"}
    optionFigures:
      - {"kind": "shapes", "transform": "rot90", "items": [{"t": "poly", "p": [[30, 88], [30, 18], [74, 18], [74, 50], [52, 50]], "open": true}, {"t": "circle", "c": [30, 88], "r": 6, "fill": "solid"}]}
      - {"kind": "shapes", "transform": "flipH", "items": [{"t": "poly", "p": [[30, 88], [30, 18], [74, 18], [74, 50], [52, 50]], "open": true}, {"t": "circle", "c": [30, 88], "r": 6, "fill": "solid"}]}
      - {"kind": "shapes", "transform": "rot270", "items": [{"t": "poly", "p": [[30, 88], [30, 18], [74, 18], [74, 50], [52, 50]], "open": true}, {"t": "circle", "c": [30, 88], "r": 6, "fill": "solid"}]}
      - {"kind": "shapes", "transform": "flipV", "items": [{"t": "poly", "p": [[30, 88], [30, 18], [74, 18], [74, 50], [52, 50]], "open": true}, {"t": "circle", "c": [30, 88], "r": 6, "fill": "solid"}]}
    answer: C
    explanation: |
      Turning anticlockwise by a quarter turn moves the top to the left and the right side to the top.
      The long stem now lies along the bottom with the black dot at its right end, and the hook rises from its left end and curls over at the top: **C**.
      A is turned clockwise. B and D are reflections, which can never be made by turning.

  - id: mw2-not-a-rotation
    topic: "Mirror or turn?"
    difficulty: 2
    q: "Three of these are the same figure turned around. Which one is a mirror image and can NOT be made by turning?"
    figure: {"kind": "shapes", "items": [{"t": "poly", "p": [[18, 16], [62, 16], [62, 38], [40, 38], [40, 84], [18, 84]], "fill": "grey"}, {"t": "circle", "c": [76, 70], "r": 8, "fill": "solid"}], "alt": "The given figure"}
    optionFigures:
      - {"kind": "shapes", "transform": "flipH", "items": [{"t": "poly", "p": [[18, 16], [62, 16], [62, 38], [40, 38], [40, 84], [18, 84]], "fill": "grey"}, {"t": "circle", "c": [76, 70], "r": 8, "fill": "solid"}]}
      - {"kind": "shapes", "transform": "rot90", "items": [{"t": "poly", "p": [[18, 16], [62, 16], [62, 38], [40, 38], [40, 84], [18, 84]], "fill": "grey"}, {"t": "circle", "c": [76, 70], "r": 8, "fill": "solid"}]}
      - {"kind": "shapes", "transform": "rot180", "items": [{"t": "poly", "p": [[18, 16], [62, 16], [62, 38], [40, 38], [40, 84], [18, 84]], "fill": "grey"}, {"t": "circle", "c": [76, 70], "r": 8, "fill": "solid"}]}
      - {"kind": "shapes", "transform": "rot270", "items": [{"t": "poly", "p": [[18, 16], [62, 16], [62, 38], [40, 38], [40, 84], [18, 84]], "fill": "grey"}, {"t": "circle", "c": [76, 70], "r": 8, "fill": "solid"}]}
    answer: A
    explanation: |
      Stand at the corner of the grey shape and turn from the long arm to the short arm. In the original the long arm points down and the short arm right, so you turn **anticlockwise**.
      In B, C and D you also turn anticlockwise, so they are turned copies. In **A** the long arm points down and the short arm left, so you turn clockwise: A is the mirror image, and no amount of turning can produce it.

  - id: mw2-letter-water
    topic: "Water images"
    difficulty: 2
    q: "Which option is the water image of this figure? Remember that the letter is reflected too."
    figure: {"kind": "shapes", "items": [{"t": "text", "at": [34, 38], "s": "R", "size": 40}, {"t": "rect", "p": [58, 58, 28, 28], "fill": "grey"}, {"t": "line", "p": [12, 88, 44, 66]}], "alt": "The given figure"}
    optionFigures:
      - {"kind": "shapes", "transform": "flipH", "items": [{"t": "text", "at": [34, 38], "s": "R", "size": 40}, {"t": "rect", "p": [58, 58, 28, 28], "fill": "grey"}, {"t": "line", "p": [12, 88, 44, 66]}]}
      - {"kind": "shapes", "transform": "rot180", "items": [{"t": "text", "at": [34, 38], "s": "R", "size": 40}, {"t": "rect", "p": [58, 58, 28, 28], "fill": "grey"}, {"t": "line", "p": [12, 88, 44, 66]}]}
      - {"kind": "shapes", "transform": "rot90", "items": [{"t": "text", "at": [34, 38], "s": "R", "size": 40}, {"t": "rect", "p": [58, 58, 28, 28], "fill": "grey"}, {"t": "line", "p": [12, 88, 44, 66]}]}
      - {"kind": "shapes", "transform": "flipV", "items": [{"t": "text", "at": [34, 38], "s": "R", "size": 40}, {"t": "rect", "p": [58, 58, 28, 28], "fill": "grey"}, {"t": "line", "p": [12, 88, 44, 66]}]}
    answer: D
    explanation: |
      In a water image left stays left, so the R and the line stay on the left side, and the R is upside down (its leg points up and to the right). The grey square moves to the top right: **D**.
      A is the mirror image (R reversed). B is a half turn, which has the R upside down but on the right side.

  - id: mw2-water-of-mirror
    topic: "Mirror or turn?"
    difficulty: 2
    q: "Take the mirror image of this figure, and then the water image of that mirror image. Which option is the result?"
    figure: {"kind": "shapes", "items": [{"t": "poly", "p": [[14, 14], [46, 14], [14, 54]], "fill": "solid"}, {"t": "poly", "p": [[54, 86], [64, 62], [74, 86], [84, 62]], "open": true}, {"t": "circle", "c": [72, 26], "r": 9, "fill": "none"}], "alt": "The given figure"}
    optionFigures:
      - {"kind": "shapes", "transform": "flipH", "items": [{"t": "poly", "p": [[14, 14], [46, 14], [14, 54]], "fill": "solid"}, {"t": "poly", "p": [[54, 86], [64, 62], [74, 86], [84, 62]], "open": true}, {"t": "circle", "c": [72, 26], "r": 9, "fill": "none"}]}
      - {"kind": "shapes", "transform": "flipV", "items": [{"t": "poly", "p": [[14, 14], [46, 14], [14, 54]], "fill": "solid"}, {"t": "poly", "p": [[54, 86], [64, 62], [74, 86], [84, 62]], "open": true}, {"t": "circle", "c": [72, 26], "r": 9, "fill": "none"}]}
      - {"kind": "shapes", "transform": "rot180", "items": [{"t": "poly", "p": [[14, 14], [46, 14], [14, 54]], "fill": "solid"}, {"t": "poly", "p": [[54, 86], [64, 62], [74, 86], [84, 62]], "open": true}, {"t": "circle", "c": [72, 26], "r": 9, "fill": "none"}]}
      - {"kind": "shapes", "items": [{"t": "poly", "p": [[14, 14], [46, 14], [14, 54]], "fill": "solid"}, {"t": "poly", "p": [[54, 86], [64, 62], [74, 86], [84, 62]], "open": true}, {"t": "circle", "c": [72, 26], "r": 9, "fill": "none"}]}
    answer: C
    explanation: |
      The mirror swaps left and right, and the water image then swaps top and bottom. Both together move every part to the opposite corner, which is a half turn.
      The black triangle goes from the top left to the bottom right, and the zigzag from the bottom right to the top left: **C**.
      D is the original figure: the two reflections do not cancel, because they flip in different directions.

  - id: mw2-turn-then-mirror
    topic: "Combined moves"
    difficulty: 3
    q: "This figure is turned a half turn (180°) and then its mirror image is taken. Which option is the final result?"
    figure: {"kind": "shapes", "items": [{"t": "line", "p": [50, 50, 50, 14], "w": 3}, {"t": "line", "p": [50, 50, 82, 64], "w": 3}, {"t": "circle", "c": [50, 50], "r": 4, "fill": "solid"}, {"t": "rect", "p": [10, 70, 24, 16], "fill": "grey"}], "alt": "The given figure"}
    optionFigures:
      - {"kind": "shapes", "transform": "flipV", "items": [{"t": "line", "p": [50, 50, 50, 14], "w": 3}, {"t": "line", "p": [50, 50, 82, 64], "w": 3}, {"t": "circle", "c": [50, 50], "r": 4, "fill": "solid"}, {"t": "rect", "p": [10, 70, 24, 16], "fill": "grey"}]}
      - {"kind": "shapes", "transform": "flipH", "items": [{"t": "line", "p": [50, 50, 50, 14], "w": 3}, {"t": "line", "p": [50, 50, 82, 64], "w": 3}, {"t": "circle", "c": [50, 50], "r": 4, "fill": "solid"}, {"t": "rect", "p": [10, 70, 24, 16], "fill": "grey"}]}
      - {"kind": "shapes", "transform": "rot180", "items": [{"t": "line", "p": [50, 50, 50, 14], "w": 3}, {"t": "line", "p": [50, 50, 82, 64], "w": 3}, {"t": "circle", "c": [50, 50], "r": 4, "fill": "solid"}, {"t": "rect", "p": [10, 70, 24, 16], "fill": "grey"}]}
      - {"kind": "shapes", "transform": "rot90", "items": [{"t": "line", "p": [50, 50, 50, 14], "w": 3}, {"t": "line", "p": [50, 50, 82, 64], "w": 3}, {"t": "circle", "c": [50, 50], "r": 4, "fill": "solid"}, {"t": "rect", "p": [10, 70, 24, 16], "fill": "grey"}]}
    answer: A
    explanation: |
      A half turn swaps both left–right and top–bottom. The mirror then swaps left and right back again. So overall only top and bottom are swapped: the result is the **water image**.
      The grey bar ends up at the top left, and the long hand points down: **A**.
      C is only the half turn, and B is only the mirror image.

  - id: mw2-mirror-turn-mirror
    topic: "Combined moves"
    difficulty: 3
    q: "Take the mirror image of this figure, turn the result a quarter turn (90°) clockwise, and then take the mirror image again. Which option do you get?"
    figure: {"kind": "shapes", "items": [{"t": "circle", "c": [26, 34], "r": 14, "fill": "none"}, {"t": "line", "p": [40, 34, 88, 34], "w": 3}, {"t": "line", "p": [70, 34, 70, 48], "w": 3}, {"t": "line", "p": [82, 34, 82, 44], "w": 3}, {"t": "rect", "p": [14, 66, 30, 18], "fill": "solid"}], "alt": "The given figure"}
    optionFigures:
      - {"kind": "shapes", "transform": "rot90", "items": [{"t": "circle", "c": [26, 34], "r": 14, "fill": "none"}, {"t": "line", "p": [40, 34, 88, 34], "w": 3}, {"t": "line", "p": [70, 34, 70, 48], "w": 3}, {"t": "line", "p": [82, 34, 82, 44], "w": 3}, {"t": "rect", "p": [14, 66, 30, 18], "fill": "solid"}]}
      - {"kind": "shapes", "transform": "flipV", "items": [{"t": "circle", "c": [26, 34], "r": 14, "fill": "none"}, {"t": "line", "p": [40, 34, 88, 34], "w": 3}, {"t": "line", "p": [70, 34, 70, 48], "w": 3}, {"t": "line", "p": [82, 34, 82, 44], "w": 3}, {"t": "rect", "p": [14, 66, 30, 18], "fill": "solid"}]}
      - {"kind": "shapes", "transform": "rot180", "items": [{"t": "circle", "c": [26, 34], "r": 14, "fill": "none"}, {"t": "line", "p": [40, 34, 88, 34], "w": 3}, {"t": "line", "p": [70, 34, 70, 48], "w": 3}, {"t": "line", "p": [82, 34, 82, 44], "w": 3}, {"t": "rect", "p": [14, 66, 30, 18], "fill": "solid"}]}
      - {"kind": "shapes", "transform": "rot270", "items": [{"t": "circle", "c": [26, 34], "r": 14, "fill": "none"}, {"t": "line", "p": [40, 34, 88, 34], "w": 3}, {"t": "line", "p": [70, 34, 70, 48], "w": 3}, {"t": "line", "p": [82, 34, 82, 44], "w": 3}, {"t": "rect", "p": [14, 66, 30, 18], "fill": "solid"}]}
    answer: D
    explanation: |
      Two mirror images cancel each other, but they also reverse the direction of any turn between them. So the clockwise quarter turn becomes a quarter turn **anticlockwise**.
      The key's handle (the ring), which was on the left, ends up at the bottom, with the teeth pointing to the right: **D**.
      A is the tempting answer (a quarter turn clockwise), but the mirrors reverse the direction of the turn.

  - id: mw2-find-original
    topic: "Combined moves"
    difficulty: 3
    q: "Meera turned a card a quarter turn (90°) clockwise, and now it looks like this. Which option shows the card before she turned it?"
    figure: {"kind": "shapes", "transform": "rot90", "items": [{"t": "poly", "p": [[18, 86], [18, 46], [64, 46]], "open": true}, {"t": "arrow", "p": [64, 46, 64, 12]}, {"t": "circle", "c": [82, 80], "r": 7, "fill": "solid"}], "alt": "The given figure"}
    optionFigures:
      - {"kind": "shapes", "transform": "rot180", "items": [{"t": "poly", "p": [[18, 86], [18, 46], [64, 46]], "open": true}, {"t": "arrow", "p": [64, 46, 64, 12]}, {"t": "circle", "c": [82, 80], "r": 7, "fill": "solid"}]}
      - {"kind": "shapes", "items": [{"t": "poly", "p": [[18, 86], [18, 46], [64, 46]], "open": true}, {"t": "arrow", "p": [64, 46, 64, 12]}, {"t": "circle", "c": [82, 80], "r": 7, "fill": "solid"}]}
      - {"kind": "shapes", "transform": "flipH", "items": [{"t": "poly", "p": [[18, 86], [18, 46], [64, 46]], "open": true}, {"t": "arrow", "p": [64, 46, 64, 12]}, {"t": "circle", "c": [82, 80], "r": 7, "fill": "solid"}]}
      - {"kind": "shapes", "transform": "flipV", "items": [{"t": "poly", "p": [[18, 86], [18, 46], [64, 46]], "open": true}, {"t": "arrow", "p": [64, 46, 64, 12]}, {"t": "circle", "c": [82, 80], "r": 7, "fill": "solid"}]}
    answer: B
    explanation: |
      To undo a quarter turn clockwise, turn the card a quarter turn **anticlockwise**. What is now at the top was on the left before, and what is now on the right was at the top.
      The arrow, now pointing right, pointed up, and the black dot, now at the bottom left, was at the bottom right: **B**.
      A is what you get by turning it clockwise again. C and D are reflections, which no turn can make.

---

This set tests whether you can picture a figure in a mirror, in water, or turned around.

## Four moves

| Move | What happens |
|---|---|
| Mirror image (mirror on the right) | Left and right swap; top and bottom stay |
| Water image | Top and bottom swap; left and right stay |
| Quarter turn clockwise (90°) | Top goes to the right, right to the bottom, bottom to the left, left to the top |
| Half turn (180°) | Every part goes to the opposite corner |

## How to check an option

Pick one or two parts that are easy to follow, such as a black dot or the point of an arrow, and ask where each should end up. Then check a second part, because one part alone can fool you.

## Worked example

A figure has a dot at the top left and an arrow pointing right along the bottom. In its mirror image the dot is at the top right and the arrow points left along the bottom. After a quarter turn clockwise, the dot is at the top right and the arrow points down along the left side.

## Combined moves

- Mirror and then water image (in either order) is the same as a half turn.
- A half turn and then a mirror image is the same as a water image.
- A turn can never give a mirror image. If an option is the mirror image of the others, it cannot be a turned copy of them.

## Common trap

Doing a quarter turn the wrong way. Clockwise means the way a clock's hands move: from 12 towards 3.
