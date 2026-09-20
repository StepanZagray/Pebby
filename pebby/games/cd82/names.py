"""Readable names for CD82's obfuscated identifiers.

Upstream ships ``third_party/arc3_games/cd82.py`` with randomised class,
method, attribute and sprite names. The file is left byte-identical; this
module is the translation table, so the rest of the package never hardcodes a
random string. Every entry was read directly out of the upstream source; the
line numbers cited are upstream line numbers.

The game in one paragraph (cd82.py:392-753): a 10x10 canvas starts black (0).
A "dial" sits on one of eight cells of a 3x3 ring (centre excluded); ACTION1-4
move it up/down/left/right along the ring. ACTION5 paints one region of the
canvas -- a half for the four edge positions, a triangle for the four corners --
with the selected colour. ACTION6 clicks: on a swatch it selects that colour,
on the small indicator that appears beside the basket at an edge position it
paints a 3x4 "cap" strip at that edge. After every paint the canvas is compared
with the target image on every cell that is not on either diagonal; a match
advances the level. Each level has a 100-action budget (ARCBaseGame.set_level
zeroes the count) and the 100th action on a level loses.
"""

import numpy as np

# --- sprite names (upstream cd82.py:36-259) ----------------------------------
SPRITE_INDICATOR = "ctwspzkygu"          # cd82.py:37, 6x5, clickable cap painter
TARGET_PREFIX = "eoqnvkspoa-"            # cd82.py:50-150, the six 10x10 targets
SPRITE_LAYOUT = "gkyfnkvrty"             # cd82.py:152, static frame chrome, layer -1
SPRITE_BASKET_DIAGONAL = "oaoosfneq-laopvne"    # cd82.py:178
SPRITE_BASKET_HORIZONTAL = "oaoosfneq-oaanwen"  # cd82.py:203
SPRITE_SWATCH = "pqkenviek"              # cd82.py:219, 5x5, centre pixel = colour
SPRITE_CANVAS = "xytrjjbyib"             # cd82.py:231, 10x10, starts all 0
SPRITE_CURSOR = "ydiwkzjkgl"             # cd82.py:248, 5x1 bar under the chosen swatch
ACTIVE_BASKET = "ActiveBasket"           # cd82.py:483, not obfuscated

# --- `Cd82` attributes (cd82.py:393-434) -------------------------------------
ATTR_BASKETS = "nicoqsvlg"       # cd82.py:393, per-dial (kind, x, y, rotation, dx, dy)
ATTR_DIAL_TO_CELL = "nfhykrqjp"  # cd82.py:403, dial index -> (row, col) on the 3x3 ring
ATTR_DIAL = "xwmfgtlso"          # cd82.py:414, dial position 0..7
ATTR_COLOR = "knqmgavuh"         # cd82.py:415, selected colour, 15 at level start
ATTR_PAINTING = "edjesyzxk"      # cd82.py:417, basket animation in flight
ATTR_HAS_INDICATOR = "yxjfgsdkm"  # cd82.py:430, level ships the indicator sprite
ATTR_BUDGET = "iewrsdwok"        # cd82.py:431, 100
ATTR_HUD = "qzgkhffci"           # cd82.py:432, the bottom-row budget bar
ATTR_CAP_PAINTING = "yfobpcuef"  # cd82.py:423, indicator animation in flight

# --- `Cd82` methods -----------------------------------------------------------
METHOD_MOVE_DIAL = "nqhfiooufi"      # cd82.py:531
METHOD_CLICK = "qbiojckwxl"          # cd82.py:551
METHOD_PAINT_REGION = "rtjwayrycq"   # cd82.py:709
METHOD_PAINT_CAP = "coublenfir"      # cd82.py:613
METHOD_CHECK_WIN = "wvrremwltt"      # cd82.py:740
METHOD_SWATCH_CLICKS = "yrfgxhebei"  # cd82.py:755, the game's own valid swatch clicks
METHOD_INDICATOR_CLICKS = "bmwcxxvjum"  # cd82.py:766, the game's own valid indicator click
METHOD_FIND_INDICATOR = "oaouwdxbxc"    # cd82.py:520

# --- actions (cd82.py:444, 642-671) ------------------------------------------
AVAILABLE_ACTIONS = (1, 2, 3, 4, 5, 6)
ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_PAINT, ACTION_CLICK = AVAILABLE_ACTIONS
ACTION_NAMES = {1: "up", 2: "down", 3: "left", 4: "right", 5: "paint", 6: "click"}

# --- the dial ring (cd82.py:403-412) ------------------------------------------
# Dial index -> (row, col) of the 3x3 ring. ACTION1/2 move the row, ACTION3/4 the
# column, clamped to 0..2; a move onto the centre (1,1) or off the grid is a
# no-op that still costs an action (cd82.py:531-549).
DIAL_CELLS = {0: (0, 1), 1: (0, 2), 2: (1, 2), 3: (2, 2), 4: (2, 1), 5: (2, 0), 6: (1, 0), 7: (0, 0)}
CELL_DIALS = {cell: dial for dial, cell in DIAL_CELLS.items()}
DIAL_COUNT = 8
EDGE_DIALS = (0, 2, 4, 6)     # halves; the indicator exists only at these
CORNER_DIALS = (1, 3, 5, 7)   # triangles
START_DIAL = 0
START_COLOR = 15              # cd82.py:449
BUDGET = 100                  # cd82.py:431; the action that reaches this count loses
MAX_ACTIONS = BUDGET - 1      # usable actions per level (set_level resets the count)

# --- geometry -----------------------------------------------------------------
FRAME_SIZE = 64
CANVAS_SIZE = 10
CANVAS_POS = (27, 34)     # cd82.py:265 and every level
TARGET_POS = (3, 3)       # cd82.py:260
SWATCH_Y = 2              # cd82.py:263
SWATCH_PITCH = 6
CURSOR_DY = 5             # cd82.py:571, cursor sits 5 below the swatch
INDICATOR_HOME = (29, 18)  # cd82.py:284, where levels place the indicator (rotation 180)

# Per-dial basket geometry (cd82.py:393-402): kind, x, y, rotation, dx, dy.
BASKETS = (
    ("horizontal", 25, 24, 180, 0, 1),
    ("diagonal", 33, 21, 0, -1, 1),
    ("horizontal", 38, 32, 270, -1, 0),
    ("diagonal", 33, 40, 90, -1, -1),
    ("horizontal", 25, 45, 0, 0, -1),
    ("diagonal", 14, 40, 180, 1, -1),
    ("horizontal", 17, 32, 90, 1, 0),
    ("diagonal", 14, 21, 270, 1, 1),
)

# Indicator placement per edge dial (cd82.py:504-509): offset from the basket
# and rotation. The base sprite is 5 rows x 6 columns (cd82.py:38-44), so its
# rendered size is 6x5 at 0/180 and 5x6 at 90/270.
INDICATOR_PLACEMENT = {0: ((4, -6), 180), 2: ((10, 4), 270), 4: ((4, 10), 0), 6: ((-6, 4), 90)}


def indicator_click(dial):
    """Display pixel that hits the indicator shown at `dial` (an edge dial).

    Mirrors cd82.py:766-777: (x + width // 2, y + height // 2) with scale 1 and
    no letterbox, since the camera is the full 64x64 frame.
    """
    (ox, oy), rotation = INDICATOR_PLACEMENT[dial]
    _, bx, by, _, _, _ = BASKETS[dial]
    width, height = (6, 5) if rotation in (0, 180) else (5, 6)
    return bx + ox + width // 2, by + oy + height // 2


def swatch_click(x, y):
    """Display pixel that hits a swatch whose top-left is (x, y) (cd82.py:761-762)."""
    return x + 2, y + 2


# --- paint regions (cd82.py:709-738 and 613-628) ------------------------------
def region_mask(dial):
    """Boolean 10x10 mask painted by ACTION5 at `dial`."""
    kind, _, _, rotation, _, _ = BASKETS[dial]
    mask = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    if kind == "horizontal":
        if rotation == 180:
            mask[0:5, :] = True
        elif rotation == 0:
            mask[5:10, :] = True
        elif rotation == 90:
            mask[:, 0:5] = True
        else:
            mask[:, 5:10] = True
    elif rotation == 180:
        for i in range(10):
            mask[i, 0:i + 1] = True
    elif rotation == 90:
        for i in range(10):
            mask[i, 9 - i:10] = True
    elif rotation == 0:
        for i in range(10):
            mask[i, i:10] = True
    else:
        for i in range(10):
            mask[i, 0:10 - i] = True
    return mask


def cap_mask(dial):
    """Boolean 10x10 mask painted by clicking the indicator at edge `dial`."""
    mask = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    if dial == 0:
        mask[0:3, 3:7] = True
    elif dial == 4:
        mask[7:10, 3:7] = True
    elif dial == 6:
        mask[3:7, 0:3] = True
    elif dial == 2:
        mask[3:7, 7:10] = True
    return mask


def compare_mask():
    """Cells the win check looks at (cd82.py:748-751): everything off both diagonals."""
    mask = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    for i in range(CANVAS_SIZE):
        mask[i, i] = False
        mask[i, 9 - i] = False
    return mask


def ring_distance(a, b):
    """Fewest ACTION1-4 presses that move the dial from `a` to `b`."""
    d = abs(a - b) % DIAL_COUNT
    return min(d, DIAL_COUNT - d)


def ring_path(a, b):
    """Action ids that move the dial from `a` to `b` along the shorter arc."""
    d = (b - a) % DIAL_COUNT
    step = 1 if d <= DIAL_COUNT - d else -1
    actions = []
    dial = a
    while dial != b:
        nxt = (dial + step) % DIAL_COUNT
        (r0, c0), (r1, c1) = DIAL_CELLS[dial], DIAL_CELLS[nxt]
        if r1 < r0:
            actions.append(ACTION_UP)
        elif r1 > r0:
            actions.append(ACTION_DOWN)
        elif c1 < c0:
            actions.append(ACTION_LEFT)
        else:
            actions.append(ACTION_RIGHT)
        dial = nxt
    return actions


def move_dial(dial, action):
    """The dial after ACTION1-4, exactly as cd82.py:531-549."""
    row, col = DIAL_CELLS[dial]
    if action == ACTION_UP:
        row = max(0, row - 1)
    elif action == ACTION_DOWN:
        row = min(2, row + 1)
    elif action == ACTION_LEFT:
        col = max(0, col - 1)
    elif action == ACTION_RIGHT:
        col = min(2, col + 1)
    else:
        return dial
    if (row, col) == (1, 1):
        return dial
    return CELL_DIALS[(row, col)]
