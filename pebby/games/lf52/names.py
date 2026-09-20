"""Readable names for LF52's obfuscated upstream implementation.

The vendored source remains untouched.  These mappings were read from
``third_party/arc3_games/lf52.py``; line references below refer to that file.

LF52 is an orthogonal peg-jump puzzle.  Clicking a peg examines the four
cardinal directions (lines 5353-5367, 5649-5669).  A direction is offered when
the adjacent cell contains a peg/obstacle and the cell two places away is an
empty board hole.  Clicking that landing marker moves the selected peg two
cells and removes the intervening peg when both have the same name
(lines 5374-5419).  Ordinary levels win when one peg remains (lines
5572-5582, 5644-5647).  Arrow actions move special yellow rail cells
(lines 5276-5325).  Later levels combine coloured pegs, permanent obstacles,
camera scrolling, scripted reset landings, and level-specific survivor rules.
The family package models those rules directly and verifies every positive
route in the native engine.
"""

# Public action ids (Lf52.__init__/jxyktkxwle, lines 5745, 5789-5845).
ACTION_RESET = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_CLICK = 6
ACTION_UNDO = 7
AVAILABLE_ACTIONS = (1, 2, 3, 4, 6, 7)

# Display/grid constants (lines 163-166, 4655 and peers).
DISPLAY = 64
TILE = 6
BACKGROUND_COLOR = 10
PADDING_COLOR = 3

# Lf52 and logical-world attributes (lines 5736-5748, 5128-5173).
ATTR_WORLD = "ikhhdzfmarl"
ATTR_ACTION_BAR = "apiwkxksucg"
ATTR_GRID = "hncnfaqaddg"
ATTR_SELECTED_TOKEN = "wpwvsglgmb"
ATTR_SELECTION_MARKERS = "zpbguihjnf"
ATTR_LOGICAL_LEVEL = "whtqurkphir"
ATTR_ACTION_COUNT = "asqvqzpfdi"
ATTR_RESET_PROMPT = "zvcnglshzcx"
ATTR_UNDO_MANAGER = "fdvqqrgrvcc"
ATTR_UNDO_STACK = "enbizandmjr"

# Grid/entity API (lines 3624-3934, 3937-4175).
CLASS_LAYOUT = "eollalrjeg"
CLASS_GRID = "nfpetofmbpr"
METHOD_GRID_ENTITIES = "xpnsbxlatu"
METHOD_ENTITIES_NAMED = "whdmasyorl"
METHOD_ENTITIES_PREFIXED = "ndtvadsrqf"
PROP_GRID_POSITION = "chahdtpdoz"
PROP_CELLS = "abvcoxnskr"

# Logical entity names and source sprite keys (lines 4367-4587).
PEG = "fozwvlovdui"
PEG_RED = "fozwvlovdui_red"
PEG_BLUE = "fozwvlovdui_blue"
PEG_GRAY = "fozwvlovdui_gray"
HOLE = "hupkpseyuim"
MOVING_HOLE = "hupkpseyuim2"
OBSTACLE = "dgxfozncuiz"
RAIL_PREFIX = "kraubslpehi"
LANDING_MARKER = "lgbyiaitpdi"
PLACEHOLDER_SPRITE = "xnpkcymhua"

# A generated Level carries the separate logical grid here.  Upstream's ten
# Level objects are only placeholders; on_set_level selects ``gridN`` from the
# module-global kciatvszkc mapping (lines 66-137, 5145, 5859-5870).
LEVEL_LAYOUT_DATA = "pebby_lf52_layout"
GENERATED_KIND = "pebby.lf52.static-peg-jump.v1"
FULL_GENERATED_KIND = "pebby.lf52.full-mechanics.v2"

PEG_KINDS = (PEG, PEG_RED, PEG_BLUE, PEG_GRAY)
SPLITS = ("train", "validation", "test")

DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # vjafsffahp, line 5002


def native_action_limit(logical_level):
    """The real loss thresholds from ``Lf52.step`` (lines 5763-5771)."""
    logical_level = int(logical_level)
    if logical_level == 1:
        return 64
    if logical_level >= 6:
        return 64 * 10
    return 64 * 5


def cell_click(origin, cell, tile=TILE):
    """Centre display pixel of a logical board cell."""
    ox, oy = origin
    x, y = cell
    return int(ox + x * tile + tile // 2), int(oy + y * tile + tile // 2)
