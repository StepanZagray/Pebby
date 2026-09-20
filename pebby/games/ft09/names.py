"""Readable names for FT09's obfuscated identifiers.

Upstream ships ``third_party/arc3_games/ft09.py`` with randomised class,
method, attribute, tag and data-key names. The file is left byte-identical;
this module is the translation table, so nothing else in this package
hardcodes a random string. Line numbers cited are upstream line numbers.

FT09 in one paragraph (ft09.py:2301-2520): a click-only Lights-Out variant.
Cells are 3x3 sprites on a 32x32 grid at a 4-pixel pitch, all starting at
palette colour 0. Clicking a cell advances it and every cell under its stencil
one step through the palette. Constraint sprites sit in empty slots: their
centre pixel is a colour, and each of the 8 border pixels says whether the
neighbouring cell must equal (pixel == 0) or differ from (pixel != 0) that
colour. The level is complete when every constraint holds after a click.
Every cell click that does not complete the level costs one unit of budget;
the level is lost when the budget hits zero. There is no randomness.
"""

# --- sprite tags -------------------------------------------------------------
TAG_CELL = "Hkx"          # ordinary cell, cycles under the level stencil (ft09.py:570)
TAG_SPECIAL_CELL = "NTi"  # cell carrying its own stencil as colour-6 pixels (ft09.py:218)
TAG_CONSTRAINT = "bsT"    # 3x3 rule sprite: centre colour + 8 match/mismatch flags (ft09.py:62)
TAG_HINT = "Ycb"          # level-0 tutorial frame that flashes on a misclick (ft09.py:1513)
TAG_ANY_CELL = "gOi"      # decorative "is a cell" tag; the engine never queries it
CELL_TAGS = (TAG_CELL, TAG_SPECIAL_CELL)

STENCIL_MARKER = 6        # pixel value marking a special cell's stencil (ft09.py:2350, 2400)
SPECIAL_BODY = 7          # special-cell body pixel in shipped sprites (overwritten to palette[0])
SPECIAL_CENTRE = 10       # special-cell centre pixel in shipped sprites (overwritten to palette[0])
CELL_BODY = 9             # ordinary-cell pixel in the shipped prototype (ft09.py:561-567)
MATCH_FLAG = 0            # constraint border pixel == 0 means "neighbour must match" (ft09.py:2427)

# --- level data keys ---------------------------------------------------------
KEY_BUDGET = "kCv"        # click budget; HUD bar along row 63 (ft09.py:2303, 2310)
KEY_PALETTE = "cwU"       # colour cycle, default [9, 8] (ft09.py:2331-2333)
KEY_STENCIL = "elp"       # 3x3 0/1 stencil for ordinary cells, default centre-only (ft09.py:2335-2338)

# --- `Ft09` attributes (ft09.py:2301+) ---------------------------------------
ATTR_HUD = "lpw"                 # `sve` budget display
ATTR_HINT = "zth"                # hint sprite or None
ATTR_FLASH_FRAMES = "our"        # remaining misclick-flash frames
ATTR_GRID_W = "pdw"
ATTR_GRID_H = "zbh"
ATTR_CONSTRAINTS = "gig"         # list of TAG_CONSTRAINT sprites
ATTR_CELLS = "fhc"               # list of TAG_CELL sprites
ATTR_SPECIAL_CELLS = "mou"       # list of TAG_SPECIAL_CELL sprites
ATTR_PALETTE = "gqb"
ATTR_STENCIL = "irw"
ATTR_CLICKED = "blr"             # sprite hit by the current click

# --- `Ft09` methods ----------------------------------------------------------
METHOD_RELOAD_BUDGET = "olv"     # ft09.py:2309
METHOD_IS_SOLVED = "cgj"         # ft09.py:2416, the 8-neighbour rule check

# --- budget HUD (`sve`, ft09.py:2272-2298) -----------------------------------
HUD_MAX = "oro"
HUD_CURRENT = "dzy"
HUD_CONSUME = "lph"              # decrements, returns False once exhausted -> lose()
HUD_REFILL = "dsl"

# --- geometry ----------------------------------------------------------------
GRID = 32           # every shipped level uses grid_size=(32, 32) (ft09.py:2062+)
FRAME = 64
SCALE = FRAME // GRID
PITCH = 4           # neighbour offsets are +-4 grid pixels (ft09.py:2380, 2422)
CELL_SIZE = 3
# Shipped cells sit at x, y in {2, 6, ..., 26}; a 7x7 lattice at pitch 4.
LATTICE = tuple(range(2, GRID - CELL_SIZE, PITCH))

# Offsets of the 8 border pixels of a constraint sprite, as (row, col) -> (dx, dy)
# in grid pixels (ft09.py:2420-2519), and of the 3x3 stencil (ft09.py:2372-2376).
NEIGHBOUR_OFFSETS = {(0, 0): (-PITCH, -PITCH), (0, 1): (0, -PITCH), (0, 2): (PITCH, -PITCH),
                     (1, 0): (-PITCH, 0), (1, 2): (PITCH, 0),
                     (2, 0): (-PITCH, PITCH), (2, 1): (0, PITCH), (2, 2): (PITCH, PITCH)}
STENCIL_OFFSETS = {(row, col): ((col - 1) * PITCH, (row - 1) * PITCH)
                   for row in range(3) for col in range(3)}
IDENTITY_STENCIL = ((0, 0, 0), (0, 1, 0), (0, 0, 0))

# --- actions -----------------------------------------------------------------
# available_actions is [6] (ft09.py:2306); RESET is id 0. A click on nothing is
# free; a click on a constraint or the gap between cells is free too.
ACTION_RESET = 0
ACTION_CLICK = 6
ACTION_IDS = (ACTION_CLICK,)


def cell_click(x, y):
    """Display coordinates that land on the centre pixel of the cell at grid (x, y)."""
    return SCALE * (x + 1), SCALE * (y + 1)
