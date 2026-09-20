"""Readable names for SB26's obfuscated identifiers.

The vendored source remains untouched.  This module is the translation table
for the exact engine adapter and the bounded generator.  Line references are
to ``third_party/arc3_games/sb26.py``.

SB26 is a sequence-building puzzle.  Coloured six-pixel tiles can be selected
with ACTION6 and moved into frame slots; selecting two occupied board cells
swaps them.  ACTION7 restores the preceding completed placement.  ACTION5
spends one unit of the native 64-unit energy budget and traverses the framed
sequence.  A tile must match the correspondingly ordered target swatch.  Some
official levels contain several frames and link tiles which recursively enter
another frame.  The full planner models those nested traversals, repeated
frame entry, the first-slot recursion guard, fixed board tiles, and cyclic
goal prefixes exactly at settled native action boundaries.
"""

# Sprite prototypes and tags (sb26.py:36-360).
SPRITE_CURSOR = "wrqpmmfhup"       # sb26.py:321-333, selected-cell cursor
SPRITE_TILE = "lngftsryyw"         # sb26.py:112-126, regular coloured tile
SPRITE_LINK = "vgszefyyyp"         # sb26.py:289-303, recursive frame link
SPRITE_SLOT = "susublrply"          # sb26.py:247-260, empty cell marker
SPRITE_GOAL = "quhhhthrri"          # sb26.py:233-246, target swatch
SPRITE_GOAL_BACKDROP = "uzxwqmkrmk"  # sb26.py:274-288
SPRITE_ENERGY_LINE = "zpwrpmkvsv"  # sb26.py:334-341, construction sentinel
SPRITE_CONNECTOR = "pqezjimbse"     # sb26.py:199-214, level-2 link tutorial cue

TAG_FRAME = "pkpgflvjel"            # sb26.py:69 and other frame prototypes
TAG_TILE = "lngftsryyw"             # sb26.py:124, 301
TAG_SLOT = "susublrply"              # sb26.py:259
TAG_CLICK = "sys_click"              # sb26.py:124, 259, 301

# The suffix is consumed by upstream as the frame's arity (sb26.py:1002).
FRAME_FOR_ARITY = {
    1: "jvkvqzheok1",                # sb26.py:53-70
    2: "jvkvqzheok2",                # sb26.py:71-88
    3: "pcrvmjfjzg3",                # sb26.py:181-198
    4: "qdmvvkvhaz4",                # sb26.py:215-232
    5: "nyqgqtujsa5",                # sb26.py:147-164
    6: "wbkmnqvtxh6",                # sb26.py:304-320
    7: "zzssdzqbbr7",                # sb26.py:342-359
}
ARITY_FOR_FRAME = {name: arity for arity, name in FRAME_FOR_ARITY.items()}

# Game attributes established by Sb26.on_set_level (sb26.py:721-776).
ATTR_INITIAL_ENERGY = "incrguxqwfjtial_energy"  # sb26.py:723
ATTR_ENERGY = "sjcuorclg"                       # sb26.py:724
ATTR_FRAMES = "qaagahahj"                       # sb26.py:726
ATTR_GOALS = "wcfyiodrx"                        # sb26.py:729
ATTR_TILES = "dkouqqads"                        # sb26.py:731
ATTR_SLOTS = "dewwplfix"                        # sb26.py:738
ATTR_SELECTION = "lqcskynzr"                    # sb26.py:739
ATTR_HISTORY = "uvawsoycr"                      # sb26.py:775

# Animation/state fields checked before symbolic extraction (sb26.py:744-769).
ATTR_SWAP_ANIMATION = "artsfnufc"
ATTR_DELAY = "modqnpqfi"
ATTR_FLASH_ANIMATION = "ftyhvmeft"
ATTR_COLOUR_ANIMATIONS = "xshdlymmy"
ATTR_MOVEMENTS = "ulzvbcvzs"
ATTR_MOVEMENT_FRAME = "jlcrtmkes"
ATTR_TILE_FILL_ANIMATION = "xjxrqgaqw"
ATTR_RESET_ANIMATION = "bbiavyren"
ATTR_WIN_ANIMATION = "lmvwmlqtw"
ATTR_FAILURE_FLASH = "japgbruyb"

# Actions (Sb26.__init__/step, sb26.py:719, 778-913).
ACTION_SUBMIT = 5
ACTION_CLICK = 6
ACTION_UNDO = 7
ACTION_IDS = (ACTION_SUBMIT, ACTION_CLICK, ACTION_UNDO)
ACTION_NAMES = {ACTION_SUBMIT: "submit", ACTION_CLICK: "click", ACTION_UNDO: "undo"}

# Native constants and geometry (sb26.py:687-697, 998-1009, 1042-1072).
FRAME_SIZE = 64
TRAY_MIN_Y = 54
TILE_SIZE = 6
CELL_PITCH = 6
FRAME_INSET = 2
CLICK_INSET = 2
INITIAL_ENERGY = 64
COLOURS = (6, 8, 9, 11, 12, 14, 15)


def tile_colour(sprite):
    """Return the semantic colour upstream compares for a regular/link tile."""
    return int(sprite.pixels[1, 1])


def goal_colour(sprite):
    """Return a target swatch's colour (the prototype is fully remapped)."""
    return int(sprite.pixels[0, 0])


def click_at(position):
    """A visible, collidable interior pixel for a six-pixel tile or slot."""
    x, y = position
    return int(x) + CLICK_INSET, int(y) + CLICK_INSET


def frame_cells(frame, arity=None):
    """Cell top-left coordinates read by upstream ``rfdjlhefnd``."""
    count = ARITY_FOR_FRAME[frame.name] if arity is None else int(arity)
    return tuple(
        (int(frame.x) + FRAME_INSET + index * CELL_PITCH, int(frame.y) + FRAME_INSET)
        for index in range(count)
    )
