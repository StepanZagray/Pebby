"""Readable names for RE86's obfuscated native identifiers.

The source references are deliberately kept here so a future source refresh can
audit the adapter without reverse-engineering the game again.
"""

# Sprite prototypes.  See third_party/arc3_games/re86.py lines 34-1613.
EMPTY_CANVAS = "0000wbshgbbxxc"

MOVABLE_RING_11 = "0030rxzjuipynt"  # lines 462-491
MOVABLE_RING_6 = "0032eqjhipvldb"   # lines 494-526
MOVABLE_RING_9_SMALL = "0033iipgjezqam"  # lines 529-559
MOVABLE_RING_12_LARGE = "0034crtnctqmun"  # lines 562-596
MOVABLE_RING_10 = "0041edqtyiekev"  # lines 708-739
MOVABLE_RING_9 = "0042qffokapnyc"   # lines 742-774
MOVABLE_RING_12 = "0043ingegpwaik"  # lines 777-806
MOVABLE_RING_8 = "0045rflckndtdm"   # lines 836-865
MOVABLE_RING_9_LARGE = "0052kdvnuoqdxw"  # lines 969-1027

# These are the ordinary translating prototypes used by the generator.  They
# have no flexible/fixed-centre tag and all fit the generator's four anchors.
GENERATED_PROTOTYPES = (
    MOVABLE_RING_11,
    MOVABLE_RING_6,
    MOVABLE_RING_9_SMALL,
    MOVABLE_RING_12_LARGE,
    MOVABLE_RING_10,
    MOVABLE_RING_9,
    MOVABLE_RING_12,
    MOVABLE_RING_8,
    MOVABLE_RING_9_LARGE,
)

# Tags.  Definitions occur on the sprite declarations above; consumers are in
# the native movement/win code at lines 1894-2157.
TAG_BACKGROUND = "0001jdldomszsf"
TAG_OBSTACLE = "0003dlchiwseii"
TAG_DYE = "0007dtbisvazhv"
TAG_MOVABLE = "0031cppcuvqlbi"
TAG_FLEXIBLE = "0036ilsgwuvbxv"
TAG_FIXED_CENTER = "0049ppblgltcfi"
TAG_TARGET = "0054xnsuqceejm"

# Native data/attribute names.  StepCounter is read at lines 1888-1892;
# animation fields are initialized at lines 1883-1886 and consumed in step().
KEY_STEP_COUNTER = "StepCounter"
ATTR_STEP_COUNTER = "xikvflgqgp"
ATTR_PENDING_DYE = "ylzrmgmdyh"
ATTR_DYE_OBJECT = "cptlsijjli"

FRAME_SIZE = 64
GRID_STEP = 3
TRANSPARENT = -1
TARGET_GUIDE = 4  # ignored by the win predicate, lines 1917-1918
SELECTED_CENTER = 0

# Native actions, declared at line 1880 and dispatched at lines 2124-2153.
ACTION_RESET = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_NEXT = 5
ACTION_CLICK = 6
MOVE_ACTIONS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)
ACTION_IDS = (*MOVE_ACTIONS, ACTION_NEXT)

ACTION_DELTAS = {
    ACTION_UP: (0, -GRID_STEP),
    ACTION_DOWN: (0, GRID_STEP),
    ACTION_LEFT: (-GRID_STEP, 0),
    ACTION_RIGHT: (GRID_STEP, 0),
}

# Important upstream implementation locations for the handoff/audit trail.
SOURCE_LINES = {
    "levels": "1615-1753",
    "main_color": "1758-1761",
    "selected_center": "1763-1777",
    "step_counter": "1826-1861",
    "game_and_actions": "1864-1881",
    "win_check": "1894-1918",
    "selection_normalization": "1920-1941",
    "movement_and_deformation": "1943-2093",
    "dye_flood": "2076-2122",
    "action_dispatch_and_completion": "2124-2158",
}
