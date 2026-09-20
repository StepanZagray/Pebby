"""Readable names for SK48's obfuscated identifiers.

All source references point to ``third_party/arc3_games/sk48.py``.
"""

# Sprite names and tags, upstream lines 28-265.
HEAD_BLUE = "ejlpqgojjt"
HEAD_WHITE = "udbuodqlxv"
HEAD_ORANGE = "xtuqlbebvk"
HEAD_BROWN = "zkekdulqku"
HEAD_PROTOTYPES = (HEAD_BLUE, HEAD_WHITE, HEAD_ORANGE, HEAD_BROWN)
CLICKABLE_HEADS = (HEAD_BLUE, HEAD_WHITE)
COLOR_PAD = "elmjchdqcn"
SEGMENT = "qtjqovumxf"
RAIL = "irkeobngyh"
BOUNDARY_SMALL = "rtwdndlhdf"
BOUNDARY_LARGE = "ksixfnredk"
BLOCKER = "mkgqjopcjn"
FOOTER = "hspquzcixt"
DIVIDER = "yukipuenar"

TAG_HEAD = "epdquznwmq"
TAG_CLICK = "sys_click"
TAG_COLOR_PAD = "elmjchdqcn"
TAG_SEGMENT = "qtjqovumxf"
TAG_RAIL = "irkeobngyh"
TAG_BOUNDARY = "jtteddgeyl"
TAG_BLOCKER = "mkgqjopcjn"

# Board constants, upstream lines 613-625 and 658-659.
FRAME_SIZE = 64
HUD_ROW = 53
CELL = 6
MOVE_BUDGET = 196
DIRECTION = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}
COLORS = (8, 9, 12, 14)

# Game attributes initialized at upstream lines 658-703.
ATTR_MAX_MOVES = "vhzjwcpmk"
ATTR_MOVES_LEFT = "qiercdohl"
ATTR_LINES = "mwfajkguqx"
ATTR_VISITED_COLORS = "vjfbwggsd"
ATTR_COLOR_PADS = "vbelzuaian"
ATTR_PAIRS = "xpmcmtbcv"
ATTR_SELECTED = "vzvypfsnt"
ATTR_HISTORY = "seghobzez"
ATTR_PENDING_MOVES = "ljprkjlji"
ATTR_PENDING_PAUSES = "pzzwlsmdt"
ATTR_WIN_ANIMATION = "lgdrixfno"

# Actions, upstream lines 746-765 and 973-982.
ACTION_RESET = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_CLICK = 6
ACTION_UNDO = 7
MOVE_ACTIONS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)
ACTION_IDS = (*MOVE_ACTIONS, ACTION_CLICK, ACTION_UNDO)
ACTION_DELTAS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}
