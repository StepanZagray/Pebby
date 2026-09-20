"""Readable names for G50T's obfuscated vendored implementation.

The comments are source-line references into ``third_party/arc3_games/g50t.py``.
They deliberately document only identifiers used by this adapter.
"""

# Native constants, upstream lines 1845-1850.
FRAME_SIZE = 64
GRID_STEP = 6
BACKGROUND_COLOR = 0
PADDING_COLOR = 2
TOGGLE_COLOR = 11

# The 64-pixel timer moves once per two actions and loses only after x < -64
# (upstream lines 2826-2828 and 2844-2849).
NATIVE_MAX_ACTIONS = 129

# Actions, upstream lines 2787-2794 and 2816-2825.
ACTION_RESET = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_REWIND = 5
ACTION_CLICK = 6
MOVE_ACTIONS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)
ACTION_IDS = (*MOVE_ACTIONS, ACTION_REWIND)
ACTION_DELTA = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}

# Prototype names used by the official first level, upstream lines 1638-1650.
SPRITE_GOAL = "gilbljmfbc"
SPRITE_CHECKPOINT = "gpkhwmwioo"
SPRITE_DOOR = "kjrcloicja"
SPRITE_SWITCH = "medyellngi"
SPRITE_CURSOR = "ovhuyqtghw"
SPRITE_TIMER = "ppfvilwwnk"
SPRITE_PLAYER = "qftsebtxuc"
SPRITE_TIMER_BACKGROUND = "xulolghsby"
SPRITE_TELEPORT_PAD = "mpreboxmgc"
SPRITE_ENEMY = "vtwcsmdoqp"

# Semantic tag enum, upstream lines 1853-1867.
TAG_ENEMY = "vtwcsmdoqp"
TAG_ENEMY_PATH = "akfoiqesdk"
TAG_BOUNDARY = "rsrdfsruqh"
TAG_PLAYER = "qftsebtxuc"
TAG_CHECKPOINT = "gpkhwmwioo"
TAG_CURSOR = "ovhuyqtghw"
TAG_DOOR = "kjrcloicja"
TAG_SWITCH = "medyellngi"
TAG_CIRCUIT = "hxztohfdlx"
TAG_TELEPORT_LINK = "hgglgttaui"
TAG_TELEPORT_PAD = "mpreboxmgc"
TAG_GOAL = "gilbljmfbc"
TAG_TIMER = "ppfvilwwnk"

# Game/controller fields, upstream lines 2494-2513 and 2780-2802.
ATTR_CONTROLLER = "vgwycxsxjz"
ATTR_ACTION_COUNTER = "ucorwtereb"
ATTR_TIMER = "twyixucrqi"
CTRL_ANIMATIONS = "hjvvibklzv"
CTRL_BOUNDARY = "afbbgvkpip"
CTRL_PLAYER = "dzxunlkwxt"
CTRL_HISTORY = "areahjypvy"
CTRL_START_X = "yugzlzepkr"
CTRL_START_Y = "vgpdqizwwm"
CTRL_GOAL = "whftgckbcu"
CTRL_ENEMIES = "kgvnkyaimw"
CTRL_GHOSTS = "rloltuowth"
CTRL_CHECKPOINTS = "drofvwhbxb"
CTRL_STAGE = "rlazdofsxb"
CTRL_DOORS = "uwxkstolmf"
CTRL_INPUTS = "hamayflsib"
CTRL_REWINDING = "dofntsemri"

# Wrapper classes, upstream lines 2111, 2142, 2179, 2264 and 2352.
CLASS_SWITCH = "lqtxaumfed"
CLASS_CIRCUIT = "alyzsfkumg"
CLASS_DOOR = "yyzqramdhd"
CLASS_TELEPORT_LINK = "crfcpstubm"
CLASS_TELEPORT_PAD = "ulhhdeoyok"

# Wrapper fields/methods, upstream lines 2111-2254.
WRAPPER_SPRITE = "xmdjbwrpmv"
SWITCH_OUTPUT = "nexhtmlmxh"
CIRCUIT_OUTPUTS = "ytztewxdin"
DOOR_ACTIVE = "dijhfchobv"
DOOR_TOGGLE = "dpdubazedr"
DOOR_DIRECTION = "hluvhlvimq"
