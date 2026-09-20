"""Readable names for LP85's obfuscated native implementation.

The mappings below point at the pinned source in
``third_party/arc3_games/lp85.py``.  They are aliases only: gameplay always
runs through the unmodified vendored ``Lp85`` class.
"""

# Core sprite prototypes, upstream lines 36-1064.
BUTTON_LEFT = {
    "A": "ahdesifykt",   # line 36
    "B": "peuddygayw",   # line 614
    "C": "ymwkwkfxct",   # line 982
}
BUTTON_RIGHT = {
    "A": "zpxgvxwdex",   # line 1054
    "B": "dvvacwfchj",   # line 238
    "C": "reafrgzfwk",   # line 674
}

TILE_YELLOW = "aydooxtcli"  # line 96, colour 9
TILE_BLUE = "ekpijbeoel"    # line 250, colour 1
TILE_RED = "kisflvfvsy"     # line 450, colour 10
TILE_ORANGE = "slefrnjymo"  # line 758, colour 2
TILE_PURPLE = "tddzizwhzf"  # line 792, colour 15
TILE_PROTOTYPES = (
    TILE_YELLOW,
    TILE_BLUE,
    TILE_RED,
    TILE_ORANGE,
    TILE_PURPLE,
)

TARGET_MARKER = "bghvgbtwcb"       # line 130, checked at +1,+1
ALT_TARGET_MARKER = "fdgmtkfrxl"   # line 260, checked at +1,+1
GOAL_TOKEN = "odkpvwbihk"          # line 580, tag ``goal``
ALT_GOAL_TOKEN = "hfikqtizdo"      # line 308, tag ``goal-o``

# Tags and level data read by Lp85, upstream lines 21363-21443.
TAG_BUTTON_PREFIX = "button_"
TAG_CLICK = "sys_click"
TAG_TILE = "tile"
TAG_TARGET_MARKER = "bghvgbtwcb"
TAG_ALT_TARGET_MARKER = "fdgmtkfrxl"
TAG_GOAL = "goal"
TAG_ALT_GOAL = "goal-o"

KEY_STEPS = "StepCounter"
KEY_LEVEL_NAME = "level_name"
# Pebby-only metadata carried inside generated Levels.  The native engine
# ignores it; env.py installs it into the module's movement-map table while a
# game is constructed or changes levels.
KEY_GENERATED_MAP = "_pebby_lp85_movement_map"

# Module globals / attributes, upstream lines 1583 and 21242-21443.
GLOBAL_MAPS = "izutyjcpih"          # raw numbered maps, lines 1583-21205
ATTR_COMPILED_MAPS = "uopmnplcnv"   # qfvvosdkqr output, line 21345
ATTR_STEP_COUNTER = "toxpunyqe"     # fonypcnqmf instance, line 21344
ATTR_CLICKABLES = "afhycvvjg"       # level buttons, line 21370
HUD_MAX_STEPS = "bnlfrvxkob"        # fonypcnqmf, line 21295
HUD_CURRENT_STEPS = "current_steps" # fonypcnqmf, line 21296

# qfvvosdkqr compiles each numbered map to these obfuscated dictionary keys,
# upstream lines 21242-21262.
MAP_POSITIONS = "qcmzcjocmj"
MAP_LENGTH = "oxbwsencfv"

FRAME_SIZE = 64
GRID_STEP = 3

ACTION_RESET = 0
ACTION_CLICK = 6
ACTION_IDS = (ACTION_CLICK,)

# Source landmarks for audits.
SOURCE_LEVELS = "lines 1091-1581"
SOURCE_RAW_MAPS = "lines 1583-21205"
SOURCE_COMPILE_MAPS = "lines 21242-21262"
SOURCE_ROTATE_MAP = "lines 21265-21286"
SOURCE_GAME = "lines 21339-21450"
