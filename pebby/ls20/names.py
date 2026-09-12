"""Readable names for LS20's obfuscated identifiers.

Upstream ships ``third_party/ls20/ls20.py`` with randomised class, method,
attribute and sprite names. The file is left byte-identical; this module is the
translation table, so the rest of Pebby never hardcodes a random string.

Every entry was read directly out of the upstream source; the line numbers cited
are upstream line numbers.
"""

# --- sprite tags (upstream `Sprite.tags`) ------------------------------------
TAG_PLAYER = "sfqyzhzkij"
TAG_WALL = "ihdgageizm"
TAG_GOAL_PAD = "rjlbuycveu"
TAG_GOAL_ICON = "kvynsvxbpi"
TAG_GOAL_RING = "vjotnebuqo"
TAG_VANISHING_RING = "vfkkzdgxzx"
TAG_GOAL_HINT_FRAME = "hoswmpiqkw"
TAG_STEP_REFILL = "npxgalaybz"
TAG_SHAPE_CYCLER = "ttfwljgohq"
TAG_COLOR_CYCLER = "soyhouuebz"
TAG_ROTATION_CYCLER = "rhsxkxzdjz"
TAG_LAUNCHER = "gbvqrjtaqo"
TAG_PATROL_RAIL = "xfmluydglp"
TAG_CARRIED_TOKEN = "wgmbtyhvbc"
TAG_HUD_PANEL = "eqatonpohu"
TAG_MATCH_FRAME = "ghizzeqtoh"

CYCLER_TAGS = {TAG_SHAPE_CYCLER: "shape", TAG_COLOR_CYCLER: "color", TAG_ROTATION_CYCLER: "rotation"}

# --- level data keys ---------------------------------------------------------
# Only `kvynsvxbpi` was renamed by the obfuscator; it means GoalShape.
KEY_GOAL_SHAPE = "kvynsvxbpi"
KEY_GOAL_COLOR = "GoalColor"
KEY_GOAL_ROTATION = "GoalRotation"
KEY_START_SHAPE = "StartShape"
KEY_START_COLOR = "StartColor"
KEY_START_ROTATION = "StartRotation"
KEY_STEP_COUNTER = "StepCounter"
KEY_STEPS_DECREMENT = "StepsDecrement"
KEY_FOG = "Fog"

# --- `Ls20` attributes (upstream ls20.py:1765+) ------------------------------
ATTR_PLAYER = "gudziatsk"          # the player Sprite
ATTR_SHAPE_INDEX = "fwckfzsyc"     # carried shape index, 0..5
ATTR_COLOR_INDEX = "hiaauhahz"     # carried colour index into COLORS
ATTR_ROTATION_INDEX = "cklxociuu"  # carried rotation index into ROTATIONS
ATTR_GOAL_PADS = "plrpelhym"
ATTR_GOAL_SOLVED = "lvrnuajbl"
ATTR_GOAL_SHAPE_INDEX = "ldxlnycps"
ATTR_GOAL_COLOR_INDEX = "yjdexjsoa"
ATTR_GOAL_ROTATION_INDEX = "ehwheiwsk"
ATTR_LIVES = "aqygnziho"           # 3 per level
ATTR_SPAWN_X = "ltwrkifkx"
ATTR_SPAWN_Y = "zyoimjaei"
ATTR_CELL_W = "gisrhqpee"          # == player.width == 5, the movement quantum
ATTR_CELL_H = "tbwnoxqgc"
ATTR_FOG = "oeuabekjf"
ATTR_LAUNCHERS = "hasivfwip"
ATTR_PATROLLERS = "wsoslqeku"
ATTR_ACTIVE_ANIMATIONS = "euemavvxz"
ATTR_REJECT_FLASH = "akoadfsur"
ATTR_DEATH_FLASH = "ebfuxzbvn"
ATTR_STEP_HUD = "_step_counter_ui"  # not obfuscated
ATTR_SHAPES = "ijessuuig"
ATTR_COLORS = "tnkekoeuk"
ATTR_ROTATIONS = "dhksvilbb"

# --- `Ls20` methods ----------------------------------------------------------
METHOD_MATCHES_GOAL = "bejndxqqzf"       # ls20.py:2039, the 3-way triple equality
METHOD_TRY_COMPLETE = "pbznecvnfr"       # ls20.py:2042
METHOD_APPLY_CELL_EFFECTS = "txnfzvzetn"  # ls20.py:1871, returns (blocked, hit_refill)
METHOD_RESET_CARRIED = "qetwzqzzik"      # ls20.py:2016

# --- step-budget HUD (`hbuhvkxlhc`) -----------------------------------------
HUD_MAX_STEPS = "osgviligwp"
HUD_STEP_COST = "efipnixsvl"
HUD_CURRENT_STEPS = "current_steps"      # not obfuscated
HUD_CONSUME = "mfyzdfvxsm"               # returns False once exhausted
HUD_REFILL = "nzukewekzr"
HUD_SET_STEPS = "kbkdzqocik"

# --- fixed palettes (upstream ls20.py:1771-1778) -----------------------------
COLORS = (12, 9, 14, 8)
ROTATIONS = (0, 90, 180, 270)
SHAPE_SPRITES = ("gngifvjddu", "fywfjzkxlm", "mkfbgalsbe", "nnjhdcanjk", "grcpfuizfp", "ubspnhafvq")
SHAPE_COUNT = len(SHAPE_SPRITES)
COLOR_COUNT = len(COLORS)
ROTATION_COUNT = len(ROTATIONS)

# --- geometry ----------------------------------------------------------------
# Frames are 64x64. The player is 5x5 and moves one whole player-width per
# action, so play happens on a 12x12 lattice inset by the HUD chrome:
# pixel_x = X_ORIGIN + 5 * column, pixel_y = Y_ORIGIN + 5 * row.
CELL = 5
X_ORIGIN = 4
Y_ORIGIN = 0
GRID_COLS = 12
GRID_ROWS = 12
FRAME_SIZE = 64

# --- actions -----------------------------------------------------------------
# Upstream ls20.py:1943-1953. available_actions is [1, 2, 3, 4]; there is no
# ACTION5/6/7 and no click action in this game.
ACTION_IDS = (1, 2, 3, 4)
ACTION_NAMES = ("up", "down", "left", "right")
ACTION_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))  # (dx, dy) in cells


def cell_to_pixel(col, row):
    return X_ORIGIN + CELL * col, Y_ORIGIN + CELL * row


def pixel_to_cell(x, y):
    return (x - X_ORIGIN) // CELL, (y - Y_ORIGIN) // CELL
