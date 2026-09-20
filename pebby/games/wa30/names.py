"""Readable names for WA30's obfuscated identifiers.

Upstream ships ``third_party/arc3_games/wa30.py`` with randomised class, method,
attribute and sprite names. The file is left byte-identical; this module is the
translation table, so the rest of the package never hardcodes a random string.

Every entry was read directly out of the upstream source; the line numbers cited
are upstream line numbers.

The game in one paragraph: the player (a 4x4 sprite) walks on a 64x64 frame in
4-pixel steps, i.e. on a 16x16 cell lattice. ACTION1-4 move up/down/left/right;
ACTION5 grabs the box the player is facing (or drops the box being held, or
destroys a thief being faced). A held box keeps its offset from the player and
is dragged rigidly. Helper robots grab loose boxes and drag them into the goal
region; thieves grab boxes and drag them into a "bad" region and can only be
stopped by ACTION5. Every action costs one step of the level's ``StepCounter``.
The level is complete when every box sits inside a goal region and nobody is
holding it; the game is lost when the steps run out first.
"""

# --- sprite prototypes (upstream ``sprites`` dict, wa30.py:34-360) ------------
SPRITE_PLAYER = "wppuejnwhl"        # wa30.py:316, 4x4, tag TAG_PLAYER, layer 1
SPRITE_BOX = "pktgsotzmw"           # wa30.py:251, 4x4, tag TAG_BOX, layer 1
SPRITE_HELPER = "byigobxzpg"        # wa30.py:62, 4x4 colour 12, tag TAG_HELPER
SPRITE_THIEF = "jqzhxgbmtz"         # wa30.py:162, 4x4 colour 15, tag TAG_THIEF
SPRITE_WALL = "uasmnkbzmm"          # wa30.py:276, 4x4, collidable, tag TAG_WALL
SPRITE_FENCE = "pmargquscu"         # wa30.py:264, 4x4, not collidable, tag TAG_FENCE
SPRITE_BIG_WALL_A = "aidclcbjcv"    # wa30.py:35, 64x20 colour 4, collidable, no tag
SPRITE_BIG_WALL_B = "cwefnfvjhr"    # wa30.py:75, 64x16 colour 4, collidable, no tag
# Goal regions: border colour 9, interior colour 2, sizes in pixels (w x h).
SPRITE_GOALS = {
    (4, 4): "xxmzyqktqy",   # wa30.py:349
    (4, 8): "ofwegeqknn",   # wa30.py:199
    (8, 4): "wkmuwhjqyo",   # wa30.py:304
    (8, 8): "ghklglzjuf",   # wa30.py:134
    (8, 12): "doijajrgdi",  # wa30.py:98
    (8, 16): "ktghqrydvd",  # wa30.py:175
    (12, 4): "jigtxgzhwt",  # wa30.py:150
    (12, 12): "peimznrlqd", # wa30.py:231
    (16, 8): "vikkhnsrzd",  # wa30.py:288
}
# Bad regions (solid colour 2), sizes in pixels.
SPRITE_BAD = {
    (4, 8): "geffskzhqq",   # wa30.py:118
    (8, 8): "ooaamfpvqr",   # wa30.py:215
    (12, 12): "xqaqifquaw", # wa30.py:329
}

# --- sprite tags -------------------------------------------------------------
TAG_PLAYER = "wbmdvjhthc"
TAG_BOX = "geezpjgiyd"
TAG_HELPER = "kdweefinfi"   # drags loose boxes into goal regions (wa30.py:1142)
TAG_THIEF = "ysysltqlke"    # drags boxes into bad regions (wa30.py:1171)
TAG_WALL = "debyzcmtnr"
TAG_FENCE = "bnzklblgdk"    # actors cannot enter; a dragged box can (wa30.py:996-1001)
TAG_GOAL = "fsjjayjoeg"     # every pixel of the sprite is a goal pixel (wa30.py:915-919)
TAG_BAD = "zqxwgacnue"      # every pixel of the sprite is a bad pixel (wa30.py:920-924)

# --- level data keys ---------------------------------------------------------
KEY_STEP_COUNTER = "StepCounter"   # wa30.py:966, actions allowed per level

# --- ``Wa30`` attributes (wa30.py:875-883) -----------------------------------
ATTR_STEP_HUD = "kuncbnslnm"       # the step-budget HUD (class at wa30.py:772)
ATTR_HELD_BY = "nsevyuople"        # dict holder Sprite -> box Sprite
ATTR_HOLDER_OF = "zmqreragji"      # dict box Sprite -> holder Sprite
ATTR_OBSTACLES = "pkbufziase"      # set of (x, y): collidable sprite origins + frame border
ATTR_GOAL_PIXELS = "wyzquhjerd"    # set of (x, y) pixels covered by goal sprites
ATTR_BAD_PIXELS = "lqctaojiby"     # set of (x, y) pixels covered by bad sprites
ATTR_FENCES = "qthdiggudy"         # set of (x, y) fence origins
ATTR_HELPER_TARGETS = "lkvghqfwan" # cells next to loose, off-goal boxes (stale between grabs)
ATTR_THIEF_TARGETS = "uuorgjazmj"  # cells next to boxes not held by a thief, off bad

# --- ``Wa30`` methods --------------------------------------------------------
METHOD_RECOMPUTE_HELPER_TARGETS = "vyltpasvhc"  # wa30.py:934
METHOD_RECOMPUTE_THIEF_TARGETS = "lgirylubbp"   # wa30.py:949
METHOD_RESET_STEPS = "xcuqvqnmiu"               # wa30.py:964
METHOD_MOVE_ACTOR_TO = "wqwsvmhhzj"             # wa30.py:970
METHOD_MOVE_ACTOR_BY = "qnmfimgpwc"             # wa30.py:987, sets rotation when not holding
METHOD_CELL_FREE = "kblzhbvysd"                 # wa30.py:993, not obstacle and not fence
METHOD_PAIR_CAN_MOVE = "fuykgiiwit"             # wa30.py:996
METHOD_IN_GOAL = "shbxbhnhjc"                   # wa30.py:1003
METHOD_IN_BAD = "ahzqkfjpsc"                    # wa30.py:1006
METHOD_HELPER_PATH_TO_BOX = "czrprbohhe"        # wa30.py:1009
METHOD_HELPER_PATH_TO_GOAL = "cyjrduhzmz"       # wa30.py:1031
METHOD_THIEF_PATH_TO_BOX = "zauouvdhta"         # wa30.py:1056
METHOD_THIEF_PATH_TO_BAD = "egqayvffim"         # wa30.py:1078
METHOD_ATTACH = "xpcvspllwr"                    # wa30.py:1103, steals from a previous holder
METHOD_DETACH = "kqrtstlzkg"                    # wa30.py:1111
METHOD_RECOLOUR = "zzppkjnqgk"                  # wa30.py:1119, purely visual
METHOD_HELPERS_ACT = "ynmgxjqkgh"               # wa30.py:1142
METHOD_BOX_HELD_BY_THIEF = "jrrltylxpp"         # wa30.py:1165
METHOD_THIEVES_ACT = "aoeyzovteg"               # wa30.py:1171
METHOD_LEVEL_COMPLETE = "ymzfopzgbq"            # wa30.py:1194
METHOD_AFTER_ACTION = "dhrikuybfo"              # wa30.py:1198: helpers, thieves, recolour
METHOD_APPLY_ACTION = "yygfcvqoyx"              # wa30.py:1203

# --- step-budget HUD (``etuniyewsy``, wa30.py:772-805) -----------------------
HUD_MAX_STEPS = "dbdarsgrbj"
HUD_CURRENT_STEPS = "current_steps"  # not obfuscated
HUD_SET_STEPS = "uwwwedmhqv"
HUD_CONSUME = "pfakmupgbr"           # decrements to a floor of 0, wa30.py:784
HUD_REFILL = "ububboesmh"
# The HUD paints pixel row 63: colour 7 for remaining budget, 4 for spent.
HUD_ROW = 63

# --- module helpers ----------------------------------------------------------
FUNC_MANHATTAN = "hbipqrhvbm"        # wa30.py:834
FUNC_ADJACENT = "mdbwmdaxuu"         # wa30.py:838, manhattan == CELL
FUNC_FACING = "vwiozbtqgi"           # wa30.py:842, sprite sits one cell ahead of an actor
FUNC_SET_BORDER_COLOUR = "uxricavavq"  # wa30.py:853
FUNC_DIRECTION_TO_ROTATION = "pjedoipwee"  # wa30.py:860
FUNC_POINT_IN_SPRITE = "anojofkynf"  # wa30.py:870

# --- constants (wa30.py:760-769) ---------------------------------------------
BACKGROUND_COLOR = 1
PADDING_COLOR = 0
CELL = 4                       # celomdfhbh: movement quantum in pixels
BOX_BORDER_IDLE = 4            # wvrpthjfsv
BOX_BORDER_FACED = 3           # vlzkmytlgh
BOX_BORDER_HELD_BY_PLAYER = 0  # hgqlwjikqr
BOX_BORDER_HELD_BY_ROBOT = 5   # qrmfayeqpo
THIEF_BORDER_IDLE = 15         # ansrsmsvzs
THIEF_BORDER_FACED = 11        # lajjveeqzb
GOAL_BORDER_COLOR = 9
GOAL_FILL_COLOR = 2
BAD_FILL_COLOR = 2
WALL_COLOR = 4

# --- geometry ----------------------------------------------------------------
FRAME_SIZE = 64
GRID_COLS = FRAME_SIZE // CELL   # 16
GRID_ROWS = FRAME_SIZE // CELL   # 16

# --- actions -----------------------------------------------------------------
# wa30.py:1203-1246. available_actions is [1, 2, 3, 4, 5]; there is no click.
ACTION_IDS = (1, 2, 3, 4, 5)
MOVE_ACTIONS = (1, 2, 3, 4)
ACTION_GRAB = 5
ACTION_NAMES = {1: "up", 2: "down", 3: "left", 4: "right", 5: "grab"}
ACTION_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}  # (dx, dy) in cells
# pjedoipwee (wa30.py:860): dy<0 -> 0, dx>0 -> 90, dy>0 -> 180, else 270.
DIRECTION_ROTATION = {(0, -1): 0, (1, 0): 90, (0, 1): 180, (-1, 0): 270}
# vwiozbtqgi (wa30.py:842): the cell an actor with this rotation is facing.
ROTATION_FACING = {0: (0, -1), 90: (1, 0), 180: (0, 1), 270: (-1, 0)}


def cell_to_pixel(col, row):
    return CELL * col, CELL * row


def pixel_to_cell(x, y):
    return x // CELL, y // CELL
