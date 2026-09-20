"""Readable names for the obfuscated TN36 source.

Mappings refer to ``third_party/arc3_games/tn36.py`` in the pinned source
identified by ``games.json`` and ``PROVENANCE.md``.
"""

SOURCE_ID = "tn36-ef4dde99"
SOURCE_FILE = "third_party/arc3_games/tn36.py"

# Public controls (Tn36.__init__, lines 2595-2603).
ACTION_RESET = 0
ACTION_CLICK = 6
ACTION_IDS = (ACTION_CLICK,)
FRAME_SIZE = 64

# Sprite-tag enum ipqpjaszxy (lines 1735-1751).
TAG_PANEL = "plljmx"
TAG_GRID = "grsysj"
TAG_PROGRAM = "takfnb"
TAG_ACTOR = "bltjrl"
TAG_WALL = "chrccc"
TAG_GATE_BODY = "laycmuofkkgm"
TAG_GATE = "laycmuommm"
TAG_PLATFORM = "wauzms"
TAG_TARGET = "taptxx"
TAG_PROGRAM_CELL = "inwola"
TAG_BIT = "Maidxz"
TAG_LOCKED = "reooao"
TAG_SELECTOR = "tozzsf"
TAG_RUN = "sucqgk"
TAG_TIMER = "sthpyh"
TAG_TIMER_BACKGROUND = "sthpyhbaatdv"

# Engine field mappings.
ATTR_CONTROLLER = "fdksqlmpki"       # Tn36.on_set_level, line 2608
ATTR_LEFT_PANEL = "mvqheosngn"       # ytkjoffamq.__init__, line 2465
ATTR_GOAL_PANEL = "bzirenxmrg"       # ytkjoffamq.__init__, line 2466
ATTR_ACTIVE = "pxbksnibsu"           # execution queue, lines 2449, 2535-2540
ATTR_SELECTORS = "miytdaqzei"        # program selectors, lines 2457, 2467-2473
ATTR_ACTOR = "htntnzkbzu"            # dimsufvezo.__init__, line 2158
ATTR_TARGET = "aqszntqeae"           # dimsufvezo.__init__, line 2159
ATTR_PROGRAM = "vupcwzjtxu"          # dimsufvezo.__init__, line 2165
ATTR_RUN = "sxhtkytekm"              # dimsufvezo.__init__, line 2163
ATTR_WALLS = "bizgpiltwm"             # dimsufvezo.__init__, line 2157
ATTR_GATES = "ekdwmirldx"             # dimsufvezo.__init__, line 2160
ATTR_PLATFORMS = "wgzwawbgew"         # dimsufvezo.__init__, line 2161
ATTR_INITIAL_X = "fwrnsvyvrz"         # dimsufvezo.mnvoffrbex, lines 2292-2297
ATTR_INITIAL_Y = "bmhxacplut"
ATTR_INITIAL_ROTATION = "qixyeojolu"
ATTR_INITIAL_SCALE = "fpofcohbab"
ATTR_INITIAL_COLOR = "nzmblccilq"
ATTR_PROGRAM_GROUPS = "rzmeklhluf"    # dalucpicjf.__init__, lines 1994-2010
ATTR_BITS = "sonocxtjtj"              # yhijbsukht, lines 1952-1967
ATTR_BIT_ON = "pyxifyfnne"            # qjdkesagtv, lines 1919-1948
ATTR_PROGRAM_LOCKED = "rpqwgvzwdv"    # dalucpicjf, lines 1986-2016

# Program opcodes (dimsufvezo.okllwtboml, lines 2168-2188).
STEP = 4
OPCODE_EFFECTS = {
    0: ("noop", 0),
    1: ("dx", -4),
    2: ("dx", 4),
    3: ("dy", 4),
    5: ("rotation", 90),
    6: ("rotation", -90),
    7: ("rotation", 180),
    8: ("scale", 1),
    9: ("scale", -1),
    10: ("dx", 8),
    11: ("dx", 8),
    12: ("dx", -8),
    13: ("dx", -8),
    14: ("color", 9),
    15: ("color", 8),
    16: ("rotation", 270),
    33: ("dy", -4),
    34: ("dx", -4),
    63: ("color", 15),
}
