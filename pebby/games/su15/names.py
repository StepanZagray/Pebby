"""Readable aliases for identifiers in the immutable vendored SU15 source.

Line references below are to ``third_party/arc3_games/su15.py``. The source
hash is pinned in ``third_party/arc3_games/PROVENANCE.md``.
"""

DISPLAY = 64
ACTION_CLICK = 6
ACTION_UNDO = 7
AVAILABLE_ACTIONS = (ACTION_CLICK, ACTION_UNDO)

# Sprite roles (upstream constants at lines 15-34; sprite definitions 36-603).
TAG_FRUIT = "zmlxwcvwb"
TAG_TARGET = "xkstxyqbs"
TAG_DECORATION = "rgjznrcin"
TAG_TUTORIAL = "ooutlqdaq"
TAG_ENEMY_1 = "ybnveypak"
TAG_ENEMY_2 = "ybnveypak2"
TAG_ENEMY_3 = "ybnveypak3"
ENEMY_TAGS = (TAG_ENEMY_1, TAG_ENEMY_2, TAG_ENEMY_3)

SPRITE_TARGET = "0008hngjwqibfi"
SPRITE_BACKGROUND = "0009mbvjylwely"
SPRITE_FRUIT_PROGRESSION = "0012qpdeinaukn"
SPRITE_ENEMY_PROGRESSION = "0017dcrmyjphec"
SPRITE_TUTORIAL = "0019oikveatxnp"
SPRITE_TUTORIAL_ARROW = "0020qkplcbivxi"

ENEMY_SPRITES = {
    1: TAG_ENEMY_1,
    2: TAG_ENEMY_2,
    3: TAG_ENEMY_3,
}
ENEMY_REQUIREMENT_KEYS = {
    1: "0030xjmmfvfpqm",
    2: "0031xcwudgivus",
    3: "0032qekmtelwqi",
}

# Engine fields initialized at upstream lines 918-967 and populated 997-1038.
ATTR_FRUITS = "lkujttxgs"
ATTR_ENEMIES = "fezhhzhih"
ATTR_TARGETS = "powykypsm"
ATTR_FRUIT_TIERS = "kqywaxhmsb"
ATTR_HISTORY = "nscnqkkvg"
ATTR_ANIMATING = "vsfwpngmx"
ATTR_ANIMATION_MODE = "qygchysnh"

# Game methods: action resolution is upstream lines 1043-1133.
METHOD_RESOLVE_CLICK = "axaxyjxqoe"       # choose nearby pieces, lines 1135-1215
METHOD_CLEAR_MOTION = "jdphqevryx"        # clear transient motion, lines 1217-1246
METHOD_MOVE_TICK = "pkrdtzfrth"           # four-tick magnetic movement, lines 1505-1571
METHOD_REFRESH_PIECES = "zmcyxbptrw"       # live fruit/enemy lists, lines 1840-1850
METHOD_CENTER = "jdeyppambj"               # integer sprite center, lines 1852-1861
METHOD_WITHIN_RADIUS = "kcqeohsztd"        # rect distance selection, lines 1888-1915
METHOD_POINT_INSIDE = "jnieciwfsv"         # target containment, lines 2014-2031
METHOD_COMPLETE = "cbdhpcilgb"             # exact target counts, lines 2033-2081
METHOD_SAVE_UNDO = "hlsimbcfgc"            # fruit/enemy snapshot, lines 2083-2092
METHOD_UNDO = "vczehveskr"                 # position/tier restore, lines 2094-2141
METHOD_VALUE_REMAINS = "xspiwkmfbs"        # required-value feasibility, line 2143+

KEY_REQUIREMENTS = "xkstxyqbs"
KEY_STEPS = "steps"
KEY_GENERATED = "__pebby_su15_generated_v1"
GENERATED_KIND = "full-nine-tier-v3"

PLAY_MIN_Y = 10
PLAY_MAX_Y = 62
# Upstream movement bounds/constants are lines 899-904; radius selection is
# implemented at 1888-1904 and target-center completion at 2033-2081.
MAGNET_RADIUS = 8
ROUTE_STEP = 6
