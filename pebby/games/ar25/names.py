"""Readable aliases for the vendored AR25 identifiers."""

DISPLAY = 64
GRID = 21

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_CYCLE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7
AVAILABLE_ACTIONS = (1, 2, 3, 4, 5, 6, 7)

# Sprite roles, defined upstream around lines 59-714.
TAG_GOAL = "0001sruqbuvukh"
TAG_MIRROR = "0003uqrdzdofso"
TAG_MOVABLE = "0006lxjtqggkmi"
TAG_FIXED = "0056icpryeujyf"
TAG_VERTICAL_MIRROR = "0054kgxrvfihgm"
TAG_HORIZONTAL_MIRROR = "0002nuguepuujf"
TAG_ROTATE_VERTICAL = "0040bwgtiqvhtu"
TAG_ROTATE_HORIZONTAL = "0044qlxgcpzowy"
TAG_REFLECT_HORIZONTAL_ONLY = "reflect_horizontal_only"
TAG_REFLECT_VERTICAL_ONLY = "0038pnuzypawco"

SPRITE_GOAL = "0001sruqbuvukh"
SPRITE_OFFICIAL_L = "0007arvfmhagbj"
SPRITE_FIXED_VERTICAL = "0055nwhypaamix"

# Ar25 fields initialized upstream around lines 1264-1385.
ATTR_STEP_INTERFACE = "lelsvjlwneo"
ATTR_HISTORY = "flqblmrxsla"
ATTR_MOVABLES = "ouurgkpbbjj"
ATTR_GOALS = "fswikrcrdmx"
ATTR_MIRRORS = "jtkyjqznbnp"
ATTR_SELECTED = "yvifanjrcyu"
ATTR_PENDING_WIN = "hujpxmlafgh"

KEY_STEPS = "StepCounter"
KEY_GENERATED = "__pebby_ar25_generated_v1"
GENERATED_KIND = "full-reflection-curriculum-v2"

TRANSPARENT = -1
