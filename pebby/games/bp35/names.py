"""Readable names for the BP35 identifiers used by this package.

The comments point at the immutable upstream implementation.  They are kept
here so engine access in the other modules is both auditable and grep-able.
"""

DISPLAY = 64

ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_CLICK = 6
ACTION_UNDO = 7
AVAILABLE_ACTIONS = (ACTION_LEFT, ACTION_RIGHT, ACTION_CLICK, ACTION_UNDO)

# BP35 owns a second scene engine inside ARCBaseGame (upstream 4470, 4558).
ATTR_WORLD = "oztjzzyqoek"
ATTR_ACTION_COUNT = "hbqwwgceeqp"  # upstream 4476, 4520-4550
ATTR_INTERFACE = "tehvqeiqsdu"

# Custom world fields (upstream 4032-4050).
ATTR_LOGICAL_LEVEL = "qswcochjodb"
ATTR_PLAYER = "twdpowducb"
ATTR_FACING_RIGHT = "ybmkdxbdko"
ATTR_MOVE_COUNT = "wjidupyeoa"
ATTR_GRAVITY_DOWN = "vivnprldht"
ATTR_GRID = "hdnrlfmyrj"
ATTR_WORLD_WIN = "nkuphphdgrp"
ATTR_WORLD_LOSS = "jrhqdvdwpsb"
ATTR_HISTORY = "lfqkneessbf"
ATTR_HISTORY_STACK = "zogplfgbcbm"  # upstream undo implementation
ATTR_CAMERA = "pevrvnrfxnw"
ATTR_CAMERA_OFFSET = "rczgvgfsfb"
ATTR_PENDING_FRAMES = "frames_to_render"

# Grid/entity API (upstream 2592-2660, 2932-3063).
ATTR_ENTITIES = "ugywcmguyv"
METHOD_ENTITIES_NAMED = "wwkbcxznzg"

WALL = "xcjjwqfzjfe"
PLAYER_RIGHT = "player_right"
PLAYER_LEFT = "player_left"
GEM = "fjlzdjxhant"
HAZARD_A = "aknlbboysnc"
HAZARD_B = "jcyhkseuorf"
SPIKE_A = "ubhhgljbnpu"
SPIKE_B = "hzusueifitk"
DESTRUCTIBLE = "qclfkhjnaac"
GROWER = "etlsaqqtjvn"
BRIDGE_SOLID = "yuuqpmlxorv"
BRIDGE_OPEN = "oonshderxef"
GRAVITY_SWITCH = "lrpkmzabbfa"

LEVEL_GRID_DATA = "__pebby_bp35_grid_v1"
GENERATED_KIND = "full-platform-v2"

LEGEND = {
    "o": [WALL],
    "n": [PLAYER_RIGHT],
    "+": [GEM],
    "x": [DESTRUCTIBLE],
    "y": [GROWER],
    "1": [BRIDGE_SOLID],
    "2": [BRIDGE_OPEN],
    "g": [GRAVITY_SWITCH],
    "v": [SPIKE_A],
    "u": [SPIKE_B],
    "m": [HAZARD_A],
    "w": [HAZARD_B],
}
GROUPS = [[WALL], [HAZARD_A], [HAZARD_B], [SPIKE_A], [SPIKE_B]]

# Logical-level action-bar loss thresholds (upstream 4411-4459).
def native_action_limit(logical_level):
    if logical_level <= 6:
        return 64
    if logical_level <= 9:
        return 128
    return 192
