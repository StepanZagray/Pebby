"""Readable names for DC22's shipped mechanics.

The vendored game intentionally uses obfuscated identifiers.  These mappings
were read from ``third_party/arc3_games/dc22.py``; the source line references
make it possible to audit the adapter without changing that pinned source.

DC22 is a two-pixel-step path puzzle.  Arrow actions move the 2x2 player over
intangible support sprites, ACTION6 clicks display pixels, and reaching the
2x2 goal advances the native game (dc22.py:10646-10892).  Clickable colour
controls cycle same-colour ``tovemc`` sprites (dc22.py:10663-10708).  The six
official levels add keys/unlocked controls, fall rollback with a 20-step
penalty, paired bridges, long cycling surfaces, movable crushers, object or
bridge carrying, pressure-revealed controls, and bridge colour cycling.
"""

# Actions: class declaration and dispatch at dc22.py:9966-9971, 10646-10663.
ACTION_RESET = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_CLICK = 6
ACTION_IDS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_CLICK)
MOVE_ACTIONS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)
MOVE_DELTAS = {
    ACTION_UP: (0, -2),
    ACTION_DOWN: (0, 2),
    ACTION_LEFT: (-2, 0),
    ACTION_RIGHT: (2, 0),
}

FRAME_SIZE = 64
MOVE_STEP = 2  # ``ndiyvmxxey`` at dc22.py:9875, used at 10866.

# Level data and prototypes: dc22.py:9625-9871 and 4092-4506.
KEY_STEPS = "StepCounter"
KEY_GENERATED_FORMAT = "PebbyDC22Format"
KEY_GENERATED_SPEC = "PebbyDC22Spec"
SPRITE_PLAYER = "plflho1"       # dc22.py:4267-4274
SPRITE_GOAL = "goknoi"          # dc22.py:4092-4098
SPRITE_FLOOR = "tacugo-plelvb"  # dc22.py:4465-4471

# Semantic tags consumed by Dc22.on_set_level/step.
TAG_PLAYER = "jfva"        # player lookup at dc22.py:10018
TAG_GOAL = "goknoi"        # goal lookup at dc22.py:10017
TAG_TOGGLE = "tovemc"      # cycled surfaces at dc22.py:10019-10067
TAG_BUTTON = "buezna"      # click target at dc22.py:10674
TAG_CLICK = "sys_click"    # click controls at dc22.py:10709
TAG_GATE_KEY = "piyqze"    # key pickup at dc22.py:10871-10879
TAG_CRUSHER = "crzsjq"     # crusher setup at dc22.py:10068-10076
TAG_BRIDGE = "tewfut"      # bridge/color objects at dc22.py:10467-10502
TAG_BRIDGE_OBJECT = "grawwq-object"  # carried object at dc22.py:10073
TAG_FALL_BLOCKER = "vcha"  # excluded as support at dc22.py:10411
TAG_PRESSURE = "njvd-rolo"  # pressure control reveal at dc22.py:10247-10287
TAG_BRIDGE_COLOR_CYCLE = "tewfut-color-cycle"  # dc22.py:10434-10474
TAG_BRIDGE_COLOR_BUTTON = "tewfut-color-buezna"  # dc22.py:10682-10684

# Dc22 fields/methods: dc22.py:9960, 9989-10019, 10293-10449.
ATTR_HUD = "ujotjblwn"
ATTR_GOAL = "hfuqkxulm"
ATTR_PLAYER = "qnnpcoyzd"
ATTR_UNDO = "sachklrxui"
ATTR_FALLING = "guspipewt"
ATTR_CRUSHER_MOVING = "fadccmsnb"
ATTR_CRUSHER_ANIMATING = "fjiyimenq"
METHOD_HIT_VISIBLE = "xodizggcom"

# HUD fields/methods: dc22.py:9890-9933.
HUD_MAX_STEPS = "hethagvldv"
HUD_CURRENT_STEPS = "current_steps"

FORMAT = "pebby.dc22.level.v2"
