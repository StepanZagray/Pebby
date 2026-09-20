"""Readable names for TU93's obfuscated identifiers.

Upstream ships ``third_party/arc3_games/tu93.py`` with randomised class, method,
attribute, tag and sprite names. The file is left byte-identical; this module is
the translation table, so nothing else in the package hardcodes a random string.

Every entry was read out of the upstream source; cited line numbers are upstream
line numbers.

Rules summary (upstream ``Tu93.step``, tu93.py:1194-1272):

* The maze is one big bitmap sprite (``TAG_MAZE``, layer -1). Node blocks sit at
  multiples of 6 px from the sprite origin and are 3x3 px; the 3x3 blocks between
  two nodes are passages when their pixel value is ``PASSAGE`` (2). Node blocks
  are painted ``NODE`` (0) and everything else is transparent (-1). Only the
  passage value is ever tested by the rules (tu93.py:1202-1246, 1128-1150).
* Exactly one head (``TAG_HEAD``) is driven. An action costs one step even when
  blocked (tu93.py:1202). On an open passage the head slides one node (6 px) and
  eats any hunter / patroller / tail whose centre coincides with its own
  (tu93.py:1080-1088, 1064-1078).
* Hunters (``TAG_HUNTER``) never turn. Once the head has settled exactly one
  node in front of a hunter, it lunges into the head's node and eats it
  (tu93.py:1090-1108).
* Patrollers (``TAG_PATROLLER``) slide one node per accepted action in their
  facing direction, regardless of walls (tu93.py:1110-1113), eat the head if
  they land on it (tu93.py:1115-1123) and then reverse when the next block in
  their facing direction is not a passage (tu93.py:1125-1151).
* Tails (``TAG_TAIL``) are dormant until the head settles exactly two nodes in
  front of one; that arms it with a rotation queue ``[r, r]`` (tu93.py:1171-1182).
  Each accepted head action appends the head's new rotation to every queue
  (tu93.py:1207-1209 etc.), armed tails slide one node in their current rotation
  (tu93.py:1153-1169), then every armed tail pops the queue head into its
  rotation. Net effect: the tail replays the head's path two moves behind, and
  eats the head if it ever lands on it.
* After the entities settle: win when every head stands exactly on an exit
  (``TAG_EXIT``); lose when no head is left or the step counter is 0
  (tu93.py:1256-1262).
"""

# --- sprite tags -------------------------------------------------------------
TAG_HUNTER = "0001haidilggfh"       # lunges at a head one node in front of it
TAG_MAZE = "0005uvnhiglpvh"         # the maze bitmap, layer -1
TAG_EXIT = "0015msvpvzxhqf"         # 3x3 colour-14 pad
TAG_HEAD = "0017unajnymcki"         # the player, layer 1
TAG_PATROLLER = "0020npxxteirsg"    # bounces along a corridor
TAG_TAIL = "0023otenflmryc"         # follows the head two moves behind

# --- sprite prototypes (upstream ``sprites`` dict) ---------------------------
SPRITE_MAZE_LEVEL1 = "0004iwcrtzivmj"  # tu93.py:71, 33x33 bitmap
SPRITE_EXIT = "0014mzhhvzrazi"         # tu93.py:424
SPRITE_HEAD = "0016ihgrljrgpq"         # tu93.py:435, pixels (9,4,9 / 9,9,9 / 9,9,9)
SPRITE_HUNTER = "0018rquzkxccdu"       # tu93.py:447, pixels (8,15,8 / ...)
SPRITE_PATROLLER = "0019zkgjxlirss"    # tu93.py:458, pixels (12,15,12 / ...)
SPRITE_TAIL = "0022ckngtnvkgw"         # tu93.py:480, pixels (13,15,13 / ...)
SPRITE_BURST_5 = "0002ebnnauydmr"      # tu93.py:45, eaten-entity animation frame
SPRITE_BURST_7 = "0003zgknydacap"      # tu93.py:57

# --- level data keys ---------------------------------------------------------
KEY_STEP_COUNTER = "StepCounter"     # not obfuscated

# --- maze raster --------------------------------------------------------------
BLOCK = 3            # ``hwthhtvyki``, tu93.py:909: block side in px
CELL = 6             # ``hcgctulqhn``, tu93.py:910: node pitch in px
TAIL_ARM_DISTANCE = 12  # ``anklfvjqkx``, tu93.py:911: head-to-tail arming distance
ACTIVE_MARK = 11     # ``ziedssriec``, tu93.py:912: pixels[0,1] of an armed entity
HUD_COLOR = 6        # ``wmwvcifkjo``, tu93.py:913: step bar colour on frame row 63
PASSAGE = 2          # the only pixel value the rules test
NODE = 0             # cosmetic: how official bitmaps paint reachable nodes
EMPTY = -1
BACKGROUND_COLOR = 5
MAZE_MARGIN = 3      # official level 1 places its bitmap at (3, 3) on a 39x39 grid
FRAME_SIZE = 64

# --- ``Tu93`` attributes (tu93.py:988-1000) ----------------------------------
ATTR_HUD = "ksulgrfyqx"              # step-counter interface
ATTR_PHASE = "kdkehgjrzq"            # 0 idle, 1 head sliding, 2 entities sliding
ATTR_TAIL_QUEUES = "ylmdnwbdyy"      # dict[Sprite, list[int]] of armed tails
HUD_MAX_STEPS = "yhzmaedply"         # tu93.py:954
HUD_CURRENT_STEPS = "current_steps"  # not obfuscated
HUD_CONSUME = "rndwkomrip"           # tu93.py:963
HUD_REFILL = "spnqceiiab"            # tu93.py:969

# --- ``Tu93`` methods ---------------------------------------------------------
METHOD_ALIGNED = "sueekttytc"        # tu93.py:915 module fn: both offsets % 6 == 0
METHOD_SAME_CENTRE = "bwrgmsbsrg"    # tu93.py:919 module fn
METHOD_SLIDE_PX = "erkaicaqdh"       # tu93.py:1013: one px in facing direction
METHOD_EAT = "uneirnujpq"            # tu93.py:1064: 3->5->7->removed
METHOD_FACING_AT = "wlhbetxehh"      # tu93.py:1047: target exactly d px ahead
METHOD_IS_ACTIVE = "rxdvicwstj"      # tu93.py:1050
METHOD_ARM = "qlzvpfwmqv"            # tu93.py:1041
METHOD_HEAD_PHASE = "ogedgxpgdy"     # tu93.py:1184
METHOD_HUNTERS_LUNGE = "ixnhjkzwic"  # tu93.py:1090
METHOD_PATROLLERS_START = "itwvxwpzyb"  # tu93.py:1110
METHOD_TAILS_START = "wcmxxknbpe"    # tu93.py:1153
METHOD_ENTITY_PHASE = "tnwgpjxbwe"   # tu93.py:1188
METHOD_PATROLLERS_BOUNCE = "rgwzxyjuqc"  # tu93.py:1125
METHOD_TAILS_ARM = "gmwsemdsae"      # tu93.py:1171

# --- actions -----------------------------------------------------------------
# available_actions is [1, 2, 3, 4] (tu93.py:1000). games.json lists the game as
# keyboard_click, but ``step`` has no ACTION6 branch: clicks are ignored.
ACTION_IDS = (1, 2, 3, 4)
ACTION_NAMES = {1: "up", 2: "down", 3: "left", 4: "right"}
# tu93.py:1207 / 1220 / 1233 / 1246: the rotation each action gives the head.
ACTION_ROTATION = {1: 0, 2: 180, 3: 270, 4: 90}
ROTATION_ACTION = {rot: action for action, rot in ACTION_ROTATION.items()}
# tu93.py:1013-1021 (``erkaicaqdh``): pixel delta per rotation, in cells here.
ROTATION_DELTA = {0: (0, -1), 180: (0, 1), 270: (-1, 0), 90: (1, 0)}
ACTION_DELTA = {action: ROTATION_DELTA[rot] for action, rot in ACTION_ROTATION.items()}
OPPOSITE = {0: 180, 180: 0, 90: 270, 270: 90}   # tu93.py:1135-1150


def cell_to_pixel(origin, cell):
    ox, oy = origin
    col, row = cell
    return ox + CELL * col, oy + CELL * row


def pixel_to_cell(origin, x, y):
    """Cell of an aligned sprite, or None if it is off the 6 px lattice."""
    ox, oy = origin
    dx, dy = x - ox, y - oy
    if dx % CELL or dy % CELL:
        return None
    return dx // CELL, dy // CELL
