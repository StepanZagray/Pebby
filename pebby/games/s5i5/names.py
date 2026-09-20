"""Readable names for S5I5's obfuscated identifiers.

Upstream ships ``third_party/arc3_games/s5i5.py`` with randomised class,
method, attribute and sprite names. The file is left byte-identical; this
module is the translation table, so nothing else in Pebby hardcodes a random
string. Every entry was read out of the upstream source; the line numbers
cited are upstream line numbers.

The game: rods (3-pixel-wide bars with a colour-3 "cap" line at their base)
carry pins. Clicking a rail extends or retracts every rod of the rail's
colour; clicking a button rotates every rod of the button's colour by 90
degrees about its base. Linked pieces (``Children`` level data, plus any pin
overlapping a rod at level start) move rigidly with their parent. A move that
makes two rods overlap is reverted on the next engine frame. The level is won
when every target has a pin at exactly its coordinates; the game is lost when
the click budget (``StepCounter``) runs out. Only ACTION6 (click) exists.
"""

# --- sprite tags (upstream `Sprite.tags`) ------------------------------------
TAG_ROD = "0001qwdmnlybkb"      # s5i5.py:2054; also frames/obstacles, which are immobile rods
TAG_PIN = "0064ocqkuqacti"      # s5i5.py:2055
TAG_RAIL = "0066ghlkyvdbgg"     # s5i5.py:2051; extend/retract control
TAG_TARGET = "0087vvmblxkzdi"   # s5i5.py:2081
TAG_BUTTON = "0089rvqdprjwpz"   # s5i5.py:2195; rotate control

# --- level data keys (not obfuscated) ----------------------------------------
KEY_STEP_COUNTER = "StepCounter"   # s5i5.py:2076
KEY_CHILDREN = "Children"          # s5i5.py:2065, list of [parent_name, child_name]

# --- module constants (s5i5.py:1942-1949) ------------------------------------
BACKGROUND_COLOR = 5
PADDING_COLOR = 3
ROD_THICKNESS = 3       # vjqemmsfmx, s5i5.py:1944; one rod "index" unit
CAP_COLOR = 3           # fbnsnwoblu, s5i5.py:1946; the base line that encodes rotation
HUD_FULL_COLOR = 3      # s5i5.py:1980, step bar on row 63
HUD_EMPTY_COLOR = 4     # s5i5.py:1981

# --- `S5i5` attributes (s5i5.py:2027-2033) -----------------------------------
ATTR_STEP_HUD = "gwiuiwqizb"      # the RenderableUserDisplay budget bar
ATTR_RAIL_RODS = "pigtralzpb"     # rail sprite -> rods it controls (colour match)
ATTR_CHILDREN = "uricqfoplr"      # rod sprite -> set of sprites moving with it
ATTR_BACKUP = "whoonmfbnp"        # sprite -> clone; non-empty means "revert next frame"

# --- `S5i5` methods ----------------------------------------------------------
METHOD_RESET_BUDGET = "zhrkjlkeib"   # s5i5.py:2074
METHOD_IS_WON = "neurwiqfry"         # s5i5.py:2080, every target has a pin at its x,y
METHOD_MOVE_TREE = "uiqzouvdxd"      # s5i5.py:2088
METHOD_BACKUP_TREE = "dxlryikffn"    # s5i5.py:2094
METHOD_ROTATION_OF = "gnpdxxlhrp"    # s5i5.py:2100, reads the cap line: 0/90/180/270
METHOD_SET_LENGTH = "nkkhgerxvq"     # s5i5.py:2110, (rod, index) rebuilds pixels, moves children
METHOD_ANY_OVERLAP = "qownxibuiy"    # s5i5.py:2138, rod-vs-rod pixel collision
METHOD_ROTATE_CHILD = "eeyirqljyp"   # s5i5.py:2145
METHOD_ROTATE_ROD = "bhgumdfgqr"     # s5i5.py:2154

# --- step-budget HUD (`gslihflgok`, s5i5.py:1952) ----------------------------
HUD_MAX_STEPS = "dazkjahdqd"
HUD_CURRENT_STEPS = "current_steps"  # not obfuscated
HUD_CONSUME = "twwpyjzobt"           # s5i5.py:1964, decrements (floor 0), True while > 0
HUD_REFILL = "yhkntwvjfr"            # s5i5.py:1970

# --- rotation encoding (s5i5.py:2100-2108) -----------------------------------
# rotation 0: cap on the bottom row, rod extends upward from its base;
# 90: cap on the left column, extends right; 180: cap on top, extends down;
# 270: cap on the right column, extends left. A rod with no cap reads as 270.
# Rotating (s5i5.py:2154-2179) applies np.rot90 to the pixels, so the cycle is
# 0 -> 270 -> 180 -> 90 -> 0 (counter-clockwise on screen).
ROTATIONS = (0, 90, 180, 270)
# (dx, dy) per unit from base towards tip, indexed by rotation.
EXTEND_DIRECTION = {0: (0, -1), 90: (1, 0), 180: (0, 1), 270: (-1, 0)}

# --- click semantics (s5i5.py:2181-2247) -------------------------------------
# Every click consumes one budget step, even a click on nothing. A button is
# looked up first, then a rail. On a rail the click offset along its long axis
# is compared with half its length: greater extends every controlled rod by one
# unit, smaller retracts (never below one unit), equal does nothing. A rail
# controls every rod whose colour (pixels[1, 1]) appears anywhere in the rail's
# pixels (s5i5.py:2058), so rod colours must avoid the rail chrome colours.
RAIL_CHROME_COLORS = (2, 4, 3)
PIN_COLOR = 13
FRAME = 64

# --- actions -----------------------------------------------------------------
ACTION_RESET = 0
ACTION_CLICK = 6
ACTION_IDS = (ACTION_CLICK,)   # upstream available_actions=[6], s5i5.py:2042
