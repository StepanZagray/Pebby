"""Readable names for R11L's obfuscated identifiers.

Upstream ships ``third_party/arc3_games/r11l.py`` with randomised class, method
and attribute names. The file is left byte-identical; this module is the
translation table, so the rest of the package never hardcodes a random string.

Every entry was read directly out of the upstream source; the line numbers
cited are upstream line numbers of that file.

How the game works (upstream ``R11l``):

* Sprites are grouped by the suffix of their name. ``roefwu-<g>`` is the
  group's *core* (a 5x5 diamond), ``roefwulewcui-<g>`` are its draggable
  *fragments* (5x5 diamonds), ``flkdtg-<g>`` is its *target* (a 7x7 diamond
  whose interior is colour -2: invisible, but collidable).
* The core is never dragged. It sits at the integer centroid of its fragments
  (``rvkbignsyr``, r11l.py:1536) and is re-centred after every drag.
* The only action is ACTION6 (click). A click inside a fragment's bounding box
  selects it; any other click drags the selected fragment so that its centre
  lands on the click, provided the fragment would not pixel-collide with a
  wall (``wakneh-*``) at the destination (``gabrtablhx``, r11l.py:1553). Only
  the destination is checked, never the path.
* After a drag, if any core pixel-collides with a hazard (``defgjl*``) the
  drag is reverted and a hazard counter increments; five hazards lose the
  game (r11l.py:1733-1756).
* A level is complete when, after a drag, every group that has a target
  (group names containing ``dirwzt`` are decoys and skipped) has its core
  pixel-colliding with its target (r11l.py:1757-1782).
* Levels 5-6 add ``puukul-*`` pickups absorbed into ``roefwu-whkxtx*`` cores.
  Pickup pixel patterns overwrite matching core pixels; otherwise-coreless
  targets require both collision and exact positive-colour-set equality
  (r11l.py:1585-1617, 1769-1780).
* 60 actions per level are counted; the 60th action loses (r11l.py:1407,
  1807-1811), so 59 are usable.
"""

# --- sprite name prefixes (r11l.py:1448-1462, 1473, 1509, 1733) --------------
PREFIX_CORE = "roefwu-"
PREFIX_FRAGMENT = "roefwulewcui-"
PREFIX_TARGET = "flkdtg-"
PREFIX_WALL = "wakneh-"
PREFIX_HAZARD = "defgjl"
PREFIX_PICKUP = "puukul-"
PREFIX_DECOR = "hawffu-"            # never referenced by game logic
PREFIX_ABSORBING_CORE = "roefwu-whkxtx"  # cores that absorb pickups (r11l.py:1508)
DECOY_GROUP_MARKER = "dirwzt"        # groups containing this are skipped in the win check (r11l.py:1763)
FLASH_TARGET = "flvobx-flkdtg"       # transient flash sprites (r11l.py:1671, 1685)
FLASH_CORE = "flvobx-roefwu"         # r11l.py:1629

# --- `R11l` attributes (r11l.py:1405-1440) -----------------------------------
ATTR_STEP_HUD = "_step_counter_ui"   # not obfuscated; class rjtqizgnlf
ATTR_MAX_ACTIONS = "_max_actions"    # 60, not obfuscated (r11l.py:1407)
ATTR_SELECTED = "wiayqaumjug"        # the selected fragment Sprite
ATTR_SELECTED_INDEX = "holbcmkehyf"
ATTR_CLICK_LATTICE = "jtqexauuzid"   # ActionInputs for every (x, y) in range(0, 64, 4); unused by logic
ATTR_GROUPS = "kacotwgjcyq"          # dict group -> {core, fragments, target}
ATTR_FRAGMENTS = "bbijaigbknc"       # all fragments, sorted by distance from origin (r11l.py:1516)
ATTR_WALLS = "tdriqoljcbs"
ATTR_ANIMATING = "yfbjozweime"
ATTR_ANIM_PROGRESS = "qvnmfoxseus"
ATTR_ANIM_FRAMES = "havofgepjpl"     # 1: a drag lands in one frame
ATTR_ANIM_FROM = "sgdntmcrxpq"
ATTR_ANIM_TO = "nqbqaxbtdej"
ATTR_WAIT_TARGET_FLASH = "npvvaucvsot"
ATTR_HAZARD_COUNT = "yledlprvvkb"    # 5 hazards lose (r11l.py:1749)
ATTR_REVERTING = "jttetcghmsb"
ATTR_HAZARD_FLASHING = "flgzyjcqcspeg"
ATTR_HAZARD_CORE = "hznupmuxgqv"
ATTR_TARGET_STATES = "xaalmogcsnh"   # per-group flash state for groups with a target
ATTR_ABSORBED = "bulmhgivatv"        # absorbing core name -> absorbed pickup names
ATTR_PICKUPS = "owuypsqbino"
ATTR_LEVEL_DONE = "uyawyyswbya"

# group dict keys (TypedDict xlpbmgyvhdc, r11l.py:1287-1292)
KEY_CORE = "roduyfsmiznvg"
KEY_FRAGMENTS = "lecfirgqbwunn"
KEY_TARGET = "gosubdcyegamj"

# --- `R11l` methods ------------------------------------------------------------
METHOD_STRIP_PREFIX = "zefcxjwlud"   # r11l.py:1442
METHOD_GROUP_NAMES = "pmmdvjqzco"    # r11l.py:1448
METHOD_SELECT = "ecernfbexd"         # r11l.py:1529, recolours 3<->0
METHOD_RECENTRE_CORE = "rvkbignsyr"  # r11l.py:1536
METHOD_WALL_BLOCKED = "gabrtablhx"   # r11l.py:1553
METHOD_START_DRAG = "hcpsunmfnx"     # r11l.py:1568
METHOD_GROUP_OF = "sehxptcyvq"       # r11l.py:1578
METHOD_ABSORB_PICKUPS = "zlkgwqnxrp" # r11l.py:1585
METHOD_COLOURS_MATCH = "ldzvchvkvp"  # r11l.py:1601
METHOD_ABSORBER_ON_TARGET = "enzibizxql"  # r11l.py:1609
METHOD_HAZARD_FLASH = "scyubqqntl"   # r11l.py:1620
METHOD_TARGET_FLASHES = "ihieafichl" # r11l.py:1646
METHOD_ANIMATE = "ltkvhywjqa"        # r11l.py:1691

# --- step-budget HUD (`rjtqizgnlf`, r11l.py:1306-1331) ------------------------
HUD_MAX_STEPS = "ddlxmmixxo"
HUD_CURRENT_STEPS = "current_steps"  # not obfuscated
HUD_SET_STEPS = "xfolsippxk"

# --- geometry ------------------------------------------------------------------
FRAME_SIZE = 64
FRAGMENT_SIZE = 5     # every fragment and core is 5x5, so a drag places top-left at click - 2
TARGET_SIZE = 7
HALF = FRAGMENT_SIZE // 2
# Upstream constructs a 4-pixel click enumeration (r11l.py:1416-1419), but
# never reads it.  ``step`` accepts every integer display coordinate 0..63.
# The planner therefore covers the full 64x64 click space; this lattice is
# retained only as a documented upstream curiosity.
LATTICE_STEP = 4
LATTICE = tuple(range(0, FRAME_SIZE, LATTICE_STEP))

# --- actions -------------------------------------------------------------------
ACTION_CLICK = 6
ACTION_IDS = (ACTION_CLICK,)   # available_actions=[6] (r11l.py:1440)
MAX_ACTIONS = 60               # r11l.py:1407
USABLE_ACTIONS = MAX_ACTIONS - 1  # the 60th action loses (r11l.py:1807-1811); verified empirically
HAZARD_LIMIT = 5

# Groups shipped upstream with a core, fragments and a single-colour target;
# used as prototypes by the generator.
SIMPLE_GROUPS = ("pumlzd", "orrqlj", "grhcew")


def group_of(name):
    """Which group a core/fragment/target sprite belongs to, or None."""
    for prefix in (PREFIX_FRAGMENT, PREFIX_CORE, PREFIX_TARGET):
        if name.startswith(prefix):
            return name[len(prefix):]
    return None


def click_to_position(x, y):
    """Top-left of a 5x5 fragment dragged so its centre lands on the click."""
    return x - HALF, y - HALF


def position_to_click(x, y):
    return x + HALF, y + HALF
