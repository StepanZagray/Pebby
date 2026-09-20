"""Readable names for SC25's obfuscated identifiers.

Upstream ships ``third_party/arc3_games/sc25.py`` with randomised sprite,
attribute and data-key names. The file is left byte-identical; this module is the
translation table so nothing else in Pebby hardcodes a random string.

Every entry was read directly out of the upstream source; cited line numbers are
upstream line numbers.
"""

# --- sprite names (upstream `sprites` dict, sc25.py:29-1377) -----------------
SPRITE_BUDGET_BAR = "action-ui"                # 1x32 bar at (62,0) scale 2; drains as toggles are spent (l.65)
SPRITE_TELEPORT_INDICATOR = "acyylh-tevyeq-inkpfx"  # 6x6 frame marking the next scale-2 teleport pad (l.79)
SPRITE_GRID_FRAME_PREFIX = "clcbko"            # invisible collidable frames around the 3x3 grid; clicks map to cells (l.93-143)
SPRITE_GRID_CELL = "clzbxlm-sptivk-slsrhr"     # 3x3 toggle cell; 9 of them at (24+5c, 49+5r) (l.144)
SPRITE_RING_OBSTACLE = "crzdcq"                # 4x4 hollow block; blocks fireballs and growth (l.150; unused by shipped levels)
SPRITE_BLOCK = "dosorb"                        # 4x3 obstacle removed when a `tagsmh` target is hit (l.167)
SPRITE_WALL_PREFIX = "duvwsv-"                 # 64x64 maze bitmaps; -1 pixels are corridors (l.238-749)
SPRITE_PICKUP = "enjehv-pahtoz"                # 2x2 non-collidable pickup; refunds 10 toggles (l.750)
SPRITE_DOOR = "exydhv"                         # 5x6 exit door; colliding with it completes the level (l.763)
SPRITE_FIREBALL = "fibcey"                     # projectile drawn while a fireball flies (l.775)
SPRITE_FIREBALL_SMALL = "fibcey-2"             # projectile used by a scale-1 player (l.785)
SPRITE_PLAYER = "pluyoo"                       # 2x2 player, scale 1 or 2 (l.795)
SPRITE_BLOCK_ALT = "seofsw-dosorb"             # second obstacle family, removed via `seofsw-tagsmh` (l.806)
SPRITE_TARGET_ALT = "seofsw-tagsmh"            # fireball target clearing every `seofsw-dosorb` (l.817)
SPRITE_TELEPORT_INDICATOR_SMALL = "smzaik-tevyeq-inkpfx"  # indicator for scale-1 pads (l.831)
SPRITE_TELEPORT_PAD_SMALL = "smzaik-tevyeq-tagsmh"        # 2x2 teleport pad used while the player is scale 1 (l.840)
SPRITE_ICON_FRAME = "sptivk-caxiiu"            # 10x10 frame behind a spell icon (l.1257)
SPRITE_ICON_PREFIX = "sptivk-"                 # `sptivk-<spell>` selects that spell (l.1273-1348)
SPRITE_GRID_PANEL = "sptivk-ui"                # 17x17 panel drawn under the 3x3 grid at (22,47) (l.1349)
SPRITE_TARGET = "tagsmh"                       # 4x4 fireball target clearing every `dosorb` (l.1361)
SPRITE_TELEPORT_PAD = "tevyeq-tagsmh"          # 4x4 teleport pad used while the player is scale 2 (l.1373)
SPRITE_DECORATION_PREFIX = "sprite-"           # decorative bitmaps; sprite-21/22/23 are collidable, 9..14 are not

# --- spells (keys of `Sc25.zzpoabuniyn`, sc25.py:1673-1689) ------------------
SPELL_TELEPORT = "tevyeq"
SPELL_GROW = "sieesc_chwjgc"
SPELL_FIRE = "fibcey"
SPELLS = (SPELL_TELEPORT, SPELL_GROW, SPELL_FIRE)  # upstream dict order; matching is tested in this order

# 3x3 bitmaps that cast each spell, as (row, col) sets. Upstream stores them as
# bool matrices; a cast happens when the toggled grid equals one of these exactly.
PATTERNS = {
    SPELL_TELEPORT: ((0, 0), (0, 1), (1, 1)),
    SPELL_GROW: ((0, 1), (1, 0), (1, 2), (2, 1)),
    SPELL_FIRE: ((0, 1), (1, 1), (2, 1)),
}

# --- level data keys (sc25.py:1410-1413) -------------------------------------
KEY_BUDGET = "slfh"          # maximum toggles (`eyxbonasvgm`); exceeding it loses
KEY_SPELLS = "efvw"          # str or list of spell names castable on this level (`jlpticwjyvy`)
KEY_UNUSED_FLAG = "rpjr"     # present on levels 1-3; never read by the game class

# --- `Sc25` attributes (sc25.py:1655-1760) -----------------------------------
ATTR_GRID = "xhhaqjfncnp"            # 3x3 list of bools, the toggled cells
ATTR_BUDGET = "eyxbonasvgm"          # int | None, max toggles
ATTR_USED = "rrinmfkkstu"            # toggles spent so far (moves and casts also count)
ATTR_SPELLS = "jlpticwjyvy"          # castable spell names for this level
ATTR_PATTERNS = "zzpoabuniyn"        # spell -> 3x3 bool matrix
ATTR_FACING = "jdmucabyqar"          # 0 up, 1 down, 2 left, 3 right; fireball direction; never reset per level
ATTR_PLAYER = "plnqvukupu"           # the player Sprite
ATTR_LEVEL_SPRITES = "lyhbotskgaq"   # sprite list captured at on_set_level, minus removed ones
ATTR_PICKUPS = "mphhxnbsevp"
ATTR_TELEPORT_PADS = "cmagnysjqzl"        # scale-2 pads
ATTR_TELEPORT_PADS_SMALL = "dkwqdbhqspz"  # scale-1 pads
ATTR_BLOCKS = "ouqsfmdpjom"               # every `dosorb`
ATTR_BLOCKS_ALT = "wrlhavqltbu"           # every `seofsw-dosorb`
ATTR_TELEPORT_INDEX = "ydpaogvspcp"       # next scale-2 pad
ATTR_TELEPORT_INDEX_SMALL = "ibmhvvyrikn" # next scale-1 pad
ATTR_SELECTED = "ijhfdcamokt"        # selected spell icon (display only; casting ignores it)
ATTR_DEMO_PENDING = "qytejzcythm"    # True on engine level 0 until the first action, which only plays a hint demo
ATTR_WALKING_OUT = "eycwbtepcvs"     # door walk-out animation in progress
ATTR_CAST_ANIM = "obrrczymkxn"
ATTR_TELEPORT_ANIM = "wmnlnlscbpq"
ATTR_GROW_ANIM = "jwlqyoqyagv"
ATTR_FIRE_ANIM = "agzbtzaakna"
ATTR_FLASH_ANIM = "vbublqskwzw"
ATTR_DEMO_ANIM = "ggotuphkheh"

# --- constants (sc25.py:1643-1652) -------------------------------------------
PICKUP_REFUND = 10           # `cribhxjrvp`; used = max(0, used - 10)
WALL_COLOR = 5

# --- geometry ------------------------------------------------------------------
GRID_ORIGIN = (24, 49)       # top-left of cell (row 0, col 0); cells are 3x3 on a 5px pitch (sc25.py:1806-1810)
GRID_PITCH = 5
ACTION_IDS = (1, 2, 3, 4, 6)
MOVE_ACTIONS = (1, 2, 3, 4)
# ACTION1..4 -> (dx, dy) unit and the facing index the game records (sc25.py:2615-2626).
MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
MOVE_FACING = {1: 0, 2: 1, 3: 2, 4: 3}
FACING_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def cell_click(row, col):
    """Display pixel that toggles grid cell (row, col); the game's own list uses these (sc25.py:1739-1749)."""
    return (GRID_ORIGIN[0] + 1 + GRID_PITCH * col, GRID_ORIGIN[1] + 1 + GRID_PITCH * row)
