"""Readable names for CN04's obfuscated identifiers.

Upstream ships ``third_party/arc3_games/cn04.py`` with randomised class, method
and attribute names. The file is left byte-identical (it matches the module
``arc_agi`` downloads); this module is the translation table, so nothing else
in Pebby hardcodes a random string. Line numbers cited are upstream lines.

Rules that matter for planning, read out of the source:

* A pin is a pixel of colour 8 or 13 (``PIN_COLORS``). Pins are located on the
  sprite's *original* pixels (``ATTR_ORIGINAL_PIXELS``) rendered at the current
  rotation (lines 949-976).
* Two pins of the *same* colour on the same grid pixel, and exactly two, count as
  matched (lines 977-988). A mixed 8/13 pair only displays as matched (colour 3,
  lines 989-1021) but does not count towards completion (lines 1023-1049).
* The level completes when every visible sprite has no unmatched pin
  (``METHOD_IS_SOLVED``), checked only after ACTION1-5 (lines 1098, 1127); the
  check sets ``ATTR_LEVEL_WON`` and the same ``perform_action`` call then runs
  ``next_level`` (lines 1052-1055).
* Sprites are never collision-tested; pieces may overlap freely.
* Sprites sharing one initial position form a stack of alternates (lines
  865-880); exactly one is visible. ACTION5 cycles a stack instead of rotating,
  transfers its position, and clamps the new dimensions (lines 1091-1097),
  bouncing at both ends via one global direction flag (lines 1133-1157).
  Clicking the selected stack on a pixel whose *display* value is 0 also cycles
  (lines 1075-1083), but neither clamps nor checks for completion.
* The n-th action of a level loses if ``n >= MaxSteps`` (lines 1057-1061 with
  ``_action_count`` incremented before ``step``), so ``MaxSteps - 1`` actions are
  usable.
"""

# --- classes -----------------------------------------------------------------
CLASS_GAME = "Cn04"                    # line 833
CLASS_STEP_HUD = "lvealyvptn"          # line 803, RenderableUserDisplay drawing the step bar on row 0

# --- step HUD (upstream lines 803-830) ---------------------------------------
HUD_MAX_STEPS = "pguodduwhg"
HUD_CURRENT_STEPS = "current_steps"
HUD_SET_MAX = "upqoakzziq"
HUD_SET_CURRENT = "tebogkqnvk"

# --- `Cn04` attributes (upstream lines 836-886) ------------------------------
ATTR_SELECTED = "xseexqzst"            # currently selected Sprite or None
ATTR_DISPLAY_BACKUP = "uysylxuqw"      # name -> pixels to restore on deselect
ATTR_ORIGINAL_PIXELS = "hlxyvcmpk"     # name -> pristine pixels (pins still 8 / 13)
ATTR_MATCHED_PINS = "iahpylgry"        # name -> {(x, y)} rendered pin coords that count for the win
ATTR_DISPLAY_PINS = "ydfurpdwv"        # name -> {(x, y)} rendered pin coords drawn as colour 3
ATTR_GREY_MASKING = "dxcfrrcpp"        # level data "GreyMasking"
ATTR_MAX_STEPS = "ojcsxidcz"           # level data "MaxSteps" (default 150)
ATTR_STEP_HUD = "kpgnbcoir"
ATTR_LAST_WAS_CLICK = "spcewphwy"      # set but never read
ATTR_STACKS = "vausolnec"              # Sprite -> list of alternates sharing its initial position
ATTR_CYCLE_FORWARD = "ztpxqonhr"       # global bounce direction for cycling alternates
ATTR_LEVEL_WON = "rqolqpqwo"           # set by the winning action; next step advances the level

# --- `Cn04` methods -----------------------------------------------------------
METHOD_SELECT = "swfljqaiqu"           # line 888
METHOD_DESELECT = "bewtoqvzlr"         # line 906
METHOD_RECOMPUTE = "uqlndqojuf"        # line 920, recolours pins and rebuilds the match sets
METHOD_IS_SOLVED = "sjwqloivve"        # line 1023
METHOD_CYCLE_ALTERNATE = "eliexourzp"  # line 1133
METHOD_HIT_TEST = "ixutchviko"         # line 1159
METHOD_PIXEL_AT = "aqrmljjiyi"         # line 1168, rendered display pixel under a grid point
METHOD_CLAMP = "rnwsvakqem"            # line 1174, keeps a sprite inside grid_size

# --- level data keys (not obfuscated) ----------------------------------------
KEY_BACKGROUND = "BackgroundColour"
KEY_MAX_STEPS = "MaxSteps"
KEY_GREY_MASKING = "GreyMasking"
DEFAULT_MAX_STEPS = 150                # line 851
GRID_SIZE = (20, 20)                   # every shipped level

# --- colours -------------------------------------------------------------------
PIN_A = 8
PIN_B = 13
PIN_COLORS = (PIN_A, PIN_B)
MATCHED = 3                            # a paired pin's display colour
GREY = 4                               # GreyMasking body colour and the default background
SELECTED_BODY = 0                      # selected piece body colour without GreyMasking
CYCLE_PIXEL = 0                        # display value that cycles a selected stack on click
TRANSPARENT = -1
TRANSPARENT_UNCLICKABLE = -2           # sprite 0029 uses -2: transparent, not clickable

# --- actions -------------------------------------------------------------------
ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 1, 2, 3, 4
ACTION_ROTATE = 5                      # cycles alternates on a stacked piece
ACTION_CLICK = 6
MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}
ACTION_IDS = (1, 2, 3, 4, 5, 6)

# --- display geometry: 20x20 grid inside the 64x64 frame ------------------------
DISPLAY_SIZE = 64
CLICK_TAG = "sys_click"


def display_geometry(grid_size=GRID_SIZE):
    """(scale, x_offset, y_offset) exactly as Camera computes them."""
    width, height = grid_size
    scale = min(DISPLAY_SIZE // width, DISPLAY_SIZE // height)
    return scale, (DISPLAY_SIZE - width * scale) // 2, (DISPLAY_SIZE - height * scale) // 2


def grid_to_display(gx, gy, grid_size=GRID_SIZE):
    """Centre display pixel of grid cell (gx, gy); Camera.display_to_grid inverts it."""
    scale, ox, oy = display_geometry(grid_size)
    return gx * scale + ox + scale // 2, gy * scale + oy + scale // 2
