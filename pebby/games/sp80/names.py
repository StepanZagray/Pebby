"""Readable names for SP80's obfuscated identifiers.

Upstream ships ``third_party/arc3_games/sp80.py`` with randomised class,
method, attribute, sprite and tag names. The file is left byte-identical; this
module is the translation table, so nothing else in Pebby hardcodes a random
string. Every entry was read out of the upstream source; the line numbers cited
are upstream line numbers.

The game: a source drips water straight down. ACTION6 selects a movable piece
(pipe or deflector), ACTION1-4 move it one cell, ACTION5 runs the flow. Water
that lands on a pipe spreads along it and falls off both ends; a deflector turns
a falling stream sideways; a stream entering a cup's mouth fills it. The level
is complete when one flow fills every cup without touching an edge sink.
"""

# --- sprite prototypes (upstream `sprites`, sp80.py:36-273) -------------------
FRAME = {16: "bodekplurlf16", 20: "bodekplurlf20", 32: "bodekplurlf32"}  # border ring, placed at (-1, -1)
WATER = "liolfvkveqg"                 # 1x1 drop, colour 6 (sp80.py:132)
PIPE = {3: "plzwjbfyfli-3", 4: "plzwjbfyfli-4", 5: "plzwjbfyfli-5", 6: "plzwjbfyfli-6",
        7: "plzwjbfyfli-7", 8: "plzwjbfyfli-8"}            # horizontal, colour 8
PIPE_VERTICAL_4 = "plzwjbfyfli-4h"                          # 1x4 vertical pipe (sp80.py:159)
SOURCE_PIPE = {5: "plzwjbfyfli-5-sowlljgtjvn", 7: "plzwjbfyfli-7-sowlljgtjvn"}  # pipe with a source in its middle
CUP = "repwkzbkhxl"                   # 3x2 cup [[11,-1,11],[11,11,11]] (sp80.py:225)
SOURCE = "sowlljgtjvn"                # 1x1 source, colour 4 (sp80.py:235)
DEFLECTOR_LEFT = "tuvkdkhdokr-lexuhyxqrsm"    # [[-1,15],[15,15]]: hole top-left, turns water LEFT (sp80.py:244)
DEFLECTOR_RIGHT = "tuvkdkhdokr-riwynidseun"   # [[15,-1],[15,15]]: hole top-right, turns water RIGHT (sp80.py:254)
SINK = "waoewejnqzc"                  # 32x1 edge sink line, colour 1 (sp80.py:264)

# --- tags --------------------------------------------------------------------
TAG_WATER = "liolfvkveqg"
TAG_PIPE = "plzwjbfyfli"
TAG_CUP = "repwkzbkhxl"
TAG_SOURCE = "sowlljgtjvn"
TAG_DEFLECTOR = "tuvkdkhdokr"
TAG_SINK = "waoewejnqzc"
MOVABLE_TAGS = (TAG_PIPE, TAG_DEFLECTOR)   # `fbrwmvzsym` lists pipes first, then deflectors (sp80.py:568-569)

# --- level data keys ---------------------------------------------------------
KEY_STEPS = "steps"                   # step budget; 0/None -> 50 (sp80.py:557)
KEY_ROTATION = "dojfslwbg"            # screen rotation in degrees; k = deg // 90 % 4 (sp80.py:564-566)

# --- `Sp80` attributes (sp80.py:497-525) -------------------------------------
ATTR_MODE = "dkvpswzsjg"              # "change" (editing) or "spill" (flow running)
ATTR_SELECTED = "vsoxmtrhqt"          # selected movable Sprite or None
ATTR_HEADS = "hmxltcipkc"             # active flow heads [(water sprite, dx, dy)]
ATTR_ADDED_WATER = "xpcxocsmmq"       # water sprites created by the current flow
ATTR_SINK_HIT = "kfdcqkodyy"          # a stream touched a sink
ATTR_FLOW_DONE = "lybfalkrdl"         # no heads left; resolving
ATTR_FILLED_CUPS = "cevwbinfgl"
ATTR_HIT_SINKS = "onoqwewztl"
ATTR_ANIM = "trhynadhiz"              # failure blink counter
ATTR_STEP_HUD = "lijmqzmgnw"
ATTR_STEPS_LEFT = "zlhbnhpcq"
ATTR_FAILED_FLOWS = "lyremoheq"       # 4 failures -> the next ACTION5 loses (sp80.py:717)
ATTR_ROTATION_K = "fahhoimkk"
ATTR_ROTATION_HUD = "jinpztikz"

# --- `Sp80` methods ------------------------------------------------------------
METHOD_MOVABLES = "fbrwmvzsym"        # movable pieces in click-resolution order (sp80.py:568)
METHOD_CUPS = "mxdlffpzkc"            # sp80.py:574
METHOD_NEAREST_PIECE = "gsvsxaspkc"   # piece minimising x^2+y^2, auto-selected (sp80.py:580)
METHOD_CAN_OCCUPY = "husluhmboo"      # y >= 3 and one cell clear of every cup (sp80.py:586)
METHOD_SELECT = "mmzrajuxwp"          # sp80.py:615
METHOD_DESELECT = "ntuflwihof"        # sp80.py:624
METHOD_START_FLOW = "vdwhttyyfq"      # sp80.py:631
METHOD_END_FLOW = "lpqbikobah"        # clears water, counts a failure, reselects (sp80.py:655)
METHOD_ROTATE_CLICK = "ojydygjkdb"    # agent display -> internal display (sp80.py:823)
METHOD_UNROTATE_CLICK = "rqlmciubuz"  # internal display -> agent display (sp80.py:834)
ATTR_ACTION_REMAP = "mfkgvxzkbj"      # k -> {agent action: internal action} (sp80.py:457)
ATTR_ACTION_UNMAP = "othselxnik"      # k -> {internal action: agent action} (sp80.py:477)

# --- actions -----------------------------------------------------------------
ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_FLOW, ACTION_CLICK = 1, 2, 3, 4, 5, 6
MOVE_ACTIONS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)
AVAILABLE_ACTIONS = (1, 2, 3, 4, 5, 6)   # sp80.py:535
# Internal (post-rotation) move deltas, sp80.py:696-704.
ACTION_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

# Agent action -> internal action per rotation k (sp80.py:457-476).
ACTION_REMAP = {
    0: {1: 1, 2: 2, 3: 3, 4: 4},
    1: {1: 4, 2: 3, 3: 1, 4: 2},
    2: {1: 2, 2: 1, 3: 4, 4: 3},
    3: {1: 3, 2: 4, 3: 2, 4: 1},
}
# Internal action -> agent action that produces it (inverse of the above).
AGENT_ACTION_FOR = {k: {v: a for a, v in table.items()} for k, table in ACTION_REMAP.items()}

MIN_PIECE_ROW = 3        # husluhmboo rejects y < 3 (sp80.py:590)
MAX_FAILED_FLOWS = 4
DISPLAY = 64


def rotate_click(k, x, y):
    """Agent display coordinates -> internal display coordinates (sp80.py:823-832)."""
    if k == 0:
        return x, y
    if k == 1:
        return DISPLAY - 1 - y, x
    if k == 2:
        return DISPLAY - 1 - x, DISPLAY - 1 - y
    return y, DISPLAY - 1 - x


def unrotate_click(k, x, y):
    """Internal display coordinates -> the agent coordinates that map onto them (sp80.py:834-843)."""
    if k == 0:
        return x, y
    if k == 1:
        return y, DISPLAY - 1 - x
    if k == 2:
        return DISPLAY - 1 - x, DISPLAY - 1 - y
    return DISPLAY - 1 - y, x
