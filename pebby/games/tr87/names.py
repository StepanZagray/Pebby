"""Readable names for TR87's obfuscated identifiers.

Upstream ships ``third_party/arc3_games/tr87.py`` with randomised class, method,
attribute and sprite names. The file is left byte-identical; this module is the
translation table, so the rest of the package never hardcodes a random string.

Every entry was read directly out of the upstream source; the line numbers cited
are upstream line numbers.

Game summary (upstream ``Tr87.step``, tr87.py:974-1022): the screen splits at the
``background`` sprite's y. Above it, rule strips (``iqrduxrukrk``) sit between a
left-hand tile sequence and a right-hand tile sequence. Below it, the first tile
row is the *source* row and the remaining tiles form the editable *target* row.
ACTION3/ACTION4 move a cursor across the editable groups (-1 / +1), ACTION1 /
ACTION2 cycle the selected group's glyph (-1 / +1 mod 7). Every action costs one
unit of a 128 (level index <= 4) or 256 budget; the solved check runs only after
ACTION1/ACTION2, and reaching budget zero without solving loses.
"""

# --- sprite tags (upstream `Sprite.tags`) ------------------------------------
TAG_TILE = "nxkictbbvzt"        # glyph tiles, tr87.py:154-425
TAG_MARKER = "tjaqvwdgkxe"      # invisible 1px double-translation markers, tr87.py:114-152
TAG_BACKING = "gyrdjxybtcm"     # 7x7 coloured squares behind tiles, tr87.py:57-103

# --- sprite names --------------------------------------------------------------
SPRITE_BACKGROUND = "background"       # 2x2 colour-3 block, scaled 32 -> splits the screen, tr87.py:38
SPRITE_STRIP = "iqrduxrukrk"           # 11x1 rule connector, tr87.py:105
SPRITE_BACKING = "gyrdjxybtcm"         # + family letter: A=colour 10, B=7, C=11
SPRITE_TILE = "nxkictbbvzt"            # + family letter + digit 1..7
SPRITE_MARKER = "jpafjzbfwiq"          # + letter + "1"/"2"; never placed in shipped levels
SPRITE_HALO = "nxkictbbvztedxeenecwqa"  # 9x9 black square shown during the win animation, tr87.py:427
SPRITE_CURSOR = "qvtymdcqear"          # + "1"/"2"/"3" (widths 5/12/19), tr87.py:444-469

FAMILIES = ("A", "B", "C")
SYMBOL_COUNT = 7                # kjgicbtgrt, tr87.py:881
TILE_SPACING = 7                # iokndxodxw, tr87.py:880: neighbouring tiles of one rule side
TILE_SIZE = 5
BACKGROUND_COLOR = 2            # tr87.py:878
PADDING_COLOR = 3               # tr87.py:879
HUD_BAR_COLOR = 1               # fiwynmpoeb, tr87.py:882
HUD_EMPTY_COLOR = 4             # efblfzysop, tr87.py:883
WIN_ANIMATION_COLORS = [5, 8, 14, 15, 6, 9, 12, 0]  # rhoqllymmn, tr87.py:884

# --- level data keys (upstream `Level.get_data`) --------------------------------
KEY_DOUBLE_TRANSLATION = "double_translation"  # tr87.py:1078; shipped level 4 and 6
KEY_ALTER_RULES = "alter_rules"                # tr87.py:943, 997, 1007; shipped level 5 and 6
KEY_TREE_TRANSLATION = "tree_translation"      # tr87.py:1063; shipped level 6
KEY_TEACHER_RULES = "_pebby_teacher_rules"    # generated-only constructive certificate data

# --- `Tr87` attributes (upstream tr87.py:910-972) ----------------------------------
ATTR_ALL_TILES = "zdwrfusvmx"          # every tile sprite, sorted by (y, x), :915
ATTR_SOURCE_ROW = "zvojhrjxxm"         # tiles below the split on its first row, :921
ATTR_TARGET_ROW = "ztgmtnnufb"         # tiles below the split on later rows (editable), :922
ATTR_RULES = "cifzvbcuwqe"             # list of (lhs tiles, rhs tiles), strip (y, x) order, :926
ATTR_CURSOR_INDEX = "qvtymdcqear_index"  # selected group, :959
ATTR_CURSOR_PARTS = "qvtymdcqear_parts"  # the two bracket sprites, :960
ATTR_BUDGET_MAX = "vfpimnmtnta"        # 128 or 256, :967
ATTR_BUDGET_LEFT = "upmkivwyrxz"       # decremented once per action, :968
ATTR_ANIMATION_STEP = "yfetxjexviz"    # -1 while playing, >= 0 during the win animation, :969
ATTR_ANIMATION_PAIRS = "pvgetmhmhgk"   # matched (source slice, target slice) pairs, :970
ATTR_ANIMATION_RULES = "hgfgmiagdcc"   # rule tiles highlighted per pair, :971
ATTR_ANIMATION_HALOS = "crvjftupafy"   # halo sprites currently shown, :972

# --- `Tr87` methods --------------------------------------------------------------
METHOD_CYCLE_TILE = "wpbnovjwkv"       # replace a tile with its digit +-1 neighbour, :1024
METHOD_PLACE_CURSOR = "pjqbnqnbsq"     # :1031
METHOD_CHECK_SOLVED = "bsqsshqpox"     # the parse-and-compare win test, :1044
METHOD_SEQUENCE_AT = "iwbhnvdaao"      # names of `seq[start:]` equal a pattern, :1107
METHOD_NEIGHBOUR_TILE = "qrkneeaawb"   # tile exactly TILE_SPACING to the left/right, :1115

# --- actions ---------------------------------------------------------------------
ACTION_CYCLE_DOWN = 1   # ACTION1: selected group digit -1, tr87.py:1005
ACTION_CYCLE_UP = 2     # ACTION2: selected group digit +1
ACTION_SELECT_PREV = 3  # ACTION3: cursor -1, tr87.py:995
ACTION_SELECT_NEXT = 4  # ACTION4: cursor +1
ACTION_IDS = (ACTION_CYCLE_DOWN, ACTION_CYCLE_UP, ACTION_SELECT_PREV, ACTION_SELECT_NEXT)
CYCLE_DELTA = {ACTION_CYCLE_DOWN: -1, ACTION_CYCLE_UP: 1}
SELECT_DELTA = {ACTION_SELECT_PREV: -1, ACTION_SELECT_NEXT: 1}

BUDGET_BY_LEVEL_INDEX = (128, 128, 128, 128, 128, 256)
# `on_set_level` seeds `random.Random` twice per level index: the first seed draws
# cosmetic rotations, the second the initial scramble (tr87.py:912, 941).
ROTATION_SEEDS = (4, 2, 1, 2, 12, 20)
SCRAMBLE_SEEDS = (7, 7, 32, 18, 23, 11)


def symbol(sprite_name):
    """'nxkictbbvztB6' -> 'B6'."""
    if not sprite_name.startswith(SPRITE_TILE) or len(sprite_name) != len(SPRITE_TILE) + 2:
        raise ValueError(f"not a tile sprite name: {sprite_name!r}")
    return sprite_name[len(SPRITE_TILE):]


def tile_sprite_name(sym):
    return SPRITE_TILE + sym


def cycle(sym, delta):
    """Upstream `wpbnovjwkv`: the digit wraps within 1..7, the family letter stays."""
    return sym[0] + str((int(sym[1]) + delta - 1) % SYMBOL_COUNT + 1)


def digit_distance(a, b):
    """Fewest single cycles taking digit `a` to digit `b`."""
    forward = (b - a) % SYMBOL_COUNT
    return min(forward, SYMBOL_COUNT - forward)
