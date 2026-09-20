"""Readable aliases for KA59's obfuscated native identifiers."""

# Sprite prototypes, upstream lines 33-40933.
PLAYER_LARGE = "0000xijqsmvawl"
EXPLOSIVE_SMALL = "0002pdebyfquah"
EXPLOSIVE_LARGE = "0004kgbrqnrpsk"
EXPLOSIVE_MIXED = "0005upuetaiejs"
PLAYER_SMALL = "0006xssgmrgoep"

TARGET_3X3 = "0009ouocpihipp"
TARGET_3X6 = "0011wpykruqziz"
TARGET_6X6 = "0012nclblyaezs"
TARGET_6X3 = "0013rdkgnuvatj"
TARGET_PROTOTYPES = (TARGET_3X3, TARGET_3X6, TARGET_6X6, TARGET_6X3)

WALL_LEVEL1 = "0014ysspdlqsqg"
BOX_3X3 = "0021xplppqqmfb"
BOX_3X6 = "0023jvrhzuyxhg"
BOX_6X3 = "0024ejsxlvzjgi"
BOX_6X6 = "0025antkezljyc"
BOX_PROTOTYPES = (BOX_3X3, BOX_3X6, BOX_6X3, BOX_6X6)
PLAYER_TARGET = "0026hnpvtkjhrp"
BOUNDARY_45 = "0028lydaygyjbu"

BOX_TO_TARGET = {
    BOX_3X3: TARGET_3X3,
    BOX_3X6: TARGET_3X6,
    BOX_6X3: TARGET_6X3,
    BOX_6X6: TARGET_6X6,
}

# Tags used by Ka59, upstream lines 41089-41456.
TAG_PLAYER = "0001uqqokjrptk"
TAG_EXPLOSIVE = "0003umnkyodpjp"
TAG_EXPLOSION = "0007zqjfknlfvm"
TAG_TARGET = "0010xzmuziohuf"
TAG_WALL = "0015qniapgwsvb"
TAG_BOX = "0022vrxelxosfy"
TAG_PLAYER_TARGET = "0027jbgxilrocf"
TAG_BOUNDARY = "0029ifoxxfvvvs"
TAG_CLICK = "sys_click"
TAG_ENEMY = "Enemy"

KEY_STEPS = "StepCounter"
ATTR_STEP_COUNTER = "urgssjskot"
ATTR_SELECTED = "prkgpeyexo"
ATTR_PENDING_PUSH = "lphmmaeepj"
ATTR_PUSH_VECTORS = "ooneovlmbq"
ATTR_ANIMATION_FRAME = "xrxdckwsth"
ATTR_PENDING_EXPLOSION = "hrknegnjkg"

FRAME_SIZE = 64
GRID_STEP = 3

ACTION_RESET = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_CLICK = 6
MOVE_ACTIONS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)
ACTION_IDS = (*MOVE_ACTIONS, ACTION_CLICK)
