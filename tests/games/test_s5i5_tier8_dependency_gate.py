"""Focused tier-8 dependency calibration against native S5I5 transitions."""

from collections import Counter

from pebby.games.s5i5 import generate as generator
from pebby.games.s5i5 import names
from pebby.games.s5i5.env import Env, official_levels
from pebby.games.s5i5.layout import extract


_REFERENCE_ACTIONS = {
    "A+": (6, 59, 11),
    "B+": (6, 45, 4),
    "B-": (6, 39, 4),
    "C+": (6, 45, 11),
    "C-": (6, 39, 11),
    "D-": (6, 22, 56),
    "E-": (6, 8, 56),
    "R": (6, 49, 18),
}
_REFERENCE_ROUTE = tuple(
    _REFERENCE_ACTIONS[word]
    for word in (
        "A+ B+ C+ D- E- R R C+ C+ E- E- C- C- R B+ D- D- B- R R "
        "A+ B- A+ D- D- A+ A+ A+ A+ A+ A+ C- A+ E- E- A+ A+ A+"
    ).split()
)


def _official_reference_spec():
    """Describe native tier 8 for test-only evidence; never generation input."""
    env = Env()
    env.set_level(7)
    children = getattr(env.game, names.ATTR_CHILDREN)
    rod_edges = sorted(
        [parent.name, child.name]
        for parent, values in children.items()
        for child in values
        if names.TAG_ROD in child.tags
    )
    pin_parent = next(
        parent.name
        for parent, values in children.items()
        if any(names.TAG_PIN in child.tags for child in values)
    )
    mechanical_names = {pin_parent}
    mechanical_names.update(name for edge in rod_edges for name in edge)
    rods = []
    obstacles = []
    for sprite in env.rods():
        row = {
            "name": sprite.name,
            "color": int(sprite.pixels[1, 1]),
            "x": int(sprite.x),
            "y": int(sprite.y),
        }
        (rods if sprite.name in mechanical_names else obstacles).append(row)

    controls = extract(env).controls
    rail_by_name = {}
    button_by_name = {}
    for control in controls:
        if control.kind in ("extend", "retract"):
            row = {"name": control.name, "color": int(control.colors[0])}
            if len(control.colors) > 1:
                row["secondary_color"] = int(control.colors[1])
            rail_by_name[control.name] = row
        else:
            button_by_name[control.name] = {
                "name": control.name,
                "color": int(control.colors[0]),
            }
    return {
        "difficulty": 8,
        "context_index": 7,
        "step_counter": 200,
        "rods": rods,
        "obstacles": obstacles,
        "pins": [{"x": int(pin.x), "y": int(pin.y)} for pin in env.pins()],
        "targets": [
            {"x": int(target.x), "y": int(target.y)} for target in env.targets()
        ],
        "children": rod_edges,
        "rails": list(rail_by_name.values()),
        "buttons": list(button_by_name.values()),
    }


def _official_level_from_reference_spec(spec):
    level = official_levels()[7]
    allowed_rods = {row["name"] for row in spec["rods"]}
    allowed_rods.update(row["name"] for row in spec.get("obstacles", ()))
    for sprite in list(level.get_sprites_by_tag(names.TAG_ROD)):
        if sprite.name not in allowed_rods:
            level.remove_sprite(sprite)
    if not spec.get("targets"):
        for sprite in list(level.get_sprites_by_tag(names.TAG_TARGET)):
            level.remove_sprite(sprite)
    level._data[names.KEY_CHILDREN] = [list(edge) for edge in spec["children"]]
    level._data[names.KEY_STEP_COUNTER] = int(spec["step_counter"])
    return level


def test_official_tier8_independent_pin_has_causal_off_pin_branch(monkeypatch):
    spec = _official_reference_spec()
    monkeypatch.setattr(generator, "build_level", _official_level_from_reference_spec)

    evidence = generator._dependency_evidence(spec, list(_REFERENCE_ROUTE))
    assert evidence is not None
    assert evidence["native_relations"]["pin_edges"] == [
        ["0043dhmhlmzfqb", [18, 3]]
    ]
    assert all(not row["pins"] for row in evidence["recursive_edges"])

    branch = evidence["branch_edges"]
    assert Counter(row["parent"] for row in branch)["0050niyswwmhla"] == 4
    companion_edge = next(
        row for row in branch if row["child"] == "0051pxvxhqyquq"
    )
    assert companion_edge["essential"] is True

    shared = next(
        row
        for row in evidence["shared_companions"]
        if row["rod"] == "0051pxvxhqyquq"
    )
    assert shared == {
        "rod": "0051pxvxhqyquq",
        "first_completion": 38,
        "reduced_actions": 36,
        "essential": True,
    }
    assert any(
        row["first_completion"] == 38 and row["reduced_actions"] < 38
        for row in evidence["recursive_edges"]
    )
    assert evidence["constraint_gate"]["reduced_actions"] == 13


def test_tier8_decorative_off_pin_edge_is_not_causal():
    spec = {
        "difficulty": 8,
        "context_index": 7,
        "step_counter": 20,
        "obstacles": [],
        "rods": [
            {"name": "pinrod", "color": 7, "x": 9, "y": 9,
             "rotation": 90, "length": 1},
            {"name": "decorroot", "color": 8, "x": 30, "y": 10,
             "rotation": 90, "length": 1},
            {"name": "decorchild", "color": 9, "x": 33, "y": 10,
             "rotation": 90, "length": 1},
        ],
        "pins": [{"x": 9, "y": 9}],
        "targets": [{"x": 12, "y": 9}],
        "children": [["decorroot", "decorchild"]],
        "rails": [
            {"name": "pinrail", "color": 7, "x": 20, "y": 50,
             "orientation": "horizontal", "style": "compact"}
        ],
        "buttons": [],
    }
    env = Env([generator.build_level(spec) for _ in range(8)])
    env.set_level(7)
    action = next(
        (names.ACTION_CLICK, *control.click)
        for control in extract(env).controls
        if control.kind == "extend"
    )
    relations = generator._native_relations(spec)
    assert generator._first_native_completion(spec, [action]) == 1
    assert generator._tier8_edge_checks(spec, [list(action)], relations) == [{
        "parent": "decorroot",
        "child": "decorchild",
        "pins": [],
        "first_completion": 1,
        "reduced_actions": 1,
        "essential": False,
    }]
