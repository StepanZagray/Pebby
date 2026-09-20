"""Focused native-semantic official-copy exclusion regressions for S5I5."""

from copy import deepcopy

import numpy as np

from pebby.games.s5i5 import generate as generator
from pebby.games.s5i5 import names
from pebby.games.s5i5.env import Env, official_levels
from pebby.games.s5i5.official_identity import (
    native_board_geometry_hash,
    native_semantic_hash,
    official_board_geometry_hashes,
    official_copy_match,
    official_copy_index,
    official_semantic_hashes,
)


def _official_generated_spec(index):
    """Test-only generated-schema reconstruction; never emitted as data."""
    env = Env()
    env.set_level(index)
    rods = list(env.rods())
    rod_set = set(rods)
    children = getattr(env.game, names.ATTR_CHILDREN)
    rail_dispatch = getattr(env.game, names.ATTR_RAIL_RODS)
    button_dispatch = {}
    for button in env.buttons():
        color = int(button.pixels[button.height // 2, button.width // 2])
        button_dispatch[button] = [
            rod for rod in rods if int(rod.pixels[1, 1]) == color
        ]
    active = set().union(*rail_dispatch.values(), *button_dispatch.values())
    for parent, values in children.items():
        rod_children = {child for child in values if child in rod_set}
        if rod_children or any(names.TAG_PIN in child.tags for child in values):
            active.add(parent)
            active.update(rod_children)

    rod_rows = []
    rotation_of = getattr(env.game, names.METHOD_ROTATION_OF)
    for rod in rods:
        if rod not in active:
            continue
        rotation = int(rotation_of(rod))
        if rod.width > rod.height and rotation not in (90, 270):
            rotation = 90
        elif rod.height > rod.width and rotation not in (0, 180):
            # Passive uncapped leaves use the source fallback 270 even though
            # their collision geometry is vertical. Generated rods need a
            # vertical value to reproduce that physical state.
            rotation = 0
        rod_rows.append({
            "name": rod.name,
            "color": int(rod.pixels[1, 1]),
            "x": int(rod.x),
            "y": int(rod.y),
            "rotation": rotation,
            "length": max(rod.width, rod.height) // names.ROD_THICKNESS,
        })

    static_sprites = [rod for rod in rods if rod not in active]
    obstacle_cells = []
    for sprite in static_sprites:
        pixels = sprite.render()
        opaque = {
            (int(sprite.x) + x, int(sprite.y) + y)
            for y in range(pixels.shape[0])
            for x in range(pixels.shape[1])
            if int(pixels[y, x]) != -1
        }
        candidates = []
        for top in range(min(y for _, y in opaque) - 2,
                         max(y for _, y in opaque) + 1):
            for left in range(min(x for x, _ in opaque) - 2,
                              max(x for x, _ in opaque) + 1):
                block = {(left + bx, top + by)
                         for bx in range(names.ROD_THICKNESS)
                         for by in range(names.ROD_THICKNESS)}
                if block <= opaque:
                    candidates.append(([left, top], block))
        special = {(33 + bx, 9 + by)
                   for bx in range(names.ROD_THICKNESS)
                   for by in range(names.ROD_THICKNESS)}
        if index == 1 and special <= opaque:
            selected = [([33, 9], special)]
            selected.extend(row for row in candidates if not row[1].intersection(special))
        else:
            selected = candidates
        assert set().union(*(block for _, block in selected)) == opaque
        obstacle_cells.append([cell for cell, _ in selected])
    obstacles = [
        {"name": f"official-static-{ordinal}", "cells": rows, "color": 15}
        for ordinal, rows in enumerate(obstacle_cells)
    ]

    rails = []
    for rail, controlled in rail_dispatch.items():
        assert controlled
        rails.append({
            "name": rail.name,
            "color": int(controlled[0].pixels[1, 1]),
            "x": int(rail.x),
            "y": int(rail.y),
            "orientation": "horizontal" if rail.width > rail.height else "vertical",
            "style": "large" if max(rail.width, rail.height) == 13 else "compact",
        })
    buttons = [{
        "name": button.name,
        "color": int(button.pixels[button.height // 2, button.width // 2]),
        "x": int(button.x),
        "y": int(button.y),
        "style": "large" if button.width == 7 else "compact",
    } for button in env.buttons()]
    return {
        "difficulty": index + 1,
        "context_index": index,
        "step_counter": env.max_steps(),
        "rods": rod_rows,
        "obstacles": obstacles,
        "pins": [{"x": int(pin.x), "y": int(pin.y)} for pin in env.pins()],
        "targets": [{"x": int(target.x), "y": int(target.y)}
                    for target in env.targets()],
        "children": sorted(
            [parent.name, child.name]
            for parent, values in children.items()
            for child in values
            if parent in active and child in active
        ),
        "rails": rails,
        "buttons": buttons,
    }


def test_all_eight_official_levels_match_the_native_deny_set():
    references = official_semantic_hashes()
    board_references = official_board_geometry_hashes()
    assert len(references) == 8
    assert len(set(references)) == 8
    assert len(board_references) == len(set(board_references)) == 8
    for index, level in enumerate(official_levels()):
        env = Env([level])
        assert native_semantic_hash(env) == references[index]
        assert official_copy_index(env) == index


def test_all_eight_generated_build_reconstructions_are_denied():
    expected_kinds = {
        2: "board_geometry", 4: "board_geometry", 7: "board_geometry",
    }
    for index in range(8):
        spec = _official_generated_spec(index)
        rebuilt = Env([generator.build_level(spec)])
        assert rebuilt.max_steps() == Env([official_levels()[index]]).max_steps()
        assert native_board_geometry_hash(rebuilt) == official_board_geometry_hashes()[index]
        assert official_copy_match(rebuilt) == (
            index, expected_kinds.get(index, "native_semantic")
        )


def test_static_collision_repartition_cannot_evade_semantic_or_board_deny():
    spec = _official_generated_spec(1)
    original = Env([generator.build_level(spec)])
    original_semantic = native_semantic_hash(original)
    original_board = native_board_geometry_hash(original)
    source = next(row for row in spec["obstacles"] if [33, 9] in row["cells"])
    target = next(row for row in spec["obstacles"] if row is not source)
    source["cells"].remove([33, 9])
    target["cells"].append([33, 9])
    repartitioned = Env([generator.build_level(spec)])
    assert native_semantic_hash(repartitioned) == original_semantic
    assert native_board_geometry_hash(repartitioned) == original_board
    assert official_copy_match(repartitioned) == (1, "native_semantic")


def test_names_control_art_and_consistent_color_aliases_cannot_evade_deny():
    reference = official_semantic_hashes()[7]
    level = official_levels()[7]
    aliases = {}
    for index, sprite in enumerate(level.get_sprites()):
        aliases[sprite.name] = f"alias-{index}"

    for sprite in level.get_sprites():
        if names.TAG_ROD in sprite.tags and int(sprite.pixels[1, 1]) == 14:
            sprite.pixels[sprite.pixels == 14] = 6
        if names.TAG_RAIL in sprite.tags:
            if 14 in sprite.pixels:
                sprite.pixels[sprite.pixels == 14] = 6
            sprite.pixels[sprite.pixels == 2] = 0
            sprite.pixels[sprite.pixels == 4] = 0
        elif names.TAG_BUTTON in sprite.tags:
            center = int(sprite.pixels[sprite.height // 2, sprite.width // 2])
            sprite.pixels[sprite.pixels == 2] = 0
            sprite.pixels[sprite.pixels == 4] = 0
            sprite.pixels[sprite.height // 2, sprite.width // 2] = center
        sprite._name = aliases[sprite.name]
    level._data[names.KEY_CHILDREN] = [
        [aliases[parent], aliases[child]]
        for parent, child in level._data[names.KEY_CHILDREN]
    ]

    aliased = Env([level])
    assert native_semantic_hash(aliased) == reference
    assert official_copy_index(aliased) == 7


def _rotate_level_quarter_turn(level):
    for sprite in level.get_sprites():
        if {names.TAG_ROD, names.TAG_PIN, names.TAG_TARGET}.intersection(sprite.tags):
            old_x, old_y, old_width = int(sprite.x), int(sprite.y), sprite.width
            sprite.set_position(old_y + 9, -old_x - old_width + 9)
            sprite.pixels = np.rot90(sprite.pixels).copy()


def _reflect_level(level):
    for sprite in level.get_sprites():
        if {names.TAG_ROD, names.TAG_PIN, names.TAG_TARGET}.intersection(sprite.tags):
            sprite.set_position(-int(sprite.x) - sprite.width + 9, int(sprite.y) + 6)
            sprite.pixels = np.fliplr(sprite.pixels).copy()


def test_translated_orientation_preserving_d4_copy_remains_denied():
    level = official_levels()[2]
    _rotate_level_quarter_turn(level)
    transformed = Env([level])
    assert native_semantic_hash(transformed) == official_semantic_hashes()[2]
    assert official_copy_index(transformed) == 2


def test_reflection_preserves_rail_only_semantics_but_not_rotation_chirality():
    rail_only = official_levels()[0]
    _reflect_level(rail_only)
    assert official_copy_index(Env([rail_only])) == 0

    rotating = official_levels()[7]
    _reflect_level(rotating)
    reflected = Env([rotating])
    assert native_semantic_hash(reflected) != official_semantic_hashes()[7]
    assert official_copy_match(reflected) == (7, "board_geometry")


def test_generated_neighbor_is_allowed_and_encoding_aliases_do_not_matter(monkeypatch):
    spec = generator.generate(0, 1, split="train")
    assert spec is not None
    env = Env([generator.build_level(spec)])
    identity = native_semantic_hash(env)
    assert official_copy_index(env) is None
    assert generator.validate_full_standard(
        spec, generator.FULL_STANDARD_CONTRACT["curriculum"][0]
    ) == []

    alias = deepcopy(spec)
    rename = {rod["name"]: f"renamed-{index}"
              for index, rod in enumerate(alias["rods"])}
    for rod in alias["rods"]:
        rod["name"] = rename[rod["name"]]
    alias["children"] = [
        [rename[parent], rename[child]] for parent, child in alias["children"]
    ]
    alias["solution"] = [[names.ACTION_CLICK, 0, 0]]
    alias["private_certificate_only"] = {"ignored": True}
    alias["obstacles"][0]["cells"].append(alias["obstacles"][0]["cells"][0])
    assert native_semantic_hash(Env([generator.build_level(alias)])) == identity

    monkeypatch.setattr(
        generator, "official_copy_match", lambda env: (0, "board_geometry")
    )
    errors = generator.validate_full_standard(
        spec, generator.FULL_STANDARD_CONTRACT["curriculum"][0]
    )
    assert errors == ["official board_geometry copy shipped tier 1"]


def test_generation_reports_conservative_official_copy_rejection(monkeypatch):
    monkeypatch.setattr(
        generator, "official_copy_match", lambda env: (0, "board_geometry")
    )
    rejections = []
    assert generator.generate(
        0, 1, attempts=23, split="train", record_rejection=rejections.append
    ) is None
    assert any(
        row["reason"] == "official_board_geometry_copy:tier1"
        for row in rejections
    )


def test_semantic_hash_preserves_dispatch_while_board_gate_is_conservative():
    env = Env([official_levels()[7]])
    rail, controlled = next(
        (rail, rods)
        for rail, rods in getattr(env.game, names.ATTR_RAIL_RODS).items()
        if len(rods) > 1
    )
    controlled.pop()
    assert native_semantic_hash(env) != official_semantic_hashes()[7]
    assert official_copy_match(env) == (7, "board_geometry")
