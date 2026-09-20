"""Exact obstacle-pixel identity and canonical obstacle admission regressions."""

from copy import deepcopy

import pytest

from pebby.games.s5i5.generate import (
    FULL_STANDARD_CONTRACT,
    _profile_errors,
    generate,
    obstacle_sprite,
    validate_full_standard,
)
from pebby.games.s5i5.generation_quality import (
    gameplay_hash,
    geometry_d4_hash,
    geometry_hash,
    solution_semantic_hash,
)


@pytest.fixture(scope="module")
def authentic_specs():
    specs = {
        difficulty: generate(0, difficulty, split="train", attempts=400)
        for difficulty in (3, 5)
    }
    assert all(specs.values())
    for difficulty, spec in specs.items():
        assert validate_full_standard(
            spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
        ) == []
        assert all(obstacle["color"] == 15 for obstacle in spec["obstacles"])
    return specs


def test_boundary_pixel_displacement_changes_identity_and_fails_canonical_schema(
    authentic_specs,
):
    original = authentic_specs[3]
    moved = deepcopy(original)
    changed = 0
    for cell in moved["obstacles"][0]["cells"]:
        if cell[0] == 64 and 0 <= cell[1] < 64:
            cell[0] = 65
            changed += 1
    assert changed

    assert geometry_hash(moved) != geometry_hash(original)
    assert geometry_d4_hash(moved) != geometry_d4_hash(original)
    assert gameplay_hash(moved) != gameplay_hash(original)
    errors = validate_full_standard(
        moved, FULL_STANDARD_CONTRACT["curriculum"][2]
    )
    assert any("canonical boundary" in error for error in errors)


def test_off_lattice_interior_collision_change_is_identity_visible_and_rejected(
    authentic_specs,
):
    original = authentic_specs[3]
    moved = deepcopy(original)
    cell = next(
        cell
        for obstacle in moved["obstacles"][1:]
        for cell in obstacle["cells"]
        if cell == [27, 3]
    )
    cell[0] = 28

    original_wall = obstacle_sprite("original", [[27, 3]])
    moved_wall = obstacle_sprite("moved", [[28, 3]])
    lattice_probe = obstacle_sprite("probe", [[30, 3]])
    assert original_wall.collides_with(lattice_probe) is False
    assert moved_wall.collides_with(lattice_probe) is True
    assert geometry_hash(moved) != geometry_hash(original)
    assert gameplay_hash(moved) != gameplay_hash(original)
    assert _profile_errors(moved) == []
    errors = validate_full_standard(
        moved, FULL_STANDARD_CONTRACT["curriculum"][2]
    )
    assert any("3-pixel arena lattice" in error for error in errors)


def test_duplicate_boundary_cell_cannot_salt_pixel_union_identity(authentic_specs):
    original = authentic_specs[3]
    duplicated = deepcopy(original)
    duplicated["obstacles"][0]["cells"].append(
        list(duplicated["obstacles"][0]["cells"][0])
    )

    # The native sprite and collision union are unchanged, so identity is too;
    # strict admission rejects the redundant source encoding.
    assert geometry_hash(duplicated) == geometry_hash(original)
    assert geometry_d4_hash(duplicated) == geometry_d4_hash(original)
    assert gameplay_hash(duplicated) == gameplay_hash(original)
    assert _profile_errors(duplicated, require_evidence=False) == []
    errors = validate_full_standard(
        duplicated, FULL_STANDARD_CONTRACT["curriculum"][2]
    )
    assert any("canonical boundary" in error for error in errors)


def test_interior_obstacle_cannot_spoof_boundary_role_off_camera(authentic_specs):
    spoofed = deepcopy(authentic_specs[3])
    interior = spoofed["obstacles"][1]
    interior["name"] = "boundary_extra"
    interior["boundary"] = True
    interior["cells"] = [[0, 60]]
    errors = validate_full_standard(
        spoofed, FULL_STANDARD_CONTRACT["curriculum"][2]
    )
    assert any("canonical schema fields" in error for error in errors)
    assert any("names/order" in error for error in errors)
    assert any("3-pixel arena lattice" in error for error in errors)


def test_pixel_identities_keep_translation_and_multicolor_controls(authentic_specs):
    original = authentic_specs[3]
    translated = deepcopy(original)
    for key in ("rods", "pins", "targets"):
        for row in translated[key]:
            row["x"] += 3
    for obstacle in translated["obstacles"]:
        obstacle["cells"] = [[x + 3, y] for x, y in obstacle["cells"]]
    for edge in translated["native_relations"]["pin_edges"]:
        edge[1][0] += 3

    assert geometry_hash(translated) == geometry_hash(original)
    assert geometry_d4_hash(translated) == geometry_d4_hash(original)
    assert gameplay_hash(translated) == gameplay_hash(original)
    assert solution_semantic_hash(translated) == solution_semantic_hash(original)

    coupled = deepcopy(original)
    coupled["rails"][0]["secondary_color"] = coupled["rails"][1]["color"]
    assert gameplay_hash(coupled) != gameplay_hash(original)
    assert solution_semantic_hash(coupled) != solution_semantic_hash(original)
