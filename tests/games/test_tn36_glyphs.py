"""Public-opcode preset glyph regressions for generated TN36 levels."""

import pytest

from pebby.games.tn36 import names
from pebby.games.tn36.env import Env, upstream
from pebby.games.tn36.generate import (
    _preset_effect_glyph, _selector_access_points, build_level, generate,
)


@pytest.mark.parametrize(("opcode", "expected"), (
    (1, ("dx", -4, "iczcramoqjhw", 10, 57, 180)),
    (33, ("dy", -4, "iczcramoqjhw", 13, 54, 270)),
    (2, ("dx", 4, "iczcramoqjhw", 12, 57, 0)),
    (3, ("dy", 4, "iczcramoqjhw", 13, 56, 90)),
    (8, ("scale", 1, "iczcrascvkkwuqelhb", 12, 56, 0)),
    (9, ("scale", -1, "iczcrascvkkwdoylbb", 14, 58, 0)),
    (5, ("rotation", 90, "iczcraroumnb", 12, 56, 0)),
    (63, ("color", 15, "iczcrapumzpq", 13, 57, 0)),
))
def test_preset_glyph_matches_public_opcode_effect(opcode, expected):
    kind, value, sprite_name, x, y, rotation = expected
    assert names.OPCODE_EFFECTS[opcode] == (kind, value)
    glyph = _preset_effect_glyph(upstream().sprites, [opcode] * 3, 10)
    assert glyph is not None
    assert (glyph.name, glyph.x, glyph.y, glyph.rotation) == (
        sprite_name, x, y, rotation,
    )
    assert "sys_click" not in glyph.tags


def test_all_generated_selectors_are_visible_and_keep_native_first_hit_mapping():
    glyph_names = {
        "iczcramoqjhw", "iczcrapumzpq", "iczcraroumnb",
        "iczcrascvkkwdoylbb", "iczcrascvkkwuqelhb",
    }
    for difficulty in range(2, 8):
        spec = generate(700 + difficulty, difficulty, split="train")
        assert spec is not None, generate.last_report
        level = build_level(spec)
        selectors = [sprite for sprite in level._sprites if "tozzsf" in sprite.tags]
        glyphs = [sprite for sprite in level._sprites if sprite.name in glyph_names]
        assert [(sprite.x, sprite.y, sprite.width, sprite.height) for sprite in selectors] == [
            (x, 54, 9, 9) for x in spec["selector_xs"]
        ]
        assert len(glyphs) == len(selectors) == len(spec["preset_programs"])

        access_points = _selector_access_points(spec["selector_xs"])
        for index, (x, access_point) in enumerate(zip(spec["selector_xs"], access_points)):
            env = Env([level.clone()])
            frame = env.render()
            assert any(
                frame[y][pixel_x] in (11, 15)
                for y in range(55, 62)
                for pixel_x in range(x + 1, x + 8)
            )
            env.perform(names.ACTION_CLICK, *access_point)
            assert env.controller.jwmpcflifn == index
            actual_program = list(env.controller.mvqheosngn.vupcwzjtxu.vkuvtkaerv)
            assert actual_program == spec["preset_programs"][index]
