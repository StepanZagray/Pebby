"""Bounded all-tier quality sample; the larger 8x8 audit lives in the family note."""

from collections import Counter
import random

from pebby.games.s5i5.generate import (
    DIFFICULTIES,
    FULL_STANDARD_CONTRACT,
    MECHANICS_INVENTORY_VERSION,
    _draft,
    _greedy_native_reduction,
    generate,
    solution_actions,
    validate_full_standard,
)
from pebby.games.s5i5.reference_profiles import PROFILES


def test_every_tier_draft_consumes_seed_entropy():
    """Rod/obstacle drafts must differ by seed before any witness search.

    Tier 4 once emitted one fixed calibrated layout for every seed, so every
    accepted row shared one geometry identity and one split.
    """
    for difficulty in DIFFICULTIES:
        layouts = set()
        for seed in range(8):
            rng = random.Random(f"{MECHANICS_INVENTORY_VERSION}:{seed}:{difficulty}")
            spec, _ = _draft(rng, difficulty)
            assert spec is not None
            layouts.add((
                tuple((rod["x"], rod["y"], rod["rotation"], rod["length"])
                      for rod in spec["rods"]),
                tuple(tuple(map(tuple, obstacle["cells"]))
                      for obstacle in spec["obstacles"]),
            ))
        assert len(layouts) >= 6, difficulty


def test_bounded_all_tier_quality_audit():
    rejections = Counter()

    def record(row):
        rejections[(row["difficulty"], row["reason"])] += 1

    for difficulty in DIFFICULTIES:
        identities = set()
        route_identities = set()
        accepted = 0
        # Attempts are bounded, so a seed can exhaust its budget (tier 5 does
        # for seed 1).  Sample seeds in order until two rows are accepted.
        for seed in range(6):
            if accepted == 2:
                break
            spec = generate(seed, difficulty, split="train", record_rejection=record)
            if spec is None:
                continue
            accepted += 1
            assert validate_full_standard(
                spec, FULL_STANDARD_CONTRACT["curriculum"][difficulty - 1]
            ) == []
            actions = solution_actions(spec)
            assert _greedy_native_reduction(spec, actions) == actions
            assert len(actions) >= PROFILES[difficulty]["action_range"][0]
            assert spec["shortcut_evidence"]["compact_outcome"] in {
                "exhaustive_clear", "bounded_unknown",
            }
            dependencies = spec["dependency_evidence"]
            assert all(row["essential"] for row in dependencies["pin_attachments"])
            if PROFILES[difficulty].get("require_linked"):
                if difficulty == 4:
                    # The official tier-4 pin is independent. All four linked
                    # shared children are calibrated causal blockers.
                    assert dependencies["recursive_edges"]
                    assert all(row["essential"] and not row["pins"]
                               for row in dependencies["recursive_edges"])
                elif difficulty == 8:
                    # The official tier-8 pin is also independent; linked work
                    # is causal through at least one off-pin branch edge.
                    assert all(not row["pins"]
                               for row in dependencies["recursive_edges"])
                    assert any(row["essential"]
                               for row in dependencies["recursive_edges"])
                else:
                    # Official tiers 5 and 7 pair one independent pin carrier
                    # with one linked carrier; tier 6 links its only pin.
                    # Every pin that rides a recursive edge must depend on
                    # an essential one, and at least one pin must be linked.
                    linked_pins = [
                        attachment["pin"]
                        for attachment in dependencies["pin_attachments"]
                        if any(attachment["pin"] in row["pins"]
                               for row in dependencies["recursive_edges"])
                    ]
                    assert linked_pins
                    for pin in linked_pins:
                        assert any(
                            row["essential"] and pin in row["pins"]
                            for row in dependencies["recursive_edges"]
                        )
            if PROFILES[difficulty].get("require_shared"):
                assert any(row["essential"] for row in dependencies["shared_companions"])
            if PROFILES[difficulty].get("require_branch"):
                assert any(row["essential"] for row in dependencies["branch_edges"])
            identities.add(spec["geometry_d4_sha256"])
            route_identities.add(spec["solution_semantic_sha256"])
        assert accepted == 2, difficulty
        assert len(identities) == 2
        assert route_identities
