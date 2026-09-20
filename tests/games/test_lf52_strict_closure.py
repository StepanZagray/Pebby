"""Focused LF52 strict-boundary and failed-seed diagnostics tests."""

import copy
import hashlib
import json

import pytest

from pebby.games.lf52 import bank
from pebby.games.lf52.generate import FULL_STANDARD_CONTRACT, generate, validate_full_standard


def _proof_hash(spec):
    spec["proof_sha256"] = hashlib.sha256(
        json.dumps(spec["proof"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def small_tier_spec():
    spec = generate(10_001, 1, attempts=64, split="train")
    assert spec is not None, generate.last_diagnostics
    assert validate_full_standard(spec, FULL_STANDARD_CONTRACT["curriculum"][0]) == []
    return spec


@pytest.mark.parametrize(
    ("path", "replacement", "message_fragment"),
    [
        (("proof", "context_index"), False, "proof.context_index"),
        (("proof", "context_index"), 0.0, "proof.context_index"),
        (
            ("solution_mechanics", "native_actions_consumed"),
            lambda value: float(value),
            "solution mechanics",
        ),
        (("coverage", "all_official_tier_mechanics"), 1, "coverage"),
        (("actual_distribution", "pegs"), lambda value: float(value), "structural distribution"),
    ],
)
def test_nested_evidence_requires_exact_json_scalar_types(
    small_tier_spec, path, replacement, message_fragment
):
    changed = copy.deepcopy(small_tier_spec)
    container, key = path
    if callable(replacement):
        replacement = replacement(changed[container][key])
    changed[container][key] = replacement
    if container == "proof":
        _proof_hash(changed)

    errors = validate_full_standard(changed, FULL_STANDARD_CONTRACT["curriculum"][0])

    assert errors
    assert any(message_fragment in error for error in errors)


@pytest.mark.parametrize(("key", "replacement"), [("expanded", "bad"), ("search_limit", None)])
def test_malformed_proof_counters_return_finite_diagnostics(small_tier_spec, key, replacement):
    changed = copy.deepcopy(small_tier_spec)
    changed["proof"][key] = replacement
    _proof_hash(changed)

    errors = validate_full_standard(changed, FULL_STANDARD_CONTRACT["curriculum"][0])

    assert 1 <= len(errors) <= 8
    assert any("proof" in error for error in errors)


@pytest.mark.parametrize("field", ["actual_distribution", "solution_mechanics", "coverage", "presentation"])
def test_malformed_evidence_containers_are_rejected_without_throwing(small_tier_spec, field):
    changed = copy.deepcopy(small_tier_spec)
    changed[field] = []

    errors = validate_full_standard(changed, FULL_STANDARD_CONTRACT["curriculum"][0])

    assert errors


def test_minimal_failed_bank_retains_bounded_hash_bound_diagnostics(tmp_path):
    specs, tried = bank.build(
        1,
        seed=800,
        difficulty=1,
        max_attempts=1,
        attempts=1,
        node_limit=1,
    )

    assert specs == [] and tried == 1
    diagnostics = bank.build.last_diagnostics
    json.dumps(diagnostics)
    assert diagnostics["failed_requested_seeds_omitted"] == 0
    assert diagnostics["serialized_bank_sha256"] == hashlib.sha256(b"").hexdigest()
    assert len(diagnostics["failed_requested_seeds"]) == 1
    failure = diagnostics["failed_requested_seeds"][0]
    assert failure["requested_seed"] == 800
    assert failure["cause"] == "search_cap"
    assert failure["details"][0]["reason"] == "search_cap"
    assert "beyond bound 1" in failure["details"][0]["details"][0]

    bank_path = bank.save(specs, tmp_path / "bank.jsonl")
    report_path = bank.save_diagnostics(specs, tmp_path / "bank.diagnostics.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["bank_sha256"] == hashlib.sha256(bank_path.read_bytes()).hexdigest()
    report_hash = report.pop("report_sha256")
    assert report_hash == hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_bank_diagnostics_reject_stale_or_unrelated_snapshots_before_write(
    small_tier_spec, tmp_path
):
    specs = [small_tier_spec]
    serialized_hash = hashlib.sha256(bank._serialized_bank(specs)).hexdigest()
    matching = {
        "serialized_bank_sha256": serialized_hash,
        "accepted": 1,
        "accepted_geometry_d4_sha256": [small_tier_spec["geometry_d4_sha256"]],
        "accepted_gameplay_sha256": [small_tier_spec["gameplay_sha256"]],
        "accepted_action_sequence_sha256": [small_tier_spec["action_sequence_sha256"]],
    }
    report_path = bank.save_diagnostics(
        specs, tmp_path / "matching.json", diagnostics=matching
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["bank_sha256"] == serialized_hash
    assert report["diagnostics"] == matching

    for key, replacement in (
        ("serialized_bank_sha256", "0" * 64),
        ("accepted", 0),
        ("accepted_geometry_d4_sha256", []),
        ("accepted_gameplay_sha256", []),
        ("accepted_action_sequence_sha256", []),
    ):
        stale = copy.deepcopy(matching)
        stale[key] = replacement
        destination = tmp_path / f"stale-{key}.json"
        with pytest.raises(ValueError, match="do not describe"):
            bank.save_diagnostics(specs, destination, diagnostics=stale)
        assert not destination.exists()

    previous = bank.build.last_diagnostics
    try:
        bank.build.last_diagnostics = {**matching, "serialized_bank_sha256": "stale"}
        destination = tmp_path / "stale-default.json"
        with pytest.raises(ValueError, match="do not describe"):
            bank.save_diagnostics(specs, destination)
        assert not destination.exists()
    finally:
        bank.build.last_diagnostics = previous
