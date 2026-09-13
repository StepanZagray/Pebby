"""The immutable reference bank's versioned-generator compatibility gate."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from pebby.level_banks import (
    LEGACY_COMPATIBILITY_MANIFEST_SHA256,
    LEGACY_COMPATIBILITY_ARCHIVE,
    LEGACY_COMPATIBILITY_CHANGED,
    LEGACY_COMPATIBILITY_RECEIPT,
    LEGACY_COMPATIBILITY_RECEIPT_SHA256,
    LevelBanks, canonical_hash,
)


ROOT = Path(__file__).resolve().parents[1]


class CompatibilityEvidenceTests(unittest.TestCase):
    def test_committable_receipt_and_archived_sources_are_self_consistent(self):
        receipt_path = LEGACY_COMPATIBILITY_RECEIPT
        self.assertTrue(receipt_path.is_file())
        self.assertEqual(hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                         LEGACY_COMPATIBILITY_RECEIPT_SHA256)
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt["archive_root"], "pebby/compatibility/reference-unequal-v1")
        expected = {
            "bank.py.txt": "dd5e7cf4d22ca714dbd2b25bbd64ca6ef1f576fd492c44b4c082bbb9cf8ee454",
            "generate.py.txt": "90a7fe0e7f24b7f1646546b402c26793650407bd53df267d2a6ca1e0f92a4c81",
        }
        self.assertEqual(receipt["archive_code_hashes"], expected)
        for name, digest in expected.items():
            path = LEGACY_COMPATIBILITY_ARCHIVE / name
            self.assertTrue(path.is_file(), path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        for relative in (
                receipt["historical_receipt"]["path"],
                receipt["generator_verification"]["path"]):
            self.assertTrue((ROOT / relative).is_file(), relative)


@unittest.skipUnless(
    (ROOT / "data/ls20-reference-unequal-v1/manifest.json").is_file(),
    "requires the local generated reference-unequal-v1 bank manifest")
class LegacyBankCompatibilityTests(unittest.TestCase):
    def test_default_bank_uses_the_pinned_compatibility_receipt(self):
        info = LevelBanks().catalogue()["banks"][0]
        self.assertEqual(info["id"], "reference-unequal-v1")
        self.assertEqual(info["status"], "complete")
        receipt = json.loads(LEGACY_COMPATIBILITY_RECEIPT.read_text())
        self.assertEqual(receipt["bank"], info["id"])
        self.assertEqual(len(receipt["fixture_replay"]["fixtures"]), 7)

    def test_receipt_tampering_fails_closed(self):
        banks = LevelBanks()
        with patch("pebby.level_banks.LEGACY_COMPATIBILITY_RECEIPT_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "Bank integrity check failed"):
                banks.catalogue()

    def test_default_manifest_is_pinned_even_when_its_listed_sources_match(self):
        banks = LevelBanks()
        manifest = json.loads((ROOT / "data/ls20-reference-unequal-v1/manifest.json").read_text())
        actual = {path: banks._hash(path) for path in manifest["code_hashes"]}
        self.assertEqual(canonical_hash(manifest), LEGACY_COMPATIBILITY_MANIFEST_SHA256)
        with self.assertRaisesRegex(ValueError, "manifest binding changed"):
            banks._legacy_source_compatibility(
                "reference-unequal-v1", manifest, "0" * 64, actual)

    def test_a_changed_manifest_listed_source_is_not_accepted_by_receipt(self):
        banks = LevelBanks()
        original = banks._hash

        def changed(path):
            value = original(path)
            if str(Path(path).resolve()) not in LEGACY_COMPATIBILITY_CHANGED and str(path).endswith(
                    "reference_generator.py"):
                return hashlib.sha256(b"changed").hexdigest()
            return value

        with patch.object(banks, "_hash", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Bank integrity check failed"):
                banks.catalogue()

    def test_missing_or_corrupt_archived_generator_fails_closed(self):
        banks = LevelBanks()
        original = banks._hash

        def missing(path):
            if Path(path).resolve() == (LEGACY_COMPATIBILITY_ARCHIVE / "bank.py.txt").resolve():
                raise FileNotFoundError(path)
            return original(path)

        with patch.object(banks, "_hash", side_effect=missing):
            with self.assertRaisesRegex(ValueError, "Bank integrity check failed"):
                banks.catalogue()

        def corrupt(path):
            if Path(path).resolve() == (LEGACY_COMPATIBILITY_ARCHIVE / "bank.py.txt").resolve():
                return "0" * 64
            return original(path)

        with patch.object(banks, "_hash", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "Bank integrity check failed"):
                banks.catalogue()

    def test_unexpected_current_generator_change_is_rejected(self):
        banks = LevelBanks()
        original = banks._hash

        def changed(path):
            if Path(path).resolve() == (ROOT / "pebby/ls20/bank.py").resolve():
                return "0" * 64
            return original(path)

        with patch.object(banks, "_hash", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Bank integrity check failed"):
                banks.catalogue()


if __name__ == "__main__":
    unittest.main()
