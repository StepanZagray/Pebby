"""``pebby.agent.glyph_model`` and its pretrainer ``pebby.agent.glyph_train``.

Covers: crop geometry agrees with the data builder; the grouped softmax and
loss/accuracy helpers; a fresh encoder is near chance and learns a synthetic
glyph fixture (rotation-asymmetric masks, colour as palette index) to 100%;
checkpoints carry provenance and reload strictly, refusing tampered format,
architecture, parameter counts or seed provenance; the pretrainer samples one
row from each of ``batch_size`` DISTINCT levels, validates schema, palette,
labels, split metadata and seed disjointness, and writes a strict checkpoint
plus report. Synthetic glyphs only, CPU only, no official level or frame.
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pebby.agent import glyph_model as gm
from pebby.agent import glyph_train as gt
from pebby.agent.glyph_model import GLYPH_CLASSES, GLYPH_SIZES, GlyphEncoder

COMBINATIONS = [(shape, colour, rotation) for shape in range(6) for colour in range(4) for rotation in range(4)]


# -------------------------------------------------------------------- fixture
def masks(seed=0):
    """Six random 6x6 masks; random masks are rotation-asymmetric with overwhelming probability."""
    rng = np.random.default_rng(seed)
    return [rng.random((6, 6)) < .5 for _ in range(6)]


def render_glyph(triple, shapes=None, background=0):
    shapes = masks() if shapes is None else shapes
    shape, colour, rotation = triple
    return np.where(np.rot90(shapes[shape], k=rotation), 8 + colour, background).astype(np.uint8)


def fixture(background=0):
    """All 96 triples, one glyph each."""
    glyphs = np.stack([render_glyph(triple, background=background) for triple in COMBINATIONS])
    triples = np.asarray(COMBINATIONS, dtype=np.int64)
    return torch.from_numpy(glyphs), torch.from_numpy(triples)


def write_glyph_npz(path, glyphs, triples, seeds, views, split, **meta_overrides):
    meta = {"format": gt.GLYPH_DATA_FORMAT, "source": "generated_only", "split": split,
            "crop": {"rows": [55, 61], "columns": [3, 9], "bounds": "half-open"}, **meta_overrides}
    np.savez(path, glyphs=np.asarray(glyphs, dtype=np.uint8), triples=np.asarray(triples, dtype=np.uint8),
             seeds=np.asarray(seeds, dtype=np.int32), views=np.asarray(views, dtype=np.uint8),
             meta=np.array(json.dumps(meta)))


def level_fixture(seeds, states=3, seed=0):
    """Per level ``states`` source states x 5 views with random triples, glyphs rendered from them."""
    rng = np.random.default_rng(seed)
    rows = []
    for level in seeds:
        for _ in range(states):
            for view in range(5):
                triple = (int(rng.integers(6)), int(rng.integers(4)), int(rng.integers(4)))
                rows.append((render_glyph(triple), triple, level, view))
    glyphs, triples, level_ids, views = zip(*rows)
    return np.stack(glyphs), np.asarray(triples), np.asarray(level_ids), np.asarray(views)


def provenance(**overrides):
    return {"source": "generated_only", "train_seeds": [1, 2, 3], "validation_seeds": [7, 8],
            "counts": {"train_rows": 3}, "results": {"validation": {"joint_accuracy": 1.}}, **overrides}


class GlyphEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_crop_geometry_matches_the_data_builder_and_the_hud_documentation(self):
        from tools.build_glyph_data import CROP
        self.assertEqual(list(gm.GLYPH_ROWS), CROP["rows"])
        self.assertEqual(list(gm.GLYPH_COLUMNS), CROP["columns"])
        self.assertEqual((gm.GLYPH_SIZE, gm.GLYPH_INPUTS, GLYPH_CLASSES), (6, 576, 14))
        frames = torch.zeros(2, 64, 64, dtype=torch.uint8)
        frames[:, 55:61, 3:9] = 9
        frames[:, 54, :] = 1
        frames[:, :, 9] = 1
        crop = gm.crop_glyph(frames)
        self.assertEqual(tuple(crop.shape), (2, 6, 6))
        self.assertTrue(bool((crop == 9).all()))
        self.assertEqual(tuple(gm.crop_glyph(torch.zeros(3, 4, 64, 64)).shape), (3, 4, 6, 6))
        with self.assertRaises(ValueError):
            gm.crop_glyph(torch.zeros(2, 63, 64))

    def test_forward_grouped_softmax_and_helpers(self):
        model = GlyphEncoder()
        self.assertEqual(model.parameter_count(), 576 * 64 + 64 + 64 * 14 + 14)
        glyphs, triples = fixture()
        logits = model(glyphs)
        self.assertEqual(tuple(logits.shape), (96, GLYPH_CLASSES))
        probabilities = GlyphEncoder.probabilities(logits)
        for scores in probabilities.split(GLYPH_SIZES, -1):
            torch.testing.assert_close(scores.sum(-1), torch.ones(96), atol=1e-6, rtol=1e-6)
        self.assertEqual(tuple(model(glyphs.view(4, 24, 6, 6)).shape), (4, 24, GLYPH_CLASSES))
        frames = torch.zeros(5, 64, 64, dtype=torch.long)
        frames[:, 55:61, 3:9] = glyphs[:5].long()
        torch.testing.assert_close(model.classify_frames(frames), logits[:5], atol=1e-6, rtol=1e-6)
        # Perfect logits score one, wrong ones zero; the loss is the three-field mean.
        perfect = torch.cat([torch.nn.functional.one_hot(triples[:, i], size).float() * 20
                             for i, size in enumerate(GLYPH_SIZES)], dim=-1)
        per_field, joint = GlyphEncoder.accuracies(perfect, triples)
        self.assertEqual(per_field.tolist(), [1., 1., 1.])
        self.assertEqual(float(joint), 1.)
        self.assertLess(float(GlyphEncoder.loss(perfect, triples)), 1e-3)
        uniform = torch.zeros(96, GLYPH_CLASSES)
        self.assertAlmostEqual(float(GlyphEncoder.loss(uniform, triples)),
                               (np.log(6) + 2 * np.log(4)) / 3, places=5)
        for bad in (torch.tensor([[6, 0, 0]]), torch.tensor([[0, 4, 0]]), torch.tensor([[0, 0, -1]])):
            with self.subTest(bad=bad.tolist()), self.assertRaises(ValueError):
                GlyphEncoder.loss(logits[:1], bad)
        with self.assertRaises(ValueError):
            GlyphEncoder.loss(logits, triples[:-1])
        with self.assertRaises(ValueError):
            model(torch.full((1, 6, 6), 16))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 5, 6))
        with self.assertRaises(ValueError):
            GlyphEncoder(hidden=0)

    def test_learns_the_synthetic_glyph_fixture_from_chance(self):
        glyphs, triples = fixture()
        self.assertEqual(len({bytes(g.numpy().tobytes()) for g in glyphs}), 96, "every triple has its own pattern")
        model = GlyphEncoder()
        with torch.no_grad():
            _, joint_before = GlyphEncoder.accuracies(model(glyphs), triples)
            loss_before = float(GlyphEncoder.loss(model(glyphs), triples))
        self.assertLess(float(joint_before), .5)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        for _ in range(300):
            loss = GlyphEncoder.loss(model(glyphs), triples)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            per_field, joint = GlyphEncoder.accuracies(model(glyphs), triples)
        self.assertEqual(per_field.tolist(), [1., 1., 1.])
        self.assertEqual(float(joint), 1.)
        self.assertLess(float(loss), .1 * loss_before)
        # Rotation is read from the pixels: rotating a glyph changes the rotation logits only if trained so.
        rotated = torch.from_numpy(np.stack([np.rot90(g.numpy(), k=1).copy() for g in glyphs]))
        expected = triples.clone()
        expected[:, 2] = (expected[:, 2] + 1) % 4
        with torch.no_grad():
            per_field, _ = GlyphEncoder.accuracies(model(rotated), expected)
        self.assertEqual(per_field.tolist(), [1., 1., 1.], "the fixture's rotations are consistent")

    def test_checkpoint_round_trip_and_strict_rejections(self):
        model = GlyphEncoder().eval()
        glyphs, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "glyph.pt"
            saved = gm.save_glyph_checkpoint(path, model, **provenance(), extra={"note": 1})
            self.assertEqual(saved["format"], gm.GLYPH_FORMAT)
            self.assertEqual(saved["parameters"], model.parameter_count())
            restored, loaded = gm.load_glyph_checkpoint(path)
            self.assertFalse(restored.training)
            self.assertEqual(loaded["train_seeds"], [1, 2, 3])
            self.assertEqual(loaded["extra"], {"note": 1})
            with torch.inference_mode():
                self.assertTrue(torch.equal(restored(glyphs), model(glyphs)))
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["glyph.pt"])
            # Metadata guards at save time.
            for bad in (dict(source="official"), dict(train_seeds=[]), dict(train_seeds=[1, 7]),
                        dict(train_seeds=[1, 1, 2]), dict(counts={}), dict(results=None),
                        dict(validation_seeds=["7"])):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    gm.save_glyph_checkpoint(path, model, **provenance(**bad))
            with self.assertRaisesRegex(ValueError, "lacks provenance"):
                gm.save_glyph_checkpoint(path, model, source="generated_only")
            with self.assertRaisesRegex(ValueError, "reserved"):
                gm.save_glyph_checkpoint(path, model, config={}, **provenance())
            with self.assertRaises(ValueError):
                gm.save_glyph_checkpoint(path, torch.nn.Linear(1, 1), **provenance())
            # Tampered files are refused at load time.
            for name, tamper in (("format", lambda c: c.update(format="pebby.ls20-world-policy.v1")),
                                 ("hidden", lambda c: c["config"].update(hidden=32)),
                                 ("config_key", lambda c: c["config"].update(bogus=1)),
                                 ("parameters", lambda c: c.update(parameters=1)),
                                 ("source", lambda c: c.update(source="mixed")),
                                 ("overlap", lambda c: c.update(validation_seeds=[1])),
                                 ("missing_results", lambda c: c.pop("results")),
                                 ("missing_weight", lambda c: c["weights"].pop("mlp.2.bias")),
                                 ("extra_weight", lambda c: c["weights"].update(other=torch.zeros(1)))):
                corrupted = dict(saved, config=dict(saved["config"]), weights=dict(saved["weights"]))
                tamper(corrupted)
                bad_path = Path(directory) / f"{name}.pt"
                torch.save(corrupted, bad_path)
                with self.subTest(name=name), self.assertRaises((ValueError, RuntimeError)):
                    gm.load_glyph_checkpoint(bad_path)


class GlyphTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._threads = torch.get_num_threads()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._threads)

    def test_level_sampler_draws_one_row_from_each_of_k_distinct_levels(self):
        seeds = np.array([5, 5, 5, 9, 9, 2, 7, 7, 7, 7])
        sampler = gt.LevelSampler(seeds)
        self.assertEqual(len(sampler), 4)
        rng = np.random.default_rng(3)
        seen_rows = set()
        for _ in range(50):
            rows = sampler.sample(4, rng)
            self.assertEqual(len(rows), 4)
            self.assertEqual(sorted(seeds[rows].tolist()), [2, 5, 7, 9], "every level exactly once")
            seen_rows.update(rows.tolist())
        self.assertEqual(seen_rows, set(range(10)), "every row of every level is reachable")
        with self.assertRaisesRegex(ValueError, "distinct levels"):
            sampler.sample(5, rng)

    def test_cli_trains_validates_and_refuses_bad_splits(self):
        train = level_fixture(seeds=range(8), states=3, seed=1)
        validation = level_fixture(seeds=range(100, 104), states=2, seed=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_glyph_npz(root / "train.npz", *train, split="train")
            write_glyph_npz(root / "validation.npz", *validation, split="validation")
            checkpoint_path, report_path = root / "out" / "glyph.pt", root / "report.json"
            flags = ["--train", str(root / "train.npz"), "--validation", str(root / "validation.npz"),
                     "--checkpoint-out", str(checkpoint_path), "--report-out", str(report_path),
                     "--steps", "30", "--batch-size", "4", "--seed", "0", "--threads", "1"]
            buffer = io.StringIO()
            with redirect_stdout(buffer), redirect_stderr(buffer):
                self.assertEqual(gt.main(flags), 0, buffer.getvalue())
            self.assertIn("30 steps x 4 distinct levels", buffer.getvalue())
            model, checkpoint = gm.load_glyph_checkpoint(checkpoint_path)
            self.assertEqual(checkpoint["source"], "generated_only")
            self.assertEqual(checkpoint["train_seeds"], list(range(8)))
            self.assertEqual(checkpoint["validation_seeds"], list(range(100, 104)))
            self.assertEqual(checkpoint["counts"], {"train_rows": 120, "train_levels": 8,
                                                    "validation_rows": 40, "validation_levels": 4})
            self.assertEqual(checkpoint["batch_unit"], "distinct_level")
            self.assertEqual((checkpoint["steps"], checkpoint["batch_size"]), (30, 4))
            self.assertEqual(len(checkpoint["hashes"]["train"]), 64)
            self.assertEqual(checkpoint["loss_trace"][-1]["step"], 30)
            for split in ("train", "validation"):
                scores = checkpoint["results"][split]
                for key in ("shape_accuracy", "color_accuracy", "rotation_accuracy", "joint_accuracy", "loss"):
                    self.assertIn(key, scores)
                self.assertEqual(sorted(scores["joint_accuracy_by_view"]), ["0", "1", "2", "3", "4"])
            report = json.loads(report_path.read_text())
            self.assertNotIn("weights", report)
            self.assertEqual(report["checkpoint"], str(checkpoint_path))
            self.assertEqual(report["results"], checkpoint["results"])
            self.assertEqual(sorted(p.name for p in checkpoint_path.parent.iterdir()), ["glyph.pt"])
            # The evaluation helper agrees with the saved results on the full split.
            loaded = gt.load_glyph_split(root / "validation.npz", "validation")
            self.assertEqual(gt.evaluate(model, loaded)["joint_accuracy"],
                             checkpoint["results"]["validation"]["joint_accuracy"])
            # Guards: each refusal exits non-zero and writes no checkpoint.
            bad_root = root / "bad"
            bad_root.mkdir()
            overlapping = level_fixture(seeds=[3, 200], states=2, seed=5)
            write_glyph_npz(root / "overlap.npz", *overlapping, split="validation")
            write_glyph_npz(root / "wrong_source.npz", *validation, split="validation", source="mixed")
            write_glyph_npz(root / "wrong_split.npz", *validation, split="train")
            write_glyph_npz(root / "wrong_crop.npz", *validation, split="validation",
                            crop={"rows": [54, 60], "columns": [3, 9]})
            glyphs, triples, seeds, views = validation
            bad_triples = triples.copy()
            bad_triples[0, 0] = 6
            write_glyph_npz(root / "bad_label.npz", glyphs, bad_triples, seeds, views, split="validation")
            bad_glyphs = glyphs.copy()
            bad_glyphs[0, 0, 0] = 16
            write_glyph_npz(root / "bad_pixel.npz", bad_glyphs, triples, seeds, views, split="validation")
            base = ["--train", str(root / "train.npz"), "--checkpoint-out", str(bad_root / "glyph.pt"),
                    "--steps", "2", "--threads", "1"]
            cases = {
                "overlap": ["--validation", str(root / "overlap.npz"), "--batch-size", "4"],
                "wrong_source": ["--validation", str(root / "wrong_source.npz"), "--batch-size", "4"],
                "wrong_split": ["--validation", str(root / "wrong_split.npz"), "--batch-size", "4"],
                "wrong_crop": ["--validation", str(root / "wrong_crop.npz"), "--batch-size", "4"],
                "bad_label": ["--validation", str(root / "bad_label.npz"), "--batch-size", "4"],
                "bad_pixel": ["--validation", str(root / "bad_pixel.npz"), "--batch-size", "4"],
                "not_power_of_two": ["--validation", str(root / "validation.npz"), "--batch-size", "3"],
                "too_large": ["--validation", str(root / "validation.npz"), "--batch-size", "2048"],
                "more_levels_than_data": ["--validation", str(root / "validation.npz"), "--batch-size", "16"],
                "min_levels": ["--validation", str(root / "validation.npz"), "--batch-size", "4",
                               "--min-train-levels", "9"],
                "missing": ["--validation", str(root / "nope.npz"), "--batch-size", "4"],
            }
            for name, extra in cases.items():
                with self.subTest(name=name), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as failure:
                        gt.main(base + extra)
                    self.assertNotEqual(failure.exception.code, 0)
            self.assertEqual(list(bad_root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
