"""Extended-bank gate and immutable-prefix orchestration, generated-only."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from pebby.ls20 import extended_curriculum as extended
from tools import extend_extended_bank as bank


class ExtendExtendedBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = extended.generate_level(20001, 1)[0]
        cls.validation = extended.generate_level(1020001, 1)[0]
        assert cls.train and cls.validation

    def fixture(self, root):
        prefixes = {}
        gate = {'status':'complete','coverage':'mixed','history':8,'epsilon':.15,'samples_per_level':16,'splits':{}}
        for split, spec in (('train', self.train), ('validation', self.validation)):
            path = root / f'{split}-prefix.jsonl'
            path.write_text(json.dumps(spec)+'\n')
            prefixes[split] = path
            output = root / f'{split}.npz'
            output.write_bytes(b'gate-fixture-only')
            gate['splits'][split] = {'source_sha256':bank.digest(path),'all_levels_win_covered':True,
                                    'levels':1,'rows':1,'output':str(output),'output_sha256':bank.digest(output),
                                    'branch_verification':{'branches':4,'expansions':1}}
        gate_path=root/'gate.json';gate_path.write_text(json.dumps(gate))
        return prefixes,gate_path,gate

    def test_prefix_proof_namespace_and_context_guards(self):
        self.assertEqual(bank.check_prefix([self.train], 'train'),20002)
        for change in ({'extended_curriculum_version':1},{'training_context_index':0},{'verification_lives':2}, {'search_truncated':True},
                       {'difficulty':2},{'gameplay_sha256':'wrong'}, {'slack_moves':7},
                       {'minimum_slack_moves':7}, {'distractor_count':0},
                       {'geometry_split':'validation'}, {'geometry_sha256':'wrong'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                bank.check_prefix([{**self.train,**change}],'train')

    def test_prefix_rejects_opposite_split_geometry_even_with_relabelled_seed(self):
        moved = {**self.validation, 'seed': self.train['seed'],
                 'training_context_index': self.train['seed'] % 7}
        with self.assertRaisesRegex(ValueError, 'geometry partition'):
            bank.check_prefix([moved], 'train')

    def test_profile_floor_is_enforced_without_silently_mixing_profiles(self):
        challenge, _ = extended.generate_level(20001, 1, quality_profile='challenge')
        self.assertIsNotNone(challenge)
        self.assertLess(challenge['minimum_slack_moves'], 8)
        self.assertEqual(bank.check_prefix([challenge], 'train', 'challenge'), 20002)
        with self.assertRaisesRegex(ValueError, 'quality profile'):
            bank.check_prefix([challenge], 'train', 'learning')
        for change in ({'budget_floor': 8}, {'minimum_slack_moves': -1},
                       {'quality_profile': 'unsupported'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                bank.check_prefix([{**challenge, **change}], 'train')

    def test_gate_binds_exact_pilot_and_collected_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            prefixes,path,gate=self.fixture(root)
            bank.check_gate(path,prefixes)
            for change in ({'status':'running'},{'epsilon':0},{'history':4}):
                path.write_text(json.dumps({**gate,**change}))
                with self.assertRaises(ValueError): bank.check_gate(path,prefixes)
            path.write_text(json.dumps(gate))
            (root/'train.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError): bank.check_gate(path,prefixes)

    def test_extension_preserves_prefix_and_dedup_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            prefixes,gate,_=self.fixture(root)
            old=root/'old.jsonl';old.write_text('')
            out=root/'out';report=root/'report.json'
            args=['--train-count','2','--validation-count','2','--train-prefix',str(prefixes['train']),
                  '--validation-prefix',str(prefixes['validation']),'--gate-report',str(gate),
                  '--out-dir',str(out),'--report',str(report),'--existing-banks',str(old)]
            with redirect_stdout(io.StringIO()): self.assertEqual(bank.main(args),0)
            for split in prefixes:
                self.assertTrue((out/f'{split}.jsonl').read_bytes().startswith(prefixes[split].read_bytes()))
                rows=bank.read_rows(out/f'{split}.jsonl')
                self.assertEqual(len(rows),2)
                self.assertEqual([row['difficulty'] for row in rows],[1,2])
                self.assertGreater(rows[1]['seed'],rows[0]['seed'])
            result=json.loads(report.read_text())
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['accepted_total'],4)
            self.assertEqual(result['dedup']['new_unique_gameplay_including_pilot'],4)
            before=(out/'train.jsonl').read_bytes()
            with self.assertRaises(ValueError): bank.main(args)
            self.assertEqual((out/'train.jsonl').read_bytes(),before)


if __name__=='__main__': unittest.main()
