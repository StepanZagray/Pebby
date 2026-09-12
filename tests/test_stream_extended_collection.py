"""Immutable growing-bank snapshots and checked worker collection."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from pebby.ls20.extended_curriculum import generate_legacy_level as generate_level
from tools import stream_extended_collection as stream


class StreamingExtendedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = generate_level(20001,1)[0]
        assert cls.spec

    def test_partial_append_is_never_consumed_and_malformed_complete_line_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bank.jsonl'
            first=json.dumps(self.spec).encode()+b'\n'
            path.write_bytes(first+b'{"seed":20002')
            self.assertEqual(stream.complete_lines(path),[first])
            path.write_bytes(first+b'broken\n')
            with self.assertRaises(ValueError): stream.complete_lines(path)

    def test_immutable_prefix_and_cross_source_dedup_guards(self):
        line=json.dumps(self.spec).encode()+b'\n'
        prefix_hash=hashlib.sha256(line).hexdigest()
        seen=set()
        result=stream.validate_bank_prefix([line],'train',prefix_hash,1,[],set(),seen)
        self.assertEqual(result,[json.loads(line)])
        stream.validate_bank_prefix([line],'train',prefix_hash,1,[line],set(),seen)
        with self.assertRaisesRegex(ValueError,'duplicates'):
            stream.validate_bank_prefix([line],'train',prefix_hash,1,[],set(),seen)
        old=json.dumps({**self.spec,'extended_curriculum_version':1}).encode()+b'\n'
        with self.assertRaisesRegex(ValueError,'contextual proof'):
            stream.validate_bank_prefix([old],'train',hashlib.sha256(old).hexdigest(),1,[],set(),set())
        changed=json.dumps({**self.spec,'seed':20002}).encode()+b'\n'
        with self.assertRaisesRegex(ValueError,'pilot prefix'):
            stream.validate_bank_prefix([changed],'train',prefix_hash,1,[line],set(),set())

    def test_worker_keeps_next_policy_labels_and_checks_all_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=Path(directory)/'pids.jsonl'
            old=os.environ.get('PEBBY_EXTENDED_COLLECTION_PID_LEDGER')
            os.environ['PEBBY_EXTENDED_COLLECTION_PID_LEDGER']=str(ledger)
            try:
                rows,proof=stream.checked_worker((self.spec,{'history':8,'samples':16,'epsilon':.15,
                                                           'coverage':'mixed','search_limit':600000}))
            finally:
                if old is None: os.environ.pop('PEBBY_EXTENDED_COLLECTION_PID_LEDGER',None)
                else: os.environ['PEBBY_EXTENDED_COLLECTION_PID_LEDGER']=old
            self.assertTrue(rows)
            self.assertIn('next_optimal',rows[0])
            counts=proof['branch_verification']
            self.assertEqual(counts['branches'],4*counts['expansions'])
            self.assertTrue(proof['win_covered'])
            identity=json.loads(ledger.read_text())
            self.assertEqual(identity,stream.process_identity(os.getpid()))


if __name__=='__main__': unittest.main()
