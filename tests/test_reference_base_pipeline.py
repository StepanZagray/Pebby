import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_reference_base_pipeline as pipeline


class ReferencePipelineTests(unittest.TestCase):
    def probe(self):
        from tools.preflight_reference_world import fresh_config
        from pebby.agent.world_model import DEFAULT_WEIGHTS
        return dict(status='complete',smoke=False,fresh_initialization=True,source_unchanged=True,
            checkpoint_saved=False,requires_fresh_fit_initialization=True,selected_batch_size=1024,
            attempts=[dict(status='complete',batch_size=1024,initial_weights_sha256='a'*64)],precision='bf16',checkpoint_encoder=True,
            checkpoint_loops=True,encoder_chunk_size=128,seed=42,config=fresh_config().__dict__,
            weights={**DEFAULT_WEIGHTS,'successor_policy':1.},optimizer=dict(lr=.0003,weight_decay=.05),
            curriculum=dict(start=pipeline.START,end=pipeline.END))

    def test_distance_head_uses_full_maximum_and_never_saturates_boundary(self):
        for high,expected in [(90,128),(127,128),(128,256),(255,256),(256,512)]:
            report=dict(status='complete',sources_unchanged=True,splits={s:dict(max_distance=high) for s in ('train','validation')})
            self.assertEqual(pipeline.distance_capacity(report),expected)
        with self.assertRaises(ValueError):pipeline.distance_capacity(dict(status='partial'))

    def test_training_cli_exactly_matches_probe_and_never_initializes_checkpoint(self):
        from pebby.agent.world_train import build_parser,CONFIG_FLAGS
        from pebby.agent.world_model import WorldModelConfig
        probe=self.probe();probe['curriculum']['end']=copy.deepcopy(pipeline.END)
        probe['curriculum']['end'][0]+=1e-16
        args=pipeline.train_arguments(probe,Path('/cache'),Path('/new.pt'),Path('/report.json'))
        parsed=build_parser().parse_args(args)
        cfg=WorldModelConfig(**{flag:getattr(parsed,flag) for flag in CONFIG_FLAGS},
            grounding=parsed.grounding,state_recall=parsed.state_recall,glyph_recall=parsed.glyph_recall,
            query_readout=parsed.query_readout,cell_recall=parsed.cell_recall)
        self.assertEqual(cfg.__dict__,probe['config'])
        self.assertEqual(parsed.batch_size,1024);self.assertEqual(parsed.epochs,10)
        self.assertEqual(parsed.select_on,'last');self.assertTrue(parsed.require_exact_distances)
        self.assertIsNone(parsed.initialize_checkpoint);self.assertIsNone(parsed.initialize_glyph_checkpoint)
        self.assertTrue(parsed.require_fresh_initialization)
        self.assertEqual(parsed.expected_initial_state_sha256,'a'*64)
        self.assertEqual({key:getattr(parsed,key+'_weight') for key in probe['weights']},probe['weights'])
        bad=copy.deepcopy(probe);bad['curriculum']['end'][6]=.35
        with self.assertRaises(ValueError):pipeline.train_arguments(bad,Path('/c'),Path('/m'),Path('/r'))

    def test_optimized_probe_flags_reach_fresh_trainer(self):
        from pebby.agent.world_train import build_parser
        probe=self.probe();probe['checkpoint_loops']=False
        probe['execution']=dict(compile_core=True,temporal_backend='math')
        args=build_parser().parse_args(pipeline.train_arguments(probe,Path('/c'),Path('/m'),Path('/r')))
        self.assertFalse(args.checkpoint_loops);self.assertTrue(args.checkpoint_encoder)
        self.assertTrue(args.compile_core);self.assertEqual(args.temporal_backend,'math')
        self.assertTrue(args.require_fresh_initialization)

    def test_pid_identity_detects_reuse_without_signalling_generation(self):
        identity=pipeline.identity(os.getpid());self.assertTrue(pipeline.same_process(identity))
        self.assertFalse(pipeline.same_process(dict(identity,start_ticks=identity['start_ticks']+1)))

    def test_real_child_stage_reaps_success_and_timeout_and_preserves_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=object.__new__(pipeline.Supervisor)
            runner.args=SimpleNamespace(out_dir=Path(directory));runner.path=Path(directory)/'pipeline.json'
            runner.child=None;runner.report=dict(sources={},stages={})
            runner.stage('tiny','timeit',['-n','1','-r','1','pass'],5,[])
            receipt=runner.report['stages']['tiny'];self.assertEqual(receipt['status'],'complete')
            self.assertFalse(pipeline.same_process(receipt['identity']))
            with patch('subprocess.Popen',side_effect=AssertionError('completed child must not restart')):
                runner.stage('tiny','timeit',[],5,[])
            with self.assertRaises(Exception):runner.stage('timeout','timeit',['-n','1','-r','1','import time; time.sleep(2)'],.05,[])
            failed=runner.report['stages']['timeout'];self.assertEqual(failed['status'],'failed')
            self.assertFalse(pipeline.same_process(failed['identity']));self.assertIsNone(runner.child)
            self.assertEqual(json.loads(runner.path.read_text())['stages']['timeout']['status'],'failed')

    def test_generated_quota_and_split_uniqueness_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            for split,quotas,start in [('train',pipeline.TRAIN,100000),('validation',pipeline.VALIDATION,1000000)]:
                rows=[]
                for tier,count in enumerate(quotas,1):
                    for _ in range(count):
                        rows.append(dict(seed=start+len(rows),split=split,difficulty=tier,difficulty_version='ls20-reference-v1'))
                (directory/(split+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
            self.assertEqual(sum(pipeline.check_bank_counts(directory)['train'].values()),10000)
            path=directory/'validation.jsonl';rows=path.read_text().splitlines();rows[1]=rows[0]
            path.write_text('\n'.join(rows)+'\n')
            with self.assertRaisesRegex(ValueError,'duplicate'):pipeline.check_bank_counts(directory)


if __name__=='__main__':unittest.main()
