import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from tools.build_reference_world_cache import MemoryAdmittedPool, admitted_build, CACHE_WORKER_OVERHEAD, checked_native_worker
from tools.regenerate_mechanism_banks import take_pending, active_reservation, job_memory, GIB


def fake_collection(task):
    spec,_=task
    time.sleep(spec.get('delay',0))
    if spec.get('fail'):raise ValueError('deliberate checked failure')
    return spec['seed']


def task(seed, tier=1, **extra):
    return (dict(seed=seed,difficulty=tier,difficulty_version='ls20-reference-v1',**extra),{})


class CacheAdmissionTests(unittest.TestCase):
    def test_shared_memory_headroom_and_tier_caps(self):
        def job(index,tier):return dict(id=str(index),mode=tier,profile='ls20-reference-v1')
        active=[job(i,6 if i%2 else 7) for i in range(8)]
        pending=[job(3,7),job(4,1)]
        self.assertEqual(take_pending(pending,active,'fair',available=100*GIB)['mode'],1)
        self.assertIsNone(take_pending([job(4,5)],[job(i,5) for i in range(4)],'fair',available=100*GIB))
        self.assertIsNone(take_pending([job(1,1)],[],'fair',available=int(4.49*GIB)))
        self.assertIsNone(take_pending([job(1,1)],[],'fair',available=8*GIB,reserved=4*GIB))
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(active_reservation(active,Path(root)),sum(map(job_memory,active)))

    def test_process_results_are_source_ordered_and_cleaned(self):
        events=[]
        with tempfile.TemporaryDirectory() as root:
            with patch('tools.build_reference_world_cache.memory_available',return_value=100*GIB):
                with MemoryAdmittedPool(2,root,events.append) as pool:
                    result=list(pool.imap(fake_collection,[task(10,delay=.15),task(11),task(12)]))
                self.assertEqual(result,[10,11,12])
                self.assertTrue(all(not process.is_alive() for process in pool.started))
                self.assertEqual(events[-1]['live_pids'],[])
                self.assertEqual(list(Path(root).glob('*.json')),[])
                self.assertLessEqual(max(e.get('active',0) for e in events),2)
                base=job_memory(dict(mode=1,profile='ls20-reference-v1'))
                self.assertTrue(all(e['job_reservation_bytes']==base+CACHE_WORKER_OVERHEAD
                                    for e in events if e.get('event')=='admitted'))
                for process in pool.started:
                    self.assertFalse(Path(f'/proc/{process.pid}').exists())

    def test_worker_error_propagates_and_siblings_stop(self):
        with tempfile.TemporaryDirectory() as root:
            with patch('tools.build_reference_world_cache.memory_available',return_value=100*GIB):
                with self.assertRaisesRegex(RuntimeError,'deliberate checked failure'):
                    with MemoryAdmittedPool(2,root) as pool:
                        list(pool.imap(fake_collection,[task(20,fail=True),task(21,delay=10)]))
                self.assertTrue(all(not process.is_alive() for process in pool.started))
                self.assertFalse(list(Path(root).glob('*.json')))

    def test_build_reuses_production_aggregation_and_worker(self):
        from pebby.agent import world_data
        from tools.stream_extended_collection import checked_worker
        original_module=world_data.multiprocessing
        def build(specs,workers,**kwargs):
            self.assertEqual(kwargs,dict(history=8,samples=32,epsilon=.15,coverage='mixed_failure'))
            self.assertIs(world_data._worker,checked_native_worker)
            with world_data.multiprocessing.get_context('spawn').Pool(workers) as pool:
                self.assertEqual(pool.workers,1)
            return {'untouched':'production result'}
        with tempfile.TemporaryDirectory() as root, patch.object(world_data,'build',side_effect=build):
            result=admitted_build([],workers=1,marker_directory=root,history=8,samples=32,epsilon=.15,coverage='mixed_failure')
        self.assertEqual(result,{'untouched':'production result'})
        self.assertIs(world_data.multiprocessing,original_module)


    def test_native_guard_rejects_before_python_search_and_restores(self):
        from pebby.agent import world_data
        from pebby.ls20 import fastplan
        from pebby.ls20.plan import Oracle
        from types import SimpleNamespace
        original = world_data.Oracle
        # Constructor setup is real, but native availability is deliberately absent.
        layout = SimpleNamespace(exact=True, refills=(), goals=[], start_cell=(0,0),
                                 start_triple=(0,0,0), max_steps=3)
        def collect(_task):
            return world_data.Oracle(layout, limit=10)
        with patch('tools.build_reference_world_cache.checked_worker', side_effect=collect), \
             patch.object(fastplan, 'available', return_value=False), \
             patch.object(fastplan, 'load_error', return_value='test unavailable'), \
             patch.object(Oracle, '_search', side_effect=AssertionError('Python fallback entered')) as python_search:
            with self.assertRaisesRegex(RuntimeError, 'fast planner unavailable: test unavailable'):
                checked_native_worker(None)
        python_search.assert_not_called()
        self.assertIs(world_data.Oracle, original)

    def test_native_guard_restores_after_collector_exception(self):
        from pebby.agent import world_data
        original = world_data.Oracle
        def fail(_task):
            self.assertIsNot(world_data.Oracle, original)
            raise ValueError('collector failed')
        with patch('tools.build_reference_world_cache.checked_worker', side_effect=fail):
            with self.assertRaisesRegex(ValueError, 'collector failed'):
                checked_native_worker(None)
        self.assertIs(world_data.Oracle, original)

    def test_real_generated_native_collection_preserves_proof_and_branches(self):
        from pebby.agent import world_data
        from pebby.ls20 import names
        from pebby.ls20.generate import FORMAT, GENERATOR_VERSION
        from pebby.ls20.plan import Oracle
        free = {(col,3) for col in range(3,9)}
        spec = dict(format=FORMAT, generator_version=GENERATOR_VERSION, seed=8, difficulty=1,
            walls=sorted({(col,row) for col in range(names.GRID_COLS)
                          for row in range(names.GRID_ROWS)}-free),
            start=(3,3), start_triple=[0,0,0], goals=[dict(cell=(6,3),triple=[0,0,0])],
            cyclers=[], launchers=[], refills=[], step_counter=20, step_cost=1, fog=False)
        original = world_data.Oracle
        with tempfile.TemporaryDirectory() as root, \
             patch.dict(os.environ, {'PEBBY_EXTENDED_COLLECTION_PID_LEDGER':str(Path(root)/'workers.jsonl')}), \
             patch.object(Oracle, '_search', side_effect=AssertionError('Python fallback entered')):
            rows, proof = checked_native_worker((spec, dict(history=8,samples=4,epsilon=0.,
                                                           coverage='mixed_failure',search_limit=10000)))
        self.assertIs(world_data.Oracle, original)
        self.assertGreater(len(rows), 0)
        self.assertEqual(proof['oracle_backend'], 'fast')
        self.assertFalse(proof['search_truncated'])
        self.assertTrue(proof['context_engine_verified'])
        self.assertTrue(proof['win_covered'])
        self.assertEqual(proof['branch_verification']['branches'], 4*len(rows))
        self.assertTrue(all(row['next_frames'].shape==(4,64,64) for row in rows))

if __name__=='__main__':unittest.main()
