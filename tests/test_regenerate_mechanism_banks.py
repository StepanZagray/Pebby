"""Small infrastructure regressions; no full banks or expensive reference oracles."""
from concurrent.futures import Future
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from pebby.ls20.extended_curriculum import ContractMismatch, gameplay_hash
from tools.generate_mechanism_pilot import MODES
from tools import regenerate_mechanism_banks as regen


def row(seed, split='train', mode='one_attribute'):
    return dict(format=regen.FORMAT, seed=seed, split=split, difficulty=1,
                pilot_mode=mode, walls=[[0, 0]], start=[seed % 11, 1],
                start_triple=[0, 0, 0], goals=[dict(cell=[3, 3], triple=[0, 0, 1])],
                cyclers=[dict(cell=[2, 2], kind='rotation')], rails=[], launchers=[],
                refills=[], step_counter=42, step_cost=1, fog=False,
                training_context_index=seed % 7, verification_level_index=seed % 7, changing_attributes=1,
                engine_verified=True, context_engine_verified=True, search_truncated=False,
                optimal_actions=1, context_optimal_actions=1, solution=[1])


class InlinePool:
    """Exercise coordinator scheduling with actual Future objects, without processes."""
    def __init__(self, **kwargs):
        self._processes = {}
        kwargs['initializer'](*kwargs['initargs'])

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)
        return future

    def shutdown(self, **kwargs):
        pass


class RegenerationJobsTests(unittest.TestCase):
    def test_reserved_blocks_and_reference_quotas(self):
        occupied = {720000, 720100, 8100001}
        train = regen.jobs_for(21, 'train', occupied)
        validation = regen.jobs_for(21, 'validation', occupied)
        seen = set(occupied)
        for job in train + validation:
            block = set(range(job['seed'], job['seed'] + 64))
            self.assertFalse(block & seen)
            seen.update(block)
        self.assertEqual([job['mode'] for job in train], list(range(1, 8)) * 3)
        self.assertEqual(train, regen.jobs_for(21, 'train', occupied))
        self.assertTrue(all(job['limit'] is None for job in train))

    def test_explicit_unequal_quotas_preserve_assignments_and_large_namespace(self):
        quotas=[2500,2500,1800,1500,1400,250,50]
        jobs=regen.jobs_for(10000,'train',{20000001},quotas=quotas,seed_range=(20000000,22000000))
        self.assertEqual([sum(j['mode']==tier for j in jobs) for tier in range(1,8)],quotas)
        self.assertEqual(jobs[0]['seed'],20000064)
        self.assertEqual(jobs,regen.jobs_for(10000,'train',{20000001},quotas=quotas,seed_range=(20000000,22000000)))
        self.assertEqual(regen.jobs_for(21,'train',set()),
                         regen.jobs_for(21,'train',set(),quotas=[3]*7))
        for invalid in ([1]*6,[1,1,1,1,1,1,0],[1]*7):
            with self.assertRaises(ValueError):regen.jobs_for(14,'train',set(),quotas=invalid)

    def test_fair_dispatch_does_not_queue_slow_jobs_over_cheap_progress(self):
        jobs=regen.jobs_for(7,'train',set())
        active=[jobs[6]]*8
        pending=[jobs[5],jobs[6],jobs[0],jobs[1]]
        self.assertEqual(regen.take_pending(pending,active,'fair'),jobs[0])
        self.assertEqual(regen.take_pending(pending,active,'fair'),jobs[1])
        self.assertIsNone(regen.take_pending(pending,active,'fair'))
        self.assertEqual(regen.take_pending(pending,[],'fair'),jobs[5])
        # FIFO remains selectable and job payloads/search budgets never mutate.
        self.assertEqual(regen.take_pending(pending,active,'fifo'),jobs[6])
        self.assertTrue(all(j['attempts']==400 and j['limit'] is None for j in jobs))

    def test_memory_admission_retains_reserve_and_tier_caps(self):
        jobs=regen.jobs_for(7,'train',set())
        pending=[jobs[6],jobs[0]]
        self.assertEqual(regen.MEMORY_RESERVE,6*regen.GIB)
        light_room=regen.MEMORY_RESERVE+regen.job_memory(jobs[0])
        heavy_room=regen.MEMORY_RESERVE+regen.job_memory(jobs[6])
        self.assertIsNone(regen.take_pending(pending,[],'fair',available=light_room))
        self.assertEqual(regen.take_pending(pending,[],'fifo',available=light_room),jobs[0])
        self.assertIsNone(regen.take_pending(pending,[],'fair',available=heavy_room-1))
        self.assertEqual(regen.take_pending(pending,[],'fair',available=heavy_room),jobs[6])
        self.assertIsNone(regen.take_pending([jobs[0]],[],'fair',available=light_room,reserved=1))
        self.assertIsNone(regen.take_pending([jobs[4]],[jobs[4]]*4,'fair',available=100*regen.GIB))
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(regen.active_reservation([jobs[6]],Path(directory)),regen.job_memory(jobs[6]))

    def test_heavy_drain_breaks_continuous_cheap_refill_starvation(self):
        jobs=regen.jobs_for(7,'train',set());light,heavy=jobs[0],jobs[6]
        active=[light]*16;pending=[heavy]+[light]*64;admitted=[]
        available=regen.MEMORY_RESERVE+regen.job_memory(heavy)+14*regen.job_memory(light)
        for _ in range(64):
            if active:active.pop(0)
            reserved=sum(regen.job_memory(job) for job in active)
            chosen=regen.take_pending(pending,active,'fair',available=available,reserved=reserved)
            if chosen is not None:active.append(chosen);admitted.append(chosen)
            if heavy in admitted:break
        self.assertIn(heavy,admitted)
        self.assertEqual(admitted[0],heavy)

    def test_running_seven_allows_smaller_heavy_backfill_with_reserve(self):
        jobs=regen.jobs_for(7,'train',set());six,seven=jobs[5],jobs[6]
        # Preserve the old heterogeneous-reservation scenario explicitly; the
        # production policy has different validated heavy estimates.
        def estimate(job):
            return {6: 5*regen.GIB, 7: 6*regen.GIB}.get(job['mode'], regen.GIB)
        with patch.object(regen, 'job_memory', side_effect=estimate):
            room=regen.MEMORY_RESERVE+estimate(seven)+estimate(six)
            pending=[seven,six]
            self.assertEqual(regen.take_pending(pending,[seven],'fair',available=room,
                reserved=estimate(seven)),six)
            self.assertEqual(pending,[seven])
            self.assertIsNone(regen.take_pending([seven,six],[seven],'fair',available=room-1,
                reserved=estimate(seven)))
            # Eight heavy workers fill the cap; a ninth cannot be admitted.
            self.assertIsNone(regen.take_pending([seven,six],
                [seven,six]*4,'fair',available=100*regen.GIB,
                reserved=4*estimate(seven)+4*estimate(six)))
            # When the running7 finishes, the queued7 has priority again.
            self.assertEqual(regen.take_pending(pending,[six],'fair',available=room,
                reserved=estimate(six)),seven)

    def test_eight_worker_ceiling_still_requires_available_memory(self):
        jobs=regen.jobs_for(7,'train',set());six,seven,light=jobs[5],jobs[6],jobs[0]
        self.assertEqual(regen.TIER_MEMORY_GIB[5:], (1.3125, 1.75))
        self.assertEqual(regen.MAX_HEAVY_WORKERS, 8)
        active=[six,six,seven,seven]
        room=regen.MEMORY_RESERVE+sum(map(regen.job_memory,active))+regen.job_memory(six)
        self.assertEqual(regen.take_pending([six],active,'fair',available=room,
            reserved=sum(map(regen.job_memory,active))),six)
        active.append(six)
        self.assertIsNone(regen.take_pending([seven],active,'fair',available=room,
            reserved=sum(map(regen.job_memory,active))))
        active=[six,seven]*4
        self.assertEqual(regen.take_pending([six,seven,light],active,'fair',
            available=100*regen.GIB,reserved=sum(map(regen.job_memory,active))),light)
        self.assertIsNone(regen.take_pending([six,seven],active,'fair',
            available=100*regen.GIB,reserved=sum(map(regen.job_memory,active))))

    def test_tier6_reservation_boundary_keeps_global_reserve(self):
        jobs=regen.jobs_for(7,'train',set());six=jobs[5]
        current=regen.job_memory(six)
        room=regen.MEMORY_RESERVE+2*current
        pending=[six]
        self.assertIsNone(regen.take_pending(pending,[six],'fair',available=room-1,
            reserved=current))
        self.assertEqual(regen.take_pending(pending,[six],'fair',available=room,
            reserved=current),six)

    def test_one_heavy_backfills_light_but_oldest_seven_cannot_be_bypassed(self):
        jobs=regen.jobs_for(7,'train',set());light,six,seven=jobs[0],jobs[5],jobs[6]
        # Preserve the old heterogeneous backfill scenario explicitly; equal
        # heavy estimates are tested above.
        def estimate(job):
            return {6: 5*regen.GIB, 7: 6*regen.GIB}.get(job['mode'], regen.GIB//4)
        with patch.object(regen, 'job_memory', side_effect=estimate):
            pending=[seven,six,light]
            # A younger6 fits beside active6; older7 does not. Backfill only light.
            available=regen.MEMORY_RESERVE+2*estimate(six)
            self.assertEqual(regen.take_pending(pending,[six],'fair',available=available,
                reserved=estimate(six)),light)
            self.assertEqual(pending,[seven,six])
            self.assertEqual(regen.take_pending(pending,[],'fair',available=available),seven)
            # Under sustained light completions, older7 gets its lane once6 ends.
            active=[six]+[light]*15;pending=[seven,six]+[light]*64
            available=regen.MEMORY_RESERVE+estimate(seven)+14*estimate(light)
            for _ in range(5):
                active.pop()
                chosen=regen.take_pending(pending,active,'fair',available=available,
                    reserved=sum(map(regen.job_memory,active)))
                self.assertEqual(chosen,light);active.append(chosen)
            active.remove(six)
            self.assertIsNone(regen.take_pending(pending,active,'fair',available=available,
                reserved=sum(map(regen.job_memory,active))))
            active.pop()
            self.assertEqual(regen.take_pending(pending,active,'fair',available=available,
                reserved=sum(map(regen.job_memory,active))),seven)

    def test_quota_namespace_dispatch_cli_binds_resumes(self):
        def generate(job):
            value=row(job['next_seed'],job['split'],job['mode'])
            value['difficulty']=job['mode']
            return dict(status='accepted',row=value,rejections=[],pid=1)
        options=['--profile','ls20-reference-v1','--train-count','8','--validation-count','7',
                 '--train-quotas','2','1','1','1','1','1','1',
                 '--validation-quotas','1','1','1','1','1','1','1',
                 '--train-seed-range','20000000','20001000','--dispatch','fair']
        with tempfile.TemporaryDirectory() as directory, patch.object(regen,'_accept',return_value=None):
            out=Path(directory)/'bank'
            self.assertEqual(self._run(out,generate,extra_args=options),0)
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['config']['train_quotas'],[2,1,1,1,1,1,1])
            self.assertEqual(manifest['config']['dispatch'],'fair')
            self.assertEqual(self._run(out,lambda _:self.fail('completed job regenerated'),resume=True,extra_args=options),0)
            changed=list(options);changed[changed.index('2')]='1'
            changed[changed.index('--train-quotas')+2]='2'
            self.assertEqual(self._run(out,generate,resume=True,extra_args=changed),1)

    def test_overlapping_namespaces_and_legacy_quota_fail_before_output(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            out=Path(directory)/'bank'
            for options in (['--train-seed-range','8100000','8200000'],
                            ['--profile','legacy-mechanism-v2','--train-quotas',*['1']*7]):
                with self.subTest(options=options),self.assertRaises(SystemExit),patch('sys.stderr',io.StringIO()):
                    regen.main(['--out-dir',str(out),*options])
                self.assertFalse(out.exists())

    def test_legacy_is_explicit_and_keeps_twelve_modes(self):
        jobs = regen.jobs_for(24, 'train', set(), profile='legacy-mechanism-v2')
        self.assertEqual([job['mode'] for job in jobs], list(MODES) * 2)

    def test_oversized_request_fails_before_starting_workers(self):
        with self.assertRaises(ValueError):
            regen.jobs_for(3000, 'train', set())

    def test_worker_inventory_and_contract_failure_are_not_silent_rejections(self):
        job = regen.jobs_for(1, 'train', set(), profile='legacy-mechanism-v2')[0]
        job['next_seed'] = job['seed']
        regen._initialize({123}, {'existing-gameplay'})
        with patch.object(regen.pilot, 'generate_one', side_effect=ContractMismatch('mismatch')) as generate:
            result = regen._generate(job)
        self.assertEqual(result['status'], 'quarantined')
        self.assertEqual(result['error_type'], 'ContractMismatch')
        self.assertEqual(generate.call_args.kwargs['seen_seeds'], {123})
        self.assertEqual(generate.call_args.kwargs['seen_specs'], {'existing-gameplay'})
        with patch.object(regen.pilot, 'generate_one', side_effect=RuntimeError('bounded seed attempts exhausted')):
            self.assertEqual(regen._generate(job)['status'], 'rejected')
        with patch.object(regen.pilot, 'generate_one', side_effect=ValueError('unexpected input')):
            self.assertEqual(regen._generate(job)['status'], 'quarantined')

    def test_worker_callback_quarantines_legacy_replay_failure(self):
        job = regen.jobs_for(1, 'train', set(), profile='legacy-mechanism-v2')[0]
        def bad(*args, **kwargs):
            kwargs['record_rejection']({'reason': 'stepwise engine oracle state mismatch'})
        regen._initialize(set(), set())
        with patch.object(regen.pilot, 'generate_one', side_effect=bad):
            result = regen._generate({**job, 'next_seed': job['seed']})
        self.assertEqual(result['status'], 'quarantined')
        self.assertEqual(len(result['rejections']), 1)

    def test_reference_worker_forwards_tier_policy_limit_and_rejection_callback(self):
        job = regen.jobs_for(1, 'validation', set())[0]
        value = row(job['seed'], 'validation')
        value.update(difficulty_version='ls20-reference-v1')
        generate = Mock(return_value=value)
        regen._initialize(set(), set())
        with patch('pebby.ls20.reference_generator.generate_level', generate):
            result = regen._generate({**job, 'next_seed': job['seed']})
        self.assertEqual(result['status'], 'accepted')
        self.assertIsNone(generate.call_args.kwargs['search_limit'])
        self.assertEqual(generate.call_args.kwargs['split'], 'validation')
        self.assertTrue(callable(generate.call_args.kwargs['record_rejection']))
        self.assertEqual(generate.call_args.args, (job['seed'], 1))

    def test_interrupted_pending_job_resumes_without_repeating_completed_sibling(self):
        calls = []
        def generate(job):
            calls.append(job['split'])
            return dict(status='accepted', row=row(job['next_seed'], job['split'], job['mode']), rejections=[], pid=1)
        completed_calls = 0
        def interrupted(futures):
            nonlocal completed_calls
            completed_calls += 1
            if completed_calls == 2:
                raise TimeoutError('injected coordinator interruption')
            return iter(futures)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'bank'
            with patch.object(regen, 'as_completed', side_effect=interrupted):
                self.assertEqual(self._run(out, generate), 1)
            saved = (out / 'jobs/train-000000.json').read_bytes()
            calls.clear()
            self.assertEqual(self._run(out, generate, resume=True), 0)
            self.assertEqual(calls, ['validation'])
            self.assertEqual((out / 'jobs/train-000000.json').read_bytes(), saved)

    def test_reference_acceptance_checks_real_profile_and_nested_proof(self):
        fixture = Path(__file__).parent / 'fixtures/ls20_reference/tier1.json'
        original = json.loads(fixture.read_text())
        job = dict(seed=original['seed'], split=original['split'],
                   profile='ls20-reference-v1', mode=1)
        self.assertIsNone(regen._accept(original, job, set(), set(), {'train': set(), 'validation': set()}))
        mutants = []
        wrong = copy.deepcopy(original)
        wrong['step_cost'] = 2
        mutants.append(wrong)
        wrong = copy.deepcopy(original)
        wrong['proof']['context_index'] = 1
        mutants.append(wrong)
        wrong = copy.deepcopy(original)
        wrong['proof']['context_engine_verified'] = 'true'
        mutants.append(wrong)
        for wrong in mutants:
            with self.subTest(mutation=wrong):
                with self.assertRaises(ValueError):
                    regen._accept(wrong, job, set(), set(), {'train': set(), 'validation': set()})

    def test_checkpoint_binds_row_checksum_job_and_sources(self):
        job = regen.jobs_for(1, 'train', set())[0]
        state = dict(status='accepted', next_seed=job['seed'] + 1,
                     failures=[], row=row(job['seed']))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'job.json'
            regen._checkpoint(path, job, 'source-config-binding', state)
            self.assertEqual(regen._restore(path, job, 'source-config-binding'), state)
            with self.assertRaises(ValueError):
                regen._restore(path, job, 'changed-source')
            corrupted = json.loads(path.read_text())
            corrupted['payload']['state']['row']['seed'] += 1
            path.write_text(json.dumps(corrupted))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                regen._restore(path, job, 'source-config-binding')

    def test_shuffle_is_reproducible_and_breaks_ordered_mode_cycle(self):
        values = [row(seed) for seed in range(48)]
        shuffled = regen.shuffled_rows(values, 'train', 'legacy-mechanism-v2')
        self.assertEqual(shuffled, regen.shuffled_rows(list(reversed(values)), 'train', 'legacy-mechanism-v2'))
        self.assertNotEqual([r['seed'] for r in shuffled], list(range(48)))
        self.assertGreater(len({r['seed'] % 12 for r in shuffled[::12]}), 1)

    def test_canonical_hash_matches_auditor_and_preserves_goal_pairings(self):
        from tools.audit_generated_banks import gameplay_hash as audit_hash
        original = row(11)
        original['goals'].append(dict(cell=[7, 7], triple=[1, 2, 3]))
        reordered = copy.deepcopy(original)
        reordered['goals'].reverse()
        self.assertEqual(gameplay_hash(original), audit_hash(reordered))
        swapped = copy.deepcopy(original)
        swapped['goals'][0]['triple'], swapped['goals'][1]['triple'] = swapped['goals'][1]['triple'], swapped['goals'][0]['triple']
        self.assertNotEqual(gameplay_hash(original), gameplay_hash(swapped))

    def _run(self, directory, generate, resume=False, inventory=None, inventory_effect=None, extra_args=()):
        args = ['--profile', 'legacy-mechanism-v2', '--train-count', '1',
                '--validation-count', '1', '--attempts', '1', '--workers', '1',
                '--out-dir', str(directory)] + (['--resume'] if resume else []) + list(extra_args)
        with patch.object(regen, '_inventory', return_value=inventory or (set(), set(), {}), side_effect=inventory_effect), \
             patch.object(regen, '_code_hashes', return_value={}), \
             patch.object(regen, '_geometry', side_effect=lambda r: str(r['seed'])), \
             patch.object(regen, '_generate', side_effect=generate), \
             patch.object(regen, 'ProcessPoolExecutor', InlinePool), redirect_stdout(io.StringIO()):
            return regen.main(args)

    def test_coordinator_duplicate_retry_preserves_other_success(self):
        calls = []
        def generate(job):
            calls.append((job['split'], job['next_seed']))
            value = row(job['next_seed'], job['split'], job['mode'])
            if job['split'] == 'train' and job['next_seed'] == job['seed']:
                value['start'] = [9, 9]
            return dict(status='accepted', row=value, rejections=[], pid=1)
        existing = row(999)
        existing['start'] = [9, 9]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'bank'
            self.assertEqual(self._run(out, generate, inventory=({999}, {gameplay_hash(existing)}, {})), 0)
            self.assertIn(('train', 720001), calls)
            self.assertEqual(len(list((out / 'jobs').glob('*.json'))), 2)
            report = json.loads((out / 'generation-report.json').read_text())
            self.assertEqual(report['accepted'], {'train': 1, 'validation': 1})
            self.assertTrue(report['workers_stopped'])
            train_job = json.loads((out / 'jobs/train-000000.json').read_text())['payload']['state']
            self.assertEqual(train_job['failures'][0]['error'], 'duplicate_seed_or_gameplay')

    def test_resume_recovers_checkpointed_success_without_regenerating(self):
        calls = []
        def generate(job):
            calls.append(job['split'])
            if job['split'] == 'validation':
                raise TimeoutError('injected worker failure')
            return dict(status='accepted', row=row(job['next_seed'], job['split'], job['mode']), rejections=[], pid=1)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'bank'
            self.assertEqual(self._run(out, generate), 1)
            saved = (out / 'jobs/train-000000.json').read_bytes()
            self.assertFalse((out / 'train.jsonl').exists())
            calls.clear()
            self.assertEqual(self._run(out, generate, resume=True), 1)
            self.assertEqual(calls, [])
            self.assertEqual((out / 'jobs/train-000000.json').read_bytes(), saved)
            report = json.loads((out / 'generation-report.json').read_text())
            self.assertEqual(report['accepted']['train'], 1)
            self.assertIn('quarantined', str(report['errors']))

    def test_resume_after_inventory_interruption_can_create_initial_manifest(self):
        def generate(job):
            return dict(status='accepted', row=row(job['next_seed'], job['split'], job['mode']), rejections=[], pid=1)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'bank'
            out.mkdir()
            (out / 'generation-report.json').write_text('{"status":"failed_closed"}')
            self.assertEqual(self._run(out, generate, resume=True), 0)
            self.assertTrue((out / 'manifest.json').exists())

    def test_inventory_rescan_tracks_new_generated_jsonl_but_not_unrelated_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            first = root / 'data/first.jsonl'
            first.write_text(json.dumps(row(123)) + '\n')
            with patch.object(regen, 'REPO_ROOT', root):
                baseline = regen._inventory(root / 'out')[2]
                (root / 'data/metadata.jsonl').write_text('{"format":"unrelated.metrics.v1"}\n')
                self.assertEqual(regen._inventory(root / 'out')[2], baseline)
                second = root / 'data/new-generated.jsonl'
                second.write_text(json.dumps(row(456)) + '\n')
                updated = regen._inventory(root / 'out')[2]
            self.assertEqual(set(updated) - set(baseline), {str(second)})

    def test_new_concurrent_inventory_prevents_publication_and_keeps_checkpoints(self):
        def generate(job):
            return dict(status='accepted', row=row(job['next_seed'], job['split'], job['mode']), rejections=[], pid=1)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'bank'
            snapshots = [(set(), set(), {}), (set(), set(), {'new-generated.jsonl': 'new-sha256'})]
            self.assertEqual(self._run(out, generate, inventory_effect=snapshots), 1)
            self.assertFalse((out / 'train.jsonl').exists())
            self.assertFalse((out / 'validation.jsonl').exists())
            self.assertEqual(len(list((out / 'jobs').glob('*.json'))), 2)
            report = json.loads((out / 'generation-report.json').read_text())
            self.assertIn('source or inventory changed', report['error'])
            self.assertEqual(report['accepted'], {'train': 1, 'validation': 1})

    def test_complete_resume_is_idempotent_and_source_change_refuses_reuse(self):
        def generate(job):
            return dict(status='accepted', row=row(job['next_seed'], job['split'], job['mode']), rejections=[], pid=1)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'bank'
            self.assertEqual(self._run(out, generate), 0)
            bank = (out / 'train.jsonl').read_bytes()
            self.assertEqual(self._run(out, lambda job: self.fail('accepted job reran'), resume=True), 0)
            self.assertEqual((out / 'train.jsonl').read_bytes(), bank)
            manifest = json.loads((out / 'manifest.json').read_text())
            manifest['code_hashes']['changed.py'] = 'modified'
            (out / 'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(self._run(out, generate, resume=True), 1)
            self.assertEqual((out / 'train.jsonl').read_bytes(), bank)


if __name__ == '__main__':
    unittest.main()
