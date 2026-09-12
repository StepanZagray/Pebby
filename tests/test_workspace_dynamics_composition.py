"""Dedicated composition only: no generic factory registration or rollout fit."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from pebby.agent.history import PolicyHistory
from pebby.agent.structured_factored_policy import ENCODER_FILES, state_digest
from pebby.agent.structured_workspace_controller import load_workspace_policy_checkpoint
from pebby.agent.world_data import history_arrays
from pebby.ls20.env import Ls20Scenario
from pebby.ls20.generate import build_level
from tools.compose_structured_workspace_dynamics import (
    WARM, WARM_SHA, INITIAL_SHA, digest, load_composition,
    public_wiring_check, validate_replacement,
)
from tools.evaluate_structured_workspace_dynamics import checked_fit, validate_fit_report, merge_sources


class CompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.warm, _ = load_workspace_policy_checkpoint(WARM, 'cpu', loops=2)
        initial = Path('checkpoints/ls20-factored-local-h4-400.pt')
        saved = torch.load(initial, map_location='cpu', weights_only=True)
        sources = {str(WARM.resolve()): WARM_SHA, str(initial.resolve()): INITIAL_SHA}
        for name in (*ENCODER_FILES, 'structured_transition.py', 'structured_global_glyph.py',
                     'structured_local_glyph.py', 'structured_workspace_controller.py'):
            path = Path('pebby/agent') / name
            sources[str(path.resolve())] = digest(path)
        manifests = {}
        for name, directory in {
            'onpolicy_train': 'data/structured-workspace-onpolicy1024-fp32-fields',
            'legacy_train': 'data/structured-field-16384/train',
            'legacy_additional_train': 'data/structured-field-additional-state-16384/train',
            'validation': 'data/structured-field-16384/validation',
        }.items():
            path = (Path(directory) / 'manifest.json').resolve()
            manifests[name] = json.loads(path.read_text())
            sources[str(path)] = digest(path)
            for key, entry in manifests[name]['arrays'].items():
                sources[str(path.with_name(key + '.npy'))] = entry['sha256']
        cls.saved = dict(saved, sources=sources, cache_manifests=manifests,
            official_inputs_used=False, frozen_encoder=True, objective='paired_onpolicy_h1_dynamics_repair',
            arm='control', updates=2, batch_size=2, smoke=True, initialize=str(initial),
            initialize_sha256=INITIAL_SHA, warmstart_workspace_sha256=WARM_SHA,
            frozen_workspace_readout=str(WARM), selection_sha256='a'*64, used_state_rows_sha256='b'*64,
            frozen_encoder_state_sha256=state_digest(cls.warm.encoder.state_dict()),
            frozen_workspace_state_sha256=state_digest(cls.warm.readout.state_dict()))
        # Alter only a dynamics parameter; this disposable fixture is not a fit.
        cls.saved['weights'] = {k:v.clone() for k,v in saved['weights'].items()}
        key = next(k for k,v in cls.saved['weights'].items() if v.is_floating_point())
        cls.saved['weights'][key].reshape(-1)[0] += .001

    def test_public_manual_action_order_and_reload_preserve_frozen_components(self):
        with Path('data/ls20-verified-train.jsonl').open() as stream:
            spec = json.loads(stream.readline())
        env = Ls20Scenario(build_level(spec), spec['training_context_index'])
        frame = env.reset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'disposable.pt'
            torch.save(self.saved, path)
            sha = digest(path)
            with self.assertRaisesRegex(ValueError, 'allow_smoke'):
                load_composition(path, sha)
            first, provenance = load_composition(path, sha, allow_smoke=True)
            second, other = load_composition(path, sha, allow_smoke=True)
            f,v,a = history_arrays([frame], [-1], 8)
            f,v,a = [torch.as_tensor(value)[None] for value in (f,v,a)]
            result = public_wiring_check(first, f,v,a)
            self.assertTrue(result['independent_four_actions_checked'])
            history = PolicyHistory(first, 'cpu'); history.observe(frame)
            torch.testing.assert_close(history.scores(), second(f, v, a)[0], rtol=0, atol=0)
            self.assertEqual(state_digest(first.state_dict()), state_digest(second.state_dict()))
            self.assertEqual(provenance, other)
            self.assertEqual(first.parameter_count(), 1123034)
            for name in ('encoder', 'readout'):
                self.assertEqual(state_digest(getattr(first,name).state_dict()),
                                 state_digest(getattr(self.warm,name).state_dict()))
            self.assertEqual(state_digest(first.dynamics.state_dict()), state_digest(self.saved['weights']))
            first.train(True)
            self.assertFalse(any(module.training for module in first.modules()))
            self.assertFalse(any(p.requires_grad for p in first.parameters()))
            with self.assertRaisesRegex(ValueError, 'replacement SHA'):
                load_composition(path, '0'*64, allow_smoke=True)

    def test_rejects_parent_state_format_order_and_source_drift(self):
        validate_replacement(self.saved, self.warm, WARM, allow_smoke=True)
        for key, value in [('format','unsupported'), ('frozen_encoder_state_sha256','0'*64),
                           ('frozen_workspace_state_sha256','0'*64),
                           ('selection_sha256','invalid'), ('initialize_sha256','0'*64)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_replacement(dict(self.saved, **{key:value}), self.warm, WARM, allow_smoke=True)
        bad = copy.deepcopy(self.saved)
        key = str(Path('pebby/agent/structured_workspace_controller.py').resolve())
        bad['sources'][key] = '0'*64
        with self.assertRaisesRegex(ValueError, 'implementation source'):
            validate_replacement(bad, self.warm, WARM, allow_smoke=True)
        bad = copy.deepcopy(self.saved)
        path = str(Path('data/structured-field-16384/train/source_rows.npy').resolve())
        bad['sources'][path] = '0'*64
        with self.assertRaisesRegex(ValueError, 'cache array'):
            validate_replacement(bad, self.warm, WARM, allow_smoke=True)

    def test_production_evaluator_rejects_disposable_report_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fit.json'
            path.write_text(json.dumps(dict(status='complete', source='generated_only',
                official_inputs_used=False, args={'smoke':True, 'batch_size':2, 'updates':2},
                source_unchanged=True, paired_initializations_exact=True, paired_schedule_exact=True)))
            with self.assertRaisesRegex(ValueError, 'production repair'):
                checked_fit(path, digest(path))


class PairedReportTests(unittest.TestCase):
    def report(self):
        arms={}
        for name in ('control','onpolicy'):
            replaced=512 if name=='onpolicy' else 0
            arms[name]=dict(status='complete',parameters=294664,trainable_parameters=294664,
                initial_state_sha256='a'*64,schedule_sha256='b'*64,
                replacement_rows=replaced*200,used_state_rows_sha256=('c' if replaced else 'd')*64,
                training=[dict(step=i,replaced=replaced) for i in range(1,201)])
        return dict(status='complete',source='generated_only',official_inputs_used=False,
            source_unchanged=True,paired_initializations_exact=True,paired_schedule_exact=True,
            args=dict(smoke=False,preflight_only=False,device='cuda',batch_size=1024,updates=200),
            replacements=512,trajectory_levels=1024,validation_levels=512,
            initialization=dict(sha256=INITIAL_SHA,state_sha256='a'*64),
            workspace_readout=dict(sha256=WARM_SHA,loaded_state_sha256='e'*64),
            frozen_encoder_state_sha256='f'*64,source_schedule_sha256='b'*64,arms=arms)

    def test_detailed_paired_witness_and_no_conflicting_source_overwrite(self):
        report=self.report();validate_fit_report(report)
        for field,value in [('initial_state_sha256','1'*64),('schedule_sha256','1'*64),
                            ('replacement_rows',100),('trainable_parameters',1),('training',[])]:
            bad=copy.deepcopy(report);bad['arms']['onpolicy'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_fit_report(bad)
        bad=copy.deepcopy(report)
        bad['arms']['onpolicy']['used_state_rows_sha256']=bad['arms']['control']['used_state_rows_sha256']
        with self.assertRaisesRegex(ValueError,'must differ'):validate_fit_report(bad)
        target={'tools/../tools/compose_structured_workspace_dynamics.py':'a'*64}
        merge_sources(target,{'tools/compose_structured_workspace_dynamics.py':'a'*64})
        self.assertEqual(len(target),1)
        with self.assertRaisesRegex(ValueError,'conflicting'):
            merge_sources(target,{'tools/compose_structured_workspace_dynamics.py':'b'*64})

    def test_saved_checkpoint_is_bound_to_completed_report(self):
        report=self.report()
        report['sources']={str(Path(__file__).resolve()):digest(__file__)}
        with tempfile.TemporaryDirectory() as directory:
            for name,arm in report['arms'].items():
                payload=dict(arm=name,updates=200,batch_size=1024,smoke=False,sources=report['sources'],
                    selection_sha256=report['source_schedule_sha256'],
                    used_state_rows_sha256=arm['used_state_rows_sha256'],
                    frozen_workspace_state_sha256='e'*64,frozen_encoder_state_sha256='f'*64)
                path=Path(directory)/(name+'.pt');torch.save(payload,path)
                arm['checkpoint']=dict(path=str(path),sha256=digest(path),strict_reload_exact=True)
            path=Path(directory)/'fit.json';path.write_text(json.dumps(report))
            checked_fit(path,digest(path))
            control=report['arms']['control']['checkpoint']
            saved=torch.load(control['path'],weights_only=True)
            saved['used_state_rows_sha256']='0'*64;torch.save(saved,control['path'])
            control['sha256']=digest(control['path']);path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'provenance differ'):checked_fit(path,digest(path))


if __name__ == '__main__':
    unittest.main()
