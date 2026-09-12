import copy
from contextlib import redirect_stderr
import io
import unittest
from unittest.mock import Mock, patch

import torch

from pebby.agent import world_train
from pebby.agent.world_runtime import configure_execution
from tests.test_world_model import make_model, make_synthetic


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_temporal_mask_gradients_and_checkpoint_keys_survive_execution_wrapper(self):
        torch.manual_seed(9)
        eager = make_model()
        candidate = copy.deepcopy(eager)
        compiled = Mock(wraps=candidate._loop)
        with patch('torch.compile', return_value=compiled) as compiler:
            configure_execution(candidate, compile_core=True, temporal_backend='math')
        self.assertEqual(compiler.call_args.args[0].__func__, type(candidate)._loop)
        self.assertEqual(eager.state_dict().keys(), candidate.state_dict().keys())
        batch = world_train.as_tensors(make_synthetic(levels=1, steps=4))
        inputs = [batch[key][:2] for key in ('frames','history_valid','previous_actions')]
        outputs = [model(*inputs) for model in (eager,candidate)]
        torch.testing.assert_close(*outputs, rtol=1e-5, atol=1e-6)
        for output in outputs:
            output.square().sum().backward()
        for left, right in zip(eager.parameters(),candidate.parameters()):
            self.assertEqual(left.grad is None,right.grad is None)
            if left.grad is not None:
                torch.testing.assert_close(left.grad,right.grad,rtol=1e-4,atol=2e-5)
        self.assertGreater(compiled.call_count,0)
        compiled.reset_mock()
        with torch.no_grad():
            candidate(*inputs)
        compiled.assert_not_called()

    def test_explicit_fresh_run_refuses_every_pretrained_initialization_before_loading(self):
        for flags in (['--initialize-checkpoint','old.pt'],
                      ['--initialize-glyph-checkpoint','old-glyph.pt'],
                      ['--initialize-cell-checkpoint','old-cell.pt'],['--cell-recall']):
            errors = io.StringIO()
            with self.subTest(flags=flags), redirect_stderr(errors), patch.object(world_train,'load_dataset') as load:
                with self.assertRaises(SystemExit):
                    world_train.main(['--train','new.npz','--require-fresh-initialization',*flags])
                load.assert_not_called()
            self.assertIn('forbids pretrained',errors.getvalue())


if __name__ == '__main__':
    unittest.main()
