import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from pebby.agent.model import load_checkpoint


class StructuredCheckpointDispatchTests(unittest.TestCase):
    def test_explicit_format_routes_to_source_bound_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.pt'
            torch.save({'format':'pebby.structured-field-policy.v1',
                        'config':{'architecture':'structured'}},path)
            sentinel=(object(),{'loaded':True})
            with patch('pebby.agent.structured_policy.load_structured_policy_checkpoint',return_value=sentinel) as loader:
                self.assertIs(load_checkpoint(path,'cpu'),sentinel)
                loader.assert_called_once_with(path,'cpu')

    def test_mismatched_or_malformed_architecture_cannot_reach_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.pt'
            for config in ({'architecture':'world'},[],None):
                torch.save({'format':'pebby.structured-field-policy.v1','config':config},path)
                with patch('pebby.agent.structured_policy.load_structured_policy_checkpoint') as loader:
                    with self.assertRaisesRegex(ValueError,'architecture'):load_checkpoint(path)
                    loader.assert_not_called()


if __name__=='__main__':unittest.main()
