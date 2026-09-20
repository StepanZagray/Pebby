"""Generated-bank strict greedy H8 evaluation for the explicit position subtype.

The ordinary evaluator owns public inputs, history resets, repeat behavior and
all game execution. This adapter changes checkpoint loading and adds provenance.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

from tools import train_reference_repair as repair


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--protocol', choices=('strict',), default='strict')
    parser.add_argument('--max-actions', type=int, choices=(300,), default=300)
    parser.add_argument('--on-stall', choices=('repeat',), default='repeat')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--report-out', type=Path, required=True)
    args = parser.parse_args(argv)
    from pebby.agent import evaluate, world_position_recall as position
    checkpoint_path, bank_path = args.checkpoint.resolve(), args.bank.resolve()
    if args.report_out.exists():
        raise FileExistsError(f'evaluation report already exists: {args.report_out}')
    sources = [checkpoint_path, bank_path, Path(__file__).resolve(),
               Path(position.__file__).resolve(), Path(evaluate.__file__).resolve()]
    guard = repair.SourceGuard(sources, [])
    loaded = {}
    original_load = position.load_checkpoint
    def load(path, device='cpu'):
        if Path(path).resolve() != checkpoint_path:
            raise ValueError('evaluation checkpoint path changed')
        model, metadata = original_load(path, device)
        if model.config()['history'] != 8:
            raise ValueError('position evaluation requires history 8')
        loaded.update(format=metadata['format'], config=metadata['config'],
                      position_source=metadata['position_source'],
                      warmup_provenance=metadata.get('warmup_provenance'),
                      variant_source=metadata.get('variant_source'))
        return model, metadata
    arguments = ['evaluate_reference_position', '--checkpoint', str(checkpoint_path),
                 '--bank', str(bank_path), '--protocol', 'strict', '--max-actions', '300',
                 '--on-stall', 'repeat', '--device', args.device, '--report-out', str(args.report_out)]
    with patch.object(evaluate, 'load_checkpoint', load), patch.object(sys, 'argv', arguments):
        evaluate.main()
    guard.verify()
    if not loaded:
        raise ValueError('evaluator did not load a position checkpoint')
    report = json.loads(args.report_out.read_text())
    report['position_evaluation_adapter'] = dict(
        module='tools.evaluate_reference_position', path=str(Path(__file__).resolve()), sha256=repair.digest(__file__),
        checkpoint_sha256=guard.hashes[str(checkpoint_path)],
        bank_sha256=guard.hashes[str(bank_path)], source_sha256=guard.hashes,
        sources_unchanged=True, protocol='strict', max_actions=300, on_stall='repeat',
        checkpoint=loaded, policy_inputs='public_frames_and_action_history_only')
    repair.write(args.report_out, report)
    return report


if __name__ == '__main__':
    main()
