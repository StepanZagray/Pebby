"""Evaluate a saved appearance component on an additional generated-only split."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from pebby.agent.cell_appearance import CellAppearance, FORMAT
from tools.train_cell_appearance import digest, evaluate, load_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--train-data', type=Path, required=True)
    parser.add_argument('--train-proof', type=Path, required=True)
    parser.add_argument('--validation-data', type=Path, required=True)
    parser.add_argument('--validation-proof', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    started = time.monotonic(); torch.set_num_threads(1)
    print('PID', os.getpid(), flush=True)
    paths = [args.checkpoint, args.train_data, args.train_proof, args.validation_data,
             args.validation_proof, Path(__file__), Path('tools/train_cell_appearance.py'),
             Path('pebby/agent/cell_appearance.py')]
    hashes = {str(path): digest(path) for path in paths}
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != FORMAT:
        raise ValueError('unsupported appearance checkpoint')
    if checkpoint['source_hashes']['pebby/agent/cell_appearance.py'] != digest('pebby/agent/cell_appearance.py'):
        raise ValueError('appearance architecture source changed')
    if checkpoint['source_hashes'][str(args.train_data)] != digest(args.train_data):
        raise ValueError('training bank differs from checkpoint provenance')
    model = CellAppearance(); model.load_state_dict(checkpoint['weights']); model.eval()
    train = load_examples(args.train_data, args.train_proof)[0]['train']
    validation = load_examples(args.validation_data, args.validation_proof)[0]['validation']
    majority = dict(roles=train['role_bits'].float().mean(0) >= .5,
                    patches={p.numpy().tobytes() for p in train['patches']},
                    attributes=torch.tensor([train['attributes'][train['goals'], j].bincount().argmax()
                                             for j in range(3)]))
    results = evaluate(model, validation, majority)
    if any(digest(path) != value for path, value in hashes.items()):
        raise ValueError('evaluation source changed')
    report = dict(status='complete', pid=os.getpid(), source_hashes=hashes, results=results,
                  parameters=model.parameter_count(), training_performed=False,
                  elapsed_seconds=time.monotonic() - started,
                  limitations=['Initial fully public7x7 neighborhoods only; no game completion estimate.',
                               'Training-bank majority and template overlap use training rows only.'])
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
