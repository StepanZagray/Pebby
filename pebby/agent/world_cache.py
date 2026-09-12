"""Source-hashed, disk-backed NPZ arrays for training larger generated banks.

NPZ cannot be memory mapped directly. Extract only requested NPY members with
bounded buffers, atomically publish a checked cache, then use copy-on-write
maps. Tensor indexing consequently copies batches, not the entire image bank.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def cached_arrays(source, cache_dir, names):
    source = Path(source)
    source_hash = digest(source)
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Include requested names so unrelated readers cannot reuse partial caches.
    names = sorted(set(names))
    if any(not name.replace('_', '').isalnum() for name in names):
        raise ValueError('cache array names must be simple identifiers')
    schema_hash = hashlib.sha256(json.dumps(names).encode()).hexdigest()[:16]
    target = root / f'{source_hash}-{schema_hash}'
    with (root / f'{target.name}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            try:
                manifest = json.loads((target / 'manifest.json').read_text())
            except (OSError, ValueError) as error:
                raise ValueError(f'cached dataset manifest is missing or corrupt: {target}') from error
            if manifest['source_sha256'] != source_hash:
                raise ValueError('cached dataset source hash mismatch')
            for name, info in manifest['arrays'].items():
                if name not in names or digest(target / f'{name}.npy') != info['sha256']:
                    raise ValueError(f'cached dataset array corrupted: {name}')
        else:
            temporary = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=root))
            try:
                manifest = {'source_sha256': source_hash, 'arrays': {}}
                with zipfile.ZipFile(source) as archive:
                    members = archive.namelist()
                    if len(members) != len(set(members)):
                        raise ValueError('duplicate NPZ members')
                    for name in names:
                        member = f'{name}.npy'
                        if member not in members:
                            continue
                        path = temporary / member
                        with archive.open(member) as src, path.open('wb') as dst:
                            shutil.copyfileobj(src, dst, length=1024 * 1024)
                        array = np.load(path, mmap_mode='r', allow_pickle=False)
                        manifest['arrays'][name] = {'sha256': digest(path),
                                                    'shape': list(array.shape),
                                                    'dtype': array.dtype.str}
                        del array
                if digest(source) != source_hash:
                    raise ValueError('source changed while building dataset cache')
                (temporary / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True))
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    arrays = {name: np.load(target / f'{name}.npy', mmap_mode='c', allow_pickle=False)
              for name in manifest['arrays']}
    # Behave like the subset of NpzFile used by load_dataset.
    class Archive(dict):
        @property
        def files(self):
            return list(self)
    yield Archive(arrays)
