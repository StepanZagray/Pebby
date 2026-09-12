"""Explicit, receipt-gated dense graph storage checkpoint migration.

The only C change allowed is the reviewed deterministic dense graph compression.
Python planner storage stays byte-identical. Scheduler proof/job code is protected.
No search is run by this migration.
"""
import argparse
import ast
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from concurrent.futures import ThreadPoolExecutor

from tools import regenerate_mechanism_banks as r
from tools.upgrade_reference_scheduler import PROTECTED, functions

ARCHIVE='fastplan-dense-storage-upgrade-v1'
FORMAT='pebby.fastplan-dense-storage-upgrade.v1'
RECEIPT_FORMAT='pebby.fastplan-dense-storage-validation.v1'
LIVE_FASTPLAN=Path(r.REPO_ROOT)/'pebby/ls20/fastplan.py'
LIVE_C=Path(r.REPO_ROOT)/'pebby/ls20/_fastplan.c'
LIVE_RUNNER=Path(r.__file__).resolve()


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def durable_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,sort_keys=True,separators=(',',':'));stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one exact match, found {count}")
    return source.replace(old, new, 1)


def compact_graph_storage(source: str) -> str:
    """Replace the hash keys, edge boundaries, and growable arrays exactly once."""
    table_old = '''typedef struct {
    uint64_t *keys;
    int32_t *index;   /* -1 marks an empty slot */
    size_t capacity;  /* power of two */
    size_t used;
} Table;

static inline size_t slot_of(uint64_t key, size_t capacity) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return (size_t)key & (capacity - 1);
}

static int table_init(Table *t, size_t capacity) {
    t->keys = (uint64_t *)malloc(capacity * sizeof(uint64_t));
    t->index = (int32_t *)malloc(capacity * sizeof(int32_t));
    if (!t->keys || !t->index) { free(t->keys); free(t->index); return -1; }
    memset(t->index, 0xff, capacity * sizeof(int32_t));
    t->capacity = capacity;
    t->used = 0;
    return 0;
}

static int table_grow(Table *t) {
    Table bigger;
    if (table_init(&bigger, t->capacity * 2) != 0)
        return -1;
    for (size_t i = 0; i < t->capacity; i++) {
        if (t->index[i] < 0)
            continue;
        size_t s = slot_of(t->keys[i], bigger.capacity);
        while (bigger.index[s] >= 0)
            s = (s + 1) & (bigger.capacity - 1);
        bigger.keys[s] = t->keys[i];
        bigger.index[s] = t->index[i];
    }
    bigger.used = t->used;
    free(t->keys);
    free(t->index);
    *t = bigger;
    return 0;
}

/* Index of `key`, or -1 if absent. */
static inline int32_t table_find(const Table *t, uint64_t key) {
    size_t s = slot_of(key, t->capacity);
    while (t->index[s] >= 0) {
        if (t->keys[s] == key)
            return t->index[s];
        s = (s + 1) & (t->capacity - 1);
    }
    return -1;
}

static int table_insert(Table *t, uint64_t key, int32_t index) {
    if ((t->used + 1) * 2 > t->capacity && table_grow(t) != 0)
        return -1;
    size_t s = slot_of(key, t->capacity);
    while (t->index[s] >= 0)
        s = (s + 1) & (t->capacity - 1);
    t->keys[s] = key;
    t->index[s] = index;
    t->used++;
    return 0;
}'''
    table_new = '''typedef struct {
    int32_t *index;   /* -1 marks an empty slot; value indexes `states` */
    size_t capacity;  /* power of two */
    size_t used;
} Table;

static inline size_t slot_of(uint64_t key, size_t capacity) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return (size_t)key & (capacity - 1);
}

static int table_init(Table *t, size_t capacity) {
    t->index = (int32_t *)malloc(capacity * sizeof(int32_t));
    if (!t->index)
        return -1;
    memset(t->index, 0xff, capacity * sizeof(int32_t));
    t->capacity = capacity;
    t->used = 0;
    return 0;
}

static int table_grow(Table *t, const uint64_t *states) {
    Table bigger;
    if (table_init(&bigger, t->capacity * 2) != 0)
        return -1;
    for (size_t i = 0; i < t->capacity; i++) {
        if (t->index[i] < 0)
            continue;
        int32_t index = t->index[i];
        size_t s = slot_of(states[index], bigger.capacity);
        while (bigger.index[s] >= 0)
            s = (s + 1) & (bigger.capacity - 1);
        bigger.index[s] = index;
    }
    bigger.used = t->used;
    free(t->index);
    *t = bigger;
    return 0;
}

/* Index of `key`, or -1 if absent. */
static inline int32_t table_find(const Table *t, const uint64_t *states, uint64_t key) {
    size_t s = slot_of(key, t->capacity);
    while (t->index[s] >= 0) {
        int32_t index = t->index[s];
        if (states[index] == key)
            return index;
        s = (s + 1) & (t->capacity - 1);
    }
    return -1;
}

static int table_insert(Table *t, const uint64_t *states, uint64_t key, int32_t index) {
    if ((t->used + 1) * 2 > t->capacity && table_grow(t, states) != 0)
        return -1;
    size_t s = slot_of(key, t->capacity);
    while (t->index[s] >= 0)
        s = (s + 1) & (t->capacity - 1);
    t->index[s] = index;
    t->used++;
    return 0;
}'''
    source = _replace_once(source, table_old, table_new, "index-only table")

    source = _replace_once(
        source,
        "    uint64_t *states = NULL;\n"
        "    int32_t *edge_end = NULL, *edge_dst = NULL, *offsets = NULL, *preds = NULL;\n"
        "    int32_t *distance = NULL, *queue = NULL;\n"
        "    uint64_t *result_states = NULL;\n"
        "    int32_t *result_distance = NULL;\n"
        "    size_t state_count = 0, state_capacity = 0, edge_count = 0, edge_capacity = 0;\n"
        "    size_t edge_end_capacity = 0;\n"
        "    Table table = {0};\n"
        "    int stop = 0;\n"
        "    size_t popped = 0;   /* how many queue entries the forward pass took out */",
        "    uint64_t *states = NULL;\n"
        "    uint8_t *edge_counts = NULL;\n"
        "    int32_t *edge_dst = NULL, *offsets = NULL, *preds = NULL;\n"
        "    int32_t *distance = NULL, *queue = NULL;\n"
        "    uint64_t *result_states = NULL;\n"
        "    int32_t *result_distance = NULL;\n"
        "    size_t state_count = 0, edge_count = 0;\n"
        "    size_t state_capacity, edge_capacity;\n"
        "    Table table = {0};\n"
        "    int stop = 0;\n"
        "    size_t popped = 0;   /* how many queue entries the forward pass took out */",
        "search declarations",
    )
    source = _replace_once(
        source,
        "    if (table_init(&table, 1 << 12) != 0)\n"
        "        return -1;\n"
        "    GROW(states, uint64_t, state_count, state_capacity);\n"
        "    states[state_count++] = start;\n"
        "    if (table_insert(&table, start, 0) != 0) goto fail;",
        "    if (limit <= 0 || limit > (int64_t)(INT32_MAX / 4))\n"
        "        return -1;\n"
        "    state_capacity = (size_t)limit;\n"
        "    edge_capacity = state_capacity * 4;\n"
        "    states = (uint64_t *)malloc(state_capacity * sizeof(uint64_t));\n"
        "    edge_counts = (uint8_t *)malloc(state_capacity * sizeof(uint8_t));\n"
        "    edge_dst = (int32_t *)malloc(edge_capacity * sizeof(int32_t));\n"
        "    if (!states || !edge_counts || !edge_dst) goto fail;\n"
        "    if (table_init(&table, 1 << 12) != 0)\n"
        "        goto fail;\n"
        "    states[state_count++] = start;\n"
        "    if (table_insert(&table, states, start, 0) != 0) goto fail;",
        "exact upfront allocations",
    )
    source = _replace_once(
        source,
        "        if (head > (size_t)INT32_MAX)\n"
        "            goto fail;\n"
        "        if (head >= edge_end_capacity) {\n"
        "            size_t bigger = edge_end_capacity ? edge_end_capacity * 2 : 4096;\n"
        "            while (bigger <= head) {\n"
        "                if (bigger > ((size_t)-1) / 2)\n"
        "                    goto fail;\n"
        "                bigger *= 2;\n"
        "            }\n"
        "            if (bigger > ((size_t)-1) / sizeof(int32_t))\n"
        "                goto fail;\n"
        "            int32_t *ends = (int32_t *)realloc(edge_end, bigger * sizeof(int32_t));\n"
        "            if (!ends) goto fail;\n"
        "            edge_end = ends;\n"
        "            edge_end_capacity = bigger;\n"
        "        }\n"
        "        uint64_t state = states[head];",
        "        uint64_t state = states[head];\n"
        "        size_t head_begin = edge_count;",
        "remove cumulative boundary growth",
    )
    source = _replace_once(
        source,
        "            edge_end[head] = (int32_t)edge_count;\n"
        "            continue;                  /* the level advances here; nothing follows */",
        "            edge_counts[head] = 0;\n"
        "            continue;                  /* the level advances here; nothing follows */",
        "won-head byte count",
    )
    source = _replace_once(
        source,
        "            int32_t index = table_find(&table, next);",
        "            int32_t index = table_find(&table, states, next);",
        "state-aware lookup",
    )
    source = _replace_once(
        source,
        "                index = (int32_t)state_count;\n"
        "                GROW(states, uint64_t, state_count, state_capacity);\n"
        "                states[state_count++] = next;\n"
        "                if (table_insert(&table, next, index) != 0) goto fail;",
        "                if (state_count >= state_capacity)\n"
        "                    goto fail;\n"
        "                index = (int32_t)state_count;\n"
        "                states[state_count++] = next;\n"
        "                if (table_insert(&table, states, next, index) != 0) goto fail;",
        "exact state append",
    )
    source = _replace_once(
        source,
        "            if (edge_count >= edge_capacity) {\n"
        "                size_t bigger = edge_capacity ? edge_capacity * 2 : 4096;\n"
        "                int32_t *dst = (int32_t *)realloc(edge_dst, bigger * sizeof(int32_t));\n"
        "                if (!dst) goto fail;\n"
        "                edge_dst = dst;\n"
        "                edge_capacity = bigger;\n"
        "            }\n"
        "            if (edge_count >= (size_t)INT32_MAX)\n"
        "                goto fail;\n"
        "            edge_dst[edge_count] = index;\n"
        "            edge_count++;",
        "            if (edge_count >= edge_capacity)\n"
        "                goto fail;\n"
        "            edge_dst[edge_count++] = index;",
        "exact edge append",
    )
    source = _replace_once(
        source,
        "        if (edge_count > (size_t)INT32_MAX)\n"
        "            goto fail;\n"
        "        edge_end[head] = (int32_t)edge_count;",
        "        size_t outgoing = edge_count - head_begin;\n"
        "        if (outgoing > UINT8_MAX)\n"
        "            goto fail;\n"
        "        edge_counts[head] = (uint8_t)outgoing;",
        "close byte count",
    )
    source = _replace_once(
        source,
        "    free(table.keys); free(table.index); table.keys = NULL; table.index = NULL;",
        "    free(table.index); table.index = NULL;",
        "forward table cleanup",
    )
    source = _replace_once(
        source,
        "/* -- growable arrays --------------------------------------------------------- */\n\n"
        "#define GROW(ptr, type, count, capacity)                                         \\\n"
        "    do {                                                                         \\\n"
        "        if ((count) >= (capacity)) {                                             \\\n"
        "            size_t bigger = (capacity) ? (capacity) * 2 : 1024;                  \\\n"
        "            type *fresh = (type *)realloc((ptr), bigger * sizeof(type));         \\\n"
        "            if (!fresh) goto fail;                                               \\\n"
        "            (ptr) = fresh;                                                       \\\n"
        "            (capacity) = bigger;                                                 \\\n"
        "        }                                                                        \\\n"
        "    } while (0)\n\n",
        "/* -- search workspaces ------------------------------------------------------- */\n\n",
        "remove obsolete grow macro",
    )
    source = _replace_once(
        source,
        "    offsets = (int32_t *)calloc(state_count + 1, sizeof(int32_t));\n"
        "    preds = (int32_t *)malloc((edge_count ? edge_count : 1) * sizeof(int32_t));\n"
        "    distance = (int32_t *)malloc(state_count * sizeof(int32_t));\n"
        "    queue = (int32_t *)malloc(state_count * sizeof(int32_t));\n"
        "    if (!offsets || !preds || !distance || !queue) goto fail;\n"
        "    for (size_t e = 0; e < edge_count; e++)\n"
        "        offsets[edge_dst[e] + 1]++;\n"
        "    for (size_t i = 0; i < state_count; i++)\n"
        "        offsets[i + 1] += offsets[i];\n"
        "    {\n"
        "        int32_t *fill = (int32_t *)malloc((state_count + 1) * sizeof(int32_t));\n"
        "        if (!fill) goto fail;\n"
        "        memcpy(fill, offsets, (state_count + 1) * sizeof(int32_t));\n"
        "        for (size_t head = 0; head < popped; head++) {\n"
        "            size_t begin = head ? (size_t)edge_end[head - 1] : 0;\n"
        "            size_t end = (size_t)edge_end[head];\n"
        "            for (size_t e = begin; e < end; e++)\n"
        "                preds[fill[edge_dst[e]]++] = (int32_t)head;\n"
        "        }\n"
        "        free(fill);\n"
        "    }\n"
        "    free(edge_end); free(edge_dst); edge_end = edge_dst = NULL;",
        "    offsets = (int32_t *)calloc(state_count + 1, sizeof(int32_t));\n"
        "    preds = (int32_t *)malloc((edge_count ? edge_count : 1) * sizeof(int32_t));\n"
        "    if (!offsets || !preds) goto fail;\n"
        "    for (size_t e = 0; e < edge_count; e++)\n"
        "        offsets[edge_dst[e] + 1]++;\n"
        "    for (size_t i = 0; i < state_count; i++)\n"
        "        offsets[i + 1] += offsets[i];\n"
        "    {\n"
        "        size_t begin = edge_count;\n"
        "        for (size_t head = popped; head-- > 0;) {\n"
        "            size_t count = (size_t)edge_counts[head];\n"
        "            if (count > begin) goto fail;\n"
        "            size_t first = begin - count;\n"
        "            for (size_t e = begin; e > first;) {\n"
        "                --e;\n"
        "                preds[--offsets[edge_dst[e] + 1]] = (int32_t)head;\n"
        "            }\n"
        "            begin = first;\n"
        "        }\n"
        "        if (begin != 0) goto fail;\n"
        "        for (size_t i = 0; i < state_count; i++)\n"
        "            offsets[i] = offsets[i + 1];\n"
        "        offsets[state_count] = (int32_t)edge_count;\n"
        "    }\n"
        "    free(edge_counts); edge_counts = NULL;\n"
        "    free(edge_dst); edge_dst = NULL;\n"
        "    distance = (int32_t *)malloc(state_count * sizeof(int32_t));\n"
        "    queue = (int32_t *)malloc(state_count * sizeof(int32_t));\n"
        "    if (!distance || !queue) goto fail;",
        "CSR cursors and phased allocations",
    )
    source = _replace_once(
        source,
        "    result_states = (uint64_t *)malloc((finite ? finite : 1) * sizeof(uint64_t));",
        "    free(preds); preds = NULL;\n"
        "    free(offsets); offsets = NULL;\n"
        "    free(queue); queue = NULL;\n"
        "    result_states = (uint64_t *)malloc((finite ? finite : 1) * sizeof(uint64_t));",
        "pre-result workspace cleanup",
    )
    source = _replace_once(
        source,
        "    free(states); free(offsets); free(preds); free(distance); free(queue);",
        "    free(states); free(distance);",
        "success cleanup",
    )
    source = _replace_once(
        source,
        "    free(table.keys); free(table.index);\n"
        "    free(states); free(edge_end); free(edge_dst); free(offsets); free(preds);",
        "    free(table.index);\n"
        "    free(states); free(edge_counts); free(edge_dst); free(offsets); free(preds);",
        "failure cleanup",
    )
    if "edge_end" in source or "table.keys" in source or "edge_src" in source:
        raise ValueError("dense storage migration left obsolete storage symbols")
    return source


def verify_python(old,new):
    if Path(old).read_bytes() != Path(new).read_bytes():
        raise ValueError('Python planner must stay byte-identical for dense storage migration')


def verify_c(old,new):
    expected=compact_graph_storage(Path(old).read_text()).encode('utf-8')
    if Path(new).read_bytes()!=expected:
        raise ValueError('C differs from exact reviewed dense graph compression')


def verify_receipt(path,old_sources,new_sources):
    receipt=json.loads(Path(path).read_text())
    if (receipt.get('format')!=RECEIPT_FORMAT or receipt.get('status')!='complete'
        or any(receipt.get(k) is not True for k in ('generated_only','tests_passed','bfs_order_unchanged','graph_and_distances_unchanged','equality_verified'))
        or receipt.get('old_sources')!=old_sources or receipt.get('new_sources')!=new_sources):
        raise ValueError('complete exact-source storage equality receipt required')
    return receipt


def snapshot_tree(source,target):
    shutil.copytree(source,target)
    for path in target.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:os.fsync(stream.fileno())
    for path in [*sorted((p for p in target.rglob('*') if p.is_dir()),reverse=True),target]:
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)


def upgrade(bank,old_fastplan,old_c,old_runner,expected_old_fastplan,expected_old_c,
            expected_old_runner,validation_receipt):
    bank=Path(bank).resolve();archive=bank/ARCHIVE
    current=[LIVE_FASTPLAN.resolve(),LIVE_C.resolve(),LIVE_RUNNER.resolve()]
    snapshots=list(map(lambda p:Path(p).resolve(),(old_fastplan,old_c,old_runner)))
    expected=[expected_old_fastplan,expected_old_c,expected_old_runner]
    old_sources=dict(zip(map(str,current),expected));new_sources={str(p):sha(p) for p in current}
    validation_receipt=Path(validation_receipt).resolve()
    if archive.is_symlink():raise ValueError('archive symlink forbidden')
    if any(sha(p)!=h for p,h in zip(snapshots,expected)):raise ValueError('old snapshot SHA mismatch')
    verify_python(snapshots[0],current[0]);verify_c(snapshots[1],current[1])
    before,after=functions(snapshots[2]),functions(current[2])
    if any(before.get(name)!=after.get(name) or before.get(name) is None for name in PROTECTED):
        raise ValueError('protected runner gameplay/proof/job code changed')
    verify_receipt(validation_receipt,old_sources,new_sources)
    validation_sha=sha(validation_receipt);tool_sha=sha(__file__)
    validator_path=Path(__file__).with_name('upgrade_reference_scheduler.py')
    validator_sha=sha(validator_path)
    with (bank/'.regeneration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt_path=archive/'upgrade.json'
        staging=bank/(ARCHIVE+'.preparing')
        if not archive.exists() and (staging/'upgrade.json').is_file():
            staged=json.loads((staging/'upgrade.json').read_text())
            if (staged.get('format')!=FORMAT or staged.get('status')!='prepared'
                or staged.get('old_sources')!=old_sources or staged.get('new_sources')!=new_sources
                or staged.get('validation_sha256')!=validation_sha or staged.get('migration_source_sha256')!=tool_sha
                or staged.get('validator_source_sha256')!=validator_sha):
                raise ValueError('prepared staging transaction source drift')
            for relative,expected_sha in staged['archive_sha256'].items():
                if sha(staging/relative)!=expected_sha:raise ValueError('prepared staging archive corruption')
            os.replace(staging,archive)
            fd=os.open(bank,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        if archive.exists():
            if not receipt_path.is_file():raise ValueError('incomplete archive without transaction; manual inspection required')
            receipt=json.loads(receipt_path.read_text())
            if (receipt.get('format')!=FORMAT or receipt['old_sources']!=old_sources or receipt['new_sources']!=new_sources
                or receipt['validation_sha256']!=validation_sha or receipt['migration_source_sha256']!=tool_sha
                or receipt.get('validator_source_sha256')!=validator_sha):
                raise ValueError('transaction source/validation receipt drift')
            for relative,expected_sha in receipt['archive_sha256'].items():
                if sha(archive/relative)!=expected_sha:raise ValueError('archived source/job corruption')
            origin=archive
        else:origin=bank
        old=json.loads((origin/'manifest.json').read_text());report=json.loads((origin/'generation-report.json').read_text())
        if any(old['code_hashes'].get(p)!=h for p,h in old_sources.items()):raise ValueError('old snapshots not bound by old manifest')
        if (report.get('manifest_sha256')!=r._hash(old)
            or any(report.get(k)!=old[k] for k in ('config','code_hashes','source_hashes'))):
            raise ValueError('stopped report does not match old manifest binding')
        if report.get('workers_stopped') is not True:raise ValueError('old workers not confirmed stopped')
        pids=[report['pid'],*report.get('worker_pids',[])]
        for pid in pids:
            if type(pid) is not int or pid<=0 or Path(f'/proc/{pid}').exists():raise ValueError('old parent/worker PID exists or invalid')
        actual=r._code_hashes(old['config']['profile'])
        if (set(actual)!=set(old['code_hashes']) or any(actual[p]!=h for p,h in old['code_hashes'].items() if p not in old_sources)
            or any(actual.get(p)!=h for p,h in new_sources.items())):
            raise ValueError('unapproved code source changed')
        seeds,fingerprints,inventory=r._inventory(bank);seeds=set(seeds);fingerprints=set(fingerprints)
        if inventory!=old['source_hashes']:raise ValueError('source inventory changed')
        new=copy.deepcopy(old);new['code_hashes']=actual
        old_binding,new_binding=r._hash(old),r._hash(new)
        config=old['config'];jobs=[]
        for split in ('train','validation'):
            jobs+=r.jobs_for(config[split+'_count'],split,seeds,attempts=config['attempts'],limit=config['limit'],
                profile=config['profile'],quotas=config.get(split+'_quotas'),seed_range=config.get(split+'_seed_range'))
        expected_jobs={j['id']:j for j in jobs};states={};rows={};geometries={'train':set(),'validation':set()}
        accepted={'train':0,'validation':0}
        for path in sorted((origin/'jobs').glob('*.json')):
            if path.stem not in expected_jobs:raise ValueError('unknown checkpoint job')
            job=expected_jobs[path.stem];state=r._restore(path,job,old_binding);states[path.stem]=state
            if state['status']=='accepted':
                if r._accept(state['row'],job,seeds,fingerprints,geometries):raise ValueError('duplicate accepted checkpoint')
                accepted[job['split']]+=1;rows[path.stem]=r._hash(state['row'])
        if accepted!=report['accepted']:raise ValueError('stopped report accepted count differs from checkpoints')
        updated=copy.deepcopy(report);updated.update(manifest_sha256=new_binding,**new,
            fastplan_dense_storage_upgrade=dict(receipt=str(receipt_path),validation_sha256=validation_sha,
                                         old_binding=old_binding,new_binding=new_binding))
        # Validate *every* live state before changing any file, including retries.
        if {p.stem for p in (bank/'jobs').glob('*.json')}!=set(states):raise ValueError('live job inventory differs from transaction')
        if json.loads((bank/'manifest.json').read_text()) not in (old,new):raise ValueError('live manifest differs from transaction')
        if json.loads((bank/'generation-report.json').read_text()) not in (report,updated):raise ValueError('live report differs from transaction')
        for name,state in states.items():
            envelope=json.loads((bank/'jobs'/(name+'.json')).read_text());payload=envelope['payload']
            if (envelope['sha256']!=r._hash(payload) or payload['format']!=r.CHECKPOINT_FORMAT
                or payload['job']!=expected_jobs[name] or payload['state']!=state or payload['binding'] not in (old_binding,new_binding)):
                raise ValueError('live job state differs from archived transaction')
        if not archive.exists():
            staging=bank/(ARCHIVE+'.preparing')
            if staging.exists():raise ValueError('unfinished archive preparation; inspect before retry')
            staging.mkdir();shutil.copyfile(bank/'manifest.json',staging/'manifest.json')
            shutil.copyfile(bank/'generation-report.json',staging/'generation-report.json')
            snapshot_tree(bank/'jobs',staging/'jobs')
            for source,name in zip(snapshots,('old-fastplan.py','old-fastplan.c','old-runner.py')):shutil.copyfile(source,staging/name)
            shutil.copyfile(__file__,staging/'upgrade_fastplan_dense_storage.py')
            shutil.copyfile(validator_path,staging/'upgrade_reference_scheduler.py')
            shutil.copyfile(validation_receipt,staging/'validation.json')
            hashes={str(p.relative_to(staging)):sha(p) for p in staging.rglob('*') if p.is_file()}
            receipt=dict(format=FORMAT,status='prepared',old_sources=old_sources,new_sources=new_sources,
                validation_sha256=validation_sha,migration_source_sha256=tool_sha,validator_source_sha256=validator_sha,
                old_binding=old_binding,new_binding=new_binding,preserved_accepted=accepted,
                accepted_row_sha256=rows,archive_sha256=hashes)
            for path in staging.rglob('*'):
                if path.is_file():
                    with path.open('rb') as stream:os.fsync(stream.fileno())
            durable_json(staging/'upgrade.json',receipt)
            os.replace(staging,archive)
            fd=os.open(bank,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        elif receipt['old_binding']!=old_binding or receipt['new_binding']!=new_binding or receipt['accepted_row_sha256']!=rows:
            raise ValueError('transaction manifest/row drift')
        if receipt['status'] not in ('prepared','complete'):raise ValueError('invalid transaction status')
        if receipt['status']=='complete':
            if json.loads((bank/'manifest.json').read_text())!=new or json.loads((bank/'generation-report.json').read_text())!=updated:
                raise ValueError('completed transaction changed')
            for name in states:r._restore(bank/'jobs'/(name+'.json'),expected_jobs[name],new_binding)
            return receipt
        # Each envelope is independent and already validated above. Keep each
        # file and directory fsync, but overlap a bounded number of writes so
        # the checkpoint transition does not serialize thousands of barriers.
        def rebind(item):
            name,state=item
            payload=dict(format=r.CHECKPOINT_FORMAT,binding=new_binding,job=expected_jobs[name],state=state)
            durable_json(bank/'jobs'/(name+'.json'),dict(payload=payload,sha256=r._hash(payload)))
        with ThreadPoolExecutor(max_workers=8) as writers:
            list(writers.map(rebind,states.items()))
        durable_json(bank/'manifest.json',new);durable_json(bank/'generation-report.json',updated)
        for name,state in states.items():
            if r._restore(bank/'jobs'/(name+'.json'),expected_jobs[name],new_binding)!=state:raise ValueError('state changed after rebinding')
        if (r._inventory(bank)[2]!=inventory or r._code_hashes(config['profile'])!=actual
            or sha(validation_receipt)!=validation_sha or sha(__file__)!=tool_sha or sha(validator_path)!=validator_sha
            or any(sha(p)!=h for p,h in new_sources.items()) or any(sha(p)!=h for p,h in zip(snapshots,expected))):
            raise ValueError('source changed during transaction')
        receipt.update(status='complete',accepted_rows_unchanged=True,checkpoint_count=len(states))
        durable_json(receipt_path,receipt)
        return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upgrade-fastplan-dense-storage',action='store_true',required=True)
    p.add_argument('--bank-dir',type=Path,required=True)
    for name in ('fastplan','c','runner'):
        p.add_argument('--old-'+name+'-snapshot',type=Path,required=True)
        p.add_argument('--expected-old-'+name+'-sha256',required=True)
    p.add_argument('--validation-receipt',type=Path,required=True)
    a=p.parse_args(argv)
    receipt=upgrade(a.bank_dir,a.old_fastplan_snapshot,a.old_c_snapshot,a.old_runner_snapshot,
        a.expected_old_fastplan_sha256,a.expected_old_c_sha256,a.expected_old_runner_sha256,a.validation_receipt)
    print(json.dumps({k:receipt[k] for k in ('status','preserved_accepted','accepted_rows_unchanged')}))

if __name__=='__main__':main()
