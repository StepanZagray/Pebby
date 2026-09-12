"""Fail-closed source migration for the dense LS20 search storage layout.

The input is the staged ABI-2 edge-boundary kernel.  This tool intentionally
uses exact, uniquely matched text fragments so a future source revision cannot
silently receive only part of the storage migration.
"""

from __future__ import annotations

import argparse
from pathlib import Path


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    result = compact_graph_storage(args.input.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
