"""Deterministically migrate the compact LS20 kernel to edge boundaries.

The transformation is deliberately textual and fail-closed.  It accepts only
the compact ABI-2 kernel revision staged for this experiment; every source
fragment must occur exactly once.  The resulting kernel keeps ``edge_dst`` in
the original action/discovery order and replaces the repeated ``edge_src``
array with one cumulative end offset per popped BFS head.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one exact match, found {count}")
    return source.replace(old, new, 1)


def compress_source_edges(source: str) -> str:
    """Return the edge-boundary kernel, refusing source drift."""
    source = _replace_once(
        source,
        "    int32_t *edge_src = NULL, *edge_dst = NULL, *offsets = NULL, *preds = NULL;",
        "    int32_t *edge_end = NULL, *edge_dst = NULL, *offsets = NULL, *preds = NULL;",
        "search declarations",
    )
    source = _replace_once(
        source,
        "    size_t state_count = 0, state_capacity = 0, edge_count = 0, edge_capacity = 0;",
        "    size_t state_count = 0, state_capacity = 0, edge_count = 0, edge_capacity = 0;\n"
        "    size_t edge_end_capacity = 0;",
        "search capacities",
    )
    source = _replace_once(
        source,
        "    for (size_t head = 0; head < state_count && !stop; head++) {\n"
        "        uint64_t state = states[head];\n"
        "        popped = head + 1;",
        "    for (size_t head = 0; head < state_count && !stop; head++) {\n"
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
        "        uint64_t state = states[head];\n"
        "        popped = head + 1;",
        "boundary allocation",
    )
    source = _replace_once(
        source,
        "        if ((int)((state >> p->shift_goals) & (uint64_t)p->mask_goals) == p->mask_goals)\n"
        "            continue;                  /* the level advances here; nothing follows */",
        "        if ((int)((state >> p->shift_goals) & (uint64_t)p->mask_goals) == p->mask_goals) {\n"
        "            /* Won heads are popped but have an empty predecessor interval. */\n"
        "            edge_end[head] = (int32_t)edge_count;\n"
        "            continue;                  /* the level advances here; nothing follows */\n"
        "        }",
        "won-head boundary",
    )
    source = _replace_once(
        source,
        "            if (edge_count >= edge_capacity) {\n"
        "                size_t bigger = edge_capacity ? edge_capacity * 2 : 4096;\n"
        "                int32_t *src = (int32_t *)realloc(edge_src, bigger * sizeof(int32_t));\n"
        "                if (!src) goto fail;\n"
        "                edge_src = src;\n"
        "                int32_t *dst = (int32_t *)realloc(edge_dst, bigger * sizeof(int32_t));\n"
        "                if (!dst) goto fail;\n"
        "                edge_dst = dst;\n"
        "                edge_capacity = bigger;\n"
        "            }\n"
        "            edge_src[edge_count] = (int32_t)head;\n"
        "            edge_dst[edge_count] = index;\n"
        "            edge_count++;",
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
        "edge destination storage",
    )
    source = _replace_once(
        source,
        "        }\n"
        "    }\n"
        "    free(table.keys); free(table.index); table.keys = NULL; table.index = NULL;",
        "        }\n"
        "        if (edge_count > (size_t)INT32_MAX)\n"
        "            goto fail;\n"
        "        edge_end[head] = (int32_t)edge_count;\n"
        "    }\n"
        "    free(table.keys); free(table.index); table.keys = NULL; table.index = NULL;",
        "close current boundary",
    )
    source = _replace_once(
        source,
        "        for (size_t e = 0; e < edge_count; e++)\n"
        "            preds[fill[edge_dst[e]]++] = edge_src[e];",
        "        for (size_t head = 0; head < popped; head++) {\n"
        "            size_t begin = head ? (size_t)edge_end[head - 1] : 0;\n"
        "            size_t end = (size_t)edge_end[head];\n"
        "            for (size_t e = begin; e < end; e++)\n"
        "                preds[fill[edge_dst[e]]++] = (int32_t)head;\n"
        "        }",
        "CSR predecessor traversal",
    )
    source = _replace_once(
        source,
        "    free(edge_src); free(edge_dst); edge_src = edge_dst = NULL;",
        "    free(edge_end); free(edge_dst); edge_end = edge_dst = NULL;",
        "CSR workspace cleanup",
    )
    source = _replace_once(
        source,
        "    free(states); free(edge_src); free(edge_dst); free(offsets); free(preds);",
        "    free(states); free(edge_end); free(edge_dst); free(offsets); free(preds);",
        "failure cleanup",
    )
    if "edge_src" in source:
        raise ValueError("edge_src remains after edge-boundary transformation")
    return source


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    transformed = compress_source_edges(args.input.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(transformed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
