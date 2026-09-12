/* Exact LS20 planner kernel: the search in `plan.Oracle._search`, in C.
 *
 * This is a transliteration of `plan.simulate` / `plan.advance` and of the
 * forward-then-reverse breadth-first search in `plan.Oracle._search`, over
 * states packed into one 64-bit word. It adds no rule and no shortcut: every
 * branch below has a counterpart in plan.py, and `fastplan.py` documents the
 * table layout it expects. `tests/test_plan_performance.py` checks the two
 * against each other transition by transition and state by state.
 *
 * Built on demand by `fastplan.py` with the local C compiler; if that fails
 * the planner simply keeps using the Python reference.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define OUTCOME_REJECTED 0
#define OUTCOME_HINT 1
#define OUTCOME_LAUNCHED 2
#define OUTCOME_WON 3
#define OUTCOME_DIED 4
#define OUTCOME_MOVED 5

typedef struct {
    /* Per (cell, action): the target cell index when it is free, else the cell itself. */
    const int32_t *target;
    const uint8_t *free_move;
    /* Per cell: goal index or -1, refill slot or -1, launch landing cell or -1. */
    const int32_t *goal_at;
    const int32_t *refill_at;
    const int32_t *launch_to;
    /* Per (tick, cell): 0 none, 1 shape, 2 color, 3 rotation (static tiles take priority). */
    const uint8_t *kind_at;
    const int32_t *next_tick;      /* per tick */
    const int32_t *goal_shape;     /* per goal */
    const int32_t *goal_color;
    const int32_t *goal_rotation;
    int32_t cells, ticks, goals, cost, max_steps, match_hint;
    int32_t shape_count, color_count, rotation_count;
    /* Packing: field = (state >> shift) & mask. `steps` is stored offset by `steps_offset`. */
    int32_t shift_shape, shift_color, shift_rotation, shift_goals, shift_taken, shift_steps, shift_tick;
    int32_t mask_cell, mask_shape, mask_color, mask_rotation, mask_goals, mask_taken, mask_steps, mask_tick;
    int32_t steps_offset;
} Params;

static inline uint64_t pack(const Params *p, int cell, int shape, int color, int rotation,
                            int goals, int taken, int steps, int tick) {
    return (uint64_t)cell
        | ((uint64_t)shape << p->shift_shape)
        | ((uint64_t)color << p->shift_color)
        | ((uint64_t)rotation << p->shift_rotation)
        | ((uint64_t)goals << p->shift_goals)
        | ((uint64_t)taken << p->shift_taken)
        | ((uint64_t)(steps + p->steps_offset) << p->shift_steps)
        | ((uint64_t)tick << p->shift_tick);
}

/* `plan.simulate`, one action at full fidelity. Returns the outcome code. */
int ls20_step(const Params *p, uint64_t state, int action, uint64_t *out) {
    int cell = (int)(state & (uint64_t)p->mask_cell);
    int shape = (int)((state >> p->shift_shape) & (uint64_t)p->mask_shape);
    int color = (int)((state >> p->shift_color) & (uint64_t)p->mask_color);
    int rotation = (int)((state >> p->shift_rotation) & (uint64_t)p->mask_rotation);
    int goals = (int)((state >> p->shift_goals) & (uint64_t)p->mask_goals);
    int taken = (int)((state >> p->shift_taken) & (uint64_t)p->mask_taken);
    int steps = (int)((state >> p->shift_steps) & (uint64_t)p->mask_steps) - p->steps_offset;
    int tick = (int)((state >> p->shift_tick) & (uint64_t)p->mask_tick);
    int moved = p->next_tick[tick];
    int rejected = 0, cycled = 0, refilled = 0;
    int position, new_tick;

    if (!p->free_move[cell * 4 + action]) {
        position = cell;                 /* wall or edge: no cell effects at all */
        new_tick = tick;
    } else {
        int target = p->target[cell * 4 + action];
        int index = p->goal_at[target];
        if (index >= 0 && !((goals >> index) & 1)
                && !(shape == p->goal_shape[index] && color == p->goal_color[index]
                     && rotation == p->goal_rotation[index]))
            rejected = 1;
        int slot = p->refill_at[target];
        if (slot >= 0 && !((taken >> slot) & 1)) {
            taken |= 1 << slot;
            steps = p->max_steps;
            refilled = 1;
        }
        int kind = p->kind_at[moved * p->cells + target];
        if (kind == 1) { shape = (shape + 1) % p->shape_count; cycled = 1; }
        else if (kind == 2) { color = (color + 1) % p->color_count; cycled = 1; }
        else if (kind == 3) { rotation = (rotation + 1) % p->rotation_count; cycled = 1; }
        position = rejected ? cell : target;
        new_tick = rejected ? tick : moved;
    }

    if (rejected) {
        *out = pack(p, position, shape, color, rotation, goals, taken, steps, new_tick);
        return OUTCOME_REJECTED;
    }
    if (p->match_hint && cycled) {
        for (int i = 0; i < p->goals; i++) {
            if (!((goals >> i) & 1) && shape == p->goal_shape[i] && color == p->goal_color[i]
                    && rotation == p->goal_rotation[i]) {
                *out = pack(p, position, shape, color, rotation, goals, taken, steps, new_tick);
                return OUTCOME_HINT;
            }
        }
    }

    if (!refilled)
        steps -= p->cost;
    int exhausted = steps < 0;

    if (!exhausted) {
        int landing = p->launch_to[position];
        if (landing >= 0) {
            position = landing;
            int slot = p->refill_at[position];
            if (slot >= 0 && !((taken >> slot) & 1)) {
                taken |= 1 << slot;
                steps = p->max_steps;
            }
            int kind = p->kind_at[new_tick * p->cells + position];
            if (kind == 1) shape = (shape + 1) % p->shape_count;
            else if (kind == 2) color = (color + 1) % p->color_count;
            else if (kind == 3) rotation = (rotation + 1) % p->rotation_count;
            *out = pack(p, position, shape, color, rotation, goals, taken, steps, new_tick);
            return OUTCOME_LAUNCHED;
        }
    }

    int index = p->goal_at[position];
    if (index >= 0 && !((goals >> index) & 1) && shape == p->goal_shape[index]
            && color == p->goal_color[index] && rotation == p->goal_rotation[index])
        goals |= 1 << index;
    *out = pack(p, position, shape, color, rotation, goals, taken, steps, new_tick);
    if (goals == p->mask_goals)
        return OUTCOME_WON;
    if (exhausted)
        return OUTCOME_DIED;
    return OUTCOME_MOVED;
}

/* `plan.advance`: 1 and the successor in *out, or 0 when nothing can follow. */
static inline int advance(const Params *p, uint64_t state, int action, uint64_t *out) {
    int outcome = ls20_step(p, state, action, out);
    if (outcome == OUTCOME_REJECTED || outcome == OUTCOME_DIED || *out == state)
        return 0;
    return 1;
}

/* -- open-addressed hash set: packed state -> discovery index ---------------- */

typedef struct {
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
}

/* -- growable arrays --------------------------------------------------------- */

#define GROW(ptr, type, count, capacity)                                         \
    do {                                                                         \
        if ((count) >= (capacity)) {                                             \
            size_t bigger = (capacity) ? (capacity) * 2 : 1024;                  \
            type *fresh = (type *)realloc((ptr), bigger * sizeof(type));         \
            if (!fresh) goto fail;                                               \
            (ptr) = fresh;                                                       \
            (capacity) = bigger;                                                 \
        }                                                                        \
    } while (0)

/* `plan.Oracle._search`.
 *
 * Forward BFS from `start` in the same order as the Python (FIFO, actions 0..3),
 * stopping with `truncated` at the same discovery as the reference does; then a
 * reverse BFS from every won state over the recorded predecessor edges.
 *
 * Outputs, all malloc'd for `ls20_free`: the packed states with a finite
 * distance and those distances (`*count` of them); `*reachable` is the size of
 * the whole discovered set. Returns 0, or -1 if memory ran out (nothing is
 * left allocated in that case).
 */
int ls20_search(const Params *p, uint64_t start, int64_t limit,
                uint64_t **out_states, int32_t **out_distance, int64_t *count,
                int64_t *reachable, int32_t *truncated) {
    uint64_t *states = NULL;
    int32_t *edge_src = NULL, *edge_dst = NULL, *offsets = NULL, *preds = NULL;
    int32_t *distance = NULL, *queue = NULL;
    uint64_t *result_states = NULL;
    int32_t *result_distance = NULL;
    size_t state_count = 0, state_capacity = 0, edge_count = 0, edge_capacity = 0;
    Table table = {0};
    int stop = 0;
    size_t popped = 0;   /* how many queue entries the forward pass took out */
    *out_states = NULL; *out_distance = NULL; *count = 0; *reachable = 0; *truncated = 0;

    if (table_init(&table, 1 << 12) != 0)
        return -1;
    GROW(states, uint64_t, state_count, state_capacity);
    states[state_count++] = start;
    if (table_insert(&table, start, 0) != 0) goto fail;

    for (size_t head = 0; head < state_count && !stop; head++) {
        uint64_t state = states[head];
        popped = head + 1;
        if ((int)((state >> p->shift_goals) & (uint64_t)p->mask_goals) == p->mask_goals)
            continue;                  /* the level advances here; nothing follows */
        for (int action = 0; action < 4; action++) {
            uint64_t next;
            if (!advance(p, state, action, &next))
                continue;
            int32_t index = table_find(&table, next);
            if (index < 0) {
                if ((int64_t)state_count >= limit) {
                    /* The reference clears its queue here, so won states still
                     * waiting in it are never collected as wins. Mirror that. */
                    *truncated = 1;
                    stop = 1;
                    break;
                }
                index = (int32_t)state_count;
                GROW(states, uint64_t, state_count, state_capacity);
                states[state_count++] = next;
                if (table_insert(&table, next, index) != 0) goto fail;
            }
            if (edge_count >= edge_capacity) {
                size_t bigger = edge_capacity ? edge_capacity * 2 : 4096;
                int32_t *src = (int32_t *)realloc(edge_src, bigger * sizeof(int32_t));
                if (!src) goto fail;
                edge_src = src;
                int32_t *dst = (int32_t *)realloc(edge_dst, bigger * sizeof(int32_t));
                if (!dst) goto fail;
                edge_dst = dst;
                edge_capacity = bigger;
            }
            edge_src[edge_count] = (int32_t)head;
            edge_dst[edge_count] = index;
            edge_count++;
        }
    }
    free(table.keys); free(table.index); table.keys = NULL; table.index = NULL;

    /* Predecessor lists in CSR form, keyed by successor. */
    offsets = (int32_t *)calloc(state_count + 1, sizeof(int32_t));
    preds = (int32_t *)malloc((edge_count ? edge_count : 1) * sizeof(int32_t));
    distance = (int32_t *)malloc(state_count * sizeof(int32_t));
    queue = (int32_t *)malloc(state_count * sizeof(int32_t));
    if (!offsets || !preds || !distance || !queue) goto fail;
    for (size_t e = 0; e < edge_count; e++)
        offsets[edge_dst[e] + 1]++;
    for (size_t i = 0; i < state_count; i++)
        offsets[i + 1] += offsets[i];
    {
        int32_t *fill = (int32_t *)malloc((state_count + 1) * sizeof(int32_t));
        if (!fill) goto fail;
        memcpy(fill, offsets, (state_count + 1) * sizeof(int32_t));
        for (size_t e = 0; e < edge_count; e++)
            preds[fill[edge_dst[e]]++] = edge_src[e];
        free(fill);
    }
    free(edge_src); free(edge_dst); edge_src = edge_dst = NULL;

    /* Reverse BFS from every won state the forward pass got round to popping. */
    size_t qhead = 0, qtail = 0, finite = 0;
    for (size_t i = 0; i < state_count; i++) {
        distance[i] = -1;
        if (i < popped
                && (int)((states[i] >> p->shift_goals) & (uint64_t)p->mask_goals) == p->mask_goals) {
            distance[i] = 0;
            queue[qtail++] = (int32_t)i;
            finite++;
        }
    }
    while (qhead < qtail) {
        int32_t s = queue[qhead++];
        for (int32_t k = offsets[s]; k < offsets[s + 1]; k++) {
            int32_t previous = preds[k];
            if (distance[previous] < 0) {
                distance[previous] = distance[s] + 1;
                queue[qtail++] = previous;
                finite++;
            }
        }
    }

    result_states = (uint64_t *)malloc((finite ? finite : 1) * sizeof(uint64_t));
    result_distance = (int32_t *)malloc((finite ? finite : 1) * sizeof(int32_t));
    if (!result_states || !result_distance) goto fail;
    {
        size_t k = 0;
        for (size_t i = 0; i < state_count; i++) {
            if (distance[i] >= 0) {
                result_states[k] = states[i];
                result_distance[k] = distance[i];
                k++;
            }
        }
    }
    free(states); free(offsets); free(preds); free(distance); free(queue);
    *out_states = result_states;
    *out_distance = result_distance;
    *count = (int64_t)finite;
    *reachable = (int64_t)state_count;
    return 0;

fail:
    free(table.keys); free(table.index);
    free(states); free(edge_src); free(edge_dst); free(offsets); free(preds);
    free(distance); free(queue); free(result_states); free(result_distance);
    *out_states = NULL; *out_distance = NULL; *count = 0; *reachable = 0; *truncated = 0;
    return -1;
}

void ls20_free(void *pointer) {
    free(pointer);
}

int ls20_abi_version(void) {
    return 2;
}

/*
 * Compact read-only index for the result arrays returned by ls20_search.
 *
 * The search result itself is already in discovery order.  Keep those two
 * native arrays and add only an open-addressed uint32-sized (int32_t) slot
 * index so Python Mapping lookups do not expand every key/value into Python
 * objects.  Positions are int32_t because the planner's accepted result
 * count is bounded well below INT32_MAX; reject larger counts explicitly.
 */
int32_t *ls20_index_build(const uint64_t *keys, size_t count, size_t *out_capacity) {
    if (!out_capacity)
        return NULL;
    *out_capacity = 0;
    if (!keys || count == 0 || count > (size_t)INT32_MAX
            || count > ((size_t)-1) / 2)
        return NULL;

    size_t capacity = 1;
    size_t target = count * 2;  /* load factor is at most one half */
    while (capacity < target) {
        if (capacity > ((size_t)-1) / 2)
            return NULL;
        capacity <<= 1;
    }
    if (capacity > ((size_t)-1) / sizeof(int32_t))
        return NULL;

    int32_t *index = (int32_t *)malloc(capacity * sizeof(int32_t));
    if (!index)
        return NULL;
    memset(index, 0xff, capacity * sizeof(int32_t));
    for (size_t position = 0; position < count; position++) {
        size_t slot = slot_of(keys[position], capacity);
        while (index[slot] >= 0)
            slot = (slot + 1) & (capacity - 1);
        index[slot] = (int32_t)position;
    }
    *out_capacity = capacity;
    return index;
}

int64_t ls20_index_lookup(const int32_t *index, size_t capacity,
                          const uint64_t *keys, size_t count, uint64_t key) {
    if (!index || !keys || capacity == 0 || count == 0)
        return -1;
    size_t slot = slot_of(key, capacity);
    for (size_t probe = 0; probe < capacity; probe++) {
        int32_t position = index[slot];
        if (position < 0)
            return -1;
        if ((size_t)position < count && keys[position] == key)
            return (int64_t)position;
        slot = (slot + 1) & (capacity - 1);
    }
    return -1;
}
