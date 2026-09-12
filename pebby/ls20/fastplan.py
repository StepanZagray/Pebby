"""Fast exact search for `plan.Oracle`, backed by a small locally built C kernel.

`plan.simulate` stays the readable reference for LS20's rules; this module
turns a `Layout` into flat lookup tables and hands the whole breadth-first
search to `_fastplan.c`, which mirrors `simulate` and `Oracle._search` line by
line over states packed into one 64-bit word. Nothing here changes what the
oracle computes: the discovered set, its size, the truncation point and every
finite distance are the ones the reference produces, and
`tests/test_plan_performance.py` checks exactly that.

The kernel is compiled once with whatever C compiler the machine has, into the
package's ignored `__pycache__` directory (or a temp directory if that is not
writable), keyed by a hash of the source, so worker processes share one build
and never rebuild. If there is no compiler, the build fails, `ctypes` cannot
load the result, or a layout does not fit the packing, `available()` or
`tables_for()` say so and `plan.Oracle` falls back to the pure-Python search.
Set `PEBBY_FASTPLAN=0` to force that fallback.
"""

from collections.abc import Mapping
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

from . import names

SOURCE = Path(__file__).with_name("_fastplan.c")
# The result Mapping now requires the native compact-index symbols below.
# Bump the ABI so an old shared object can never be accepted accidentally.
ABI_VERSION = 2
KIND_CODES = {"shape": 1, "color": 2, "rotation": 3}
OUTCOMES = ("rejected", "hint", "launched", "won", "died", "moved")
FIELD_LIMIT = 62  # bits available for one packed state (keeps every key a positive int64)


class Unsupported(Exception):
    """This layout cannot be packed for the kernel; use the reference search."""


# -- building and loading the kernel -----------------------------------------------------------------

_lock = threading.Lock()
_library = None
_load_error = None


class Params(ctypes.Structure):
    _fields_ = [
        ("target", ctypes.POINTER(ctypes.c_int32)),
        ("free_move", ctypes.POINTER(ctypes.c_uint8)),
        ("goal_at", ctypes.POINTER(ctypes.c_int32)),
        ("refill_at", ctypes.POINTER(ctypes.c_int32)),
        ("launch_to", ctypes.POINTER(ctypes.c_int32)),
        ("kind_at", ctypes.POINTER(ctypes.c_uint8)),
        ("next_tick", ctypes.POINTER(ctypes.c_int32)),
        ("goal_shape", ctypes.POINTER(ctypes.c_int32)),
        ("goal_color", ctypes.POINTER(ctypes.c_int32)),
        ("goal_rotation", ctypes.POINTER(ctypes.c_int32)),
        ("cells", ctypes.c_int32), ("ticks", ctypes.c_int32), ("goals", ctypes.c_int32),
        ("cost", ctypes.c_int32), ("max_steps", ctypes.c_int32), ("match_hint", ctypes.c_int32),
        ("shape_count", ctypes.c_int32), ("color_count", ctypes.c_int32),
        ("rotation_count", ctypes.c_int32),
        ("shift_shape", ctypes.c_int32), ("shift_color", ctypes.c_int32),
        ("shift_rotation", ctypes.c_int32), ("shift_goals", ctypes.c_int32),
        ("shift_taken", ctypes.c_int32), ("shift_steps", ctypes.c_int32),
        ("shift_tick", ctypes.c_int32),
        ("mask_cell", ctypes.c_int32), ("mask_shape", ctypes.c_int32),
        ("mask_color", ctypes.c_int32), ("mask_rotation", ctypes.c_int32),
        ("mask_goals", ctypes.c_int32), ("mask_taken", ctypes.c_int32),
        ("mask_steps", ctypes.c_int32), ("mask_tick", ctypes.c_int32),
        ("steps_offset", ctypes.c_int32),
    ]


def _build_dir():
    candidates = [Path(__file__).parent / "__pycache__",
                  Path(tempfile.gettempdir()) / f"pebby-fastplan-{os.getuid() if hasattr(os, 'getuid') else 'u'}"]
    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if os.access(directory, os.W_OK):
                return directory
        except OSError:
            continue
    return None


def library_path():
    """Where the kernel for this source, interpreter and platform lives (built or not)."""
    directory = _build_dir()
    if directory is None:
        return None
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
    tag = f"{sys.platform}-{sys.implementation.cache_tag}"
    suffix = ".dll" if sys.platform == "win32" else ".so"
    return directory / f"_fastplan-{tag}-abi{ABI_VERSION}-{digest}{suffix}"


def _compile(path):
    compilers = [c for c in (os.environ.get("CC"), "cc", "gcc", "clang") if c and shutil.which(c)]
    if not compilers:
        raise OSError("no C compiler found on PATH (tried CC, cc, gcc, clang)")
    errors = []
    for compiler in compilers:
        # Build to a private name and rename into place, so concurrent workers
        # racing to build the same file never see a half-written library.
        scratch = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        command = [compiler, "-O2", "-std=c99", "-fPIC", "-shared", "-o", str(scratch), str(SOURCE)]
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f"{compiler}: {error}")
            continue
        if done.returncode == 0 and scratch.is_file():
            os.replace(scratch, path)
            return
        errors.append(f"{compiler}: exit {done.returncode}: {done.stderr.strip()[:500]}")
        try:
            scratch.unlink()
        except OSError:
            pass
    raise OSError("; ".join(errors))


def _load():
    global _library, _load_error
    if _library is not None or _load_error is not None:
        return _library
    with _lock:
        if _library is not None or _load_error is not None:
            return _library
        try:
            if os.environ.get("PEBBY_FASTPLAN", "1") == "0":
                raise OSError("disabled by PEBBY_FASTPLAN=0")
            path = library_path()
            if path is None:
                raise OSError("no writable build directory for the kernel")
            if not path.is_file():
                _compile(path)
            library = ctypes.CDLL(str(path))
            library.ls20_abi_version.restype = ctypes.c_int
            library.ls20_abi_version.argtypes = []
            if library.ls20_abi_version() != ABI_VERSION:
                raise OSError("kernel ABI mismatch")
            library.ls20_step.restype = ctypes.c_int
            library.ls20_step.argtypes = [ctypes.POINTER(Params), ctypes.c_uint64, ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_uint64)]
            library.ls20_search.restype = ctypes.c_int
            library.ls20_search.argtypes = [
                ctypes.POINTER(Params), ctypes.c_uint64, ctypes.c_int64,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_uint64)),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
                ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int32)]
            library.ls20_free.restype = None
            library.ls20_free.argtypes = [ctypes.c_void_p]
            library.ls20_index_build.restype = ctypes.POINTER(ctypes.c_int32)
            library.ls20_index_build.argtypes = [
                ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t)]
            library.ls20_index_lookup.restype = ctypes.c_int64
            library.ls20_index_lookup.argtypes = [
                ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_uint64]
            _library = library
        except (OSError, AttributeError) as error:
            _load_error = str(error)
    return _library


def available():
    """True when the kernel is built and loaded; `load_error()` says why if not."""
    return _load() is not None


def load_error():
    _load()
    return _load_error


# -- tables ------------------------------------------------------------------------------------------

class Tables:
    """Everything the kernel needs to know about one layout, plus the state codec.

    Cells are indexed over a universe that covers every cell a planner state can
    name: the 12x12 lattice, the start cell, and every launcher landing cell
    (upstream never bounds-checks a launch, so a landing may lie off-lattice).
    """

    def __init__(self, layout, refills):
        cells = [(col, row) for row in range(layout.rows) for col in range(layout.cols)]
        seen = set(cells)
        for extra in (layout.start_cell, *self._landings(layout)):
            extra = tuple(extra)
            if extra not in seen:
                seen.add(extra)
                cells.append(extra)
        self.cells = cells
        self.cell_index = {cell: index for index, cell in enumerate(cells)}
        self.refills = tuple(refills)
        self.goal_count = len(layout.goals)
        self.cost, self.max_steps = layout.step_cost, layout.max_steps
        self.ticks = len(layout.moving_cyclers)
        if self.ticks != layout.tick_span or self.ticks < 1:
            raise Unsupported("patroller schedule does not match tick_span")
        self.steps_offset = self.cost
        widths = (self._width(len(cells) - 1), self._width(names.SHAPE_COUNT - 1),
                  self._width(names.COLOR_COUNT - 1), self._width(names.ROTATION_COUNT - 1),
                  self.goal_count, len(self.refills),
                  self._width(self.max_steps + self.steps_offset), self._width(self.ticks - 1))
        if any(width > 30 for width in widths) or sum(widths) > FIELD_LIMIT:
            raise Unsupported("state does not fit the kernel's packed word")
        if self.cost < 0 or self.max_steps < 0:
            raise Unsupported("negative budget")
        shifts, total = [], 0
        for width in widths:
            shifts.append(total)
            total += width
        (self.shift_cell, self.shift_shape, self.shift_color, self.shift_rotation,
         self.shift_goals, self.shift_taken, self.shift_steps, self.shift_tick) = shifts
        (self.mask_cell, self.mask_shape, self.mask_color, self.mask_rotation,
         self.mask_goals, self.mask_taken, self.mask_steps, self.mask_tick) = [
            (1 << width) - 1 for width in widths]

        count = len(cells)
        target = (ctypes.c_int32 * (count * 4))()
        free_move = (ctypes.c_uint8 * (count * 4))()
        goal_at = (ctypes.c_int32 * count)(*[layout.goal_at.get(cell, -1) for cell in cells])
        refill_at = (ctypes.c_int32 * count)(
            *[self.refills.index(cell) if cell in layout.refills else -1 for cell in cells])
        launch_to = (ctypes.c_int32 * count)(*[self._launch(layout, cell) for cell in cells])
        for index, cell in enumerate(cells):
            for action, (dx, dy) in enumerate(names.ACTION_DELTAS):
                probe = (cell[0] + dx, cell[1] + dy)
                if layout.free(probe):
                    target[index * 4 + action] = self.cell_index[probe]
                    free_move[index * 4 + action] = 1
                else:
                    target[index * 4 + action] = index
        kinds = []
        for tick in range(self.ticks):
            kinds.extend(KIND_CODES.get(layout.cycler_at(cell, tick), 0) for cell in cells)
        kind_at = (ctypes.c_uint8 * (self.ticks * count))(*kinds)
        next_tick = (ctypes.c_int32 * self.ticks)(*[layout.next_tick(t) for t in range(self.ticks)])
        if any(not 0 <= next_tick[t] < self.ticks for t in range(self.ticks)):
            raise Unsupported("tick schedule leaves its range")
        goals = max(self.goal_count, 1)
        goal_shape = (ctypes.c_int32 * goals)(*[triple[0] for _, triple in layout.goals] or [-1])
        goal_color = (ctypes.c_int32 * goals)(*[triple[1] for _, triple in layout.goals] or [-1])
        goal_rotation = (ctypes.c_int32 * goals)(*[triple[2] for _, triple in layout.goals] or [-1])
        # Keep the buffers alive as long as the struct that points into them.
        self._buffers = (target, free_move, goal_at, refill_at, launch_to, kind_at, next_tick,
                         goal_shape, goal_color, goal_rotation)
        self.params = Params(
            target=target, free_move=free_move, goal_at=goal_at, refill_at=refill_at,
            launch_to=launch_to, kind_at=kind_at, next_tick=next_tick,
            goal_shape=goal_shape, goal_color=goal_color, goal_rotation=goal_rotation,
            cells=count, ticks=self.ticks, goals=self.goal_count, cost=self.cost,
            max_steps=self.max_steps, match_hint=int(bool(layout.match_hint)),
            shape_count=names.SHAPE_COUNT, color_count=names.COLOR_COUNT,
            rotation_count=names.ROTATION_COUNT,
            shift_shape=self.shift_shape, shift_color=self.shift_color,
            shift_rotation=self.shift_rotation, shift_goals=self.shift_goals,
            shift_taken=self.shift_taken, shift_steps=self.shift_steps, shift_tick=self.shift_tick,
            mask_cell=self.mask_cell, mask_shape=self.mask_shape, mask_color=self.mask_color,
            mask_rotation=self.mask_rotation, mask_goals=self.mask_goals,
            mask_taken=self.mask_taken, mask_steps=self.mask_steps, mask_tick=self.mask_tick,
            steps_offset=self.steps_offset)

    @staticmethod
    def _width(largest):
        return max(1, int(largest).bit_length()) if largest > 0 else 0

    @staticmethod
    def _landings(layout):
        for launcher in layout.launchers:
            if launcher["distance"] <= 0:
                continue
            dx, dy = launcher["delta"]
            for trigger in launcher["triggers"]:
                yield (trigger[0] + dx * launcher["distance"], trigger[1] + dy * launcher["distance"])

    def _launch(self, layout, cell):
        # First launcher in list order whose triggers hold the cell and that throws.
        for launcher in layout.launchers:
            if cell in launcher["triggers"] and launcher["distance"] > 0:
                dx, dy = launcher["delta"]
                return self.cell_index[(cell[0] + dx * launcher["distance"],
                                        cell[1] + dy * launcher["distance"])]
        return -1

    # -- codec ----------------------------------------------------------------

    def pack(self, state):
        """The packed word for a planner state tuple, or None if it cannot be one."""
        try:
            cell, shape, color, rotation, goals, taken, steps, tick = state
            index = self.cell_index.get(cell)
        except (TypeError, ValueError):
            return None
        if index is None:
            return None
        stored = steps + self.steps_offset
        if not (0 <= shape <= self.mask_shape and 0 <= color <= self.mask_color
                and 0 <= rotation <= self.mask_rotation and 0 <= goals <= self.mask_goals
                and 0 <= taken <= self.mask_taken and 0 <= stored <= self.mask_steps
                and 0 <= tick <= self.mask_tick):
            return None
        return (index | (shape << self.shift_shape) | (color << self.shift_color)
                | (rotation << self.shift_rotation) | (goals << self.shift_goals)
                | (taken << self.shift_taken) | (stored << self.shift_steps)
                | (tick << self.shift_tick))

    def unpack(self, key):
        return (self.cells[key & self.mask_cell],
                (key >> self.shift_shape) & self.mask_shape,
                (key >> self.shift_color) & self.mask_color,
                (key >> self.shift_rotation) & self.mask_rotation,
                (key >> self.shift_goals) & self.mask_goals,
                (key >> self.shift_taken) & self.mask_taken,
                ((key >> self.shift_steps) & self.mask_steps) - self.steps_offset,
                (key >> self.shift_tick) & self.mask_tick)


def tables_for(layout, refills):
    """`Tables` for the layout, or None if the kernel cannot represent it."""
    try:
        return Tables(layout, refills)
    except Unsupported:
        return None


# -- results -----------------------------------------------------------------------------------------

class PackedDistances(Mapping):
    """`Oracle._distance` over packed keys: planner-state tuple -> optimal actions left.

    Behaves as a read-only dict keyed by the same tuples the reference search
    stores, without ever materialising them; a tuple outside the layout's state
    space is simply absent.
    """

    __slots__ = ("_tables", "_keys", "_values", "_index", "_capacity", "_count",
                 "_library", "_lock")

    def __init__(self, tables, keys, values, index, capacity, count, library):
        self._tables = tables
        self._keys, self._values = keys, values
        self._index, self._capacity, self._count = index, int(capacity), int(count)
        self._library = library
        # Search arrays are immutable and safe for concurrent readers.  The
        # small lock only serializes explicit close() against a reader so a
        # caller cannot free a native buffer while ctypes is using it.
        self._lock = threading.Lock()

    @classmethod
    def from_native(cls, tables, keys, values, count, library):
        """Take ownership of native search arrays and build their compact index."""
        count = int(count)
        if count < 0:
            raise ValueError("negative search result count")
        capacity = ctypes.c_size_t()
        index = None
        try:
            if count:
                index = library.ls20_index_build(
                    keys, ctypes.c_size_t(count), ctypes.byref(capacity))
                if not index:
                    raise MemoryError("fast planner result index allocation failed")
            result = cls(tables, keys, values, index, capacity.value, count, library)
            # Ownership has moved to result.  The caller must not free these pointers.
            return result
        except BaseException:
            if index:
                library.ls20_free(index)
            raise

    def close(self):
        """Release the C result buffers; safe to call more than once."""
        with self._lock:
            library = self._library
            keys, values, index = self._keys, self._values, self._index
            self._keys = self._values = self._index = None
            self._capacity = self._count = 0
            if library is not None:
                if index:
                    library.ls20_free(index)
                if keys:
                    library.ls20_free(keys)
                if values:
                    library.ls20_free(values)
            self._library = None

    def __del__(self):  # pragma: no cover - exercised by repeated-free smoke tests
        try:
            self.close()
        except Exception:
            # Interpreter shutdown can tear down ctypes before this object.  A
            # best-effort leak is safer than calling into a half-destroyed DLL.
            pass

    def _value_for_key(self, key):
        """Return a copied distance while holding the native-buffer lock."""
        with self._lock:
            if self._count == 0 or self._index is None:
                return None
            position = int(self._library.ls20_index_lookup(
                self._index, self._capacity, self._keys, self._count,
                ctypes.c_uint64(key)))
            if position < 0:
                return None
            # Copy the int32 value before close() can free the arrays.
            return int(self._values[position])

    def __getitem__(self, state):
        key = self._tables.pack(state)
        if key is None:
            raise KeyError(state)
        value = self._value_for_key(key)
        if value is None:
            raise KeyError(state)
        return value

    def get(self, state, default=None):
        key = self._tables.pack(state)
        if key is None:
            return default
        value = self._value_for_key(key)
        return default if value is None else value

    def __contains__(self, state):
        key = self._tables.pack(state)
        return key is not None and self._value_for_key(key) is not None

    def __len__(self):
        return self._count

    def __iter__(self):
        unpack = self._tables.unpack

        def iterator():
            position = 0
            while True:
                with self._lock:
                    if position >= self._count or self._keys is None:
                        return
                    key = int(self._keys[position])
                yield unpack(key)
                position += 1

        return iterator()

    def __repr__(self):
        return f"PackedDistances({len(self)} states)"


def search(tables, start, limit):
    """Run the kernel. Returns (distances, reachable, truncated); raises MemoryError."""
    library = _load()
    if library is None:
        raise OSError(_load_error)
    key = tables.pack(start)
    if key is None:
        raise Unsupported("start state does not fit the packed word")
    states = ctypes.POINTER(ctypes.c_uint64)()
    distance = ctypes.POINTER(ctypes.c_int32)()
    count, reachable, truncated = ctypes.c_int64(), ctypes.c_int64(), ctypes.c_int32()
    status = library.ls20_search(ctypes.byref(tables.params), key, int(limit),
                                 ctypes.byref(states), ctypes.byref(distance),
                                 ctypes.byref(count), ctypes.byref(reachable), ctypes.byref(truncated))
    if status != 0:
        raise MemoryError("fast planner ran out of memory")
    owned = False
    try:
        result = PackedDistances.from_native(tables, states, distance, count.value, library)
        owned = True
        return result, reachable.value, bool(truncated.value)
    finally:
        if not owned:
            if states:
                library.ls20_free(states)
            if distance:
                library.ls20_free(distance)


def step(tables, state, action):
    """The kernel's `plan.simulate`: (next_state, outcome) for a state tuple.

    Only for differential tests; the planner itself keeps `plan.simulate` for
    anything that is not the exhaustive search. `state` must be one the search
    would expand: a won state with an overdrawn budget is never stepped by the
    reference, and charging it again would leave the packed budget field.
    """
    library = _load()
    if library is None:
        raise OSError(_load_error)
    key = tables.pack(state)
    if key is None:
        raise ValueError(f"state {state!r} is outside the packed space")
    out = ctypes.c_uint64()
    outcome = library.ls20_step(ctypes.byref(tables.params), key, int(action), ctypes.byref(out))
    return tables.unpack(out.value), OUTCOMES[outcome]
