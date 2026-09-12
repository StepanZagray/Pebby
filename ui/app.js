/* The generated-level viewer.

   The server is stateless: it holds no game. This page owns the level spec and
   the list of actions taken so far, and every board it shows is the server
   replaying that whole list from the start. That is what makes undo exact and
   what lets the solution scrubber jump to any step — a prefix of the level's
   own stored solution is just another action list.

   Every request goes through HostAI. When the page is embedded, HostAI injects
   the bridge and we call it; standalone, we post the same envelope ourselves. */
(function () {
  "use strict";

  const MODEL = "pebby:latest";
  const REQUEST_TIMEOUT = 60000;
  const TRAJECTORY_CONCURRENCY = 6;
  const AUTOPLAY_DELAY = 260;
  const SCRUB_DEBOUNCE = 90;

  const KEY_ACTIONS = { ArrowUp: 1, ArrowDown: 2, ArrowLeft: 3, ArrowRight: 4 };
  const STATE_WORDS = {
    NOT_PLAYED: ["Ready", ""],
    NOT_FINISHED: ["Playing", ""],
    WIN: ["Completed", "good"],
    GAME_OVER: ["Game over", "bad"],
  };
  const DIFFICULTY_WORDS = {
    1: "1 — one attribute, sparse walls",
    2: "2 — two attributes, a decoy cycler",
    3: "3 — three attributes, a launcher",
    4: "4 — refills, 2 budget per move",
    5: "5 — two goals, two launchers",
  };
  const CYCLER_WORDS = { shape: "shape", color: "colour", rotation: "rotation" };

  const state = {
    info: null,
    level: null,
    source: null,       // {kind:"generated", seed, difficulty} | {kind:"shipped", index}
    actions: [],
    frame: null,
    status: null,
    revision: 0,
    busy: false,
    trajectory: null,   // {level, cells, frames, statuses}
    trajectoryPending: false,
    autoplay: null,     // identity token for the running solution playback
    overlays: { lattice: false, features: true, route: false },
  };

  let infer = null;
  const el = {};
  const ids = [
    "transport", "model", "level-title", "game-state", "board", "board-overlay",
    "board-summary", "action-count", "ov-lattice", "ov-features", "ov-route",
    "source-fields", "seed", "difficulty", "prev-seed", "generate", "next-seed",
    "random-seed", "shipped", "load-shipped", "source-error", "facts", "features",
    "solution-summary", "scrub", "solution-play", "solution-end", "solution-note",
    "undo", "reset", "carried-swatch", "carried", "goals", "live", "status",
  ];

  function bind() {
    ids.forEach(function (id) {
      el[id] = document.getElementById(id);
    });
  }

  /* ---- transport ---- */

  async function connect() {
    if (window.hostai) {
      await window.hostai.ready();
      infer = async function (input) {
        let first = null;
        const events = await window.hostai.infer(input, {
          onEvent: function (event) { if (first === null) first = event; },
        });
        if (Array.isArray(events) && events.length) return events[0];
        if (first !== null) return first;
        throw new Error("HostAI returned no result.");
      };
      el.transport.textContent = "HostAI bridge";
      el.transport.className = "pill good";
    } else {
      infer = async function (input) {
        const response = await fetch("/hostai/infer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: MODEL, input: input }),
          signal: AbortSignal.timeout(REQUEST_TIMEOUT),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Request failed.");
        return result;
      };
      el.transport.textContent = "HostAI over HTTP";
      el.transport.className = "pill";
    }
  }

  /* ---- helpers ---- */

  function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
  }

  function cellWords(cell) {
    return "col " + cell[0] + ", row " + cell[1];
  }

  // A carried token is (shape, colour, rotation); the colour is an index into
  // the level's four colours, which are themselves ARC palette indices.
  function describeTriple(triple) {
    const info = state.info;
    if (!info || !triple) return { text: "—", hex: "#000000" };
    const arc = info.triple.colors[triple[1]];
    const name = info.palette_names[arc] || ("colour " + triple[1]);
    const rotation = info.triple.rotations[triple[2]];
    return {
      text: "shape " + triple[0] + " · " + name + " · " + rotation + "°",
      hex: info.palette[arc] || "#000000",
    };
  }

  function facts(target, rows) {
    target.replaceChildren();
    rows.forEach(function (row) {
      if (row[1] === null || row[1] === undefined) return;
      const term = document.createElement("dt");
      term.textContent = row[0];
      const value = document.createElement("dd");
      value.textContent = row[1];
      if (row[2]) value.className = row[2];
      target.append(term, value);
    });
  }

  function featureGroup(title, items) {
    if (!items.length) return null;
    const section = document.createElement("div");
    section.className = "feature-group";
    const heading = document.createElement("h3");
    heading.textContent = title + " (" + items.length + ")";
    section.append(heading);
    const list = document.createElement("ul");
    items.forEach(function (item) {
      const entry = document.createElement("li");
      if (item.hex) {
        const chip = document.createElement("span");
        chip.className = "swatch small";
        window.PebbyBoard.swatch(chip, item.hex);
        entry.append(chip);
      }
      const text = document.createElement("span");
      text.textContent = item.text;
      entry.append(text);
      list.append(entry);
    });
    section.append(list);
    return section;
  }

  /* ---- rendering ---- */

  function render() {
    const info = state.info;
    if (!info) return;
    window.PebbyBoard.draw(el.board, state.frame, info.palette);
    window.PebbyBoard.overlay(el["board-overlay"], {
      grid: info.grid,
      level: state.level,
      show: state.overlays,
      cells: state.trajectory ? state.trajectory.cells : null,
      progress: state.actions.length,
    });
    renderTitle();
    renderAnatomy();
    renderSolution();
    renderStatus();
    renderControls();
  }

  function renderTitle() {
    const source = state.source;
    if (!source) {
      el["level-title"].textContent = "No level";
    } else if (source.kind === "shipped") {
      el["level-title"].textContent = "Shipped level " + (source.index + 1);
    } else {
      el["level-title"].textContent = "Generated · seed " + source.seed +
        " · difficulty " + source.difficulty;
    }
    el["action-count"].textContent = state.actions.length +
      (state.actions.length === 1 ? " action" : " actions");
  }

  function renderAnatomy() {
    const level = state.level;
    if (!level) {
      facts(el.facts, []);
      el.features.replaceChildren();
      return;
    }
    const shipped = "shipped" in level;
    const status = state.status;
    const fog = level.fog !== undefined ? level.fog : (status ? status.fog : undefined);
    facts(el.facts, [
      ["Difficulty", shipped ? "—" : level.difficulty],
      ["Seed", shipped ? "—" : level.seed],
      ["Step budget", level.step_counter],
      ["Cost per move", level.step_cost !== undefined ? level.step_cost
        : (status ? status.step_cost : null)],
      ["Fog of war", fog === undefined ? null : (fog ? "yes" : "no")],
      ["Optimal actions", level.optimal_actions === undefined ? null : level.optimal_actions],
      ["Budget slack", level.slack_moves === undefined ? null : level.slack_moves + " moves"],
      ["Reachable states", level.reachable_states === undefined
        ? null : level.reachable_states.toLocaleString()],
      ["Search truncated", level.search_truncated === undefined
        ? null : (level.search_truncated ? "yes" : "no"),
        level.search_truncated ? "warn" : ""],
      ["Walls", level.walls ? level.walls.length : null],
    ]);

    const groups = [];
    if (level.start) {
      const token = describeTriple(level.start_triple);
      groups.push(featureGroup("Start", [
        { text: cellWords(level.start) + " carrying " + token.text, hex: token.hex },
      ]));
    }
    groups.push(featureGroup("Goal pads", (level.goals || []).map(function (goal) {
      const token = describeTriple(goal.triple);
      return { text: cellWords(goal.cell) + " needs " + token.text, hex: token.hex };
    })));
    groups.push(featureGroup("Cyclers", (level.cyclers || []).map(function (item) {
      return { text: cellWords(item.cell) + " · " + (CYCLER_WORDS[item.kind] || item.kind) };
    })));
    groups.push(featureGroup("Launchers", (level.launchers || []).map(function (item) {
      return { text: cellWords(item.cell) + " · flings " + directionWords(item.delta) };
    })));
    groups.push(featureGroup("Refills", (level.refills || []).map(function (cell) {
      return { text: cellWords(cell) };
    })));
    el.features.replaceChildren.apply(el.features, groups.filter(Boolean));
    if (!groups.filter(Boolean).length) {
      const note = document.createElement("p");
      note.className = "muted";
      note.textContent = shipped
        ? "Shipped levels are described by the engine, not by a generated spec."
        : "This level declares no features.";
      el.features.append(note);
    }
  }

  function directionWords(delta) {
    if (!delta) return "—";
    if (delta[0] === 0 && delta[1] < 0) return "up";
    if (delta[0] === 0 && delta[1] > 0) return "down";
    if (delta[1] === 0 && delta[0] < 0) return "left";
    if (delta[1] === 0 && delta[0] > 0) return "right";
    return "(" + delta[0] + ", " + delta[1] + ")";
  }

  function renderSolution() {
    const level = state.level;
    const solution = level && Array.isArray(level.solution) ? level.solution : null;
    if (!solution || !solution.length) {
      el["solution-summary"].textContent = level && "shipped" in level
        ? "Shipped levels carry no stored solution here."
        : "—";
      el.scrub.max = 0;
      el.scrub.value = 0;
      el.scrub.disabled = true;
      el["solution-play"].disabled = true;
      el["solution-end"].disabled = true;
      el["solution-note"].textContent = "";
      return;
    }
    el["solution-summary"].textContent = solution.length +
      " actions, proved optimal and replayed to a win by the generator.";
    el.scrub.max = solution.length;
    el.scrub.disabled = state.busy;
    const onSolution = isSolutionPrefix();
    if (onSolution) el.scrub.value = state.actions.length;
    el["solution-play"].disabled = state.busy;
    el["solution-end"].disabled = state.busy;
    el["solution-play"].textContent = state.autoplay ? "Stop" : "Play solution";
    if (state.trajectoryPending) {
      el["solution-note"].textContent = "Replaying every step…";
    } else if (state.trajectory) {
      el["solution-note"].textContent = "Every step cached — scrubbing is instant.";
    } else if (!onSolution) {
      el["solution-note"].textContent = "Hand moves have diverged from the solution.";
    } else {
      el["solution-note"].textContent = "";
    }
  }

  function isSolutionPrefix() {
    const solution = state.level && state.level.solution;
    if (!solution) return false;
    if (state.actions.length > solution.length) return false;
    return state.actions.every(function (action, index) { return action === solution[index]; });
  }

  function renderStatus() {
    const status = state.status;
    if (!status) {
      el["game-state"].textContent = "—";
      el["game-state"].className = "pill";
      el.carried.textContent = "—";
      el.goals.replaceChildren();
      facts(el.live, []);
      el["board-summary"].textContent = "";
      return;
    }
    const words = STATE_WORDS[status.state] || [status.state, ""];
    el["game-state"].textContent = words[0];
    el["game-state"].className = "pill " + words[1];

    const token = describeTriple(status.triple);
    el.carried.textContent = token.text;
    window.PebbyBoard.swatch(el["carried-swatch"], token.hex);

    el.goals.replaceChildren();
    (status.goal_triples || []).forEach(function (triple, index) {
      const solved = (status.goals_solved || [])[index];
      const entry = document.createElement("li");
      if (solved) entry.className = "solved";
      const chip = document.createElement("span");
      chip.className = "swatch small";
      const described = describeTriple(triple);
      window.PebbyBoard.swatch(chip, described.hex);
      const text = document.createElement("span");
      text.className = "goal-text";
      text.textContent = described.text;
      const badge = document.createElement("span");
      badge.className = "badge " + (solved ? "good" : "");
      badge.textContent = solved ? "cleared" : "open";
      entry.append(chip, text, badge);
      el.goals.append(entry);
    });

    const moves = status.step_cost ? Math.floor(status.steps_left / status.step_cost) : null;
    facts(el.live, [
      ["Steps left", status.steps_left + (moves === null ? "" : " (" + moves + " moves)")],
      ["Lives", "●".repeat(status.lives) + "○".repeat(Math.max(0, 3 - status.lives))],
      ["Player cell", cellWords(status.player_cell)],
      ["Goals cleared", (status.goals_solved || []).filter(Boolean).length + " / " +
        (status.goals_solved || []).length],
    ]);

    const open = (status.goals_solved || []).filter(function (done) { return !done; }).length;
    el["board-summary"].textContent = words[0] + ". Carrying " + token.text + ". " +
      open + " goals open. " + status.steps_left + " steps left, " + status.lives + " lives.";
  }

  function renderControls() {
    const ready = Boolean(state.info) && !state.busy;
    el["source-fields"].disabled = !ready;
    const playable = ready && Boolean(state.level) && state.status && !state.status.finished;
    document.querySelectorAll(".pad").forEach(function (button) {
      button.disabled = !playable;
    });
    el.undo.disabled = !ready || !state.actions.length;
    el.reset.disabled = !ready || !state.actions.length;
    el.board.setAttribute("aria-busy", state.busy ? "true" : "false");
  }

  function fillMenus() {
    const info = state.info;
    el.difficulty.replaceChildren();
    info.difficulties.forEach(function (value) {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = DIFFICULTY_WORDS[value] || String(value);
      if (value === 3) option.selected = true;
      el.difficulty.append(option);
    });
    el.shipped.replaceChildren();
    for (let index = 0; index < info.shipped_levels; index += 1) {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = "Level " + (index + 1);
      el.shipped.append(option);
    }
    el.seed.max = String(info.max_seed);
  }

  /* ---- operations ---- */

  function showError(message) {
    el["source-error"].textContent = message;
    el["source-error"].hidden = false;
  }

  function clearError() {
    el["source-error"].hidden = true;
  }

  async function withBusy(work) {
    state.busy = true;
    renderControls();
    try {
      return await work();
    } finally {
      state.busy = false;
      renderControls();
    }
  }

  async function load(source) {
    stopAutoplay();
    await withBusy(async function () {
      const request = source.kind === "shipped"
        ? { op: "shipped", index: source.index }
        : { op: "generate", seed: source.seed, difficulty: source.difficulty };
      let result;
      try {
        result = await infer(request);
      } catch (error) {
        // A rejected seed must not destroy the level already on screen.
        showError(String(error.message || error));
        return;
      }
      clearError();
      state.revision += 1;
      state.source = source;
      state.level = result.level;
      state.frame = result.frame;
      state.status = result.status;
      state.actions = [];
      state.trajectory = null;
      state.trajectoryPending = false;
      el.scrub.value = 0;
      el.status.textContent = source.kind === "shipped"
        ? "Loaded shipped level " + (source.index + 1) + "."
        : "Generated seed " + source.seed + " at difficulty " + source.difficulty + ".";
    });
    render();
    if (state.overlays.route) ensureTrajectory();
  }

  async function commit(actions) {
    const revision = state.revision;
    const level = state.level;
    if (!level) return;
    // Serve a cached step instantly when the whole solution is already replayed.
    const cached = trajectoryStep(actions);
    if (cached) {
      state.actions = actions;
      state.frame = cached.frame;
      state.status = cached.status;
      render();
      return;
    }
    await withBusy(async function () {
      let result;
      try {
        result = await infer({ op: "play", level: level, actions: actions });
      } catch (error) {
        showError(String(error.message || error));
        return;
      }
      if (revision !== state.revision) return;  // a new level landed meanwhile
      clearError();
      state.actions = actions;
      state.frame = result.frame;
      state.status = result.status;
    });
    render();
  }

  function trajectoryStep(actions) {
    const cache = state.trajectory;
    if (!cache || cache.level !== state.level) return null;
    const solution = state.level.solution || [];
    if (actions.length > solution.length) return null;
    for (let index = 0; index < actions.length; index += 1) {
      if (actions[index] !== solution[index]) return null;
    }
    return { frame: cache.frames[actions.length], status: cache.statuses[actions.length] };
  }

  function move(action) {
    if (state.busy || !state.level || (state.status && state.status.finished)) return;
    commit(state.actions.concat([action]));
  }

  /* Replay every prefix of the stored solution once, so the scrubber becomes
     instant and the route overlay has a cell for each step. The server is
     stateless, so this is the only way to see the path the solution walks. */
  async function ensureTrajectory() {
    const level = state.level;
    const solution = level && level.solution;
    if (!solution || !solution.length) return null;
    if (state.trajectory && state.trajectory.level === level) return state.trajectory;
    if (state.trajectoryPending) return null;
    const revision = state.revision;
    state.trajectoryPending = true;
    renderSolution();
    const indices = [];
    for (let k = 0; k <= solution.length; k += 1) indices.push(k);
    const results = new Array(indices.length);
    let next = 0;
    const workers = [];
    const width = Math.min(TRAJECTORY_CONCURRENCY, indices.length);
    for (let w = 0; w < width; w += 1) {
      workers.push((async function () {
        while (true) {
          const at = next;
          next += 1;
          if (at >= indices.length) return;
          results[at] = await infer({
            op: "play", level: level, actions: solution.slice(0, indices[at]),
          });
        }
      }()));
    }
    try {
      await Promise.all(workers);
    } catch (error) {
      state.trajectoryPending = false;
      showError(String(error.message || error));
      renderSolution();
      return null;
    }
    state.trajectoryPending = false;
    if (revision !== state.revision) return null;
    state.trajectory = {
      level: level,
      frames: results.map(function (result) { return result.frame; }),
      statuses: results.map(function (result) { return result.status; }),
      cells: results.map(function (result) { return result.status.player_cell; }),
    };
    render();
    return state.trajectory;
  }

  function stopAutoplay() {
    state.autoplay = null;
  }

  async function playSolution() {
    if (state.autoplay) {
      stopAutoplay();
      renderSolution();
      return;
    }
    const solution = state.level && state.level.solution;
    if (!solution || !solution.length) return;
    const cache = await ensureTrajectory();
    if (!cache) return;
    const token = {};
    state.autoplay = token;
    renderSolution();
    let step = isSolutionPrefix() ? state.actions.length : 0;
    if (step >= solution.length) step = 0;
    while (state.autoplay === token && step < solution.length) {
      step += 1;
      const cached = cache.frames[step];
      if (!cached) break;
      state.actions = solution.slice(0, step);
      state.frame = cache.frames[step];
      state.status = cache.statuses[step];
      el.scrub.value = step;
      render();
      await new Promise(function (resolve) { setTimeout(resolve, AUTOPLAY_DELAY); });
    }
    if (state.autoplay === token) stopAutoplay();
    renderSolution();
  }

  /* ---- wiring ---- */

  // Generate reports a bad seed instead of clamping it, so a typo is visible
  // rather than quietly answering about a different level.
  function readSeed() {
    const raw = el.seed.value.trim();
    const value = Number(raw);
    if (!raw || !Number.isInteger(value) || value < 0 || value > state.info.max_seed) return null;
    return value;
  }

  // Stepping is a different intent from typing, so it clamps rather than refusing.
  function steppedSeed(delta) {
    const typed = readSeed();
    const base = typed !== null ? typed
      : (state.source && state.source.kind === "generated" ? state.source.seed : 0);
    return clamp(base + delta, 0, state.info.max_seed);
  }

  function currentDifficulty() {
    return Number(el.difficulty.value) || 1;
  }

  function generateAt(seed) {
    el.seed.value = String(seed);
    return load({ kind: "generated", seed: seed, difficulty: currentDifficulty() });
  }

  function wire() {
    el.generate.addEventListener("click", function () {
      const seed = readSeed();
      if (seed === null) {
        showError("Seed must be a whole number between 0 and " + state.info.max_seed + ".");
        return;
      }
      generateAt(seed);
    });
    el["prev-seed"].addEventListener("click", function () { generateAt(steppedSeed(-1)); });
    el["next-seed"].addEventListener("click", function () { generateAt(steppedSeed(1)); });
    el["random-seed"].addEventListener("click", function () {
      generateAt(Math.floor(Math.random() * (state.info.max_seed + 1)));
    });
    el.difficulty.addEventListener("change", function () { generateAt(steppedSeed(0)); });
    el["load-shipped"].addEventListener("click", function () {
      load({ kind: "shipped", index: Number(el.shipped.value) || 0 });
    });

    document.querySelectorAll(".pad").forEach(function (button) {
      button.addEventListener("click", function () {
        move(Number(button.dataset.action));
      });
    });
    el.undo.addEventListener("click", function () {
      if (state.actions.length) commit(state.actions.slice(0, -1));
    });
    el.reset.addEventListener("click", function () {
      stopAutoplay();
      commit([]);
    });

    let scrubTimer = null;
    el.scrub.addEventListener("input", function () {
      stopAutoplay();
      const step = Number(el.scrub.value) || 0;
      const level = state.level;
      const solution = level && level.solution;
      if (!solution) return;
      const actions = solution.slice(0, clamp(step, 0, solution.length));
      if (trajectoryStep(actions)) {
        commit(actions);
        return;
      }
      if (scrubTimer) clearTimeout(scrubTimer);
      scrubTimer = setTimeout(function () {
        if (state.level === level) commit(actions);
      }, SCRUB_DEBOUNCE);
    });
    el["solution-play"].addEventListener("click", playSolution);
    el["solution-end"].addEventListener("click", function () {
      stopAutoplay();
      const solution = state.level && state.level.solution;
      if (solution) {
        el.scrub.value = solution.length;
        commit(solution.slice());
      }
    });

    [["ov-lattice", "lattice"], ["ov-features", "features"], ["ov-route", "route"]]
      .forEach(function (pair) {
        el[pair[0]].addEventListener("change", function () {
          state.overlays[pair[1]] = el[pair[0]].checked;
          render();
          if (pair[1] === "route" && el[pair[0]].checked) ensureTrajectory();
        });
      });

    document.addEventListener("keydown", function (event) {
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      const action = KEY_ACTIONS[event.key];
      if (!action) return;
      const target = event.target;
      if (target && target.closest &&
          target.closest("input, select, textarea, [contenteditable], dialog")) return;
      event.preventDefault();
      stopAutoplay();
      move(action);
    });

    window.addEventListener("resize", function () {
      if (state.info) render();
    });
  }

  /* ---- start ---- */

  async function start() {
    bind();
    wire();
    try {
      await connect();
      state.info = await infer({ op: "info" });
    } catch (error) {
      el.transport.textContent = "Disconnected";
      el.transport.className = "pill bad";
      el.status.textContent = "Could not reach the model: " + (error.message || error);
      return;
    }
    el.model.textContent = MODEL + " · " + state.info.game + " · " + state.info.ruleset;
    fillMenus();
    el["ov-lattice"].checked = state.overlays.lattice;
    el["ov-features"].checked = state.overlays.features;
    el["ov-route"].checked = state.overlays.route;
    await load({ kind: "generated", seed: 7, difficulty: 3 });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
}());
