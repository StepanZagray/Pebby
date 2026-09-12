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
  const PAGE_SIZE = 50;

  const KEY_ACTIONS = { ArrowUp: 1, ArrowDown: 2, ArrowLeft: 3, ArrowRight: 4 };
  const SOURCES = ["bank", "generate", "shipped"];
  // A level's kind names where it came from; a pane is where you ask for one.
  const PANE_OF_KIND = { bank: "bank", generated: "generate", shipped: "shipped" };
  const STATE_WORDS = {
    NOT_PLAYED: ["Ready", ""],
    NOT_FINISHED: ["Playing", ""],
    WIN: ["Completed", "good"],
    GAME_OVER: ["Game over", "bad"],
  };
  const CYCLER_WORDS = { shape: "shape", color: "colour", rotation: "rotation" };

  const state = {
    info: null,
    level: null,
    source: null,       // {kind:"generated", seed, difficulty} | {kind:"shipped", index}
    banks: [],
    bankPage: null,
    actions: [],
    frame: null,
    status: null,
    revision: 0,
    busy: false,
    pending: null,      // which section asked for the request in flight
    trajectory: null,   // {level, cells, frames, statuses}
    trajectoryPending: false,
    autoplay: null,     // identity token for the running solution playback
    overlays: { lattice: false, features: true, route: false },
    bannerOff: false,   // the finish card was dismissed for the run on screen
  };

  let infer = null;
  const el = {};
  const ids = [
    "transport", "model", "level-title", "game-state", "board", "board-overlay",
    "board-summary", "action-count", "ov-lattice", "ov-features", "ov-route",
    "board-wrap", "board-banner", "banner-mark", "banner-title", "banner-sub",
    "banner-primary", "banner-dismiss",
    "level-panel", "level-working", "level-error",
    "src-bank", "src-generate", "src-shipped",
    "pane-bank", "pane-generate", "pane-shipped",
    "gen-fields", "seed", "roll-seed", "difficulty",
    "generate", "generate-label", "prev-seed", "next-seed", "gen-hint",
    "shipped-fields", "shipped", "load-shipped", "load-shipped-label",
    "bank-fields", "bank", "bank-field", "bank-summary",
    "bank-split", "bank-difficulty", "bank-level", "load-bank", "load-bank-label",
    "refresh-bank", "bank-prev", "bank-next", "bank-page",
    "facts", "features",
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
    renderBanner();
    renderControls();
  }

  function renderTitle() {
    const source = state.source;
    if (!source) {
      el["level-title"].textContent = "No level";
    } else if (source.kind === "shipped") {
      el["level-title"].textContent = "Shipped level " + (source.index + 1);
    } else if (source.kind === "bank") {
      el["level-title"].textContent = "Bank · seed " + state.level.seed +
        " · tier " + state.level.difficulty;
    } else {
      el["level-title"].textContent = "Generated · seed " + source.seed +
        " · tier " + source.difficulty;
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
      ["Tier", shipped ? "—" : level.difficulty],
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

  /* What comes after the level on screen, or null when nothing does.

     "Next" means the next one in the source this level came from: the next row
     of the bank listing as it is filtered right now, or the next seed at the
     same tier. The shipped campaign advances inside its own replay — finishing
     level 3 rolls straight into level 4 without a win — so a win there is the
     end of the campaign and there is nothing after it. */
  function nextSource() {
    const source = state.source;
    if (!source || !state.info) return null;
    if (source.kind === "generated") {
      return source.seed >= state.info.max_seed ? null
        : { kind: "generated", seed: source.seed + 1, difficulty: source.difficulty };
    }
    if (source.kind !== "bank") return null;
    const page = state.bankPage;
    if (!page || !page.levels.length) return null;
    const at = page.levels.findIndex(function (row) { return row.id === source.id; });
    if (at >= 0) {
      if (at + 1 < page.levels.length) {
        return { kind: "bank", bank: page.bank, id: page.levels[at + 1].id };
      }
      return page.offset + page.levels.length < page.total
        ? { kind: "bank-page", bank: page.bank, offset: page.offset + page.levels.length }
        : null;
    }
    // No row matches: the listing moved on while this level was being played, a
    // filter changed or a page turned. What is on screen now is the sequence.
    return { kind: "bank", bank: page.bank, id: page.levels[0].id };
  }

  async function goNextLevel() {
    const next = nextSource();
    if (!next || state.busy) return;
    state.bannerOff = true;   // the card has been answered; the next board is the reply
    renderBanner();
    if (next.kind === "bank-page") {
      await browseBank(next.offset, false);
      const page = state.bankPage;
      if (!page || !page.levels.length) return;
      el["bank-level"].value = page.levels[0].id;
      await load({ kind: "bank", bank: page.bank, id: page.levels[0].id });
      return;
    }
    if (next.kind === "bank") {
      el["bank-level"].value = next.id;
      await load(next);
      return;
    }
    el.difficulty.value = String(next.difficulty);
    await generateAt(next.seed);
  }

  /* The finish card. A run ends rarely but matters every time, so the end of it
     is announced over the board itself rather than only in the corner pill. It
     is dismissible: the coloured frame stays behind it as the quiet reminder. */
  function renderBanner() {
    const status = state.status;
    const won = Boolean(status) && status.state === "WIN";
    const over = Boolean(status) && status.state === "GAME_OVER";
    el["board-wrap"].classList.toggle("is-win", won);
    el["board-wrap"].classList.toggle("is-over", over);
    el["board-banner"].classList.toggle("is-over", over);

    if (!won && !over) {
      state.bannerOff = false;  // the next finish is a fresh event worth showing
      el["board-banner"].hidden = true;
      return;
    }
    if (state.bannerOff) {
      el["board-banner"].hidden = true;
      return;
    }

    const source = state.source;
    const named = Boolean(source) && source.kind !== "shipped" && Boolean(state.level);
    const moves = state.actions.length + (state.actions.length === 1 ? " action" : " actions");
    el["banner-mark"].textContent = won ? "\u2713" : "\u2715";
    el["banner-title"].textContent = won ? "Level complete" : "Game over";
    if (won) {
      el["banner-sub"].textContent = "Solved in " + moves +
        (named ? " · seed " + state.level.seed + " · tier " + state.level.difficulty : "");
    } else {
      el["banner-sub"].textContent = (status.lives ? "Out of steps" : "No lives left") +
        " after " + moves + ".";
    }
    el["banner-primary"].textContent = !won ? "Try again"
      : nextSource() ? "Next level" : "Play again";
    el["banner-dismiss"].textContent = won ? "Stay here" : "Dismiss";
    el["board-banner"].hidden = false;
  }

  /* A level takes a round trip to build, so the panel that asked for it says so
     while it works. Only an unusable page dims; a working one stays readable. */
  function renderWork() {
    const pending = state.pending;
    // Playing a move is not level work, so it leaves the panel alone.
    const working = pending === "generate" || pending === "shipped" || pending === "bank";
    el["level-panel"].classList.toggle("is-working", working);
    el["level-panel"].classList.toggle("offline", !state.info);
    el["level-panel"].setAttribute("aria-busy", working ? "true" : "false");
    el["level-working"].textContent = pending === "generate" ? "generating…" : "loading…";
    el["level-working"].hidden = !working;
    el["generate-label"].textContent = pending === "generate" ? "Generating…" : "Generate";
    el["load-shipped-label"].textContent = pending === "shipped" ? "Loading…" : "Load level";
    el["load-bank-label"].textContent = pending === "bank" ? "Loading…" : "Load level";
  }

  /* The three sources differ only in where a level comes from, so they share a
     panel and only the chosen one is on show. */
  function showSource(name) {
    SOURCES.forEach(function (source) {
      el["src-" + source].checked = source === name;
      el["pane-" + source].hidden = source !== name;
    });
  }

  // The seed and difficulty boxes only state an intent; this says so whenever
  // they have drifted from the level actually on the board.
  function renderGenHint() {
    const source = state.source;
    let dirty = false;
    // Mid-request the boxes already describe the level being built, so asking
    // for another press would be wrong.
    if (state.info && state.pending !== "generate" && source && source.kind === "generated") {
      const seed = readSeed();
      dirty = seed !== source.seed || currentDifficulty() !== source.difficulty;
    }
    el["gen-hint"].hidden = !dirty;
    el.generate.classList.toggle("dirty", dirty);
  }

  function renderControls() {
    const ready = Boolean(state.info) && !state.busy;
    el["gen-fields"].disabled = !ready;
    el["shipped-fields"].disabled = !ready;
    el["bank-fields"].disabled = !ready;
    const page = state.bankPage;
    el["load-bank"].disabled = !page || !page.levels.length;
    el["bank-prev"].disabled = !page || page.offset === 0;
    el["bank-next"].disabled = !page || page.offset + page.levels.length >= page.total;
    renderWork();
    renderGenHint();
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
      option.textContent = "Tier " + value;
      if (value === 3) option.selected = true;
      el.difficulty.append(option);
    });
    el.shipped.replaceChildren();
    for (let index = 0; index < info.shipped_levels; index += 1) {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = String(index + 1);
      el.shipped.append(option);
    }
    el.seed.max = String(info.max_seed);
    el["bank-difficulty"].replaceChildren(new Option("All tiers", ""));
    info.difficulties.forEach(function (value) {
      el["bank-difficulty"].append(new Option("Tier " + value, String(value)));
    });
    showSource("bank");
  }

  /* ---- operations ---- */

  // One slot under the panel: only one source is ever on show, so the error is
  // always beneath the control that caused it.
  function showError(message) {
    el["level-error"].textContent = message;
    el["level-error"].hidden = false;
  }

  function clearError() {
    el["level-error"].hidden = true;
  }

  async function withBusy(kind, work) {
    state.busy = true;
    state.pending = kind;
    renderControls();
    try {
      return await work();
    } finally {
      state.busy = false;
      state.pending = null;
      renderControls();
    }
  }

  /* Take a level result onto the board. load() gets one by asking; boot already
     has one in hand, and neither needs to know how the other got there. */
  function adoptLevel(source, result) {
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
      : source.kind === "bank" ? "Loaded accepted " + result.level.split + " level · seed " +
        result.level.seed + " · " + (result.bank_status === "complete" ? "complete bank." : "partial bank.")
        : "Generated seed " + source.seed + " at tier " + source.difficulty + ".";
  }

  async function load(source) {
    stopAutoplay();
    showSource(PANE_OF_KIND[source.kind]);
    await withBusy(source.kind === "generated" ? "generate" : source.kind, async function () {
      const request = source.kind === "shipped"
        ? { op: "shipped", index: source.index }
        : source.kind === "bank" ? { op: "bank_level", bank: source.bank, id: source.id }
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
      adoptLevel(source, result);
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
    await withBusy("play", async function () {
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

  function dismissBanner() {
    state.bannerOff = true;
    renderBanner();
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

  /* Showing a page and fetching one are separate jobs: the opening screen
     arrives with its first page already inside the boot response, so it renders
     without asking for anything. */
  function showBankPage(page) {
    state.bankPage = page;
    el["bank-level"].replaceChildren();
    page.levels.forEach(function (level) {
      el["bank-level"].append(new Option("Seed " + level.seed + " · tier " + level.difficulty +
        " · " + level.optimal_actions + " actions", level.id));
    });
    el["bank-page"].textContent = page.total
      ? (page.offset + 1) + "–" + (page.offset + page.levels.length) + " of " + page.total.toLocaleString() + " accepted levels"
      : "No accepted levels match these filters yet. Refresh as generation progresses.";
  }

  async function fetchBankPage(offset) {
    const bank = el.bank.value;
    state.bankPage = null;
    el["bank-level"].replaceChildren();
    el["bank-page"].textContent = "";
    if (!bank) return;
    const request = { op: "bank_levels", bank: bank, split: el["bank-split"].value,
      offset: offset, limit: PAGE_SIZE };
    if (el["bank-difficulty"].value) request.difficulty = Number(el["bank-difficulty"].value);
    showBankPage(await infer(request));
  }

  function showBanks(banks) {
    const selected = el.bank.value;
    state.banks = banks;
    el.bank.replaceChildren();
    banks.forEach(function (bank) {
      el.bank.append(new Option(bank.label, bank.id, false, bank.id === selected));
    });
    // One bank is the usual case, and a chooser with one entry is noise.
    el["bank-field"].hidden = state.banks.length < 2;
  }

  function describeBank() {
    const bank = state.banks.find(function (item) { return item.id === el.bank.value; });
    el["bank-summary"].textContent = bank
      ? bank.label + " · " +
        (bank.status === "complete" ? "complete" : bank.status === "unavailable" ? "unavailable" : "partial") +
        " · " + (bank.accepted.train + bank.accepted.validation).toLocaleString() + " accepted levels"
      : "No generated banks are available yet.";
  }

  async function browseBank(offset, refresh) {
    if (state.busy) return;
    await withBusy("bank", async function () {
      clearError();
      try {
        if (refresh) showBanks((await infer({ op: "banks" })).banks);
        describeBank();
        await fetchBankPage(offset);
      } catch (error) {
        state.bankPage = null;
        el["bank-level"].replaceChildren();
        el["bank-page"].textContent = "Levels could not be loaded. Try Refresh.";
        showError(String(error.message || error));
      }
    });
  }

  async function loadAcceptedLevel() {
    if (!el["bank-level"].value || state.busy) return;
    await load({ kind: "bank", bank: el.bank.value, id: el["bank-level"].value });
  }

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

  function randomSeed() {
    return Math.floor(Math.random() * (state.info.max_seed + 1));
  }

  function generateAt(seed) {
    el.seed.value = String(seed);
    renderGenHint();
    return load({ kind: "generated", seed: seed, difficulty: currentDifficulty() });
  }

  function wire() {
    SOURCES.forEach(function (name) {
      el["src-" + name].addEventListener("change", function () {
        if (el["src-" + name].checked) showSource(name);
      });
    });
    el["refresh-bank"].addEventListener("click", function () { browseBank(0, true); });
    ["bank", "bank-split", "bank-difficulty"].forEach(function (id) {
      el[id].addEventListener("change", function () { browseBank(0, false); });
    });
    el["load-bank"].addEventListener("click", loadAcceptedLevel);
    el["bank-prev"].addEventListener("click", function () {
      if (state.bankPage) browseBank(Math.max(0, state.bankPage.offset - PAGE_SIZE), false);
    });
    el["bank-next"].addEventListener("click", function () {
      if (state.bankPage) browseBank(state.bankPage.offset + PAGE_SIZE, false);
    });
    // The die only fills the box. Nothing in this panel fetches a level except
    // the buttons under it, so a stray click never replaces what you are reading.
    el["roll-seed"].addEventListener("click", function () {
      if (!state.info) return;
      el.seed.value = String(randomSeed());
      renderGenHint();
      el.seed.focus();
      el.seed.select();
    });
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
    el.seed.addEventListener("input", renderGenHint);
    el.seed.addEventListener("keydown", function (event) {
      if (event.key !== "Enter") return;
      event.preventDefault();
      el.generate.click();
    });
    el.difficulty.addEventListener("change", renderGenHint);
    el["load-shipped"].addEventListener("click", function () {
      load({ kind: "shipped", index: Number(el.shipped.value) || 0 });
    });

    el["banner-primary"].addEventListener("click", function () {
      const status = state.status;
      if (status && status.state === "WIN" && nextSource()) {
        goNextLevel();
        return;
      }
      state.bannerOff = true;
      stopAutoplay();
      commit([]);
    });
    el["banner-dismiss"].addEventListener("click", dismissBanner);
    el["board-banner"].addEventListener("click", function (event) {
      if (event.target === el["board-banner"]) dismissBanner();
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
      const target = event.target;
      const inField = Boolean(target && target.closest &&
        target.closest("input, select, textarea, [contenteditable], dialog"));
      if (!el["board-banner"].hidden) {
        if (event.key === "Escape") {
          dismissBanner();
          return;
        }
        // Enter takes the card's offer, so a finished level is one key from the
        // next one. A focused button answers for itself.
        if (event.key === "Enter" && !inField && (!target || target.tagName !== "BUTTON")) {
          event.preventDefault();
          el["banner-primary"].click();
          return;
        }
      }
      const action = KEY_ACTIONS[event.key];
      if (!action || inField) return;
      event.preventDefault();
      stopAutoplay();
      move(action);
    });

    window.addEventListener("resize", function () {
      if (state.info) render();
    });
  }

  /* ---- start ---- */

  /* One request paints the opening screen.

     This used to be four, each waiting on the one before it: info, then banks,
     then the first page of rows, then that row's level. Only the last two are
     genuinely dependent, and the server can follow that chain itself without
     paying for a round trip between each link. */
  async function start() {
    bind();
    wire();
    let boot;
    try {
      await connect();
      boot = await infer({ op: "boot", limit: PAGE_SIZE });
    } catch (error) {
      el.transport.textContent = "Disconnected";
      el.transport.className = "pill bad";
      el.status.textContent = "Could not reach the model: " + (error.message || error);
      return;
    }
    state.info = boot.info;
    el.model.textContent = MODEL + " · " + state.info.game + " · " + state.info.ruleset;
    fillMenus();
    el["ov-lattice"].checked = state.overlays.lattice;
    el["ov-features"].checked = state.overlays.features;
    el["ov-route"].checked = state.overlays.route;
    showBanks(boot.banks);
    describeBank();
    if (boot.page) showBankPage(boot.page);
    if (!boot.level) {
      // An empty or unreadable bank still deserves something to play.
      await load({ kind: "generated", seed: 7, difficulty: 3 });
      return;
    }
    el["bank-level"].value = boot.level.id;
    adoptLevel({ kind: "bank", bank: boot.page.bank, id: boot.level.id }, boot.level);
    render();
    if (state.overlays.route) ensureTrajectory();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
}());
