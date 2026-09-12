// Browser regression for the generated-level viewer.
//
// Run only against a dedicated browser on a verified private test display.
// Requires playwright-core plus explicit PEBBY_TEST_CDP and PEBBY_TEST_ORIGIN;
// never point these at a normal desktop browser's debugging endpoint.
const { chromium } = require(process.env.PEBBY_PLAYWRIGHT || 'playwright-core');
const assert = require('node:assert/strict');
const { mkdir } = require('node:fs/promises');
const path = require('node:path');

(async () => {
  assert(process.env.PEBBY_TEST_CDP && process.env.PEBBY_TEST_ORIGIN,
    'Set PEBBY_TEST_CDP and PEBBY_TEST_ORIGIN for an isolated test browser and server.');
  const out = process.env.PEBBY_SCREENSHOT_DIR;
  const browser = await chromium.connectOverCDP(process.env.PEBBY_TEST_CDP);
  let failed = false;
  try {
    const context = browser.contexts()[0];
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const text = selector => page.locator(selector).textContent();
    const count = () => page.locator('#action-count').textContent();
    const shot = async name => {
      if (!out) return;
      await mkdir(out, { recursive: true });
      await page.evaluate(() => { window.scrollTo(0, 0); document.querySelector('.rail').scrollTop = 0; });
      await page.waitForTimeout(120);
      await page.screenshot({ path: path.join(out, name) });
    };
    // The source fieldset is enabled exactly when no replay is in flight, so it
    // is the honest "the UI is idle again" signal.
    const settled = () => page.waitForFunction(
      () => !document.getElementById('source-fields').disabled, null, { timeout: 120000 });
    const actionsAre = n => page.waitForFunction(
      want => document.getElementById('action-count').textContent === want,
      `${n} action${n === 1 ? '' : 's'}`, { timeout: 60000 });
    // Ask the server directly, so assertions are checked against the engine
    // rather than against the page's own idea of what happened.
    const ask = input => page.evaluate(async body => {
      const response = await fetch('/hostai/infer', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'pebby:latest', input: body }),
      });
      return response.json();
    }, input);

    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto(process.env.PEBBY_TEST_ORIGIN + '/ui/index.html');
    await settled();

    // --- startup -------------------------------------------------------
    assert.match(await text('#transport'), /HostAI/, 'the page must report its HostAI transport');
    assert.match(await text('#model'), /pebby:latest · ls20/);
    assert.equal(await text('#level-title'), 'Generated · seed 7 · difficulty 3');
    assert.equal(await count(), '0 actions');
    assert.equal(await text('#game-state'), 'Ready');
    assert.equal(await page.locator('#board').evaluate(c => `${c.width}x${c.height}`), '64x64');

    // --- the anatomy panel reports the spec the server actually sent -----
    const generated = await ask({ op: 'generate', seed: 7, difficulty: 3 });
    const level = generated.level;
    const facts = await page.locator('#facts').evaluate(list => {
      const rows = {};
      const nodes = [...list.children];
      for (let i = 0; i < nodes.length; i += 2) rows[nodes[i].textContent] = nodes[i + 1].textContent;
      return rows;
    });
    assert.equal(facts['Seed'], String(level.seed));
    assert.equal(facts['Difficulty'], String(level.difficulty));
    assert.equal(facts['Optimal actions'], String(level.optimal_actions));
    assert.equal(facts['Walls'], String(level.walls.length));
    assert.equal(facts['Step budget'], String(level.step_counter));
    assert.equal(facts['Fog of war'], level.fog ? 'yes' : 'no');
    const groups = await page.locator('.feature-group h3').allTextContents();
    assert.ok(groups.some(g => g === `Goal pads (${level.goals.length})`),
      `goal count must be shown, saw ${JSON.stringify(groups)}`);
    if (level.cyclers.length) assert.ok(groups.some(g => g === `Cyclers (${level.cyclers.length})`));
    if (level.launchers.length) assert.ok(groups.some(g => g === `Launchers (${level.launchers.length})`));

    // --- overlays actually draw -----------------------------------------
    const overlayInk = () => page.locator('#board-overlay').evaluate(canvas => {
      const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      let lit = 0;
      for (let i = 3; i < data.length; i += 4) if (data[i] > 8) lit += 1;
      return lit;
    });
    const withFeatures = await overlayInk();
    assert.ok(withFeatures > 0, 'the features overlay must put ink on the overlay canvas');
    await page.locator('#ov-features').uncheck();
    assert.equal(await overlayInk(), 0, 'unchecking features must clear the overlay');
    await page.locator('#ov-lattice').check();
    const withLattice = await overlayInk();
    assert.ok(withLattice > 0, 'the lattice overlay must draw the 12x12 grid');
    await page.locator('#ov-features').check();
    assert.ok(await overlayInk() > withLattice, 'features add ink on top of the lattice');
    await page.locator('#ov-lattice').uncheck();

    // --- seed stepping ---------------------------------------------------
    await page.locator('#next-seed').click();
    await settled();
    assert.equal(await page.locator('#seed').inputValue(), '8');
    assert.equal(await text('#level-title'), 'Generated · seed 8 · difficulty 3');
    await page.locator('#prev-seed').click();
    await settled();
    assert.equal(await text('#level-title'), 'Generated · seed 7 · difficulty 3');
    assert.equal(await count(), '0 actions', 'a new level starts with an empty history');

    // --- hand play, undo as an exact prefix replay ------------------------
    await page.keyboard.press('ArrowLeft');
    await actionsAre(1);
    await settled();
    await page.locator('.pad.up').click();
    await actionsAre(2);
    await settled();
    await page.keyboard.press('ArrowRight');
    await actionsAre(3);
    await settled();
    const afterTwo = await ask({ op: 'play', level, actions: [3, 1] });
    await page.locator('#undo').click();
    await actionsAre(2);
    await settled();
    const shownCell = await page.locator('#live').evaluate(list => {
      const nodes = [...list.children];
      for (let i = 0; i < nodes.length; i += 2) {
        if (nodes[i].textContent === 'Player cell') return nodes[i + 1].textContent;
      }
      return null;
    });
    assert.equal(shownCell, `col ${afterTwo.status.player_cell[0]}, row ${afterTwo.status.player_cell[1]}`,
      'undo must replay the prefix exactly, not guess at an inverse move');
    await page.locator('#reset').click();
    await actionsAre(0);
    await settled();
    assert.equal(await page.locator('#undo').isDisabled(), true);

    // --- native fields keep their own arrow keys -------------------------
    await page.locator('#seed').focus();
    const seed = Number(await page.locator('#seed').inputValue());
    await page.keyboard.press('ArrowUp');
    assert.equal(Number(await page.locator('#seed').inputValue()), seed + 1);
    assert.equal(await count(), '0 actions', 'arrow keys inside a field must not move the player');
    await page.locator('#seed').fill(String(seed));

    // --- the stored solution completes the level -------------------------
    assert.match(await text('#solution-summary'), new RegExp(`^${level.solution.length} actions`));
    await page.locator('#solution-end').click();
    await actionsAre(level.solution.length);
    await settled();
    assert.equal(await text('#game-state'), 'Completed',
      "the generator's stored solution must finish the level it was proved on");
    await shot('viewer-solved.png');

    // --- the route overlay replays every step ----------------------------
    await page.locator('#ov-route').check();
    await page.waitForFunction(
      () => document.getElementById('solution-note').textContent.includes('cached'),
      null, { timeout: 120000 });
    assert.ok(await overlayInk() > 0, 'the route overlay must draw a path');

    // A cached scrub must agree with the engine, not just look plausible.
    const half = Math.floor(level.solution.length / 2);
    await page.locator('#scrub').fill(String(half));
    await page.locator('#scrub').dispatchEvent('input');
    await actionsAre(half);
    await settled();
    const halfway = await ask({ op: 'play', level, actions: level.solution.slice(0, half) });
    const scrubbedCell = await page.locator('#live').evaluate(list => {
      const nodes = [...list.children];
      for (let i = 0; i < nodes.length; i += 2) {
        if (nodes[i].textContent === 'Player cell') return nodes[i + 1].textContent;
      }
      return null;
    });
    assert.equal(scrubbedCell, `col ${halfway.status.player_cell[0]}, row ${halfway.status.player_cell[1]}`,
      'a cached scrub step must match the engine replaying the same prefix');
    await shot('viewer-route.png');
    await page.emulateMedia({ colorScheme: 'dark' });
    await page.waitForTimeout(200);
    await shot('viewer-dark.png');
    await page.emulateMedia({ colorScheme: 'light' });
    await page.waitForTimeout(200);

    // --- shipped levels load, and say plainly that they carry no solution --
    await page.locator('#shipped').selectOption('5');
    await page.locator('#load-shipped').click();
    await settled();
    assert.equal(await text('#level-title'), 'Shipped level 6');
    assert.match(await text('#solution-summary'), /no stored solution/);
    assert.equal(await page.locator('#solution-end').isDisabled(), true);
    assert.equal(await page.locator('#scrub').isDisabled(), true);

    // --- a rejected seed reports itself without destroying the level ------
    await page.locator('#seed').fill('-4');
    await page.locator('#generate').click();
    await page.waitForFunction(() => !document.getElementById('source-error').hidden,
      null, { timeout: 30000 });
    await settled();
    assert.equal(await text('#level-title'), 'Shipped level 6', 'a bad seed must not clear the level');
    await page.locator('#seed').fill('11');
    await page.locator('#generate').click();
    await settled();
    assert.equal(await page.locator('#source-error').isHidden(), true, 'a working request retires the error');
    assert.equal(await text('#level-title'), 'Generated · seed 11 · difficulty 3');
    await shot('viewer-desktop.png');

    // --- layout ----------------------------------------------------------
    const fits = await page.evaluate(() => {
      const rect = document.querySelector('.board-wrap').getBoundingClientRect();
      return rect.bottom <= window.innerHeight + 1 && Math.abs(rect.width - rect.height) < 2;
    });
    assert.ok(fits, 'the board must stay square and fit 1280x800');
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(300);
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert.ok(overflow <= 1, `no horizontal overflow at 390px, saw ${overflow}`);
    const pad = await page.locator('.pad.up').boundingBox();
    assert.ok(pad.height >= 36, `the d-pad must stay tappable at 390px, saw ${pad.height}`);
    await shot('viewer-mobile.png');

    assert.deepEqual(errors, [], 'the page must raise no uncaught errors');

    // --- the injected-bridge path -----------------------------------------
    // Everything above rode the standalone fallback. HostAI instead injects
    // window.hostai, and the page must prefer it. This is a stand-in bridge,
    // not a gateway: it proves the seam in app.js is wired and used, and says
    // nothing about a real HostAI host.
    const bridged = await context.newPage();
    const bridgeErrors = [];
    bridged.on('pageerror', error => bridgeErrors.push(error.message));
    await bridged.addInitScript(() => {
      window.__bridgeCalls = 0;
      window.__bridgeReady = false;
      window.hostai = {
        ready: async () => { window.__bridgeReady = true; },
        infer: async (input, options) => {
          window.__bridgeCalls += 1;
          const response = await fetch('/hostai/infer', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: 'pebby:latest', input }),
          });
          const result = await response.json();
          if (!response.ok) throw new Error(result.error || 'Request failed.');
          if (options && options.onEvent) options.onEvent(result);
          return [result];
        },
      };
    });
    await bridged.setViewportSize({ width: 1280, height: 800 });
    await bridged.goto(process.env.PEBBY_TEST_ORIGIN + '/ui/index.html');
    await bridged.waitForFunction(
      () => !document.getElementById('source-fields').disabled, null, { timeout: 120000 });
    assert.equal(await bridged.locator('#transport').textContent(), 'HostAI bridge',
      'the page must prefer the injected bridge over its own fetch');
    assert.equal(await bridged.evaluate(() => window.__bridgeReady), true,
      'the page must await hostai.ready() before inferring');
    assert.ok(await bridged.evaluate(() => window.__bridgeCalls) >= 2,
      'info and the first level must both travel over the bridge');
    assert.equal(await bridged.locator('#level-title').textContent(),
      'Generated · seed 7 · difficulty 3');
    await bridged.keyboard.press('ArrowLeft');
    await bridged.waitForFunction(
      () => document.getElementById('action-count').textContent === '1 action', null, { timeout: 30000 });
    assert.deepEqual(bridgeErrors, [], 'the bridged page must raise no uncaught errors');
    await bridged.close();

    console.log('ui_viewer.cjs: OK');
  } catch (error) {
    failed = true;
    console.error('ui_viewer.cjs: FAIL\n' + (error && error.stack || error));
  } finally {
    await browser.close().catch(() => {});
  }
  process.exit(failed ? 1 : 0);
})();
