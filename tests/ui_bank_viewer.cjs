// Requires a dedicated browser on a verified private test display.
const { chromium } = require(process.env.PEBBY_PLAYWRIGHT || 'playwright-core');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

(async () => {
  assert(process.env.PEBBY_TEST_CDP && process.env.PEBBY_TEST_ORIGIN);
  const out = process.env.PEBBY_SCREENSHOT_DIR;
  assert(out, 'An evidence directory is required.');
  await fs.mkdir(out, { recursive: true });
  const browser = await chromium.connectOverCDP(process.env.PEBBY_TEST_CDP);
  const page = await browser.contexts()[0].newPage();
  const errors = [];
  const requested = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => {
    if (request.url().endsWith('/hostai/infer')) requested.push(request.postDataJSON().input.op);
  });
  const text = id => page.locator('#' + id).textContent();
  const settled = () => page.waitForFunction(() =>
    !document.getElementById('bank-fields').disabled, null, { timeout: 120000 });
  const ask = input => page.evaluate(async body => {
    const response = await fetch('/hostai/infer', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'pebby:latest', input: body }),
    });
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  }, input);
  const screenshot = async name => {
    await page.evaluate(() => {
      window.scrollTo(0, 0);
      document.querySelector('.rail').scrollTop = 0;
    });
    await page.screenshot({ path: path.join(out, name), fullPage: true });
  };
  const played = [];
  try {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto(process.env.PEBBY_TEST_ORIGIN + '/ui/index.html');
    await settled();
    await page.waitForFunction(() => document.getElementById('level-title').textContent.startsWith('Bank ·'));
    assert.match(await text('bank-summary'), /accepted levels/);
    assert.equal(await page.locator('#bank-difficulty option').count(), 8);
    assert.equal(await text('action-count'), '0 actions');
    await screenshot('bank-desktop.png');

    // Pagination and filters select stored rows without changing the board.
    const initialTitle = await text('level-title');
    await page.locator('#bank-next').click();
    await settled();
    assert.match(await text('bank-page'), /^51–100 of /);
    assert.equal(await text('level-title'), initialTitle);
    await page.locator('#bank-prev').click();
    await settled();
    assert.match(await text('bank-page'), /^1–50 of /);

    // Play one accepted row from each tier and split using its exact stored route.
    for (const split of ['train', 'validation']) {
      await page.locator('#bank-split').selectOption(split);
      await settled();
      for (let tier = 1; tier <= 7; tier++) {
        await page.locator('#bank-difficulty').selectOption(String(tier));
        await settled();
        const id = await page.locator('#bank-level').inputValue();
        assert(id, `An accepted ${split} tier ${tier} row must be available.`);
        const bank = await page.evaluate(() => document.getElementById('bank').value);
        const accepted = await ask({ op: 'bank_level', bank, id });
        assert.equal(accepted.level.difficulty, tier);
        assert.equal(accepted.level.split, split);
        await page.locator('#load-bank').focus();
        await page.keyboard.press('Enter');
        await settled();
        assert.equal(await text('level-title'), `Bank · seed ${accepted.level.seed} · tier ${tier}`);
        assert.equal(await text('action-count'), '0 actions');
        const factRows = await page.locator('#facts').evaluate(list => {
          const entries = [...list.children];
          return Object.fromEntries(entries.filter((_, i) => i % 2 === 0)
            .map((entry, i) => [entry.textContent, entries[i * 2 + 1].textContent]));
        });
        assert.equal(factRows.Seed, String(accepted.level.seed));
        assert.equal(factRows.Tier, String(tier));
        assert.equal(factRows['Optimal actions'], String(accepted.level.optimal_actions));
        await page.locator('#solution-end').click();
        await settled();
        await page.waitForFunction(() => document.getElementById('game-state').textContent === 'Completed');
        assert.equal(await text('action-count'), `${accepted.level.solution.length} actions`);
        assert.equal(await text('banner-primary'), 'Next level');
        await page.locator('#banner-dismiss').click();
        assert.equal(await page.locator('#board-banner').isHidden(), true);
        played.push({ split, tier, id, seed: accepted.level.seed, actions: accepted.level.solution.length });
      }
    }
    await screenshot('bank-tier7.png');

    // A delayed bank request must report work while preserving the current board.
    const currentTitle = await text('level-title');
    await page.route('**/hostai/infer', async route => {
      if (route.request().postDataJSON().input.op === 'bank_levels')
        await new Promise(resolve => setTimeout(resolve, 650));
      await route.continue();
    });
    await page.locator('#refresh-bank').click();
    await page.waitForFunction(() => document.getElementById('level-panel').getAttribute('aria-busy') === 'true');
    assert.equal(await text('level-title'), currentTitle);
    await screenshot('bank-loading.png');
    await settled();
    await page.unroute('**/hostai/infer');

    // A failed catalogue or an empty filtered page cannot erase the current game.
    await page.route('**/hostai/infer', async route => {
      if (route.request().postDataJSON().input.op === 'bank_levels')
        return route.fulfill({ status: 400, contentType: 'application/json',
          body: JSON.stringify({ error: 'Accepted row failed integrity validation.' }) });
      return route.continue();
    });
    await page.locator('#refresh-bank').click();
    await settled();
    assert.match(await text('level-error'), /integrity validation/);
    assert.equal(await text('level-title'), currentTitle);
    assert(await page.locator('#load-bank').isDisabled());
    await screenshot('bank-error.png');
    await page.unroute('**/hostai/infer');
    await page.route('**/hostai/infer', async route => {
      if (route.request().postDataJSON().input.op === 'bank_levels')
        return route.fulfill({ status: 200, contentType: 'application/json',
          body: JSON.stringify({ bank: 'reference-unequal-v1', bank_status: 'partial', total: 0, offset: 0, levels: [] }) });
      return route.continue();
    });
    await page.locator('#refresh-bank').click();
    await settled();
    assert.match(await text('bank-page'), /No accepted levels match/);
    assert(await page.locator('#load-bank').isDisabled());
    assert.equal(await text('level-title'), currentTitle);
    await screenshot('bank-empty.png');
    await page.unroute('**/hostai/infer');
    await page.locator('#refresh-bank').click();
    await settled();
    assert(await page.locator('#level-error').isHidden());
    assert(!(await page.locator('#load-bank').isDisabled()));

    // The finish card walks the listing: a completed bank level offers the row
    // after it, and taking the offer keeps the panel's own selection in step.
    await page.locator('#bank-difficulty').selectOption('1');
    await settled();
    const rows = await page.locator('#bank-level option').evaluateAll(list => list.map(o => o.value));
    assert(rows.length > 1, 'the listing needs a row after the first one to advance to');
    await page.locator('#load-bank').click();
    await settled();
    await page.locator('#solution-end').click();
    await settled();
    await page.waitForFunction(() => document.getElementById('game-state').textContent === 'Completed');
    assert.equal(await text('banner-primary'), 'Next level');
    await page.locator('#banner-primary').click();
    await settled();
    assert.equal(await page.evaluate(() => document.getElementById('bank-level').value), rows[1],
      'Next level must load the next accepted row and select it in the panel');
    assert.equal(await text('action-count'), '0 actions');
    assert.equal(await page.locator('#board-banner').isHidden(), true);

    // At the end of a page the card turns it rather than stopping there.
    await page.locator('#bank-level').selectOption(rows[rows.length - 1]);
    await page.locator('#load-bank').click();
    await settled();
    await page.locator('#solution-end').click();
    await settled();
    await page.waitForFunction(() => document.getElementById('game-state').textContent === 'Completed');
    assert.equal(await text('banner-primary'), 'Next level');
    await page.locator('#banner-primary').click();
    await settled();
    assert.match(await text('bank-page'), /^51–100 of /,
      'the last row of a page must lead to the next page');
    assert.equal(await page.evaluate(() => document.getElementById('bank-level').selectedIndex), 0,
      'and land on the first row of the page it turned to');
    assert.equal(await text('action-count'), '0 actions');

    for (const width of [320, 768, 1024, 1440]) {
      await page.setViewportSize({ width, height: 1000 });
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
      assert(!overflow, `Horizontal overflow at ${width}px`);
      await screenshot(`bank-width-${width}.png`);
    }
    assert(!requested.includes('generate'), 'Browsing accepted levels must never regenerate a level.');
    assert.deepEqual(errors, []);
    assert(process.env.PEBBY_TEST_GRIM && process.env.PEBBY_TEST_XDG_RUNTIME_DIR &&
      process.env.PEBBY_TEST_WAYLAND_DISPLAY, 'Private compositor capture settings are required.');
    execFileSync(process.env.PEBBY_TEST_GRIM, ['-o', 'HEADLESS-1', path.join(out, 'compositor.png')], {
      env: { PATH: '/usr/bin:/bin', XDG_RUNTIME_DIR: process.env.PEBBY_TEST_XDG_RUNTIME_DIR,
        WAYLAND_DISPLAY: process.env.PEBBY_TEST_WAYLAND_DISPLAY }, timeout: 10000,
    });
    await fs.writeFile(path.join(out, 'browser-results.json'), JSON.stringify({ status: 'passed', played,
      pageErrors: errors, requests: requested.length, regenerationRequests: 0,
      testedWidths: [320, 768, 1024, 1440], loadingErrorEmptyRecovery: true }, null, 2) + '\n');
    console.log(JSON.stringify({ status: 'passed', played: played.length, regenerationRequests: 0 }));
  } finally {
    await page.close();
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
