/* Rendering for the generated-level viewer.

   Two layers. `draw` blits the engine's 64x64 frame of palette indices, one
   game pixel per ImageData quadruple, and CSS scales it with nearest-neighbour
   so nothing is resampled. `overlay` annotates that frame on a second canvas
   using the level's own logical 12x12 lattice, which the frame does not encode
   in any way a reader can pick out by eye.

   Everything `overlay` draws is in frame-pixel units (0..64) and scaled once by
   the context, so a line width is a fraction of a game pixel at any zoom. */
(function () {
  "use strict";

  const MISSING = [255, 0, 255];  // absent from LS20 output, so a bad index shows

  // Markers are stroked twice, a dark halo under a bright accent, because they
  // sit on top of arbitrary palette colours and must read against all of them.
  const HALO = "rgba(0, 0, 0, 0.78)";
  const ACCENT = {
    start: "#ffffff",
    goal: "#f472b6",
    cycler: "#c4b5fd",
    launcher: "#fbbf24",
    refill: "#34d399",
    route: "#38bdf8",
    routeDone: "#fde047",
  };

  let cachedPalette = null;
  let cachedTable = null;

  function table(palette) {
    if (palette === cachedPalette) return cachedTable;
    cachedTable = palette.map(function (hex) {
      return [
        parseInt(hex.slice(1, 3), 16),
        parseInt(hex.slice(3, 5), 16),
        parseInt(hex.slice(5, 7), 16),
      ];
    });
    cachedPalette = palette;
    return cachedTable;
  }

  function draw(canvas, frame, palette) {
    const context = canvas.getContext("2d", { alpha: false });
    if (!frame || !frame.length || !frame[0] || !frame[0].length) {
      context.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const rows = frame.length;
    const cols = frame[0].length;
    if (canvas.width !== cols || canvas.height !== rows) {
      canvas.width = cols;
      canvas.height = rows;
    }
    const rgb = table(palette);
    const image = context.createImageData(cols, rows);
    const data = image.data;
    let at = 0;
    for (let r = 0; r < rows; r += 1) {
      const row = frame[r];
      for (let c = 0; c < cols; c += 1) {
        const colour = rgb[row[c]] || MISSING;
        data[at] = colour[0];
        data[at + 1] = colour[1];
        data[at + 2] = colour[2];
        data[at + 3] = 255;
        at += 4;
      }
    }
    context.putImageData(image, 0, 0);
  }

  function swatch(element, hex) {
    element.style.background = hex;
  }

  /* ---- overlay ---- */

  function centre(grid, cell) {
    return [
      grid.x_origin + grid.cell * cell[0] + grid.cell / 2,
      grid.y_origin + grid.cell * cell[1] + grid.cell / 2,
    ];
  }

  function stroked(context, accent, width, path) {
    context.beginPath();
    path(context);
    context.lineWidth = width + 0.32;
    context.strokeStyle = HALO;
    context.stroke();
    context.lineWidth = width;
    context.strokeStyle = accent;
    context.stroke();
  }

  function ring(context, x, y, radius) {
    stroked(context, ACCENT.start, 0.28, function (c) {
      c.arc(x, y, radius, 0, Math.PI * 2);
    });
  }

  function diamond(context, x, y, reach) {
    stroked(context, ACCENT.goal, 0.3, function (c) {
      c.moveTo(x, y - reach);
      c.lineTo(x + reach, y);
      c.lineTo(x, y + reach);
      c.lineTo(x - reach, y);
      c.closePath();
    });
  }

  // The three cycler kinds get three silhouettes rather than three colours,
  // because a colour-coded cycler competes with the palette underneath it.
  function cycler(context, x, y, kind, reach) {
    stroked(context, ACCENT.cycler, 0.28, function (c) {
      if (kind === "shape") {
        c.moveTo(x, y - reach);
        c.lineTo(x + reach, y + reach * 0.8);
        c.lineTo(x - reach, y + reach * 0.8);
        c.closePath();
      } else if (kind === "color") {
        c.arc(x, y, reach * 0.92, 0, Math.PI * 2);
      } else {
        c.arc(x, y, reach * 0.92, -Math.PI * 0.6, Math.PI * 1.1);
      }
    });
    if (kind === "rotation") {
      // an arrowhead, so the open arc is not mistaken for a broken circle
      const tip = [x + reach * 0.92 * Math.cos(Math.PI * 1.1), y + reach * 0.92 * Math.sin(Math.PI * 1.1)];
      stroked(context, ACCENT.cycler, 0.26, function (c) {
        c.moveTo(tip[0] - reach * 0.55, tip[1] - reach * 0.1);
        c.lineTo(tip[0], tip[1]);
        c.lineTo(tip[0] + reach * 0.15, tip[1] - reach * 0.6);
      });
    }
  }

  function launcher(context, x, y, delta, reach) {
    const length = Math.hypot(delta[0], delta[1]) || 1;
    const ux = delta[0] / length;
    const uy = delta[1] / length;
    const px = -uy;
    const py = ux;
    stroked(context, ACCENT.launcher, 0.3, function (c) {
      c.moveTo(x - ux * reach, y - uy * reach);
      c.lineTo(x + ux * reach, y + uy * reach);
      c.moveTo(x + ux * reach * 0.1 + px * reach * 0.72, y + uy * reach * 0.1 + py * reach * 0.72);
      c.lineTo(x + ux * reach, y + uy * reach);
      c.lineTo(x + ux * reach * 0.1 - px * reach * 0.72, y + uy * reach * 0.1 - py * reach * 0.72);
    });
  }

  function refill(context, x, y, reach) {
    stroked(context, ACCENT.refill, 0.3, function (c) {
      c.moveTo(x - reach, y);
      c.lineTo(x + reach, y);
      c.moveTo(x, y - reach);
      c.lineTo(x, y + reach);
    });
  }

  function lattice(context, grid) {
    const left = grid.x_origin;
    const top = grid.y_origin;
    const right = left + grid.cell * grid.cols;
    const bottom = top + grid.cell * grid.rows;
    context.beginPath();
    for (let c = 0; c <= grid.cols; c += 1) {
      const x = left + grid.cell * c;
      context.moveTo(x, top);
      context.lineTo(x, bottom);
    }
    for (let r = 0; r <= grid.rows; r += 1) {
      const y = top + grid.cell * r;
      context.moveTo(left, y);
      context.lineTo(right, y);
    }
    context.lineWidth = 0.12;
    context.strokeStyle = "rgba(255, 255, 255, 0.26)";
    context.stroke();
    context.beginPath();
    context.rect(left, top, right - left, bottom - top);
    context.lineWidth = 0.22;
    context.strokeStyle = "rgba(255, 255, 255, 0.5)";
    context.stroke();
  }

  function route(context, grid, cells, progress) {
    if (!cells || cells.length < 2) return;
    const points = cells.map(function (cell) { return centre(grid, cell); });
    const walked = Math.max(0, Math.min(progress, points.length - 1));
    const line = function (from, to, accent, width, alpha) {
      if (to <= from) return;
      context.save();
      context.globalAlpha = alpha;
      stroked(context, accent, width, function (c) {
        c.moveTo(points[from][0], points[from][1]);
        for (let i = from + 1; i <= to; i += 1) c.lineTo(points[i][0], points[i][1]);
      });
      context.restore();
    };
    line(walked, points.length - 1, ACCENT.route, 0.4, 0.62);
    line(0, walked, ACCENT.routeDone, 0.46, 0.95);
    // where the scrubber currently sits
    const head = points[walked];
    stroked(context, ACCENT.routeDone, 0.26, function (c) {
      c.arc(head[0], head[1], 0.85, 0, Math.PI * 2);
    });
  }

  /* `plan` is { grid, level, show:{lattice,features,route}, cells, progress }. */
  function overlay(canvas, plan) {
    const ratio = window.devicePixelRatio || 1;
    const size = Math.max(1, Math.round((canvas.clientWidth || 0) * ratio));
    if (canvas.width !== size || canvas.height !== size) {
      canvas.width = size;
      canvas.height = size;
    }
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!plan || !plan.grid || !plan.level || !size) return;
    const grid = plan.grid;
    const level = plan.level;
    const show = plan.show || {};

    context.save();
    context.scale(size / grid.frame_size, size / grid.frame_size);
    context.lineJoin = "round";
    context.lineCap = "round";

    if (show.lattice) lattice(context, grid);

    if (show.features) {
      const reach = grid.cell * 0.34;
      (level.refills || []).forEach(function (cell) {
        const point = centre(grid, cell);
        refill(context, point[0], point[1], reach);
      });
      (level.launchers || []).forEach(function (item) {
        const point = centre(grid, item.cell);
        launcher(context, point[0], point[1], item.delta, reach * 1.12);
      });
      (level.cyclers || []).forEach(function (item) {
        const point = centre(grid, item.cell);
        cycler(context, point[0], point[1], item.kind, reach);
      });
      (level.goals || []).forEach(function (item) {
        const point = centre(grid, item.cell);
        diamond(context, point[0], point[1], reach * 1.2);
      });
      if (level.start) {
        const point = centre(grid, level.start);
        ring(context, point[0], point[1], reach * 1.18);
      }
    }

    if (show.route) route(context, grid, plan.cells, plan.progress || 0);

    context.restore();
  }

  window.PebbyBoard = { draw: draw, swatch: swatch, overlay: overlay, ACCENT: ACCENT };
}());
