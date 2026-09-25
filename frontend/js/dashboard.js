/**
 * TrafficEye – dashboard page.
 *
 * Loads GET /api/stats and renders:
 *   - four stat tiles (total, no helmet, triple riding, total fines)
 *   - a doughnut of violations by type, with a legend that shows every value
 *   - a two-line daily trend for the last 7 days (plus a table view)
 *   - the five most recent violations
 *
 * Chart choices follow one rule set: each violation type keeps the same
 * validated colour everywhere, text is always in ink colours (never the series
 * colour), lines are 2px with 8px points, and every value is also readable
 * without hovering (legend numbers, end labels, table view).
 */
(function () {
  "use strict";

  const TYPES = [
    { key: "NO_HELMET", field: "no_helmet", label: "No Helmet" },
    { key: "TRIPLE_RIDING", field: "triple_riding", label: "Triple Riding" },
  ];
  const AUTO_REFRESH_MS = 60000;

  const $ = (id) => document.getElementById(id);
  const els = {
    content: $("dashboard-content"),
    empty: $("dashboard-empty"),
    error: $("dashboard-error"),
    errorText: $("dashboard-error-text"),
    retry: $("retry-btn"),
    refresh: $("refresh-btn"),
    updatedAt: $("updated-at"),
    statTotal: $("stat-total"),
    statTotalSub: $("stat-total-sub"),
    statNoHelmet: $("stat-no-helmet"),
    statNoHelmetSub: $("stat-no-helmet-sub"),
    statTriple: $("stat-triple"),
    statTripleSub: $("stat-triple-sub"),
    statFines: $("stat-fines"),
    statFinesSub: $("stat-fines-sub"),
    typeCanvas: $("type-chart"),
    typeTotal: $("type-total"),
    typeLegend: $("type-legend"),
    trendCanvas: $("trend-chart"),
    trendLegend: $("trend-legend"),
    trendTable: $("trend-table"),
    recentList: $("recent-list"),
  };

  const state = { typeChart: null, trendChart: null, trend: [], loading: false, loadedOnce: false };

  // Colours come from the CSS custom properties so there is one source of truth.
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const COLORS = {
    surface: css("--viz-surface") || "#0f172a",
    grid: css("--viz-grid") || "rgba(148,163,184,0.12)",
    axis: css("--viz-axis") || "#94a3b8",
    ink: css("--viz-ink") || "#e2e8f0",
  };
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (window.Chart) {
    Chart.defaults.font.family = "Inter, system-ui, sans-serif";
    Chart.defaults.color = COLORS.axis;
  }

  // -------------------------------------------------------------------------
  // Formatting
  // -------------------------------------------------------------------------
  const percent = (part, whole) => (whole > 0 ? Math.round((part / whole) * 100) : 0);

  function dayLabel(isoDay, isLast) {
    if (isLast) return "Today";
    return TE.parseDay(isoDay).toLocaleDateString("en-IN", { weekday: "short", day: "numeric" });
  }

  function fullDay(isoDay) {
    return TE.parseDay(isoDay).toLocaleDateString("en-IN", { weekday: "long", day: "numeric", month: "short" });
  }

  /** Count up to a number on first load (skipped for reduced motion). */
  function animateNumber(el, target, format) {
    if (reduceMotion || state.loadedOnce) {
      el.textContent = format(target);
      return;
    }
    const duration = 700;
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = format(Math.round(target * eased));
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  // -------------------------------------------------------------------------
  // Chart.js plugins
  // -------------------------------------------------------------------------
  /** Vertical hairline at the hovered day, so both series read off one position. */
  const crosshairPlugin = {
    id: "teCrosshair",
    afterDatasetsDraw(chart) {
      const active = chart.tooltip && chart.tooltip.getActiveElements();
      if (!active || !active.length) return;
      const { ctx, chartArea } = chart;
      const x = active[0].element.x;
      ctx.save();
      ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();
      ctx.restore();
    },
  };

  /** Series name next to each line's last point (text in ink colour, nudged apart if close). */
  const endLabelPlugin = {
    id: "teEndLabels",
    afterDatasetsDraw(chart) {
      const { ctx } = chart;
      const placed = [];
      chart.data.datasets.forEach((dataset, index) => {
        const meta = chart.getDatasetMeta(index);
        if (meta.hidden || !meta.data.length) return;
        const point = meta.data[meta.data.length - 1];
        let y = point.y;
        placed.forEach((otherY) => {
          if (Math.abs(otherY - y) < 15) y = otherY + (y >= otherY ? 15 : -15);
        });
        placed.push(y);
        ctx.save();
        ctx.font = "500 12px Inter, system-ui, sans-serif";
        ctx.fillStyle = COLORS.ink;
        ctx.textBaseline = "middle";
        ctx.fillText(`${dataset.label} ${dataset.data[dataset.data.length - 1]}`, point.x + 10, y);
        ctx.restore();
      });
    },
  };

  const tooltipStyle = {
    backgroundColor: "rgba(2, 6, 23, 0.95)",
    borderColor: "rgba(148, 163, 184, 0.25)",
    borderWidth: 1,
    padding: 10,
    titleColor: COLORS.ink,
    bodyColor: COLORS.ink,
    footerColor: COLORS.axis,
    titleFont: { weight: "600" },
    footerFont: { weight: "500" },
    usePointStyle: true,
    boxPadding: 4,
  };

  // -------------------------------------------------------------------------
  // Rendering
  // -------------------------------------------------------------------------
  function renderTiles(stats) {
    const money = (n) => TE.formatCurrency(n);
    const plain = (n) => n.toLocaleString("en-IN");
    animateNumber(els.statTotal, stats.total, plain);
    animateNumber(els.statNoHelmet, stats.no_helmet, plain);
    animateNumber(els.statTriple, stats.triple_riding, plain);
    animateNumber(els.statFines, stats.total_fines, money);

    els.statTotalSub.textContent = `${stats.today} today`;
    els.statNoHelmetSub.textContent = `${percent(stats.no_helmet, stats.total)}% of all violations`;
    els.statTripleSub.textContent = `${percent(stats.triple_riding, stats.total)}% of all violations`;
    els.statFinesSub.textContent = `Across ${stats.total} e-challan${stats.total === 1 ? "" : "s"}`;
  }

  function renderTypeChart(stats) {
    const values = TYPES.map((t) => stats.by_type[t.key] || 0);
    const colors = TYPES.map((t) => TE.violationColor(t.key));
    els.typeTotal.textContent = stats.total.toLocaleString("en-IN");

    els.typeLegend.innerHTML = TYPES.map((t, i) => `
      <li class="flex items-center justify-between gap-3 text-sm">
        <span class="flex items-center gap-2 text-slate-200"><span class="series-swatch ${TE.swatchClass(t.key)}"></span>${t.label}</span>
        <span class="num text-slate-300"><b class="text-white">${values[i]}</b> · ${percent(values[i], stats.total)}%</span>
      </li>`).join("");

    if (state.typeChart) {
      state.typeChart.data.datasets[0].data = values;
      state.typeChart.update();
      return;
    }
    state.typeChart = new Chart(els.typeCanvas, {
      type: "doughnut",
      data: {
        labels: TYPES.map((t) => t.label),
        datasets: [
          {
            data: values,
            backgroundColor: colors,
            hoverBackgroundColor: colors,
            borderColor: COLORS.surface,       // 2px surface gap between segments
            borderWidth: 2,
            hoverOffset: 6,
          },
        ],
      },
      options: {
        maintainAspectRatio: false,
        cutout: "70%",
        animation: { duration: reduceMotion ? 0 : 700 },
        plugins: {
          legend: { display: false },
          tooltip: {
            ...tooltipStyle,
            callbacks: {
              // Read the chart's current data so the % stays right after a refresh.
              label: (item) => {
                const total = item.dataset.data.reduce((sum, value) => sum + value, 0);
                return ` ${item.label}: ${item.raw} (${percent(item.raw, total)}%)`;
              },
            },
          },
        },
      },
    });
  }

  function renderTrendChart(stats) {
    const trend = stats.daily_trend;
    state.trend = trend;                  // tooltips read this, so they stay current after refreshes
    const labels = trend.map((d, i) => dayLabel(d.date, i === trend.length - 1));

    els.trendLegend.innerHTML = TYPES.map((t) =>
      `<span class="flex items-center gap-2"><span class="series-swatch ${TE.swatchClass(t.key)}"></span>${t.label}</span>`
    ).join("");

    els.trendTable.innerHTML = trend.map((d) => `
      <tr>
        <td>${fullDay(d.date)}</td>
        <td class="num">${d.no_helmet}</td>
        <td class="num">${d.triple_riding}</td>
        <td class="num font-semibold text-white">${d.count}</td>
      </tr>`).join("");

    const datasets = TYPES.map((t) => {
      const color = TE.violationColor(t.key);
      return {
        label: t.label,
        data: trend.map((d) => d[t.field]),
        borderColor: color,
        backgroundColor: color,
        borderWidth: 2,
        pointRadius: 4,                 // 8px markers
        pointHoverRadius: 6,
        pointBorderColor: COLORS.surface, // 2px surface ring around each point
        pointBorderWidth: 2,
        pointHitRadius: 14,
        // Smooth but never overshooting: a curve must not suggest counts that never happened.
        cubicInterpolationMode: "monotone",
      };
    });

    if (state.trendChart) {
      state.trendChart.data.labels = labels;
      datasets.forEach((ds, i) => { state.trendChart.data.datasets[i].data = ds.data; });
      state.trendChart.update();
      return;
    }

    state.trendChart = new Chart(els.trendCanvas, {
      type: "line",
      data: { labels, datasets },
      plugins: [crosshairPlugin, endLabelPlugin],
      options: {
        maintainAspectRatio: false,
        animation: { duration: reduceMotion ? 0 : 700 },
        interaction: { mode: "index", intersect: false },
        layout: { padding: { top: 8, right: 118 } },   // room for the end labels
        scales: {
          x: {
            grid: { display: false },
            border: { color: "rgba(148, 163, 184, 0.3)" },
            ticks: { color: COLORS.axis },
          },
          y: {
            beginAtZero: true,
            grace: "10%",
            ticks: { precision: 0, color: COLORS.axis },
            grid: { color: COLORS.grid },
            border: { display: false },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            ...tooltipStyle,
            callbacks: {
              title: (items) => fullDay(state.trend[items[0].dataIndex].date),
              footer: (items) => `Total: ${state.trend[items[0].dataIndex].count}`,
            },
          },
        },
      },
    });
  }

  function renderRecent(records) {
    if (!records.length) {
      els.recentList.innerHTML = '<li class="py-4 text-sm text-slate-400">No violations yet.</li>';
      return;
    }
    els.recentList.innerHTML = records.map((r) => {
      const plate = r.plate_number && r.plate_number !== "Not detected"
        ? `<span class="font-mono text-slate-200">${TE.escapeHtml(r.plate_number)}</span>`
        : '<span class="text-slate-500">Plate not detected</span>';
      return `
        <li class="flex flex-wrap items-center gap-4 py-3">
          <img src="${TE.escapeHtml(r.evidence_url)}" alt="" loading="lazy"
               class="w-16 h-10 object-cover rounded-lg ring-1 ring-white/10 bg-slate-950" />
          <div class="min-w-0 flex-1">
            <div class="flex flex-wrap items-center gap-2">${TE.typeChip(r.type)} ${plate}</div>
            <p class="mt-1 text-xs text-slate-500">${TE.escapeHtml(r.challan_id)} · ${TE.formatDateTime(r.created_at)}</p>
          </div>
          <span class="num text-sm font-semibold text-amber-200">${TE.formatCurrency(r.fine)}</span>
          <a href="${TE.escapeHtml(r.report_url)}?download=false" target="_blank" rel="noopener" class="btn-ghost btn-sm">Challan</a>
        </li>`;
    }).join("");
  }

  // -------------------------------------------------------------------------
  // Loading
  // -------------------------------------------------------------------------
  function showState(which) {
    els.error.hidden = which !== "error";
    els.empty.hidden = which !== "empty";
    els.content.hidden = which !== "content";
  }

  async function load() {
    if (state.loading) return;
    state.loading = true;
    // Keep the previous render visible (dimmed) while refetching - no layout jump.
    if (state.loadedOnce) els.content.classList.add("is-refreshing");
    els.refresh.disabled = true;

    try {
      if (!window.Chart) throw new TE.ApiError("Charts unavailable", "Chart.js could not be loaded from the CDN. Check your internet connection.");
      const stats = await TE.apiFetch("/api/stats");
      if (stats.total === 0) {
        showState("empty");
      } else {
        showState("content");
        renderTiles(stats);
        renderTypeChart(stats);
        renderTrendChart(stats);
        renderRecent(stats.recent);
        state.loadedOnce = true;
      }
      els.updatedAt.textContent = `Updated ${new Date().toLocaleTimeString("en-IN")}`;
    } catch (err) {
      if (state.loadedOnce) {
        TE.toast.error(err.error || "Refresh failed", err.detail || err.message);
      } else {
        els.errorText.textContent = err.detail || err.message;
        showState("error");
      }
    } finally {
      state.loading = false;
      els.refresh.disabled = false;
      els.content.classList.remove("is-refreshing");
    }
  }

  els.refresh.addEventListener("click", load);
  els.retry.addEventListener("click", load);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && state.loadedOnce) load();
  });
  setInterval(() => {
    if (document.visibilityState === "visible") load();
  }, AUTO_REFRESH_MS);

  load();
})();
