/**
 * TrafficEye – shared frontend helpers (loaded first on every page).
 *
 *  - Navbar with Home / page links and live model status dots (/api/health)
 *  - Toast notifications (success / error / warning / info), no library needed
 *  - apiFetch(): fetch wrapper that understands {success:false, error, detail}
 *  - Small formatting helpers shared by the page scripts
 *
 * Everything is exposed on window.TE so page scripts can use it.
 */
(function () {
  "use strict";

  const NAV_LINKS = [
    { id: "home", label: "Home", href: "/" },
    { id: "dashboard", label: "Dashboard", href: "/dashboard" },
    { id: "history", label: "History", href: "/history" },
  ];

  const MODEL_STATUS = [
    { key: "yolo", label: "YOLO", field: "yolo_loaded" },
    { key: "helmet", label: "Helmet", field: "helmet_model_loaded" },
    { key: "ocr", label: "OCR", field: "ocr_loaded" },
  ];

  const HEALTH_REFRESH_MS = 30000;

  // -------------------------------------------------------------------------
  // Formatting helpers
  // -------------------------------------------------------------------------
  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatBytes(bytes) {
    const value = Number(bytes) || 0;
    if (value < 1024) return `${value} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
    return `${(value / (1024 * 1024)).toFixed(2)} MB`;
  }

  function formatPercent(value, digits = 0) {
    return `${((Number(value) || 0) * 100).toFixed(digits)}%`;
  }

  const rupeeFormatter = new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  });

  function formatCurrency(amount) {
    return rupeeFormatter.format(Number(amount) || 0);
  }

  function formatDateTime(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "—";
    return date.toLocaleString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function titleCase(text) {
    return String(text ?? "")
      .toLowerCase()
      .replace(/[_-]+/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());
  }

  const VIOLATION_LABELS = {
    NO_HELMET: "No Helmet",
    TRIPLE_RIDING: "Triple Riding",
  };

  function violationLabel(type) {
    return VIOLATION_LABELS[type] || titleCase(type);
  }

  // Each violation type keeps ONE colour on every page (defined in style.css).
  const VIOLATION_COLOR_VARS = {
    NO_HELMET: "--viz-no-helmet",
    TRIPLE_RIDING: "--viz-triple-riding",
  };

  function violationColor(type) {
    const cssVar = VIOLATION_COLOR_VARS[type];
    const value = cssVar ? getComputedStyle(document.documentElement).getPropertyValue(cssVar).trim() : "";
    return value || "#94a3b8";
  }

  function swatchClass(type) {
    return type === "TRIPLE_RIDING" ? "swatch-triple-riding" : type === "NO_HELMET" ? "swatch-no-helmet" : "";
  }

  /** Coloured dot + label; the text stays in ink colour so colour is never the only cue. */
  function typeChip(type) {
    return `<span class="type-chip"><span class="series-swatch ${swatchClass(type)}"></span>${escapeHtml(violationLabel(type))}</span>`;
  }

  /** "2026-09-25" -> local Date (new Date("2026-09-25") would be parsed as UTC). */
  function parseDay(text) {
    const [year, month, day] = String(text).split("-").map(Number);
    return new Date(year, (month || 1) - 1, day || 1);
  }

  function todayIso() {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
  }

  function debounce(fn, waitMs) {
    let timer = null;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), waitMs);
    };
  }

  // -------------------------------------------------------------------------
  // API wrapper
  // -------------------------------------------------------------------------
  class ApiError extends Error {
    constructor(error, detail = "", status = 0) {
      super(detail ? `${error}: ${detail}` : error);
      this.name = "ApiError";
      this.error = error;
      this.detail = detail;
      this.status = status;
    }
  }

  /**
   * fetch() that always resolves to parsed JSON or throws an ApiError whose
   * .error / .detail come straight from the backend's error format.
   */
  async function apiFetch(url, options = {}) {
    let response;
    try {
      response = await fetch(url, options);
    } catch (networkError) {
      throw new ApiError("Network error", "Could not reach the TrafficEye server. Is uvicorn running?");
    }

    let payload = null;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      try {
        payload = await response.json();
      } catch (parseError) {
        payload = null;
      }
    }

    if (!response.ok || (payload && payload.success === false)) {
      throw new ApiError(
        (payload && payload.error) || `Request failed (${response.status})`,
        (payload && payload.detail) || response.statusText || "",
        response.status
      );
    }
    return payload;
  }

  // -------------------------------------------------------------------------
  // Toasts
  // -------------------------------------------------------------------------
  const TOAST_ICONS = {
    success: '<path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"/>',
    error: '<path stroke-linecap="round" stroke-linejoin="round" d="m9.75 9.75 4.5 4.5m0-4.5-4.5 4.5M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"/>',
    warning: '<path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z"/>',
    info: '<path stroke-linecap="round" stroke-linejoin="round" d="m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z"/>',
  };

  function getToastContainer() {
    let container = document.getElementById("toast-container");
    if (!container) {
      container = document.createElement("div");
      container.id = "toast-container";
      document.body.appendChild(container);
    }
    container.setAttribute("aria-live", "polite");
    return container;
  }

  function dismissToast(toastEl) {
    if (!toastEl || toastEl.classList.contains("leaving")) return;
    toastEl.classList.add("leaving");
    toastEl.addEventListener("animationend", () => toastEl.remove(), { once: true });
  }

  function showToast(type, title, message = "", duration = 5000) {
    const kind = TOAST_ICONS[type] ? type : "info";
    const toastEl = document.createElement("div");
    toastEl.className = `toast toast-${kind}`;
    toastEl.setAttribute("role", kind === "error" ? "alert" : "status");
    toastEl.innerHTML = `
      <svg class="toast-icon w-5 h-5 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke-width="1.8" stroke="currentColor">${TOAST_ICONS[kind]}</svg>
      <div class="min-w-0">
        <p class="text-sm font-semibold text-white">${escapeHtml(title)}</p>
        ${message ? `<p class="text-xs text-slate-300 mt-0.5 break-words">${escapeHtml(message)}</p>` : ""}
      </div>
      <button type="button" class="toast-close" aria-label="Dismiss">&times;</button>`;

    toastEl.querySelector(".toast-close").addEventListener("click", () => dismissToast(toastEl));
    getToastContainer().appendChild(toastEl);
    if (duration > 0) setTimeout(() => dismissToast(toastEl), duration);
    return toastEl;
  }

  const toast = {
    success: (title, message, duration) => showToast("success", title, message, duration),
    error: (title, message, duration) => showToast("error", title, message, duration ?? 7000),
    warning: (title, message, duration) => showToast("warning", title, message, duration ?? 6000),
    info: (title, message, duration) => showToast("info", title, message, duration),
  };

  // -------------------------------------------------------------------------
  // Navbar
  // -------------------------------------------------------------------------
  const LOGO_SVG = `
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke="url(#te-logo-gradient)">
      <defs>
        <linearGradient id="te-logo-gradient" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#f87171"/><stop offset="1" stop-color="#fbbf24"/>
        </linearGradient>
      </defs>
      <path stroke-linecap="round" stroke-linejoin="round" d="M2.04 12.32a1 1 0 0 1 0-.64C3.42 7.51 7.36 4.5 12 4.5c4.64 0 8.57 3.01 9.96 7.18.07.21.07.43 0 .64C20.58 16.49 16.64 19.5 12 19.5c-4.64 0-8.57-3.01-9.96-7.18Z"/>
      <circle cx="12" cy="12" r="3"/>
    </svg>`;

  function statusPillsHtml() {
    return MODEL_STATUS.map(
      (m) => `<span class="status-pill" data-status="${m.key}" title="Checking ${m.label}...">
                <span class="status-dot pending"></span>${m.label}
              </span>`
    ).join("");
  }

  function renderNavbar() {
    const header = document.getElementById("navbar");
    if (!header) return;
    const active = document.body.dataset.page || "home";

    const links = (extraClass) =>
      NAV_LINKS.map(
        (link) => `<a href="${link.href}" class="nav-link ${extraClass} ${link.id === active ? "active" : ""}"
                      ${link.id === active ? 'aria-current="page"' : ""}>${link.label}</a>`
      ).join("");

    header.innerHTML = `
      <nav class="fixed top-0 inset-x-0 z-50 border-b border-white/5 bg-slate-950/70 backdrop-blur-xl">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          <a href="/" class="flex items-center gap-2.5 shrink-0">
            <span class="logo-mark">${LOGO_SVG}</span>
            <span class="font-display text-lg font-bold tracking-tight text-white">Traffic<span class="text-gradient">Eye</span></span>
          </a>
          <div class="hidden md:flex items-center gap-1">${links("")}</div>
          <div class="hidden md:flex items-center gap-2" aria-label="Model status">${statusPillsHtml()}</div>
          <div class="md:hidden">
            <button id="nav-toggle" type="button" class="btn-ghost btn-sm" aria-expanded="false" aria-controls="mobile-menu" aria-label="Toggle menu">
              <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke-width="1.8" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5"/>
              </svg>
            </button>
          </div>
        </div>
        <div id="mobile-menu" class="md:hidden hidden border-t border-white/5 px-4 py-4 space-y-3 bg-slate-950/90">
          <div class="flex flex-col gap-1">${links("block")}</div>
          <div class="flex flex-wrap gap-2 pt-2 border-t border-white/5" aria-label="Model status">${statusPillsHtml()}</div>
        </div>
      </nav>`;

    const toggle = document.getElementById("nav-toggle");
    const menu = document.getElementById("mobile-menu");
    toggle.addEventListener("click", () => {
      const open = menu.classList.toggle("hidden") === false;
      toggle.setAttribute("aria-expanded", String(open));
    });
  }

  // -------------------------------------------------------------------------
  // Model status (GET /api/health)
  // -------------------------------------------------------------------------
  let lastHealth = null;

  function updateStatusPills(health) {
    MODEL_STATUS.forEach((model) => {
      const loaded = Boolean(health && health[model.field]);
      const error = health && health.errors ? health.errors[model.key] : "";
      let tooltip;
      if (!health) tooltip = "Server unreachable";
      else if (loaded) tooltip = `${model.label} model loaded`;
      else tooltip = error || `${model.label} model not loaded`;

      document.querySelectorAll(`[data-status="${model.key}"]`).forEach((pill) => {
        pill.title = tooltip;
        const dot = pill.querySelector(".status-dot");
        if (dot) dot.className = `status-dot ${loaded ? "ok" : "off"}`;
      });
    });
  }

  async function refreshHealth() {
    try {
      lastHealth = await apiFetch("/api/health");
    } catch (err) {
      lastHealth = null;
    }
    updateStatusPills(lastHealth);
    document.dispatchEvent(new CustomEvent("te:health", { detail: lastHealth }));
    return lastHealth;
  }

  // -------------------------------------------------------------------------
  // Public API + page bootstrap
  // -------------------------------------------------------------------------
  window.TE = {
    apiFetch,
    ApiError,
    toast,
    escapeHtml,
    formatBytes,
    formatPercent,
    formatCurrency,
    formatDateTime,
    titleCase,
    violationLabel,
    violationColor,
    swatchClass,
    typeChip,
    parseDay,
    todayIso,
    debounce,
    refreshHealth,
    getHealth: () => lastHealth,
  };

  document.addEventListener("DOMContentLoaded", () => {
    renderNavbar();
    refreshHealth();
    setInterval(refreshHealth, HEALTH_REFRESH_MS);
  });
})();
