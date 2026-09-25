/**
 * TrafficEye – violation history page.
 *
 *  - Loads GET /api/violations with the current filters (plate search, type, date)
 *  - Filters live in the URL (?search=&type=&date=) so a filtered view survives a refresh
 *  - Evidence thumbnails open a full-size preview modal
 *  - Challans download from /api/report/{id}
 *  - Delete asks for confirmation, then calls DELETE /api/violations/{id}
 *  - Loading (skeleton rows), empty and error states
 */
(function () {
  "use strict";

  const SEARCH_DEBOUNCE_MS = 300;
  const SKELETON_ROWS = 6;

  const $ = (id) => document.getElementById(id);
  const els = {
    search: $("filter-search"),
    type: $("filter-type"),
    date: $("filter-date"),
    clear: $("clear-filters"),
    summary: $("result-summary"),
    tableCard: $("table-card"),
    body: $("history-body"),
    empty: $("history-empty"),
    emptyTitle: $("history-empty-title"),
    emptyText: $("history-empty-text"),
    emptyActions: $("history-empty-actions"),
    error: $("history-error"),
    errorText: $("history-error-text"),
    retry: $("retry-btn"),
    evidenceModal: $("evidence-modal"),
    evidenceTitle: $("evidence-title"),
    evidenceMeta: $("evidence-meta"),
    evidenceImage: $("evidence-image"),
    evidenceSource: $("evidence-source"),
    evidenceOpen: $("evidence-open"),
    evidenceChallan: $("evidence-challan"),
    confirmModal: $("confirm-modal"),
    confirmTitle: $("confirm-title"),
    confirmDelete: $("confirm-delete"),
  };

  const state = {
    rows: [],
    requestId: 0,          // ignore responses from outdated requests (fast typing)
    pendingDelete: null,   // record waiting for confirmation
    lastFocus: null,       // element to refocus when a modal closes
  };

  // -------------------------------------------------------------------------
  // Filters <-> URL
  // -------------------------------------------------------------------------
  function currentFilters() {
    return {
      search: els.search.value.trim(),
      type: els.type.value,
      date: els.date.value,
    };
  }

  const hasFilters = (f) => Boolean(f.search || f.type || f.date);

  function readFiltersFromUrl() {
    const params = new URLSearchParams(window.location.search);
    els.search.value = params.get("search") || "";
    const type = params.get("type") || "";
    els.type.value = [...els.type.options].some((o) => o.value === type) ? type : "";
    els.date.value = /^\d{4}-\d{2}-\d{2}$/.test(params.get("date") || "") ? params.get("date") : "";
  }

  function writeFiltersToUrl(filters) {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => value && params.set(key, value));
    const query = params.toString();
    window.history.replaceState(null, "", query ? `?${query}` : window.location.pathname);
  }

  // -------------------------------------------------------------------------
  // Rendering
  // -------------------------------------------------------------------------
  function skeletonRows() {
    const cell = (w) => `<td><span class="skeleton block h-4" style="width:${w}"></span></td>`;
    const row = `<tr>${cell("3rem")}${cell("7rem")}${cell("6rem")}${cell("7rem")}${cell("3.5rem")}
      <td><span class="skeleton block" style="width:4.5rem;height:2.75rem"></span></td>
      ${cell("8rem")}${cell("4rem")}${cell("2rem")}</tr>`;
    return row.repeat(SKELETON_ROWS);
  }

  function rowHtml(r) {
    const percent = Math.round(r.confidence * 100);
    const plate = r.plate_number && r.plate_number !== "Not detected"
      ? `<span class="plate">${TE.escapeHtml(r.plate_number)}</span>`
      : '<span class="plate missing">Not detected</span>';
    return `
      <tr data-id="${r.id}">
        <td>
          <div class="num font-semibold text-white">#${r.id}</div>
          <div class="font-mono text-[11px] text-slate-500">${TE.escapeHtml(r.challan_id)}</div>
        </td>
        <td>${TE.typeChip(r.type)}</td>
        <td>
          <div class="flex items-center gap-2">
            <div class="conf-bar w-20"><span style="width:${percent}%"></span></div>
            <span class="num text-slate-300">${percent}%</span>
          </div>
        </td>
        <td>${plate}</td>
        <td class="num text-amber-200 font-medium">${TE.formatCurrency(r.fine)}</td>
        <td>
          <button type="button" class="thumb-button" data-action="preview" data-id="${r.id}"
                  aria-label="Enlarge evidence for violation #${r.id}">
            <img src="${TE.escapeHtml(r.evidence_url)}" alt="" loading="lazy" />
          </button>
        </td>
        <td class="num text-slate-300">${TE.formatDateTime(r.created_at)}</td>
        <td>
          <a href="${TE.escapeHtml(r.report_url)}" class="btn-ghost btn-sm" download title="Download e-challan PDF">
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3"/></svg>
            PDF
          </a>
        </td>
        <td>
          <button type="button" class="icon-button danger" data-action="delete" data-id="${r.id}"
                  aria-label="Delete violation #${r.id}" title="Delete">
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0"/></svg>
          </button>
        </td>
      </tr>`;
  }

  function renderSummary() {
    const count = state.rows.length;
    const fines = state.rows.reduce((sum, r) => sum + r.fine, 0);
    els.summary.textContent = count
      ? `Showing ${count} violation${count === 1 ? "" : "s"} · ${TE.formatCurrency(fines)} in fines`
      : "";
  }

  function renderEmpty(filters) {
    const filtered = hasFilters(filters);
    els.emptyTitle.textContent = filtered ? "No violations match these filters" : "No violations recorded yet";
    els.emptyText.innerHTML = filtered
      ? "Try a different plate, type or date."
      : 'Analyze a traffic image, or load demo data with <code class="text-amber-300">python -m backend.seed</code>.';
    els.emptyActions.innerHTML = filtered
      ? '<button type="button" class="btn-ghost btn-sm" data-action="clear">Clear filters</button>'
      : '<a href="/" class="btn-primary btn-sm">Analyze an image</a>';
    els.empty.hidden = false;
  }

  function renderRows(filters) {
    els.empty.hidden = true;
    if (!state.rows.length) {
      els.body.innerHTML = "";
      renderEmpty(filters);
    } else {
      els.body.innerHTML = state.rows.map(rowHtml).join("");
    }
    renderSummary();
  }

  // -------------------------------------------------------------------------
  // Loading
  // -------------------------------------------------------------------------
  async function load() {
    const filters = currentFilters();
    writeFiltersToUrl(filters);
    const requestId = ++state.requestId;

    els.error.hidden = true;
    els.tableCard.hidden = false;
    els.empty.hidden = true;
    els.body.innerHTML = skeletonRows();
    els.summary.textContent = "Loading…";

    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => value && params.set(key, value));

    try {
      const data = await TE.apiFetch(`/api/violations?${params.toString()}`);
      if (requestId !== state.requestId) return;       // a newer request is in flight
      state.rows = data.violations;
      renderRows(filters);
    } catch (err) {
      if (requestId !== state.requestId) return;
      state.rows = [];
      els.tableCard.hidden = true;
      els.errorText.textContent = err.detail || err.message;
      els.error.hidden = false;
      els.summary.textContent = "";
      TE.toast.error(err.error || "Could not load violations", err.detail || err.message);
    }
  }

  function clearFilters() {
    els.search.value = "";
    els.type.value = "";
    els.date.value = "";
    load();
    els.search.focus();
  }

  // -------------------------------------------------------------------------
  // Modals (evidence preview + delete confirmation)
  // -------------------------------------------------------------------------
  function openModal(modal, focusTarget) {
    state.lastFocus = document.activeElement;
    modal.hidden = false;
    document.body.classList.add("modal-open");
    (focusTarget || modal.querySelector("[data-close]")).focus();
  }

  function closeModal(modal) {
    if (modal.hidden) return;
    modal.hidden = true;
    if (els.evidenceModal.hidden && els.confirmModal.hidden) document.body.classList.remove("modal-open");
    if (state.lastFocus && document.contains(state.lastFocus)) state.lastFocus.focus();
  }

  function findRow(id) {
    return state.rows.find((r) => r.id === Number(id));
  }

  function openEvidence(record) {
    const plate = record.plate_number !== "Not detected" ? record.plate_number : "plate not detected";
    els.evidenceTitle.textContent = `${TE.violationLabel(record.type)} · ${plate}`;
    els.evidenceMeta.textContent =
      `${record.challan_id} · ${TE.formatDateTime(record.created_at)} · confidence ${Math.round(record.confidence * 100)}%`;
    els.evidenceImage.src = record.evidence_url;
    els.evidenceImage.alt = `Annotated evidence for violation #${record.id}`;
    els.evidenceSource.textContent = `Source file: ${record.source_file}`;
    els.evidenceOpen.href = record.evidence_url;
    els.evidenceChallan.href = record.report_url;
    openModal(els.evidenceModal);
  }

  function askDelete(record) {
    state.pendingDelete = record;
    els.confirmTitle.textContent = `Delete violation #${record.id}?`;
    els.confirmDelete.disabled = false;
    els.confirmDelete.textContent = "Delete";
    openModal(els.confirmModal, els.confirmModal.querySelector("[data-close]"));   // Cancel is the safe default
  }

  async function confirmDelete() {
    const record = state.pendingDelete;
    if (!record) return;
    els.confirmDelete.disabled = true;
    els.confirmDelete.textContent = "Deleting…";

    try {
      const result = await TE.apiFetch(`/api/violations/${record.id}`, { method: "DELETE" });
      closeModal(els.confirmModal);
      state.pendingDelete = null;
      TE.toast.success(
        `Violation #${record.id} deleted`,
        result.evidence_deleted
          ? "Record, e-challan and evidence image removed."
          : "Record and e-challan removed. The evidence image is kept because another violation uses it."
      );

      const row = els.body.querySelector(`tr[data-id="${record.id}"]`);
      state.rows = state.rows.filter((r) => r.id !== record.id);
      const finish = () => renderRows(currentFilters());
      if (row) {
        row.classList.add("removing");
        setTimeout(finish, 300);
      } else {
        finish();
      }
    } catch (err) {
      els.confirmDelete.disabled = false;
      els.confirmDelete.textContent = "Delete";
      TE.toast.error(err.error || "Delete failed", err.detail || err.message);
      if (err.status === 404) {
        closeModal(els.confirmModal);
        load();                                        // someone else already deleted it
      }
    }
  }

  // -------------------------------------------------------------------------
  // Event wiring
  // -------------------------------------------------------------------------
  function init() {
    readFiltersFromUrl();

    els.search.addEventListener("input", TE.debounce(load, SEARCH_DEBOUNCE_MS));
    els.type.addEventListener("change", load);
    els.date.addEventListener("change", load);
    els.clear.addEventListener("click", clearFilters);
    els.retry.addEventListener("click", load);

    // One delegated listener for every row button.
    els.body.addEventListener("click", (event) => {
      const button = event.target.closest("[data-action]");
      if (!button) return;
      const record = findRow(button.dataset.id);
      if (!record) return;
      if (button.dataset.action === "preview") openEvidence(record);
      if (button.dataset.action === "delete") askDelete(record);
    });
    els.emptyActions.addEventListener("click", (event) => {
      if (event.target.closest('[data-action="clear"]')) clearFilters();
    });

    els.confirmDelete.addEventListener("click", confirmDelete);
    [els.evidenceModal, els.confirmModal].forEach((modal) => {
      modal.addEventListener("click", (event) => {
        // Close on the backdrop or any [data-close] button, not on clicks inside the panel.
        if (event.target === modal || event.target.closest("[data-close]")) closeModal(modal);
      });
    });
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      closeModal(els.confirmModal);
      closeModal(els.evidenceModal);
    });

    load();
  }

  init();
})();
