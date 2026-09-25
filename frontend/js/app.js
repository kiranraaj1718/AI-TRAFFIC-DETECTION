/**
 * TrafficEye – home page logic.
 *
 *  1. Pick an image (drag & drop or browse) and validate it in the browser.
 *  2. Show a preview, then POST it to /api/detect/image.
 *  3. While waiting, rotate through processing step messages.
 *  4. Render the annotated evidence, violations and detected objects.
 */
(function () {
  "use strict";

  const IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"];
  const VIDEO_EXTENSIONS = [".mp4"];
  const MAX_MB = { image: 10, video: 50 };
  const ENDPOINTS = { image: "/api/detect/image", video: "/api/detect/video" };

  const STEP_MESSAGES = [
    "Analyzing traffic...",
    "Detecting vehicles...",
    "Checking violations...",
    "Reading number plate...",
    "Generating evidence...",
  ];
  const STEP_INTERVAL_MS = 1400;

  // How each rider helmet status is shown in the "Motorcycles & riders" panel.
  const RIDER_STATUS = {
    HELMET: {
      label: "Helmet",
      classes: "bg-green-500/15 text-green-300 ring-green-500/30",
      hint: "Helmet detected on this rider's head region.",
    },
    NO_HELMET: {
      label: "No helmet",
      classes: "bg-red-500/15 text-red-300 ring-red-500/40",
      hint: "Bare head detected on this rider's head region.",
    },
    UNKNOWN: {
      label: "Unknown",
      classes: "bg-slate-500/15 text-slate-300 ring-slate-500/30",
      hint: "No helmet or head detection matched this rider, so no helmet violation is reported.",
    },
  };

  const HELMET_CHECK_MESSAGES = {
    skipped: {
      title: "Helmet model not loaded — helmet check skipped",
      text: "Place a YOLOv8 helmet model at backend/models/helmet.pt and restart the server to enable no-helmet detection. Riders are shown as Unknown.",
    },
    failed: {
      title: "Helmet check failed for this image",
      text: "The helmet model raised an error, so riders are shown as Unknown and no helmet violations were reported. See the server logs.",
    },
  };

  const $ = (id) => document.getElementById(id);
  const els = {
    ctaImage: $("cta-image"),
    ctaVideo: $("cta-video"),
    previewVideo: $("preview-video"),
    noEvidence: $("no-evidence"),
    statObjectsLabel: $("stat-objects-label"),
    motorcyclesCard: $("motorcycles-card"),
    detectionsCard: $("detections-card"),
    analyzeSection: $("analyze"),
    dropZone: $("drop-zone"),
    fileInput: $("file-input"),
    previewPanel: $("preview-panel"),
    previewImage: $("preview-image"),
    scanOverlay: $("scan-overlay"),
    fileName: $("file-name"),
    fileMeta: $("file-meta"),
    processing: $("processing"),
    stepText: $("step-text"),
    stepDots: $("step-dots"),
    analyzeBtn: $("analyze-btn"),
    changeBtn: $("change-btn"),
    results: $("results"),
    resultTitle: $("result-title"),
    resultMeta: $("result-meta"),
    helmetWarning: $("helmet-warning"),
    helmetWarningTitle: $("helmet-warning-title"),
    helmetWarningText: $("helmet-warning-text"),
    motorcyclesPanel: $("motorcycles-panel"),
    evidenceLink: $("evidence-link"),
    evidenceOpen: $("evidence-open"),
    resultImage: $("result-image"),
    statViolations: $("stat-violations"),
    statObjects: $("stat-objects"),
    statTime: $("stat-time"),
    violationsPanel: $("violations-panel"),
    classChips: $("class-chips"),
    detectionList: $("detection-list"),
    newAnalysisBtn: $("new-analysis-btn"),
  };

  const state = {
    kind: "image",           // "image" or "video", decided by the file extension
    file: null,
    previewUrl: null,
    busy: false,
    stepTimer: null,
  };

  // -------------------------------------------------------------------------
  // File selection + client-side validation
  // -------------------------------------------------------------------------
  function getExtension(name) {
    const dot = name.lastIndexOf(".");
    return dot >= 0 ? name.slice(dot).toLowerCase() : "";
  }

  function kindOf(file) {
    const ext = getExtension(file.name);
    if (IMAGE_EXTENSIONS.includes(ext)) return "image";
    if (VIDEO_EXTENSIONS.includes(ext)) return "video";
    return null;
  }

  function validateFile(file) {
    if (!file) return "No file selected.";
    const kind = kindOf(file);
    if (!kind) return "Only JPG, PNG and MP4 files are allowed.";
    if (file.size === 0) return "The selected file is empty.";
    if (file.size > MAX_MB[kind] * 1024 * 1024) {
      return `The file is ${TE.formatBytes(file.size)}. Maximum ${kind} size is ${MAX_MB[kind]} MB.`;
    }
    return null;
  }

  function selectFile(file) {
    const problem = validateFile(file);
    if (problem) {
      TE.toast.error("Invalid file", problem);
      return;
    }

    state.file = file;
    state.kind = kindOf(file);
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = URL.createObjectURL(file);

    const isVideo = state.kind === "video";
    els.previewImage.classList.toggle("hidden", isVideo);
    els.previewVideo.classList.toggle("hidden", !isVideo);
    if (isVideo) {
      els.previewVideo.src = state.previewUrl;
      els.previewImage.removeAttribute("src");
    } else {
      els.previewVideo.removeAttribute("src");
      els.previewImage.src = state.previewUrl;
    }
    els.fileName.textContent = file.name;
    els.fileMeta.textContent = `${TE.formatBytes(file.size)} · ${file.type || getExtension(file.name).slice(1).toUpperCase()}`;
    els.dropZone.classList.add("hidden");
    els.previewPanel.classList.remove("hidden");
    els.previewPanel.classList.add("fade-in");
  }

  function openFilePicker() {
    if (state.busy) return;
    els.fileInput.value = ""; // lets the user pick the same file again
    els.fileInput.click();
  }

  // -------------------------------------------------------------------------
  // Processing indicator
  // -------------------------------------------------------------------------
  function renderStepDots() {
    els.stepDots.innerHTML = STEP_MESSAGES.map(() => '<span class="step-dot"></span>').join("");
  }

  function showStep(index) {
    els.stepText.textContent = STEP_MESSAGES[index];
    // Restart the fade animation for each new message.
    els.stepText.classList.remove("step-text");
    void els.stepText.offsetWidth;
    els.stepText.classList.add("step-text");
    [...els.stepDots.children].forEach((dot, i) => dot.classList.toggle("active", i === index));
  }

  function setBusy(busy) {
    state.busy = busy;
    els.analyzeBtn.disabled = busy;
    els.changeBtn.disabled = busy;
    els.processing.classList.toggle("hidden", !busy);
    els.scanOverlay.classList.toggle("hidden", !busy);

    clearInterval(state.stepTimer);
    if (busy) {
      let step = 0;
      showStep(step);
      state.stepTimer = setInterval(() => {
        step = (step + 1) % STEP_MESSAGES.length;
        showStep(step);
      }, STEP_INTERVAL_MS);
    }
  }

  // -------------------------------------------------------------------------
  // API call
  // -------------------------------------------------------------------------
  async function analyze() {
    if (!state.file || state.busy) return;

    const form = new FormData();
    form.append("file", state.file, state.file.name);

    setBusy(true);
    try {
      const data = await TE.apiFetch(ENDPOINTS[state.kind], { method: "POST", body: form });
      if (state.kind === "video") renderVideoResult(data);
      else renderResult(data);

      const count = data.violations.length;
      if (count > 0) {
        TE.toast.warning(`${count} violation${count > 1 ? "s" : ""} detected`, "Evidence and challan details are shown below.");
      } else if (state.kind === "video") {
        TE.toast.success("Analysis complete", `No violations in ${data.frames_processed} analysed frames.`);
      } else {
        TE.toast.success(
          "Analysis complete",
          `${data.detections.length} object${data.detections.length === 1 ? "" : "s"} detected in ${(data.processing_time_ms / 1000).toFixed(2)} s.`
        );
      }
    } catch (err) {
      TE.toast.error(err.error || "Analysis failed", err.detail || err.message);
    } finally {
      setBusy(false);
    }
  }

  // -------------------------------------------------------------------------
  // Result rendering
  // -------------------------------------------------------------------------
  function showEvidence(url, emptyMessage) {
    els.evidenceLink.classList.toggle("hidden", !url);
    els.noEvidence.classList.toggle("hidden", Boolean(url));
    els.evidenceOpen.classList.toggle("hidden", !url);
    if (url) {
      els.resultImage.src = url;
      els.evidenceLink.href = url;
      els.evidenceOpen.href = url;
    } else {
      els.noEvidence.textContent = emptyMessage;
    }
  }

  function renderVideoResult(data) {
    const violations = data.violations || [];
    els.resultTitle.textContent = violations.length
      ? `${violations.length} unique violation${violations.length > 1 ? "s" : ""} found`
      : "No violations detected";
    els.resultMeta.textContent =
      `${data.filename} · ${data.duration_seconds}s at ${data.fps} fps · every ${data.frame_interval}th frame`;
    showEvidence(violations[0] && violations[0].evidence_url,
      `No violations were found in ${data.frames_processed} analysed frames.`);

    els.statViolations.textContent = violations.length;
    els.statViolations.className = `font-display text-2xl font-bold mt-1 ${violations.length ? "text-red-400" : "text-green-400"}`;
    els.statObjectsLabel.textContent = "Frames";
    els.statObjects.textContent = data.frames_processed;
    els.statTime.textContent = `${(data.processing_time_ms / 1000).toFixed(1)}s`;

    const skipped = !data.helmet_model_loaded;
    els.helmetWarning.classList.toggle("hidden", !skipped);
    if (skipped) {
      els.helmetWarningTitle.textContent = HELMET_CHECK_MESSAGES.skipped.title;
      els.helmetWarningText.textContent = HELMET_CHECK_MESSAGES.skipped.text;
    }

    renderViolations(violations, skipped ? "skipped" : "enabled");
    if (violations.length) {
      els.violationsPanel.insertAdjacentHTML(
        "afterbegin",
        `<p class="text-xs text-slate-400">${data.candidates_found} sightings across frames merged into ${violations.length} unique violation${violations.length > 1 ? "s" : ""}.</p>`
      );
    }
    els.motorcyclesCard.classList.add("hidden");
    els.detectionsCard.classList.add("hidden");

    els.results.classList.remove("hidden");
    els.results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderResult(data) {
    const violations = data.violations || [];
    const detections = data.detections || [];
    els.statObjectsLabel.textContent = "Objects";
    els.motorcyclesCard.classList.remove("hidden");
    els.detectionsCard.classList.remove("hidden");
    showEvidence(data.evidence_url, "");

    els.resultTitle.textContent = violations.length
      ? `${violations.length} violation${violations.length > 1 ? "s" : ""} found`
      : "No violations detected";
    els.resultMeta.textContent = `${data.filename} · ${data.image_width}×${data.image_height}px`;

    els.resultImage.src = data.evidence_url;
    els.evidenceLink.href = data.evidence_url;
    els.evidenceOpen.href = data.evidence_url;

    els.statViolations.textContent = violations.length;
    els.statViolations.className = `font-display text-2xl font-bold mt-1 ${violations.length ? "text-red-400" : "text-green-400"}`;
    els.statObjects.textContent = detections.length;
    els.statTime.textContent = `${(data.processing_time_ms / 1000).toFixed(2)}s`;

    const helmetMessage = HELMET_CHECK_MESSAGES[data.helmet_check];
    els.helmetWarning.classList.toggle("hidden", !helmetMessage);
    if (helmetMessage) {
      els.helmetWarningTitle.textContent = helmetMessage.title;
      els.helmetWarningText.textContent = helmetMessage.text;
    }

    renderViolations(violations, data.helmet_check);
    renderMotorcycles(data.motorcycles || []);
    renderDetections(detections, data.object_counts || {});

    els.results.classList.remove("hidden");
    els.results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderViolations(violations, helmetCheck) {
    if (!violations.length) {
      const detail =
        helmetCheck === "enabled"
          ? "No triple riding or riding without a helmet was found in this image."
          : "No triple riding was found. Helmets were not checked for this image.";
      els.violationsPanel.innerHTML = `
        <div class="glass success-panel p-5 flex items-start gap-4 fade-in">
          <div class="grid place-items-center w-11 h-11 rounded-xl bg-green-500/15 ring-1 ring-green-500/40 shrink-0">
            <svg class="w-6 h-6 text-green-400" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" d="m4.5 12.75 6 6 9-13.5"/>
            </svg>
          </div>
          <div>
            <p class="font-display font-semibold text-green-300">No violations detected</p>
            <p class="text-sm text-slate-400 mt-0.5">${detail}</p>
          </div>
        </div>`;
      return;
    }
    els.violationsPanel.innerHTML = violations.map(violationCard).join("");
  }

  function violationCard(violation) {
    const percent = Math.round((Number(violation.confidence) || 0) * 100);
    const plate = violation.plate_number && violation.plate_number !== "Not detected" ? violation.plate_number : null;
    const reportUrl = violation.report_url ? TE.escapeHtml(violation.report_url) : "";
    const challanButton = reportUrl
      ? `<div class="flex flex-wrap gap-2">
           <a href="${reportUrl}?download=false" class="btn-ghost btn-sm" target="_blank" rel="noopener">View</a>
           <a href="${reportUrl}" class="btn-primary btn-sm" download>
             <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor">
               <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3"/>
             </svg>
             Download Challan
           </a>
         </div>`
      : "";
    const challanId = violation.challan_id
      ? `<p class="text-[11px] font-mono text-slate-400 mt-1">Challan ${TE.escapeHtml(violation.challan_id)}</p>`
      : "";

    return `
      <article class="glass violation-card p-5 fade-in">
        <div class="flex items-start justify-between gap-3">
          <div>
            <p class="text-xs uppercase tracking-wider text-red-300/80">Violation</p>
            <h4 class="font-display text-lg font-semibold text-white">${TE.escapeHtml(TE.violationLabel(violation.type))}</h4>
            ${violation.description ? `<p class="text-sm text-slate-300 mt-0.5">${TE.escapeHtml(violation.description)}</p>` : ""}
            ${challanId}
            ${violation.time_seconds != null
              ? `<p class="text-[11px] text-slate-400 mt-0.5">At ${violation.time_seconds}s · ${violation.merged_detections} sighting${violation.merged_detections === 1 ? "" : "s"} merged</p>`
              : ""}
          </div>
          <span class="fine-badge">${TE.formatCurrency(violation.fine)}</span>
        </div>
        <div class="mt-4">
          <div class="flex justify-between text-xs text-slate-400 mb-1.5">
            <span>Confidence</span><span class="text-slate-200 font-medium">${percent}%</span>
          </div>
          <div class="conf-bar"><span style="width: ${percent}%"></span></div>
        </div>
        <div class="mt-4 flex flex-wrap items-end justify-between gap-3">
          <div>
            <p class="text-xs text-slate-400">Number plate</p>
            <span class="plate ${plate ? "" : "missing"}">${TE.escapeHtml(plate || "Not detected")}</span>
          </div>
          ${challanButton}
        </div>
      </article>`;
  }

  function riderChip(rider, index) {
    const status = RIDER_STATUS[rider.helmet_status] || RIDER_STATUS.UNKNOWN;
    const confidence = rider.helmet_confidence != null ? ` ${Math.round(rider.helmet_confidence * 100)}%` : "";
    return `
      <span class="inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ${status.classes}"
            title="${TE.escapeHtml(status.hint)}">
        Rider ${index + 1}: ${status.label}${confidence}
      </span>`;
  }

  function renderMotorcycles(motorcycles) {
    if (!motorcycles.length) {
      els.motorcyclesPanel.innerHTML = '<p class="text-sm text-slate-400">No motorcycles detected.</p>';
      return;
    }
    els.motorcyclesPanel.innerHTML = motorcycles
      .map((moto, i) => {
        const violating = moto.violations.length > 0;
        const riders = moto.riders.length
          ? moto.riders.map(riderChip).join("")
          : '<span class="text-xs text-slate-500">No riders linked (parked?)</span>';
        const verdict = violating
          ? moto.violations.map((type) => TE.escapeHtml(TE.violationLabel(type))).join(" + ")
          : "No violation";
        const plate = moto.plate_number ? ` · Plate: ${TE.escapeHtml(moto.plate_number)}` : "";
        return `
          <div class="rounded-xl p-3 ring-1 ${violating ? "ring-red-500/40 bg-red-950/20" : "ring-white/5 bg-slate-950/40"}">
            <div class="flex items-center justify-between gap-2 text-sm">
              <span class="font-medium text-slate-200">Motorcycle ${i + 1}
                <span class="text-slate-500 font-normal">· ${Math.round(moto.confidence * 100)}%</span></span>
              <span class="text-xs font-medium ${violating ? "text-red-300" : "text-green-300"}">${verdict}</span>
            </div>
            <p class="mt-1 text-xs text-slate-400">${moto.rider_count} rider${moto.rider_count === 1 ? "" : "s"}${plate}</p>
            <div class="mt-2 flex flex-wrap gap-1.5">${riders}</div>
          </div>`;
      })
      .join("");
  }

  function renderDetections(detections, counts) {
    const chips = Object.entries(counts)
      .filter(([, count]) => count > 0)
      .map(([name, count]) => `<span class="chip"><b>${count}</b> ${TE.escapeHtml(TE.titleCase(name))}</span>`);
    els.classChips.innerHTML = chips.length
      ? chips.join("")
      : '<span class="text-sm text-slate-400">No people or vehicles were detected.</span>';

    els.detectionList.innerHTML = detections
      .map((det) => {
        const percent = Math.round(det.confidence * 100);
        return `
          <li class="rounded-lg bg-slate-950/40 ring-1 ring-white/5 px-3 py-2.5">
            <div class="flex items-center justify-between text-sm">
              <span class="font-medium text-slate-200">${TE.escapeHtml(TE.titleCase(det.class))}</span>
              <span class="text-slate-400">${percent}%</span>
            </div>
            <div class="conf-bar good mt-2"><span style="width: ${percent}%"></span></div>
            <p class="mt-1.5 text-[11px] text-slate-500 font-mono">bbox [${det.bbox.join(", ")}]</p>
          </li>`;
      })
      .join("");
  }

  function resetForNewAnalysis() {
    els.results.classList.add("hidden");
    els.analyzeSection.scrollIntoView({ behavior: "smooth", block: "start" });
    openFilePicker();
  }

  // -------------------------------------------------------------------------
  // Event wiring
  // -------------------------------------------------------------------------
  function wireDropZone() {
    const zone = els.dropZone;

    zone.addEventListener("click", openFilePicker);
    zone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openFilePicker();
      }
    });

    ["dragenter", "dragover"].forEach((type) =>
      zone.addEventListener(type, (event) => {
        event.preventDefault();
        zone.classList.add("dragover");
      })
    );
    ["dragleave", "dragend"].forEach((type) =>
      zone.addEventListener(type, (event) => {
        if (!zone.contains(event.relatedTarget)) zone.classList.remove("dragover");
      })
    );
    zone.addEventListener("drop", (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
      const files = event.dataTransfer && event.dataTransfer.files;
      if (!files || !files.length) return;
      if (files.length > 1) TE.toast.info("One file at a time", "Only the first file will be analyzed.");
      selectFile(files[0]);
    });

    // Stop the browser from opening a file dropped outside the drop zone.
    ["dragover", "drop"].forEach((type) => window.addEventListener(type, (event) => event.preventDefault()));
  }

  function init() {
    renderStepDots();
    wireDropZone();

    els.fileInput.addEventListener("change", () => {
      if (els.fileInput.files.length) selectFile(els.fileInput.files[0]);
    });
    els.analyzeBtn.addEventListener("click", analyze);
    els.changeBtn.addEventListener("click", openFilePicker);
    els.newAnalysisBtn.addEventListener("click", resetForNewAnalysis);
    [els.ctaImage, els.ctaVideo].forEach((button) =>
      button.addEventListener("click", () => {
        els.analyzeSection.scrollIntoView({ behavior: "smooth", block: "start" });
        openFilePicker();
      })
    );

    // Tell the user early if detection is impossible.
    document.addEventListener(
      "te:health",
      (event) => {
        const health = event.detail;
        if (!health) {
          TE.toast.error("Server unreachable", "Start the backend with: uvicorn backend.main:app --reload");
        } else if (!health.yolo_loaded) {
          TE.toast.error("YOLO model not loaded", health.errors?.yolo || "Detection is unavailable. Check the server logs.");
        }
      },
      { once: true }
    );
  }

  init();
})();
