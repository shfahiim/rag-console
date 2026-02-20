const statusEl = document.getElementById("status");
const uploadForm = document.getElementById("upload-form");
const queryForm = document.getElementById("query-form");
const fileInput = document.getElementById("file-input");
const queryInput = document.getElementById("query-input");
const queryButton = document.getElementById("query-button");
const matchesEl = document.getElementById("matches");
const answerEl = document.getElementById("answer");
const sourcesEl = document.getElementById("sources");
const indexedFileEl = document.getElementById("indexed-file");
const chunkCountEl = document.getElementById("chunk-count");
const uploadProgressBarEl = document.getElementById("upload-progress-bar");
const uploadPercentLabelEl = document.getElementById("upload-percent-label");
const uploadProgressMessageEl = document.getElementById("upload-progress-message");
const uploadStepEls = Array.from(document.querySelectorAll("#upload-steps li"));
const evidencePreviewEl = document.getElementById("evidence-preview");
const evidenceTitleEl = document.getElementById("evidence-title");
const evidenceMetaEl = document.getElementById("evidence-meta");
const evidenceTextEl = document.getElementById("evidence-text");
const evidenceCloseEl = document.getElementById("evidence-close");
const uploadButton = uploadForm?.querySelector('button[type="submit"]');
const defaultUploadButtonLabel = uploadButton?.textContent?.trim() || "Index Document";
const defaultQueryButtonLabel = queryButton?.textContent?.trim() || "Send";

const UPLOAD_STAGE_ORDER = [
  "uploading",
  "uploaded",
  "extracting",
  "pipeline",
  "chunking",
  "embedding",
  "indexing",
  "complete",
];

let activeQueryRun = 0;
let activeUploadRunId = 0;
let uploadPollTimer = null;
let currentCitationMap = new Map();
let queryReady = false;
let queryBusy = false;
let queryFeedbackTimer = null;

async function readResponsePayload(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return await response.json();
  }
  const text = await response.text();
  return { error: text || `Request failed with status ${response.status}` };
}

function setStatus(message) {
  setStatusWithKind(message, "info");
}

function setStatusWithKind(message, kind) {
  if (!statusEl) {
    queryButton.title = String(message || "");
    queryButton.dataset.statusKind = kind || "info";
    return;
  }
  statusEl.textContent = message;
  statusEl.dataset.kind = kind || "info";
  statusEl.classList.remove("is-bump");
  // Force a reflow so the bump can retrigger.
  void statusEl.offsetWidth;
  statusEl.classList.add("is-bump");
  setTimeout(() => statusEl.classList.remove("is-bump"), 180);
}

function syncQueryControlState() {
  const enabled = queryReady && !queryBusy;
  queryInput.disabled = !enabled;
  queryButton.disabled = !enabled;
  queryButton.dataset.busy = String(queryBusy);
  if (queryBusy) {
    queryButton.classList.remove("is-success", "is-error");
    queryButton.textContent = "Generating...";
    return;
  }
  if (!queryButton.classList.contains("is-success") && !queryButton.classList.contains("is-error")) {
    queryButton.textContent = defaultQueryButtonLabel;
  }
}

function setQueryBusy(isBusy) {
  queryBusy = Boolean(isBusy);
  syncQueryControlState();
}

function clearQueryFeedback() {
  if (queryFeedbackTimer) {
    clearTimeout(queryFeedbackTimer);
    queryFeedbackTimer = null;
  }
  queryButton.classList.remove("is-success", "is-error");
}

function flashQueryFeedback(kind) {
  clearQueryFeedback();
  if (!queryReady || queryBusy) return;

  const isSuccess = kind === "success";
  queryButton.classList.add(isSuccess ? "is-success" : "is-error");
  queryButton.textContent = isSuccess ? "Done" : "Retry";

  queryFeedbackTimer = setTimeout(() => {
    queryButton.classList.remove("is-success", "is-error");
    if (!queryBusy) {
      queryButton.textContent = defaultQueryButtonLabel;
    }
    queryFeedbackTimer = null;
  }, 900);
}

function setUploadBusy(isBusy) {
  if (!uploadButton) return;
  const busy = Boolean(isBusy);
  uploadForm.setAttribute("aria-busy", String(busy));
  uploadButton.disabled = busy;
  uploadButton.dataset.busy = String(busy);
  uploadButton.textContent = busy ? "Indexing..." : defaultUploadButtonLabel;
  fileInput.disabled = busy;
}

function clearAnswer() {
  answerEl.classList.remove("is-updated");
  answerEl.textContent = "";
}

function setAnswerText(text) {
  clearAnswer();
  answerEl.textContent = text;
}

function appendInline(container, text, citationMap) {
  // Safe: we only create text nodes and specific elements (no HTML parsing).
  const parts = String(text).split("`");
  for (let i = 0; i < parts.length; i += 1) {
    const segment = parts[i];
    if (i % 2 === 1) {
      const code = document.createElement("code");
      code.textContent = segment;
      container.appendChild(code);
      continue;
    }

    const boldSplit = segment.split("**");
    for (let j = 0; j < boldSplit.length; j += 1) {
      const boldSeg = boldSplit[j];
      const target = j % 2 === 1 ? document.createElement("strong") : null;
      const host = target || container;

      // Highlight citations like [doc:idx:hash] as pills.
      const citeRegex = /\[[^\]]+\]/g;
      let last = 0;
      let m;
      while ((m = citeRegex.exec(boldSeg)) !== null) {
        const start = m.index;
        const end = start + m[0].length;
        if (start > last) host.appendChild(document.createTextNode(boldSeg.slice(last, start)));
        const raw = m[0];
        const chunkId = raw.slice(1, -1);
        const n = citationMap?.get(chunkId);
        if (n) {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "cite cite-btn";
          btn.dataset.chunkId = chunkId;
          btn.textContent = `[${n}]`;
          host.appendChild(btn);
        } else {
          const cite = document.createElement("span");
          cite.className = "cite cite-raw";
          cite.textContent = raw;
          host.appendChild(cite);
        }
        last = end;
      }
      if (last < boldSeg.length) host.appendChild(document.createTextNode(boldSeg.slice(last)));

      if (target) container.appendChild(target);
    }
  }
}

function renderAnswerMarkdown(text) {
  clearAnswer();
  const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");

  let i = 0;
  let para = [];

  function flushParagraph() {
    if (para.length === 0) return;
    const p = document.createElement("p");
    appendInline(p, para.join("\n").trim(), currentCitationMap);
    answerEl.appendChild(p);
    para = [];
  }

  function readList(kind) {
    const list = document.createElement(kind);
    while (i < lines.length) {
      const line = lines[i];
      const ulMatch = line.match(/^\s*[-*]\s+(.*)$/);
      const olMatch = line.match(/^\s*\d+\.\s+(.*)$/);
      const match = kind === "ul" ? ulMatch : olMatch;
      if (!match) break;
      const li = document.createElement("li");
      appendInline(li, match[1], currentCitationMap);
      list.appendChild(li);
      i += 1;
    }
    answerEl.appendChild(list);
  }

  while (i < lines.length) {
    const line = lines[i];

    // Code fence
    if (line.startsWith("```")) {
      flushParagraph();
      const fenceLang = line.slice(3).trim();
      i += 1;
      const codeLines = [];
      while (i < lines.length && !lines[i].startsWith("```")) {
        codeLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length && lines[i].startsWith("```")) i += 1;

      const pre = document.createElement("pre");
      const code = document.createElement("code");
      if (fenceLang) code.dataset.lang = fenceLang;
      code.textContent = codeLines.join("\n");
      pre.appendChild(code);
      answerEl.appendChild(pre);
      continue;
    }

    // Blank line -> paragraph break
    if (!line.trim()) {
      flushParagraph();
      i += 1;
      continue;
    }

    // Headings
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushParagraph();
      const level = Math.min(6, heading[1].length);
      const h = document.createElement(`h${Math.max(1, level)}`);
      appendInline(h, heading[2], currentCitationMap);
      answerEl.appendChild(h);
      i += 1;
      continue;
    }

    // Blockquote
    const quote = line.match(/^\s*>\s?(.*)$/);
    if (quote) {
      flushParagraph();
      const bq = document.createElement("blockquote");
      appendInline(bq, quote[1], currentCitationMap);
      answerEl.appendChild(bq);
      i += 1;
      continue;
    }

    // Lists
    if (/^\s*[-*]\s+/.test(line)) {
      flushParagraph();
      readList("ul");
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      flushParagraph();
      readList("ol");
      continue;
    }

    para.push(line);
    i += 1;
  }

  flushParagraph();
  answerEl.classList.add("is-updated");
}

function shortPath(path) {
  if (!path) return "";
  const parts = String(path).split(/[\\/]/);
  return parts[parts.length - 1];
}

function stageIndex(stage) {
  return UPLOAD_STAGE_ORDER.indexOf(stage);
}

function paintUploadSteps(stage, error) {
  const currentIndex = stageIndex(stage);
  for (const stepEl of uploadStepEls) {
    const idx = stageIndex(stepEl.dataset.step || "");
    stepEl.classList.remove("is-pending", "is-active", "is-complete", "is-error");

    if (error) {
      if (idx < currentIndex) {
        stepEl.classList.add("is-complete");
      } else if (idx === currentIndex || (currentIndex < 0 && idx === 0)) {
        stepEl.classList.add("is-error");
      } else {
        stepEl.classList.add("is-pending");
      }
      continue;
    }

    if (stage === "idle" || currentIndex < 0) {
      stepEl.classList.add("is-pending");
      continue;
    }

    if (idx < currentIndex) {
      stepEl.classList.add("is-complete");
    } else if (idx === currentIndex) {
      stepEl.classList.add(stage === "complete" ? "is-complete" : "is-active");
    } else {
      stepEl.classList.add("is-pending");
    }
  }
}

function renderUploadProgress(progress) {
  const rawStage = progress?.stage || "idle";
  const stage = rawStage === "error" ? progress?.failed_stage || "uploading" : rawStage;
  const percent = Math.max(0, Math.min(100, Number(progress?.percent || 0)));
  const message = progress?.message || "Waiting for upload.";
  const error = progress?.error || null;

  uploadProgressBarEl.style.width = `${percent}%`;
  uploadPercentLabelEl.textContent = `${percent}%`;
  uploadProgressMessageEl.textContent = error ? `${message} (${error})` : message;
  paintUploadSteps(stage, error);
}

function renderMatches(matches) {
  matchesEl.innerHTML = "";
  if (!matches || matches.length === 0) {
    matchesEl.textContent = "No matches available.";
    return;
  }

  let i = 0;
  for (const match of matches) {
    const card = document.createElement("article");
    card.className = "match";
    card.style.animationDelay = `${Math.min(180, i * 14)}ms`;
    card.dataset.chunkId = match.chunk_id || "";
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", `Open match #${match.rank}`);

    const meta = document.createElement("div");
    meta.className = "match-meta";
    const score = Number(match.rrf_score ?? 0).toFixed(5);
    meta.textContent = `#${match.rank} | score ${score} | ${shortPath(match.source_path)}`;

    const text = document.createElement("p");
    text.className = "match-text";
    text.textContent = match.text_preview || "";

    card.appendChild(meta);
    card.appendChild(text);
    matchesEl.appendChild(card);
    i += 1;
  }
}

matchesEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const card = target.closest?.(".match");
  if (!(card instanceof HTMLElement)) return;
  const chunkId = card.dataset?.chunkId;
  if (chunkId) openEvidencePreview(chunkId);
});

matchesEl.addEventListener("keydown", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  if (event.key !== "Enter" && event.key !== " ") return;
  const card = target.closest?.(".match");
  if (!(card instanceof HTMLElement)) return;
  event.preventDefault();
  const chunkId = card.dataset?.chunkId;
  if (chunkId) openEvidencePreview(chunkId);
});

function renderSources(sources) {
  sourcesEl.innerHTML = "";
  if (!sources || sources.length === 0) {
    return;
  }

  sourcesEl.style.animation = "none";
  void sourcesEl.offsetWidth;
  sourcesEl.style.animation = "reveal 0.35s ease both";

  let n = 1;
  currentCitationMap = new Map();
  for (const source of sources) {
    currentCitationMap.set(source.chunk_id, n);
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "source-link";
    btn.dataset.chunkId = source.chunk_id;
    btn.textContent = `[${n}] ${shortPath(source.source_path)} :: ${source.chunk_id}`;
    li.appendChild(btn);
    sourcesEl.appendChild(li);
    n += 1;
  }
}

async function openEvidencePreview(chunkId) {
  if (!chunkId) return;
  try {
    const response = await fetch(`/api/chunk?chunk_id=${encodeURIComponent(chunkId)}`);
    const payload = await readResponsePayload(response);
    if (!response.ok) {
      throw new Error(payload.error || "Failed to load evidence.");
    }

    evidenceTitleEl.textContent = `Evidence [${currentCitationMap.get(chunkId) || "?"}]`;
    evidenceMetaEl.textContent = `${shortPath(payload.source_path)} | chunk ${payload.chunk_index}`;
    evidenceTextEl.textContent = payload.text || "";
    evidencePreviewEl.hidden = false;
    evidencePreviewEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    setStatusWithKind(error.message, "error");
  }
}

function closeEvidencePreview() {
  evidencePreviewEl.hidden = true;
  evidenceTextEl.textContent = "";
}

evidenceCloseEl.addEventListener("click", closeEvidencePreview);

answerEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const chunkId = target.dataset?.chunkId;
  if (chunkId && target.classList.contains("cite-btn")) {
    openEvidencePreview(chunkId);
  }
});

sourcesEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const chunkId = target.dataset?.chunkId;
  if (chunkId && target.classList.contains("source-link")) {
    openEvidencePreview(chunkId);
  }
});

function setIndexedMeta(filename, chunks) {
  indexedFileEl.textContent = filename || "None";
  chunkCountEl.textContent = `Chunks: ${Number(chunks || 0)}`;
}

function setQueryEnabled(enabled) {
  queryReady = Boolean(enabled);
  if (!queryReady) {
    clearQueryFeedback();
  }
  syncQueryControlState();
}

function stopUploadProgressPolling() {
  if (uploadPollTimer) {
    clearInterval(uploadPollTimer);
    uploadPollTimer = null;
  }
}

async function fetchUploadProgress() {
  try {
    const response = await fetch("/api/upload-progress");
    const payload = await readResponsePayload(response);
    if (!response.ok) return null;

    if (payload.run_id && payload.run_id > activeUploadRunId) {
      activeUploadRunId = payload.run_id;
    }
    if (activeUploadRunId && payload.run_id && payload.run_id < activeUploadRunId) {
      return payload;
    }

    renderUploadProgress(payload);
    setUploadBusy(Boolean(payload.active));

    if (!payload.active && ["idle", "complete", "error"].includes(payload.stage)) {
      stopUploadProgressPolling();
    }
    return payload;
  } catch (_error) {
    return null;
  }
}

function startUploadProgressPolling(runIdHint = 0) {
  if (runIdHint > activeUploadRunId) {
    activeUploadRunId = runIdHint;
  }
  stopUploadProgressPolling();
  fetchUploadProgress();
  uploadPollTimer = setInterval(fetchUploadProgress, 550);
}

async function loadInitialStatus() {
  try {
    const response = await fetch("/api/status");
    const payload = await readResponsePayload(response);
    if (!response.ok) return;

    setIndexedMeta(payload.indexed_file || "None", payload.chunk_count || 0);
    renderUploadProgress(payload.upload_progress || { stage: "idle", percent: 0, message: "Waiting for upload." });
    setUploadBusy(Boolean(payload.upload_progress?.active));
    setQueryEnabled(Boolean(payload.ready));

    if (payload.upload_progress?.run_id) {
      activeUploadRunId = payload.upload_progress.run_id;
    }
    if (payload.upload_progress?.active) {
      setQueryEnabled(false);
      startUploadProgressPolling(activeUploadRunId);
    }
  } catch (_error) {
    // Non-critical on first paint.
  }
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files?.[0];
  if (!file) {
    setStatus("Pick a file first.");
    return;
  }

  const nextRunId = activeUploadRunId + 1;
  activeUploadRunId = nextRunId;
  setUploadBusy(true);
  renderUploadProgress({
    run_id: nextRunId,
    stage: "uploading",
    message: "Uploading file to server...",
    percent: 6,
    active: true,
    error: null,
  });
  startUploadProgressPolling(nextRunId);
  setQueryEnabled(false);
  setStatusWithKind("Upload started. Tracking indexing stages...", "info");

  const data = new FormData();
  data.append("file", file);

  try {
    const response = await fetch("/api/upload", { method: "POST", body: data });
    const payload = await readResponsePayload(response);
    if (!response.ok) {
      throw new Error(payload.error || "Upload failed.");
    }

    if (payload.run_id && payload.run_id > activeUploadRunId) {
      activeUploadRunId = payload.run_id;
    }
    setIndexedMeta(payload.filename, payload.chunks_indexed);
    await fetchUploadProgress();
    setQueryEnabled(true);
    setStatusWithKind(
      `Indexed ${payload.filename} | chunks: ${payload.chunks_indexed} | ${payload.ingest_seconds}s`
      , "ok"
    );
  } catch (error) {
    stopUploadProgressPolling();
    renderUploadProgress({
      run_id: activeUploadRunId,
      stage: "error",
      message: "Upload failed.",
      percent: 100,
      active: false,
      error: error.message,
    });
    setQueryEnabled(false);
    setStatusWithKind(error.message, "error");
  } finally {
    setUploadBusy(false);
  }
});

queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) {
    setStatus("Enter a query.");
    return;
  }

  const runId = ++activeQueryRun;
  let queryOutcome = null;
  setQueryBusy(true);
  setStatusWithKind("Finding top 20 matches...", "info");
  setAnswerText("Generating response...");
  matchesEl.textContent = "Searching...";
  sourcesEl.innerHTML = "";

  try {
    const retrieveResponse = await fetch("/api/retrieve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const retrievePayload = await readResponsePayload(retrieveResponse);
    if (!retrieveResponse.ok) {
      throw new Error(retrievePayload.error || "Retrieve failed.");
    }
    if (runId !== activeQueryRun) {
      return;
    }

    renderMatches(retrievePayload.matches || []);
    setStatusWithKind("Top matches ready. Generating LLM response...", "info");

    const answerResponse = await fetch("/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const answerPayload = await readResponsePayload(answerResponse);
    if (!answerResponse.ok) {
      throw new Error(answerPayload.error || "Answer failed.");
    }
    if (runId !== activeQueryRun) {
      return;
    }

    renderSources(answerPayload.sources || []);
    renderAnswerMarkdown(answerPayload.answer || "");
    setStatusWithKind("Query complete.", "ok");
    queryOutcome = "success";
  } catch (error) {
    if (runId !== activeQueryRun) {
      return;
    }
    setAnswerText("");
    setStatusWithKind(error.message, "error");
    queryOutcome = "error";
  } finally {
    if (runId === activeQueryRun) {
      setQueryBusy(false);
      if (queryOutcome) {
        flashQueryFeedback(queryOutcome);
      }
    }
  }
});

setUploadBusy(false);
setQueryEnabled(false);
loadInitialStatus();
