(function () {
  "use strict";

  const state = {
    bundleMeta: null,
    caseSummaries: [],
    filteredIndices: [],
    caseIndex: 0,
    suspiciousOnly: false,
    undecidedOnly: false,
    hideDropped: false,
    showPitchLabels: true,
    fileMap: new Map(),
    detailCache: new Map(),
    legacyPayload: null,
    decisions: {},
    renderVersion: 0,
    bundleLabel: "",
  };

  const elements = {
    openDirBtn: document.getElementById("openDirBtn"),
    openLegacyBtn: document.getElementById("openLegacyBtn"),
    prevBtn: document.getElementById("prevBtn"),
    nextBtn: document.getElementById("nextBtn"),
    markKeepBtn: document.getElementById("markKeepBtn"),
    markDropBtn: document.getElementById("markDropBtn"),
    clearDecisionBtn: document.getElementById("clearDecisionBtn"),
    exportDecisionsBtn: document.getElementById("exportDecisionsBtn"),
    statusText: document.getElementById("statusText"),
    metaSummary: document.getElementById("metaSummary"),
    searchInput: document.getElementById("searchInput"),
    searchBtn: document.getElementById("searchBtn"),
    suspiciousOnlyToggle: document.getElementById("suspiciousOnlyToggle"),
    undecidedOnlyToggle: document.getElementById("undecidedOnlyToggle"),
    hideDroppedToggle: document.getElementById("hideDroppedToggle"),
    showPitchLabelsToggle: document.getElementById("showPitchLabelsToggle"),
    bundleDirInput: document.getElementById("bundleDirInput"),
    legacyFileInput: document.getElementById("legacyFileInput"),
    caseTitle: document.getElementById("caseTitle"),
    caseSubtitle: document.getElementById("caseSubtitle"),
    caseMeta: document.getElementById("caseMeta"),
    caseDecision: document.getElementById("caseDecision"),
    bundleInfo: document.getElementById("bundleInfo"),
    timelineContainer: document.getElementById("timelineContainer"),
    basicInfo: document.getElementById("basicInfo"),
    keyInfo: document.getElementById("keyInfo"),
    phraseInfo: document.getElementById("phraseInfo"),
    flagList: document.getElementById("flagList"),
    frameTable: document.getElementById("frameTable"),
    boundaryTable: document.getElementById("boundaryTable"),
    tooltip: document.getElementById("tooltip"),
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function setStatus(message) {
    elements.statusText.textContent = message;
  }

  function midiPitchLabel(pitch) {
    const safePitch = Number(pitch) || 0;
    const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    const octave = Math.floor(safePitch / 12) - 1;
    return `${names[((safePitch % 12) + 12) % 12]}${octave}`;
  }

  function keyColor(keyName) {
    if (!keyName || keyName === "uncertain") {
      return "rgba(141, 141, 141, 0.78)";
    }
    const rootOrder = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    const normalized = String(keyName);
    let root = normalized;
    let isMinor = false;
    if (normalized.includes(":")) {
      const parts = normalized.split(":");
      root = parts[0];
      isMinor = parts[1] === "min";
    } else if (normalized.endsWith("_MAJ")) {
      root = normalized.slice(4, -4).replaceAll("_SHARP", "#").replaceAll("_", "");
    } else if (normalized.endsWith("_MIN")) {
      root = normalized.slice(4, -4).replaceAll("_SHARP", "#").replaceAll("_", "");
      isMinor = true;
    }
    const rootIndex = rootOrder.indexOf(root);
    const hue = rootIndex >= 0 ? rootIndex * 30 : 0;
    const lightness = isMinor ? 44 : 58;
    return `hsla(${hue}, 62%, ${lightness}%, 0.80)`;
  }

  function supportOpacity(value) {
    const safe = Math.max(0, Number(value) || 0);
    return Math.min(0.95, Math.max(0.20, safe / 2.5));
  }

  function decisionForCase(caseId) {
    return state.decisions[String(caseId)] || "undecided";
  }

  function setDecision(caseId, decision) {
    if (decision === "undecided") {
      delete state.decisions[String(caseId)];
    } else {
      state.decisions[String(caseId)] = decision;
    }
  }

  function attachTooltip(node, htmlContent) {
    node.addEventListener("mouseenter", () => {
      elements.tooltip.style.display = "block";
      elements.tooltip.innerHTML = htmlContent;
    });
    node.addEventListener("mousemove", (event) => {
      elements.tooltip.style.left = `${event.clientX + 14}px`;
      elements.tooltip.style.top = `${event.clientY + 14}px`;
    });
    node.addEventListener("mouseleave", () => {
      elements.tooltip.style.display = "none";
    });
  }

  function createKvRows(rows) {
    return rows
      .map(([label, value]) => `<div class="label">${escapeHtml(label)}</div><div>${value}</div>`)
      .join("");
  }

  function normalizeRelativePath(file) {
    const raw = String(file.webkitRelativePath || file.name || "").replaceAll("\\", "/");
    if (!raw.includes("/")) {
      return raw;
    }
    const parts = raw.split("/");
    return parts.slice(1).join("/");
  }

  async function parseJsonFile(file) {
    return JSON.parse(await file.text());
  }

  function resetBundleState() {
    state.bundleMeta = null;
    state.caseSummaries = [];
    state.filteredIndices = [];
    state.caseIndex = 0;
    state.fileMap = new Map();
    state.detailCache = new Map();
    state.legacyPayload = null;
    state.decisions = {};
    state.bundleLabel = "";
  }

  async function loadBundleFromDirectory(fileList) {
    resetBundleState();
    const files = Array.from(fileList);
    for (const file of files) {
      state.fileMap.set(normalizeRelativePath(file), file);
    }
    state.bundleLabel = files.length ? String(files[0].webkitRelativePath || "").split("/")[0] : "";
    const indexFile = state.fileMap.get("index.json");
    if (!indexFile) {
      setStatus("目录中未找到 index.json，请选择构建脚本输出的完整目录。");
      return;
    }
    const indexPayload = await parseJsonFile(indexFile);
    state.bundleMeta = indexPayload.meta || {};
    state.caseSummaries = Array.isArray(indexPayload.cases) ? indexPayload.cases : [];
    setStatus(`已加载数据目录：${state.bundleLabel || "未命名目录"}`);
    render();
  }

  async function loadBundleFromSingleJson(file) {
    resetBundleState();
    const payload = await parseJsonFile(file);
    if (!Array.isArray(payload.cases)) {
      setStatus("选中的 JSON 不是合法的 review 数据。");
      return;
    }
    state.bundleMeta = payload.meta || {};
    if (payload.meta && Number(payload.meta.bundle_version) === 2) {
      setStatus("index.json 需要配合同目录下的 cases/ 一起打开，请改用“选择数据目录”。");
      return;
    }
    state.legacyPayload = payload;
    state.caseSummaries = payload.cases.map((item) => {
      const bars = Array.isArray(item.bars) ? item.bars : [];
      const notes = Array.isArray(item.notes) ? item.notes : [];
      return {
        case_id: item.case_id,
        source_kind: item.source_kind,
        title: item.title,
        subtitle: item.subtitle,
        source_path: item.source_path,
        meta: item.meta || {},
        stats: {
          bar_count: bars.length,
          note_count: notes.length,
          token_count: Array.isArray(item.tokens) ? item.tokens.length : 0,
          positions_per_bar: Number(payload.meta?.positions_per_bar || 32),
        },
        key_summary: {
          initial_key: item.key_analysis?.initial_key || "uncertain",
          dominant_key: item.key_analysis?.dominant_key || "uncertain",
          dominant_key_coverage: Number(item.key_analysis?.dominant_key_coverage || 0),
          timeline_summary: item.key_analysis?.timeline_summary || "uncertain",
          modulation_count: Array.isArray(item.key_analysis?.modulation_points) ? item.key_analysis.modulation_points.length : 0,
          segment_count: Array.isArray(item.key_analysis?.segments) ? item.key_analysis.segments.length : 0,
          frame_count: Array.isArray(item.key_analysis?.frames) ? item.key_analysis.frames.length : 0,
        },
        phrase_summary: {
          boundary_count: Array.isArray(item.phrase_analysis?.boundaries) ? item.phrase_analysis.boundaries.length : 0,
          phrase_span_count: Array.isArray(item.phrase_analysis?.phrase_spans) ? item.phrase_analysis.phrase_spans.length : 0,
          mean_phrase_bars: Number(item.phrase_analysis?.mean_phrase_bars || 0),
        },
        debug_flags: item.debug_flags || {},
      };
    });
    setStatus(`已加载单文件 JSON：${file.name}`);
    render();
  }

  async function currentCaseDetail(summary) {
    if (!summary) {
      return null;
    }
    const cacheKey = String(summary.case_id);
    if (state.detailCache.has(cacheKey)) {
      return state.detailCache.get(cacheKey);
    }
    let detail = null;
    if (state.legacyPayload) {
      detail = (state.legacyPayload.cases || []).find((item) => String(item.case_id) === cacheKey) || null;
    } else if (summary.detail_path) {
      const file = state.fileMap.get(String(summary.detail_path));
      if (!file) {
        setStatus(`缺少详情文件：${summary.detail_path}`);
        return null;
      }
      detail = await parseJsonFile(file);
    }
    if (detail) {
      state.detailCache.set(cacheKey, detail);
    }
    return detail;
  }

  function buildFilteredIndices() {
    state.filteredIndices = [];
    state.caseSummaries.forEach((item, index) => {
      const suspicious = Boolean(item?.debug_flags?.is_suspicious);
      const decision = decisionForCase(item.case_id);
      if (state.suspiciousOnly && !suspicious) {
        return;
      }
      if (state.undecidedOnly && decision !== "undecided") {
        return;
      }
      if (state.hideDropped && decision === "drop") {
        return;
      }
      state.filteredIndices.push(index);
    });
    if (state.caseIndex >= state.filteredIndices.length) {
      state.caseIndex = Math.max(0, state.filteredIndices.length - 1);
    }
  }

  function currentSummary() {
    if (!state.filteredIndices.length) {
      return null;
    }
    return state.caseSummaries[state.filteredIndices[state.caseIndex]] || null;
  }

  function updateDecisionBadge(summary) {
    const decision = decisionForCase(summary.case_id);
    elements.caseDecision.className = `decision-pill ${decision}`;
    elements.caseDecision.textContent = {
      undecided: "未标记",
      keep: "保留",
      drop: "剔除",
    }[decision];
  }

  function renderBundleInfo() {
    const sourceSummary = state.bundleMeta?.source_summary || {};
    const rows = [
      ["bundle_version", escapeHtml(state.bundleMeta?.bundle_version ?? "legacy")],
      ["来源模式", escapeHtml(sourceSummary.mode || "unknown")],
      ["目录标签", escapeHtml(state.bundleLabel || "单文件")],
      ["case_count", escapeHtml(state.bundleMeta?.case_count ?? 0)],
      ["可疑样本", escapeHtml(state.bundleMeta?.suspicious_case_count ?? 0)],
    ];
    elements.bundleInfo.innerHTML = createKvRows(rows);
  }

  function renderPanels(summary, detail) {
    const flags = summary.debug_flags || {};
    const basicRows = [
      ["来源类型", `<code>${escapeHtml(summary.source_kind)}</code>`],
      ["源文件", `<code>${escapeHtml(summary.source_path)}</code>`],
      ["case_id", `<code>${escapeHtml(summary.case_id)}</code>`],
      ["小节数", escapeHtml(summary.stats?.bar_count ?? 0)],
      ["音符数", escapeHtml(summary.stats?.note_count ?? 0)],
      ["token 数", escapeHtml(summary.stats?.token_count ?? 0)],
    ];
    const keyRows = [
      ["initial_key", `<code>${escapeHtml(summary.key_summary?.initial_key)}</code>`],
      ["dominant_key", `<code>${escapeHtml(summary.key_summary?.dominant_key)}</code>`],
      ["主调覆盖率", `${((Number(summary.key_summary?.dominant_key_coverage) || 0) * 100).toFixed(1)}%`],
      ["转调次数", escapeHtml(summary.key_summary?.modulation_count ?? 0)],
      ["时间线摘要", `<code>${escapeHtml(summary.key_summary?.timeline_summary)}</code>`],
    ];
    const phraseRows = [
      ["边界数", escapeHtml(summary.phrase_summary?.boundary_count ?? 0)],
      ["乐句段数", escapeHtml(summary.phrase_summary?.phrase_span_count ?? 0)],
      ["平均句长", `${Number(summary.phrase_summary?.mean_phrase_bars || 0).toFixed(2)} bars`],
      ["密集边界", escapeHtml((flags.dense_phrase_boundaries || []).length)],
      ["超长句", escapeHtml((flags.long_phrase_spans || []).length)],
    ];
    elements.basicInfo.innerHTML = createKvRows(basicRows);
    elements.keyInfo.innerHTML = createKvRows(keyRows);
    elements.phraseInfo.innerHTML = createKvRows(phraseRows);

    const flagNames = Array.isArray(flags.flag_names) ? flags.flag_names : [];
    elements.flagList.innerHTML = flagNames.length
      ? flagNames.map((item) => `<span class="pill danger">${escapeHtml(item)}</span>`).join("")
      : '<span class="pill">未发现可疑标记</span>';

    const meta = summary.meta || {};
    const metaParts = [];
    if (meta.row_id !== undefined && meta.row_id !== null) {
      metaParts.push(`row_id=${meta.row_id}`);
    }
    if (meta.bucket) {
      metaParts.push(`bucket=${meta.bucket}`);
    }
    if (meta.token_origin) {
      metaParts.push(`token_origin=${meta.token_origin}`);
    }
    elements.caseMeta.textContent = metaParts.join(" | ");
    updateDecisionBadge(summary);
  }

  function renderTables(detail) {
    const frameRows = (detail.key_analysis?.frames || []).map((frame) => `
      <tr>
        <td><code>${escapeHtml(frame.start_bar)}:${escapeHtml(frame.start_pos)}</code></td>
        <td><code>${escapeHtml(frame.end_bar)}:${escapeHtml(frame.end_pos)}</code></td>
        <td><code>${escapeHtml(frame.best_key)}</code></td>
        <td><code>${escapeHtml(frame.raw_key)}</code></td>
        <td>${Number(frame.best_score).toFixed(3)}</td>
        <td>${Number(frame.margin_to_second).toFixed(3)}</td>
        <td>${Number(frame.smoothed_support).toFixed(3)}</td>
      </tr>
    `).join("");
    elements.frameTable.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>起点</th>
            <th>终点</th>
            <th>best_key</th>
            <th>raw_key</th>
            <th>best_score</th>
            <th>margin</th>
            <th>support</th>
          </tr>
        </thead>
        <tbody>${frameRows || '<tr><td colspan="7">无调性帧</td></tr>'}</tbody>
      </table>
    `;

    const scoreByBar = new Map((detail.phrase_analysis?.boundary_scores || []).map((item) => [Number(item.bar_index), item]));
    const boundaryRows = (detail.phrase_analysis?.boundaries || []).map((boundary) => {
      const scoreItem = scoreByBar.get(Number(boundary.bar_index));
      return `
        <tr>
          <td><code>${escapeHtml(boundary.bar_index)}</code></td>
          <td><code>${escapeHtml(boundary.anchor_pos)}</code></td>
          <td><code>${escapeHtml(boundary.source_rule || "-")}</code></td>
          <td>${boundary.score !== null && boundary.score !== undefined ? Number(boundary.score).toFixed(3) : (scoreItem ? Number(scoreItem.score).toFixed(3) : "-")}</td>
          <td>${escapeHtml((boundary.source_reasons || (scoreItem ? scoreItem.reasons : []) || []).join(", ")) || "-"}</td>
        </tr>
      `;
    }).join("");
    elements.boundaryTable.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>边界 bar</th>
            <th>anchor_pos</th>
            <th>规则</th>
            <th>score</th>
            <th>reasons</th>
          </tr>
        </thead>
        <tbody>${boundaryRows || '<tr><td colspan="5">无乐句边界</td></tr>'}</tbody>
      </table>
    `;
  }

  function renderTimeline(summary, detail) {
    const notes = Array.isArray(detail.notes) ? detail.notes : [];
    const keyFrames = Array.isArray(detail.key_analysis?.frames) ? detail.key_analysis.frames : [];
    const keySegments = Array.isArray(detail.key_analysis?.segments) ? detail.key_analysis.segments : [];
    const phraseBoundaries = Array.isArray(detail.phrase_analysis?.boundaries) ? detail.phrase_analysis.boundaries : [];
    const bars = Array.isArray(detail.bars) ? detail.bars : [];
    const positionsPerBar = Number(state.bundleMeta?.positions_per_bar || summary.stats?.positions_per_bar || 32);
    const barCount = Math.max(1, bars.length || Math.max(...notes.map((item) => Number(item.end_bar) || 0), 0));
    const totalUnits = Math.max(
      barCount * positionsPerBar,
      ...notes.map((item) => Number(item.end_unit) || 0),
      ...keySegments.map((item) => Number(item.end_unit) || 0),
      ...keyFrames.map((item) => Number(item.end_unit) || 0),
    );
    const wrapper = elements.timelineContainer.parentElement;
    const wrapperHeight = wrapper ? wrapper.clientHeight : 0;
    const width = Math.max(1100, totalUnits * 8);
    const height = Math.max(560, wrapperHeight > 0 ? wrapperHeight - 4 : 0);
    const paddingLeft = 60;
    const paddingRight = 24;
    const usableWidth = width - paddingLeft - paddingRight;
    const frameBandTop = 10;
    const frameBandHeight = 28;
    const segmentBandTop = 46;
    const segmentBandHeight = 28;
    const rollTop = 92;
    const boundaryLabelBandHeight = 48;
    const rollHeight = Math.max(420, height - rollTop - boundaryLabelBandHeight);
    const boundaryLabelY = rollTop + rollHeight + 20;
    const minPitch = notes.length ? Math.min(...notes.map((item) => Number(item.pitch) || 0)) : 48;
    const maxPitch = notes.length ? Math.max(...notes.map((item) => Number(item.pitch) || 0)) : 72;
    const pitchSpan = Math.max(1, maxPitch - minPitch + 1);

    function xForUnit(unit) {
      return paddingLeft + ((Number(unit) || 0) / Math.max(1, totalUnits)) * usableWidth;
    }

    function yForPitch(pitch) {
      const relative = (maxPitch - (Number(pitch) || 0)) / pitchSpan;
      return rollTop + (relative * (rollHeight - 16));
    }

    let svg = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;
    svg += `<rect x="0" y="0" width="${width}" height="${height}" fill="rgba(255,255,255,0.65)" rx="18" />`;

    for (let barIndex = 0; barIndex <= barCount; barIndex += 1) {
      const unit = barIndex * positionsPerBar;
      const x = xForUnit(unit);
      svg += `<line x1="${x}" y1="${frameBandTop}" x2="${x}" y2="${rollTop + rollHeight}" stroke="rgba(80, 63, 43, 0.18)" stroke-width="${barIndex === 0 ? 1.6 : 1}" />`;
      if (barIndex < barCount) {
        svg += `<text x="${x + 4}" y="${rollTop - 8}" fill="rgba(80,63,43,0.72)" font-size="11">B${barIndex}</text>`;
      }
    }

    for (let pitch = minPitch; pitch <= maxPitch; pitch += 1) {
      const y = yForPitch(pitch);
      svg += `<line x1="${paddingLeft}" y1="${y + 8}" x2="${width - paddingRight}" y2="${y + 8}" stroke="rgba(91, 108, 124, 0.08)" stroke-width="1" />`;
    }

    keyFrames.forEach((frame, index) => {
      const x = xForUnit(frame.start_unit);
      const frameWidth = Math.max(2, xForUnit(frame.end_unit) - x);
      const fill = frame.is_uncertain ? "rgba(141,141,141,0.72)" : keyColor(frame.best_key);
      svg += `<rect class="frame-cell" data-frame-index="${index}" x="${x}" y="${frameBandTop}" width="${frameWidth}" height="${frameBandHeight}" rx="5" fill="${fill}" fill-opacity="${supportOpacity(frame.smoothed_support)}" />`;
    });

    keySegments.forEach((segment, index) => {
      const x = xForUnit(segment.start_unit);
      const segmentWidth = Math.max(2, xForUnit(segment.end_unit) - x);
      const fill = keyColor(segment.key);
      svg += `<rect class="segment-cell" data-segment-index="${index}" x="${x}" y="${segmentBandTop}" width="${segmentWidth}" height="${segmentBandHeight}" rx="6" fill="${fill}" />`;
      svg += `<text x="${x + 6}" y="${segmentBandTop + 18}" fill="rgba(255,255,255,0.96)" font-size="12">${escapeHtml(segment.key)}</text>`;
    });

    notes.forEach((note, index) => {
      const x = xForUnit(note.start_unit);
      const noteWidth = Math.max(3, xForUnit(note.end_unit) - x);
      const y = yForPitch(note.pitch);
      const label = escapeHtml(midiPitchLabel(note.pitch));
      svg += `<g class="note-cell" data-note-index="${index}">`;
      svg += `<rect x="${x}" y="${y}" width="${noteWidth}" height="14" rx="3" fill="rgba(91,108,124,0.84)" stroke="rgba(53,65,77,0.92)" stroke-width="1" />`;
      if (state.showPitchLabels) {
        svg += `<text x="${x + 2}" y="${y + 10.5}" fill="rgba(255,255,255,0.96)" font-size="9">${label}</text>`;
      }
      svg += `</g>`;
    });

    phraseBoundaries.forEach((boundary, index) => {
      const x = xForUnit(boundary.unit);
      const stroke = boundary.anchor_kind === "mid_bar" ? "rgba(79,138,107,0.96)" : "rgba(35,79,109,0.96)";
      const dash = boundary.anchor_kind === "mid_bar" ? "5 5" : "";
      svg += `<line class="phrase-boundary" data-boundary-index="${index}" x1="${x}" y1="${segmentBandTop}" x2="${x}" y2="${rollTop + rollHeight}" stroke="${stroke}" stroke-width="2.4" stroke-dasharray="${dash}" />`;
      if (boundary.source_label) {
        svg += `<text class="boundary-label" data-boundary-index="${index}" x="${x}" y="${boundaryLabelY}" text-anchor="middle" fill="rgba(80,63,43,0.78)" font-size="10">${escapeHtml(boundary.source_label)}</text>`;
      }
    });

    svg += `<text x="12" y="${frameBandTop + 18}" fill="rgba(80,63,43,0.72)" font-size="12">调性帧</text>`;
    svg += `<text x="12" y="${segmentBandTop + 18}" fill="rgba(80,63,43,0.72)" font-size="12">稳定调性</text>`;
    svg += `<text x="12" y="${rollTop + 18}" fill="rgba(80,63,43,0.72)" font-size="12">音符</text>`;
    svg += `</svg>`;

    elements.timelineContainer.innerHTML = svg;
    const svgRoot = elements.timelineContainer.querySelector("svg");
    if (!svgRoot) {
      return;
    }
    svgRoot.querySelectorAll(".note-cell").forEach((node) => {
      const note = notes[Number(node.getAttribute("data-note-index"))];
      attachTooltip(
        node,
        `<strong>音符</strong><br/>音高: <code>${escapeHtml(midiPitchLabel(note.pitch))}</code> / <code>${escapeHtml(note.pitch)}</code><br/>起点: <code>${escapeHtml(note.start_bar)}:${escapeHtml(note.start_pos)}</code><br/>终点: <code>${escapeHtml(note.end_bar)}:${escapeHtml(note.end_pos)}</code><br/>力度桶: <code>${escapeHtml(note.velocity_bin)}</code>`,
      );
    });
    svgRoot.querySelectorAll(".frame-cell").forEach((node) => {
      const frame = keyFrames[Number(node.getAttribute("data-frame-index"))];
      attachTooltip(
        node,
        `<strong>调性帧</strong><br/>区间: <code>${escapeHtml(frame.start_bar)}:${escapeHtml(frame.start_pos)} → ${escapeHtml(frame.end_bar)}:${escapeHtml(frame.end_pos)}</code><br/>best: <code>${escapeHtml(frame.best_key)}</code><br/>raw: <code>${escapeHtml(frame.raw_key)}</code><br/>score: <code>${Number(frame.best_score).toFixed(3)}</code><br/>margin: <code>${Number(frame.margin_to_second).toFixed(3)}</code>`,
      );
    });
    svgRoot.querySelectorAll(".segment-cell").forEach((node) => {
      const segment = keySegments[Number(node.getAttribute("data-segment-index"))];
      attachTooltip(
        node,
        `<strong>稳定调性段</strong><br/>key: <code>${escapeHtml(segment.key)}</code><br/>区间: <code>${escapeHtml(segment.start_bar)}:${escapeHtml(segment.start_pos)} → ${escapeHtml(segment.end_bar)}:${escapeHtml(segment.end_pos)}</code><br/>长度: <code>${Number(segment.length_bars).toFixed(2)} bars</code>`,
      );
    });
    svgRoot.querySelectorAll(".phrase-boundary").forEach((node) => {
      const boundary = phraseBoundaries[Number(node.getAttribute("data-boundary-index"))];
      attachTooltip(
        node,
        `<strong>乐句边界</strong><br/>bar: <code>${escapeHtml(boundary.bar_index)}</code><br/>anchor_pos: <code>${escapeHtml(boundary.anchor_pos)}</code><br/>类型: <code>${escapeHtml(boundary.anchor_kind)}</code><br/>规则: <code>${escapeHtml(boundary.source_rule || "-")}</code><br/>命中: <code>${escapeHtml((boundary.source_reasons || []).join(", ") || "-")}</code>`,
      );
    });
    svgRoot.querySelectorAll(".boundary-label").forEach((node) => {
      const boundary = phraseBoundaries[Number(node.getAttribute("data-boundary-index"))];
      if (!boundary) {
        return;
      }
      attachTooltip(
        node,
        `<strong>乐句边界规则</strong><br/>规则: <code>${escapeHtml(boundary.source_rule || "-")}</code><br/>命中: <code>${escapeHtml((boundary.source_reasons || []).join(", ") || "-")}</code>`,
      );
    });
  }

  async function render() {
    buildFilteredIndices();
    renderBundleInfo();
    const totalAll = state.caseSummaries.length;
    const totalFiltered = state.filteredIndices.length;
    elements.metaSummary.textContent = `样本 ${totalFiltered} / ${totalAll}`;
    elements.prevBtn.disabled = !totalFiltered || state.caseIndex <= 0;
    elements.nextBtn.disabled = !totalFiltered || state.caseIndex >= totalFiltered - 1;
    elements.markKeepBtn.disabled = !totalFiltered;
    elements.markDropBtn.disabled = !totalFiltered;
    elements.clearDecisionBtn.disabled = !totalFiltered;
    elements.exportDecisionsBtn.disabled = totalAll === 0;

    const summary = currentSummary();
    if (!summary) {
      setStatus("没有符合筛选条件的样本");
      elements.caseTitle.textContent = "没有符合筛选条件的样本";
      elements.caseSubtitle.textContent = "可以调整筛选开关或重新加载数据包。";
      elements.caseMeta.textContent = "";
      elements.caseDecision.className = "decision-pill undecided";
      elements.caseDecision.textContent = "未标记";
      elements.timelineContainer.innerHTML = '<div class="empty">没有可显示的数据</div>';
      elements.basicInfo.innerHTML = "";
      elements.keyInfo.innerHTML = "";
      elements.phraseInfo.innerHTML = "";
      elements.flagList.innerHTML = '<span class="pill">空</span>';
      elements.frameTable.innerHTML = "";
      elements.boundaryTable.innerHTML = "";
      return;
    }

    const absoluteIndex = state.filteredIndices[state.caseIndex] + 1;
    setStatus(`当前 ${state.caseIndex + 1} / ${totalFiltered}（总 ${totalAll}，原始索引 ${absoluteIndex}）`);
    elements.caseTitle.textContent = summary.title || "未命名样本";
    elements.caseSubtitle.textContent = summary.subtitle || "";

    const version = ++state.renderVersion;
    elements.timelineContainer.innerHTML = '<div class="empty">正在加载当前样本详情...</div>';
    const detail = await currentCaseDetail(summary);
    if (version !== state.renderVersion || !detail) {
      return;
    }

    renderPanels(summary, detail);
    renderTimeline(summary, detail);
    renderTables(detail);
  }

  function moveCase(delta) {
    if (!state.filteredIndices.length) {
      return;
    }
    const nextIndex = Math.min(
      Math.max(0, state.caseIndex + delta),
      state.filteredIndices.length - 1,
    );
    state.caseIndex = nextIndex;
    render();
  }

  function jumpToSearch() {
    const query = String(elements.searchInput.value || "").trim().toLowerCase();
    if (!query) {
      return;
    }
    const matchIndex = state.caseSummaries.findIndex((item) => {
      const meta = item.meta || {};
      return [
        item.case_id,
        item.title,
        item.subtitle,
        item.source_path,
        meta.row_id,
        meta.midi_path,
      ].some((field) => String(field ?? "").toLowerCase().includes(query));
    });
    if (matchIndex < 0) {
      setStatus(`未找到：${query}`);
      return;
    }
    buildFilteredIndices();
    const filteredPos = state.filteredIndices.indexOf(matchIndex);
    if (filteredPos >= 0) {
      state.caseIndex = filteredPos;
      render();
      return;
    }
    state.caseIndex = 0;
    state.suspiciousOnly = false;
    state.undecidedOnly = false;
    state.hideDropped = false;
    elements.suspiciousOnlyToggle.checked = false;
    elements.undecidedOnlyToggle.checked = false;
    elements.hideDroppedToggle.checked = false;
    buildFilteredIndices();
    const fallbackPos = state.filteredIndices.indexOf(matchIndex);
    if (fallbackPos >= 0) {
      state.caseIndex = fallbackPos;
      render();
    }
  }

  function exportDecisions() {
    const decisions = state.caseSummaries
      .map((summary) => {
        const decision = decisionForCase(summary.case_id);
        if (decision === "undecided") {
          return null;
        }
        return {
          case_id: summary.case_id,
          decision,
          source_kind: summary.source_kind,
          source_path: summary.source_path,
          title: summary.title,
          subtitle: summary.subtitle,
          relative_path: summary.meta?.relative_path || null,
          midi_path: summary.meta?.midi_path || null,
          row_id: summary.meta?.row_id ?? null,
        };
      })
      .filter(Boolean);
    const payload = {
      meta: {
        bundle_label: state.bundleLabel || "single-json",
        case_count: state.caseSummaries.length,
        exported_decision_count: decisions.length,
      },
      decisions,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "annotation_review_decisions.json";
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function markCurrentDecision(decision) {
    const summary = currentSummary();
    if (!summary) {
      return;
    }
    setDecision(summary.case_id, decision);
    render();
  }

  elements.openDirBtn.addEventListener("click", () => elements.bundleDirInput.click());
  elements.openLegacyBtn.addEventListener("click", () => elements.legacyFileInput.click());
  elements.bundleDirInput.addEventListener("change", async (event) => {
    const files = event.target.files;
    if (files && files.length) {
      await loadBundleFromDirectory(files);
    }
    event.target.value = "";
  });
  elements.legacyFileInput.addEventListener("change", async (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) {
      await loadBundleFromSingleJson(file);
    }
    event.target.value = "";
  });
  elements.prevBtn.addEventListener("click", () => moveCase(-1));
  elements.nextBtn.addEventListener("click", () => moveCase(1));
  elements.markKeepBtn.addEventListener("click", () => markCurrentDecision("keep"));
  elements.markDropBtn.addEventListener("click", () => markCurrentDecision("drop"));
  elements.clearDecisionBtn.addEventListener("click", () => markCurrentDecision("undecided"));
  elements.exportDecisionsBtn.addEventListener("click", exportDecisions);
  elements.searchBtn.addEventListener("click", jumpToSearch);
  elements.searchInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      jumpToSearch();
    }
  });
  elements.suspiciousOnlyToggle.addEventListener("change", () => {
    state.suspiciousOnly = Boolean(elements.suspiciousOnlyToggle.checked);
    state.caseIndex = 0;
    render();
  });
  elements.undecidedOnlyToggle.addEventListener("change", () => {
    state.undecidedOnly = Boolean(elements.undecidedOnlyToggle.checked);
    state.caseIndex = 0;
    render();
  });
  elements.hideDroppedToggle.addEventListener("change", () => {
    state.hideDropped = Boolean(elements.hideDroppedToggle.checked);
    state.caseIndex = 0;
    render();
  });
  elements.showPitchLabelsToggle.addEventListener("change", () => {
    state.showPitchLabels = Boolean(elements.showPitchLabelsToggle.checked);
    render();
  });

  document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement) {
      if (event.key === "Escape") {
        event.target.blur();
      }
      return;
    }
    if (event.key === "ArrowLeft") {
      moveCase(-1);
    } else if (event.key === "ArrowRight") {
      moveCase(1);
    } else if (event.key.toLowerCase() === "f") {
      event.preventDefault();
      elements.searchInput.focus();
      elements.searchInput.select();
    } else if (event.key.toLowerCase() === "k") {
      markCurrentDecision("keep");
    } else if (event.key.toLowerCase() === "d" || event.key === "Delete") {
      markCurrentDecision("drop");
    } else if (event.key.toLowerCase() === "u") {
      markCurrentDecision("undecided");
    }
  });

  renderBundleInfo();
})();
