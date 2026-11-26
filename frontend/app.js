(function () {
  "use strict";

  const CFG = window.ZHIJIANG_CONFIG || {};
  const API_BASE = (CFG.API_BASE || "").replace(/\/+$/, "");

  const MEMBER_THEMES = {
    "沐霂": { color: "#A1D29A", bg: "#F2F9F1", ink: "#2E6A29", fan: "小沐标", official: "沐霂是MUMU呀" },
    "又一": { color: "#E3BA09", bg: "#FDF9E6", ink: "#8A6A00", fan: "酥酥又", official: "又一充电中" },
    "恬豆": { color: "#A7C9D3", bg: "#F1F7F9", ink: "#2C6475", fan: "恬豆包", official: "恬豆发芽了" },
    "梨安": { color: "#E38691", bg: "#FDF2F4", ink: "#9E3342", fan: "向心梨", official: "梨安不迷路" }
  };

  const ICON_MIC = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>`;
  const ICON_SCREEN = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="2" y="3" width="20" height="14" rx="2"></rect><line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line></svg>`;

  function formatSpeakerBadge(label, speakerName, sim, fallbackUpName, speakerLocked) {
    const simText = speakerLocked
      ? "（人工已校对）"
      : (typeof sim === "number" && !Number.isNaN(sim)
        ? `（声纹相似度 ${Math.round(sim * 100)}%）`
        : "");
    const resolvedName = (speakerName || "").trim() || (fallbackUpName || "").trim();
    if (label === "host") {
      const t = getTheme(resolvedName);
      const display = resolvedName ? `${esc(resolvedName)}原声` : "主播原声";
      return `<span class="badge badge-source-host" style="--src-color:${t.color};--src-bg:${t.bg};--src-ink:${t.ink};" title="${speakerLocked ? "人工校对" : "四成员声纹识别"}：${display}${simText}">${ICON_MIC}${display}</span>`;
    }
    if (label === "mixed") {
      const display = resolvedName ? `${esc(resolvedName)}原声 · 含混音` : "主播原声 · 含混音";
      return `<span class="badge badge-source-mixed" title="${speakerLocked ? "人工校对" : "四成员声纹识别"}：本句包含${resolvedName || "主播"}插嘴及外部音轨拼接${simText}">${ICON_MIC}${display}</span>`;
    }
    if (label === "external") {
      return `<span class="badge badge-source-external" title="${speakerLocked ? "人工校对" : "四成员声纹识别"}：与四位主播均不匹配，可能来自视频、影视或游戏音轨${simText}">${ICON_SCREEN}主播无关音轨</span>`;
    }
    return "";
  }

  function formatCtxSourcePill(label, speakerName) {
    const name = (speakerName || "").trim();
    if (label === "host") {
      const t = getTheme(name);
      return `<span class="ctx-src ctx-src-host" style="--src-color:${t.color};--src-bg:${t.bg};--src-ink:${t.ink};">${esc(name || "主播")}</span>`;
    }
    if (label === "mixed") {
      return `<span class="ctx-src ctx-src-mixed">${esc(name ? `${name}·混` : "混音")}</span>`;
    }
    if (label === "external") {
      return `<span class="ctx-src ctx-src-external">无关音轨</span>`;
    }
    return "";
  }

  const state = {
    selectedUps: new Set(),
    currentPage: 1,
    totalPages: 0,
    totalHits: 0,
    pageLimit: 20,
    hasSearched: false,
    announcementsExpanded: false,
    members: [],
    stats: null,
    site: {
      author_name: CFG.AUTHOR_NAME || "M1NG",
      author_home_url: CFG.AUTHOR_HOME_URL || CFG.AUTHOR_BILIBILI_URL || "",
      github_url: CFG.GITHUB_REPO_URL || "https://github.com/M1NGrr/SiXiWanZi-Database-Test",
      upstream_author_name: CFG.UPSTREAM_AUTHOR_NAME || "没品人类之一",
      upstream_author_url: CFG.UPSTREAM_AUTHOR_URL || "https://space.bilibili.com/396896138",
      upstream_repo_url: CFG.UPSTREAM_REPO_URL || "https://github.com/RicojrH/zhijiang-database"
    }
  };

  const $ = (id) => document.getElementById(id);

  function esc(str) {
    return String(str || "").replace(/[&<>"]/g, (ch) => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[ch]
    ));
  }

  function formatNumber(n) {
    return Number(n || 0).toLocaleString("zh-CN");
  }

  function getTheme(upName) {
    return MEMBER_THEMES[upName] || {
      color: "#4B6373",
      bg: "#F1F4F6",
      ink: "#2F4554",
      fan: "四禧丸子",
      official: upName || ""
    };
  }

  function applyThemeAccent() {
    const root = document.documentElement;
    const activeUp = state.selectedUps.size === 1 ? Array.from(state.selectedUps)[0] : "";
    const t = getTheme(activeUp);
    root.style.setProperty("--accent", t.color);
    root.style.setProperty("--accent-bg", t.bg);
    root.style.setProperty("--accent-ink", t.ink);
    updateSearchPlaceholder();
  }

  function updateSearchPlaceholder() {
    const input = $("qInput");
    if (!input) return;
    const ups = Array.from(state.selectedUps);
    if (ups.length === 1) {
      const upName = ups[0];
      const upStat = state.stats && state.stats.by_up
        ? state.stats.by_up.find((u) => u.up_name === upName)
        : null;
      const countText = upStat && upStat.subtitle_lines >= 10000
        ? ` ${Math.round(upStat.subtitle_lines / 10000)} 万+ 条`
        : (upStat ? ` ${formatNumber(upStat.subtitle_lines)} 条` : "");
      input.placeholder = `在 ${upName} 的${countText}字幕中搜索台词、热梗或名场面…（空格分隔多词）`;
    } else if (ups.length > 1) {
      input.placeholder = `在 ${ups.join("、")} 的字幕中搜索台词、热梗或名场面…（空格分隔多词）`;
    } else {
      input.placeholder = "搜台词、搜热梗、搜名场面… 如：黑话 / 爆 / 羞死了 / 四禧丸子（空格分隔多词）";
    }
  }

  function highlightText(text, q, expandedMap) {
    let out = esc(text);
    const terms = new Set((q || "").trim().split(/\s+/).filter(Boolean));
    if (expandedMap) {
      Object.values(expandedMap).forEach((arr) => {
        (arr || []).forEach((v) => terms.add(v));
      });
    }
    const sorted = Array.from(terms).sort((a, b) => b.length - a.length);
    for (const t of sorted) {
      const safe = t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      if (!safe) continue;
      out = out.replace(new RegExp(safe, "gi"), (m) => `<mark>${m}</mark>`);
    }
    return out;
  }

  function hasActiveQueryCondition() {
    const q = $("qInput").value.trim();
    const bvid = $("bvidFilter").value.trim();
    const df = $("dateFrom").value;
    const dt = $("dateTo").value;
    return Boolean(q || bvid || df || dt);
  }

  function showLandingMode() {
    state.hasSearched = false;
    $("landingBoard").classList.remove("hidden");
    $("resultsSection").classList.add("hidden");
  }

  function showResultsMode() {
    state.hasSearched = true;
    $("landingBoard").classList.add("hidden");
    $("resultsSection").classList.remove("hidden");
  }

  function renderMemberChips() {
    const wrap = $("memberStrip");
    const members = state.members.length
      ? state.members
      : Object.keys(MEMBER_THEMES).map((up_name) => ({ up_name, ...MEMBER_THEMES[up_name] }));

    const allActive = state.selectedUps.size === 0 ? "active" : "";
    const allBtn = `
      <span class="strip-label">成员（可多选）</span>
      <button type="button" class="member-chip ${allActive}"
        data-up="" style="--chip-color:#576690;--chip-bg:#F1F3F8;--chip-ink:#1B2232;">
        全部成员
      </button>
    `;
    const memberBtns = members.map((m) => {
      const t = getTheme(m.up_name);
      const color = m.color || t.color;
      const active = state.selectedUps.has(m.up_name) ? "active" : "";
      return `
        <button type="button" class="member-chip ${active}" data-up="${esc(m.up_name)}"
          style="--chip-color:${color};--chip-bg:${t.bg};--chip-ink:${t.ink};">
          <span class="color-dot" style="background:${color};"></span>
          ${esc(m.up_name)}
        </button>
      `;
    }).join("");

    wrap.innerHTML = allBtn + memberBtns;
    wrap.querySelectorAll(".member-chip").forEach((btn) => {
      btn.addEventListener("click", () => toggleMember(btn.dataset.up || "", false));
    });
  }

  function toggleMember(upName, focusInput, exclusive) {
    if (!upName) {
      state.selectedUps.clear();
    } else if (exclusive) {
      state.selectedUps.clear();
      state.selectedUps.add(upName);
    } else {
      if (state.selectedUps.has(upName)) {
        state.selectedUps.delete(upName);
      } else {
        state.selectedUps.add(upName);
      }
      const totalMembers = state.members.length || Object.keys(MEMBER_THEMES).length;
      if (state.selectedUps.size >= totalMembers) {
        state.selectedUps.clear();
      }
    }
    applyThemeAccent();
    renderMemberChips();
    if (hasActiveQueryCondition()) {
      runSearch(1, false);
    } else {
      showLandingMode();
      if (focusInput) {
        $("qInput").focus();
      }
    }
  }

  function formatLocalDateTime(isoStr) {
    if (!isoStr) return "";
    const d = new Date(isoStr);
    if (Number.isNaN(d.getTime())) return String(isoStr);
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }

  function renderAnnouncements(items) {
    const card = $("announcementCard");
    const listEl = $("announcementList");
    const metaEl = $("announcementMeta");
    const toggleBtn = $("announcementToggle");
    if (!card || !listEl) return;

    const list = Array.isArray(items) ? items : [];
    if (!list.length) {
      card.classList.add("hidden");
      listEl.innerHTML = "";
      return;
    }

    card.classList.remove("hidden");
    const previewCount = 3;
    const expanded = state.announcementsExpanded;
    const visible = expanded ? list : list.slice(0, previewCount);

    if (metaEl) {
      metaEl.textContent = list.length > 1 ? `共 ${list.length} 条动态` : (list[0].date || "");
    }

    listEl.innerHTML = visible.map((it) => {
      const level = ["info", "change", "warn"].includes(it.level) ? it.level : "info";
      const tag = esc(it.tag || "公告");
      const dateHtml = it.date ? `<span class="announcement-date">${esc(it.date)}</span>` : "";
      return `
        <li class="announcement-item">
          <span class="announcement-tag tag-${level}">${tag}</span>
          <span class="announcement-text">${esc(it.content)}</span>
          ${dateHtml}
        </li>
      `;
    }).join("");

    if (toggleBtn) {
      if (list.length > previewCount) {
        toggleBtn.classList.remove("hidden");
        toggleBtn.textContent = expanded ? "收起较早公告" : `展开全部 ${list.length} 条公告`;
      } else {
        toggleBtn.classList.add("hidden");
      }
    }
  }

  function renderLandingStatsAndCards() {
    const st = state.stats;
    if (st) {
      $("statTotalLines").textContent = formatNumber(st.subtitles_total);
      $("statTotalParts").textContent = `${formatNumber(st.video_parts_ok)} 个分P`;
      $("statDateRange").textContent = st.date_min && st.date_max
        ? `${st.date_min} ~ ${st.date_max}`
        : "—";

      const latestEl = $("statLatestVideo");
      if (latestEl) {
        const lv = st.latest_video;
        if (lv && lv.bvid) {
          const t = getTheme(lv.up_name);
          const cleanTitle = (lv.title || lv.bvid).replace(/^【直播回放】\s*/, "");
          const partSuffix = lv.page && lv.page > 1 ? ` · P${lv.page}` : "";
          const fullTitle = `${lv.up_name ? `[${lv.up_name}] ` : ""}${lv.title || lv.bvid}${partSuffix}`;
          const upHtml = lv.up_name
            ? `<span class="stat-latest-up" style="--latest-ink:${t.ink};">[${esc(lv.up_name)}]</span>`
            : "";
          const href = esc(lv.video_url || `https://www.bilibili.com/video/${lv.bvid}/`);
          latestEl.innerHTML =
            `<span class="stat-latest-label">最新录播：</span>` +
            `<a href="${href}" target="_blank" rel="noopener noreferrer" title="${esc(fullTitle)}">` +
            `${upHtml}${esc(cleanTitle)}${esc(partSuffix)}</a>`;
        } else {
          latestEl.innerHTML = "";
        }
      }

      const updatedEl = $("statUpdatedAt");
      if (updatedEl) {
        const ts = formatLocalDateTime(st.latest_updated_at);
        updatedEl.textContent = ts ? `数据库同步时间：${ts}` : "";
      }

      renderAnnouncements(st.announcements);
    }

    const byUpMap = {};
    if (st && st.by_up) {
      st.by_up.forEach((row) => { byUpMap[row.up_name] = row; });
    }

    const members = state.members.length
      ? state.members
      : Object.keys(MEMBER_THEMES).map((up_name) => ({ up_name, ...MEMBER_THEMES[up_name] }));

    $("memberGrid").innerHTML = members.map((m) => {
      const t = getTheme(m.up_name);
      const color = m.color || t.color;
      const stat = byUpMap[m.up_name];
      const partsText = stat ? `${formatNumber(stat.ok_parts)} 个分P` : "—";
      const linesText = stat ? `${formatNumber(stat.subtitle_lines)} 条字幕` : "—";
      return `
        <button type="button" class="member-card" data-up="${esc(m.up_name)}"
          style="--m-color:${color};--m-bg:${t.bg};--m-ink:${t.ink};">
          <div class="member-card-top">
            <div class="member-id">
              <span class="member-avatar">${esc(m.up_name.slice(0, 1))}</span>
              <div style="min-width:0;">
                <div class="member-name">${esc(m.up_name)}</div>
                <div class="member-official">${esc(m.official_name || t.official)}</div>
              </div>
            </div>
            <span class="fan-pill">${esc(m.fan_name || t.fan)}</span>
          </div>
          <div class="member-card-stats">
            <span>${partsText}</span>
            <span>${linesText}</span>
          </div>
        </button>
      `;
    }).join("");

    $("memberGrid").querySelectorAll(".member-card").forEach((card) => {
      card.addEventListener("click", () => {
        toggleMember(card.dataset.up || "", true, true);
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
    });

    const hotTagsEl = $("hotTags");
    if (hotTagsEl) {
      const hotList = CFG.HOT_KEYWORDS || [];
      hotTagsEl.innerHTML = hotList.map((kw) => (
        `<button type="button" class="hot-tag" data-kw="${esc(kw)}">${esc(kw)}</button>`
      )).join("");
      hotTagsEl.querySelectorAll(".hot-tag").forEach((tag) => {
        tag.addEventListener("click", () => {
          $("qInput").value = tag.dataset.kw || "";
          syncClearBtn();
          runSearch(1, false);
        });
      });
    }
  }

  function buildPageNumbers(current, total) {
    if (total <= 7) {
      return Array.from({ length: total }, (_, i) => i + 1);
    }
    const pages = [1];
    let left = Math.max(2, current - 2);
    let right = Math.min(total - 1, current + 2);
    if (current <= 3) right = Math.min(total - 1, 5);
    if (current >= total - 2) left = Math.max(2, total - 4);
    if (left > 2) pages.push("...");
    for (let i = left; i <= right; i++) pages.push(i);
    if (right < total - 1) pages.push("...");
    pages.push(total);
    return pages;
  }

  function renderPager(containerId, position) {
    const el = $(containerId);
    if (!el) return;
    if (state.totalPages <= 1) {
      el.classList.add("hidden");
      el.innerHTML = "";
      return;
    }
    el.classList.remove("hidden");
    const cur = state.currentPage;
    const total = state.totalPages;
    const nums = buildPageNumbers(cur, total);

    const numButtons = nums.map((item) => {
      if (item === "...") {
        return `<span class="page-ellipsis">…</span>`;
      }
      const active = item === cur ? "active" : "";
      return `<button type="button" class="page-btn ${active}" data-page="${item}">${item}</button>`;
    }).join("");

    el.innerHTML = `
      <div class="pager-nav">
        <button type="button" class="page-btn" data-page="1" ${cur <= 1 ? "disabled" : ""}>首页</button>
        <button type="button" class="page-btn" data-page="${cur - 1}" ${cur <= 1 ? "disabled" : ""}>上一页</button>
        ${numButtons}
        <button type="button" class="page-btn" data-page="${cur + 1}" ${cur >= total ? "disabled" : ""}>下一页</button>
        <button type="button" class="page-btn" data-page="${total}" ${cur >= total ? "disabled" : ""}>末页</button>
      </div>
      <div class="pager-jump">
        <span>跳至</span>
        <input type="number" min="1" max="${total}" value="${cur}" aria-label="跳转页码" class="jump-input" />
        <span>/ ${formatNumber(total)} 页</span>
        <button type="button" class="page-btn jump-go">跳转</button>
      </div>
    `;

    el.querySelectorAll("button[data-page]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = parseInt(btn.dataset.page, 10);
        if (!Number.isNaN(target) && target >= 1 && target <= total && target !== cur) {
          runSearch(target, position === "bottom");
        }
      });
    });

    const jInput = el.querySelector(".jump-input");
    const jGo = el.querySelector(".jump-go");
    const triggerJump = () => {
      const val = parseInt(jInput.value, 10);
      if (!Number.isNaN(val)) {
        const clamped = Math.max(1, Math.min(total, val));
        runSearch(clamped, position === "bottom");
      }
    };
    jGo.addEventListener("click", triggerJump);
    jInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") triggerJump();
    });
  }

  async function runSearch(page, scrollToResults) {
    if (!hasActiveQueryCondition()) {
      showLandingMode();
      return;
    }

    state.pageLimit = parseInt($("pageSizeFilter").value || "20", 10);
    const targetPage = Math.max(1, page || 1);
    const offset = (targetPage - 1) * state.pageLimit;
    const q = $("qInput").value.trim();

    const params = new URLSearchParams();
    if (q) params.set("q", q);
    if (state.selectedUps.size > 0) {
      params.set("up_name", Array.from(state.selectedUps).join(","));
    }
    const df = $("dateFrom").value;
    const dt = $("dateTo").value;
    const bv = $("bvidFilter").value.trim();
    const sort = $("sortFilter").value;
    const fuzzy = $("fuzzyToggle").checked;
    const sourceEl = $("speakerSourceFilter");
    const speakerSource = sourceEl ? sourceEl.value : "all";

    if (df) params.set("date_from", df);
    if (dt) params.set("date_to", dt);
    if (bv) params.set("bvid", bv);
    params.set("sort", sort);
    params.set("fuzzy", fuzzy ? "true" : "false");
    if (speakerSource && speakerSource !== "all") {
      params.set("speaker_source", speakerSource);
    }
    params.set("limit", String(state.pageLimit));
    params.set("offset", String(offset));

    showResultsMode();
    $("hitList").innerHTML = `<div class="state-box">正在检索语料库…</div>`;
    $("pagerTop").classList.add("hidden");
    $("pagerBottom").classList.add("hidden");

    try {
      const t0 = performance.now();
      const resp = await fetch(`${API_BASE}/api/v1/search?${params.toString()}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const elapsedMs = Math.round(performance.now() - t0);

      const p = data.pagination || {};
      state.totalHits = p.total || 0;
      state.currentPage = p.page || targetPage;
      state.totalPages = p.total_pages || (state.totalHits > 0 ? Math.ceil(state.totalHits / state.pageLimit) : 0);

      const startIdx = state.totalHits === 0 ? 0 : offset + 1;
      const endIdx = offset + (data.items ? data.items.length : 0);
      $("resultSummaryText").textContent =
        `共命中 ${formatNumber(state.totalHits)} 条字幕（第 ${state.currentPage} / ${Math.max(1, state.totalPages)} 页，显示 ${formatNumber(startIdx)}-${formatNumber(endIdx)}，耗时 ${elapsedMs}ms）`;

      const expEl = $("expandedHint");
      const expEntries = Object.entries(data.fuzzy_expanded || {});
      if (expEntries.length) {
        expEl.textContent = "已同音扩展：" + expEntries.map(([k, v]) => `${k} → ${v.join(" / ")}`).join("；");
        expEl.classList.remove("hidden");
      } else {
        expEl.classList.add("hidden");
      }

      if (!data.items || !data.items.length) {
        $("hitList").innerHTML = `
          <div class="state-box">
            没有找到匹配的字幕条目。可以尝试缩短关键词、开启“同音/别称模糊扩展”或切换至“全部成员”。
          </div>
        `;
        return;
      }

      $("hitList").innerHTML = data.items.map((it) => {
        const t = getTheme(it.up_name);
        const spkBadge = formatSpeakerBadge(it.speaker_label, it.speaker_name, it.speaker_sim, it.up_name, it.speaker_locked);
        const cardMod = it.speaker_label === "external" ? " is-external" : "";
        return `
          <article class="hit-card${cardMod}" style="--m-color:${t.color};--m-bg:${t.bg};--m-ink:${t.ink};">
            <div class="hit-meta">
              <span class="badge badge-member">
                <span class="color-dot" style="background:${t.color};"></span>
                ${esc(it.up_name)}
              </span>
              <span class="badge badge-time">${esc(it.start_hms)} ~ ${esc(it.end_hms)}</span>
              ${spkBadge}
              <span>${esc(it.pub_date)}</span>
              <span>${esc(it.bvid)} · P${it.page}</span>
            </div>
            <div class="hit-content">${highlightText(it.content, q, data.fuzzy_expanded)}</div>
            <div class="hit-foot">
              <div class="hit-title">${esc(it.title)}${it.part_title && it.part_title !== it.title ? " · " + esc(it.part_title) : ""}</div>
              <div class="hit-actions">
                <button type="button" class="act-link ctx-btn"
                  data-id="${it.id}" data-bvid="${esc(it.bvid)}" data-page="${it.page}" data-start="${it.start_sec}">
                  前后上下文
                </button>
                <button type="button" class="act-link copy-btn" data-copy="${esc(it.jump_url)}">
                  复制链接
                </button>
                <a class="act-link" href="${esc(it.jump_url)}" target="_blank" rel="noopener noreferrer"
                  title="打开 B 站原视频，若播放器恢复历史进度可手动拖至 ${esc(it.start_hms)}">
                  B站原片 (${esc(it.start_hms)}) ↗
                </a>
              </div>
            </div>
            <div class="ctx-drawer hidden" id="ctx-${it.id}"></div>
          </article>
        `;
      }).join("");

      $("hitList").querySelectorAll(".ctx-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          toggleContext(
            parseInt(btn.dataset.id, 10),
            btn.dataset.bvid,
            parseInt(btn.dataset.page, 10),
            parseFloat(btn.dataset.start)
          );
        });
      });

      $("hitList").querySelectorAll(".copy-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          navigator.clipboard.writeText(btn.dataset.copy || "").then(() => {
            const orig = btn.textContent;
            btn.textContent = "已复制";
            setTimeout(() => { btn.textContent = orig; }, 1200);
          });
        });
      });

      renderPager("pagerTop", "top");
      renderPager("pagerBottom", "bottom");

      if (scrollToResults) {
        const sec = $("resultsSection");
        const topY = sec.getBoundingClientRect().top + window.scrollY - 76;
        window.scrollTo({ top: Math.max(0, topY), behavior: "smooth" });
      }
    } catch (err) {
      $("hitList").innerHTML = `<div class="state-box">检索请求失败：${esc(err.message)}。请确认后端 API 服务已正常运行。</div>`;
    }
  }

  async function toggleContext(subId, bvid, page, startSec) {
    const box = $(`ctx-${subId}`);
    if (!box) return;
    if (!box.classList.contains("hidden")) {
      box.classList.add("hidden");
      return;
    }
    box.classList.remove("hidden");
    box.textContent = "正在加载前后 25 秒上下文…";
    try {
      const qs = new URLSearchParams({
        page: String(page),
        start_sec: String(Math.max(0, startSec - 25)),
        end_sec: String(startSec + 25),
        limit: "30"
      });
      const r = await fetch(`${API_BASE}/api/v1/videos/${encodeURIComponent(bvid)}/subtitles?${qs.toString()}`);
      const data = await r.json();
      if (!data.items || !data.items.length) {
        box.textContent = "暂无上下文记录。";
        return;
      }
      box.innerHTML = data.items.map((x) => `
        <div class="ctx-row ${x.id === subId ? "is-hit" : ""} ${x.speaker_label === "external" ? "is-external" : ""}">
          <span class="ctx-ts">${Number(x.start_sec).toFixed(1)}s</span>
          ${formatCtxSourcePill(x.speaker_label, x.speaker_name)}
          <span>${esc(x.content)}</span>
        </div>
      `).join("");
    } catch (e) {
      box.textContent = "加载上下文失败。";
    }
  }

  function renderAuthorAndGithub() {
    const author = state.site.author_name || "M1NG";
    const homeUrl = CFG.AUTHOR_HOME_URL || state.site.author_bilibili_url || state.site.author_home_url || "";
    const ghUrl = CFG.GITHUB_REPO_URL || state.site.github_url || "";
    const upAuthor = state.site.upstream_author_name || "没品人类之一";
    const upAuthorUrl = state.site.upstream_author_url || "https://space.bilibili.com/396896138";
    const upRepoUrl = state.site.upstream_repo_url || "https://github.com/RicojrH/zhijiang-database";

    const footerAuthor = $("footerAuthorLink");
    const modalAuthor = $("modalAuthorLink");
    footerAuthor.textContent = author;
    modalAuthor.textContent = author;
    if (homeUrl) {
      footerAuthor.href = homeUrl;
      modalAuthor.href = homeUrl;
    } else {
      footerAuthor.removeAttribute("href");
      modalAuthor.removeAttribute("href");
    }

    const footerGh = $("footerGithubLink");
    const modalGh = $("modalGithubLink");
    if (ghUrl) {
      footerGh.href = ghUrl;
      footerGh.classList.remove("hidden");
      modalGh.href = ghUrl;
      modalGh.textContent = ghUrl;
    } else {
      footerGh.classList.add("hidden");
      modalGh.removeAttribute("href");
      modalGh.textContent = "仓库创建后可在 frontend/config.js 填入 GitHub 链接";
    }

    const footerUp = $("footerUpstreamLink");
    if (footerUp) footerUp.href = upRepoUrl;
    const modalUpRepo = $("modalUpstreamRepoLink");
    if (modalUpRepo) modalUpRepo.href = upRepoUrl;
    const modalUpAuthor = $("modalUpstreamAuthorLink");
    if (modalUpAuthor) {
      modalUpAuthor.textContent = `B站 @${upAuthor}`;
      modalUpAuthor.href = upAuthorUrl;
    }
  }

  function syncClearBtn() {
    $("clearInputBtn").classList.toggle("hidden", !$("qInput").value);
  }

  let lastStatsFetchAt = 0;

  async function refreshStats(force) {
    const now = Date.now();
    if (!force && now - lastStatsFetchAt < 30000) return;
    lastStatsFetchAt = now;
    try {
      const sRes = await fetch(`${API_BASE}/api/v1/stats`, { cache: "no-store" });
      if (sRes.ok) {
        state.stats = await sRes.json();
        renderLandingStatsAndCards();
        updateSearchPlaceholder();
      }
    } catch (err) {
      // 忽略后台静默刷新失败
    }
  }

  function resetAll() {
    $("qInput").value = "";
    $("dateFrom").value = "";
    $("dateTo").value = "";
    $("bvidFilter").value = "";
    $("sortFilter").value = "date_desc";
    $("fuzzyToggle").checked = true;
    if ($("speakerSourceFilter")) $("speakerSourceFilter").value = "all";
    state.selectedUps.clear();
    applyThemeAccent();
    renderMemberChips();
    syncClearBtn();
    showLandingMode();
    refreshStats(true);
  }

  async function init() {
    renderMemberChips();
    renderLandingStatsAndCards();
    renderAuthorAndGithub();
    applyThemeAccent();

    $("searchForm").addEventListener("submit", (e) => {
      e.preventDefault();
      if (!hasActiveQueryCondition()) {
        $("qInput").focus();
        return;
      }
      runSearch(1, false);
    });

    $("qInput").addEventListener("input", syncClearBtn);
    $("clearInputBtn").addEventListener("click", () => {
      $("qInput").value = "";
      syncClearBtn();
      $("qInput").focus();
      if (!hasActiveQueryCondition()) showLandingMode();
    });

    $("resetBtn").addEventListener("click", resetAll);
    $("brandHome").addEventListener("click", resetAll);
    const annToggle = $("announcementToggle");
    if (annToggle) {
      annToggle.addEventListener("click", () => {
        state.announcementsExpanded = !state.announcementsExpanded;
        if (state.stats) renderAnnouncements(state.stats.announcements);
      });
    }

    ["dateFrom", "dateTo", "sortFilter", "pageSizeFilter", "speakerSourceFilter", "fuzzyToggle"].forEach((id) => {
      if (!$(id)) return;
      $(id).addEventListener("change", () => {
        if (hasActiveQueryCondition()) runSearch(1, false);
      });
    });

    $("bvidFilter").addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        runSearch(1, false);
      }
    });

    // 关于 / 作者弹窗
    const openModal = () => $("aboutModal").classList.remove("hidden");
    const closeModal = () => $("aboutModal").classList.add("hidden");
    $("aboutBtn").addEventListener("click", openModal);
    $("footerAboutBtn").addEventListener("click", (e) => { e.preventDefault(); openModal(); });
    $("closeModalBtn").addEventListener("click", closeModal);
    $("aboutModal").addEventListener("click", (e) => {
      if (e.target === $("aboutModal")) closeModal();
    });

    try {
      const [mRes, sRes] = await Promise.all([
        fetch(`${API_BASE}/api/v1/members`, { cache: "no-store" }),
        fetch(`${API_BASE}/api/v1/stats`, { cache: "no-store" })
      ]);
      if (mRes.ok) {
        const mData = await mRes.json();
        state.members = mData.members || [];
        if (mData.site) {
          state.site = { ...state.site, ...mData.site };
        }
      }
      if (sRes.ok) {
        state.stats = await sRes.json();
        lastStatsFetchAt = Date.now();
      }
    } catch (err) {
      // 后端不可达时保留默认成员卡片
    }

    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible" && !state.hasSearched) {
        refreshStats(false);
      }
    });

    renderMemberChips();
    renderLandingStatsAndCards();
    renderAuthorAndGithub();
    updateSearchPlaceholder();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
