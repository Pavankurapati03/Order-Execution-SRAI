
window.sendChoiceMessage = function(txt) {
  if (txt && txt.toLowerCase().includes("executive")) {
    var solUrls = {};
    try {
      solUrls = JSON.parse(localStorage.getItem("quantellix_solution_urls") || "{}");
    } catch(e) {}
    if (solUrls["executive"]) {
      window.open(solUrls["executive"], "_blank");
      return;
    }
    var execItem = document.getElementById("btnExecutiveDashboard_executive");
    if (execItem && execItem.getAttribute("data-dash-url")) {
      window.open(execItem.getAttribute("data-dash-url"), "_blank");
      return;
    }
    window.open("/portal/executive_dashboard", "_blank");
    return;
  }
  var solMatch = txt.match(/View\s+(E\d+)\s+Dashboard/i);
  if (solMatch) {
    var solKey = solMatch[1].toLowerCase();
    var solUrls = {};
    try {
      solUrls = JSON.parse(localStorage.getItem("quantellix_solution_urls") || "{}");
    } catch(e) {}
    if (solUrls[solKey]) {
      window.open(solUrls[solKey], "_blank");
      return;
    }
  }
  if (txt === 'View Dashboard' || (txt.indexOf('Dashboard') !== -1 && txt.indexOf('View') !== -1)) {
    var savedUrl = localStorage.getItem("quantellix_dashboard_url");
    if (savedUrl) {
      window.open(savedUrl, "_blank");
      return;
    }
  }
  var input = document.getElementById("chatInput");
  if (input) {
    input.value = txt;
    if (typeof sendMessage === 'function') {
      sendMessage();
    } else {
      var btn = document.getElementById("btnSend");
      if (btn) btn.click();
    }
  }
};

window.submitManualFeatures = function() {
  var checkboxes = document.querySelectorAll('#manual-features-form input[type="checkbox"]:checked');
  var selected = [];
  for (var i = 0; i < checkboxes.length; i++) {
    selected.push(checkboxes[i].value);
  }
  if (selected.length === 0) {
    alert("Please select at least one actionable feature.");
    return;
  }
  var input = document.getElementById("chatInput");
  if (input) {
    input.value = "MANUAL_FEATURES: " + selected.join(", ");
    if (typeof sendMessage === 'function') {
      sendMessage();
    } else {
      var btn = document.getElementById("btnSend");
      if (btn) btn.click();
    }
  }
};


window.submitManualFeatures = function() {
    var checkboxes = document.querySelectorAll('#manual-features-form input[type="checkbox"]:checked');
    var selected = [];
    for (var i = 0; i < checkboxes.length; i++) {
        selected.push(checkboxes[i].value);
    }
    if (selected.length === 0) {
        alert("Please select at least one actionable feature.");
        return;
    }
    var chatInput = document.getElementById("chatInput") || document.getElementById("chat-message-input") || document.querySelector('input[type="text"]');
    var btnSend = document.getElementById("btnSend") || document.querySelector('button[type="submit"]');
    if (chatInput) {
        chatInput.value = "MANUAL_FEATURES:" + selected.join(",");
        if (btnSend) {
            btnSend.click();
        }
    }
};

window.sendQuickOption = function(text) {
    var chatInput = document.getElementById("chatInput") || document.getElementById("chat-message-input") || document.querySelector('input[type="text"]');
    var btnSend = document.getElementById("btnSend") || document.querySelector('button[type="submit"]');
    if (chatInput) {
        chatInput.value = text;
        if (btnSend) {
            btnSend.click();
        }
    }
};

/* Quantellix.AI Modern Chatbot Page JavaScript */
document.addEventListener("DOMContentLoaded", function () {
  const chatMessages = document.getElementById("chatMessages");
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const btnSend = document.getElementById("btnSend");
  const btnRestart = document.getElementById("btnRestart");
  const btnThemeToggle = document.getElementById("btnThemeToggle");

  function scrollToBottom(smooth = false) {
    if (!chatMessages) return;
    const doScroll = () => {
      chatMessages.scrollTo({
        top: chatMessages.scrollHeight,
        behavior: smooth ? 'smooth' : 'auto'
      });
    };
    doScroll();
    requestAnimationFrame(doScroll);
    setTimeout(doScroll, 60);
    setTimeout(doScroll, 180);
    setTimeout(doScroll, 350);
  }
  scrollToBottom();

  // Theme Toggle
  if (btnThemeToggle) {
    const savedTheme = localStorage.getItem("quantellix_theme") || "light";
    document.documentElement.setAttribute("data-theme", savedTheme);
    btnThemeToggle.textContent = savedTheme === "dark" ? "☀️" : "🌙";

    btnThemeToggle.addEventListener("click", function () {
      const current = document.documentElement.getAttribute("data-theme") || "light";
      const nextTheme = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", nextTheme);
      localStorage.setItem("quantellix_theme", nextTheme);
      btnThemeToggle.textContent = nextTheme === "dark" ? "☀️" : "🌙";
    });
  }

  function lockEdaButtons() {
    if (btnViewEDA) {
      btnViewEDA.className = "side-btn-disabled";
      btnViewEDA.title = "Locked until dataset upload and EDA completion";
      btnViewEDA.onclick = null;
      if (badgeViewEDA) {
        badgeViewEDA.textContent = "Locked";
      }
    }
    document.querySelectorAll(".btn-executive-dash-item").forEach(item => {
      item.className = "side-btn-disabled btn-executive-dash-item";
      item.title = "Locked until analysis is executed";
      item.onclick = null;
      item.style.cursor = "default";
      const badge = item.querySelector(".badgeExecutiveDashboard");
      if (badge) {
        badge.textContent = "Locked";
        badge.style.backgroundColor = "";
        badge.style.color = "";
      }
    });
    if (btnExecutiveDashboard) {
      btnExecutiveDashboard.className = "side-btn-disabled btn-executive-dash-item";
      btnExecutiveDashboard.title = "Locked until you pick a problem";
      btnExecutiveDashboard.onclick = null;
      if (badgeExecutiveDashboard) {
        badgeExecutiveDashboard.textContent = "Locked";
      }
    }
    lockMasterButtons();
  }

  // Restart Button
  if (btnRestart) {
    btnRestart.addEventListener("click", async function () {
      if (!confirm("Are you sure you want to restart the chat session?")) return;
      try {
        localStorage.removeItem("quantellix_attached_files");
        localStorage.removeItem("quantellix_files_added");
        localStorage.removeItem("quantellix_eda_ready");
        localStorage.removeItem("quantellix_dashboard_url");
        localStorage.removeItem("quantellix_solution_urls");
        localStorage.removeItem("quantellix_master_url");
        lockEdaButtons();
        await fetch("/api/chat/restart", { method: "POST" });
        window.location.href = window.location.pathname;
      } catch (err) {
        console.error("Failed to restart:", err);
        window.location.href = window.location.pathname;
      }
    });
  }

  // Recent Files & Dataset Attachment
  const btnAttachFile = document.getElementById("btnAttachFile");
  const fileAttachInput = document.getElementById("fileAttachInput");
  const recentFilesContainer = document.getElementById("recentFilesContainer");
  const recentFilesBadge = document.getElementById("recentFilesBadge");
  const btnViewEDA = document.getElementById("btnViewEDA");
  const badgeViewEDA = document.getElementById("badgeViewEDA");
  const btnExecutiveDashboard = document.getElementById("btnExecutiveDashboard");
  const badgeExecutiveDashboard = document.getElementById("badgeExecutiveDashboard");
  const masterReadyBox = document.getElementById("masterReadyBox");
  const masterLockedBtn = document.getElementById("masterLockedBtn");
  const btnDownloadMasterActive = document.getElementById("btnDownloadMasterActive");

  function unlockMasterButtons(downloadUrl) {
    if (masterReadyBox && masterLockedBtn) {
      masterLockedBtn.style.display = "none";
      masterReadyBox.style.display = "block";
    }
    if (btnDownloadMasterActive && downloadUrl) {
      btnDownloadMasterActive.setAttribute("href", downloadUrl);
    }
  }

  function lockMasterButtons() {
    if (masterReadyBox && masterLockedBtn) {
      masterReadyBox.style.display = "none";
      masterLockedBtn.style.display = "flex";
    }
  }

  function unlockEdaButtons(sessionId, viewUrl) {
    if (btnViewEDA) {
      btnViewEDA.className = "side-btn-active";
      btnViewEDA.title = "View Comprehensive EDA Dashboard";
      if (badgeViewEDA) {
        badgeViewEDA.textContent = "Ready";
      }
      btnViewEDA.onclick = function (e) {
        if (e) e.preventDefault();
        const sid = sessionId || btnViewEDA.getAttribute("data-session-id") || "";
        const targetUrl = viewUrl || (sid ? `/portal/eda/report/${sid}` : "/portal/eda/report");
        window.open(targetUrl, "_blank");
      };
    }
  }

  function unlockAllExecutiveDashboards(dashboardUrlsMap, primaryUrl, primaryTitle) {
    const dashItems = document.querySelectorAll(".btn-executive-dash-item");
    if (dashItems && dashItems.length > 0) {
      dashItems.forEach(item => {
        const slug = (item.getAttribute("data-sol-slug") || "").toLowerCase();
        const code = (item.getAttribute("data-sol-code") || "").toLowerCase();
        let targetUrl = null;
        if (dashboardUrlsMap) {
          targetUrl = dashboardUrlsMap[slug] || dashboardUrlsMap[code] || dashboardUrlsMap[slug.toUpperCase()] || dashboardUrlsMap[code.toUpperCase()];
        }
        if (!targetUrl) {
          targetUrl = item.getAttribute("data-dash-url");
        }
        if (!targetUrl && primaryUrl) {
          targetUrl = primaryUrl;
        }

        if (targetUrl) {
          item.className = "side-btn-active btn-executive-dash-item";
          item.title = `Open Dashboard`;
          item.style.cursor = "pointer";
          item.onclick = function (e) {
            if (e) e.preventDefault();
            window.open(targetUrl, "_blank");
          };
          const badge = item.querySelector(".badgeExecutiveDashboard");
          if (badge) {
            badge.textContent = "Open";
            badge.style.backgroundColor = slug === "executive" ? "#f59e0b" : "#10b981";
            badge.style.color = "#ffffff";
            badge.style.fontWeight = "800";
          }
          if (slug === "executive") {
            item.style.border = "1.5px solid #f59e0b";
            item.style.background = "#fffbeb";
          }
        }
      });
    }

    if (btnExecutiveDashboard) {
      btnExecutiveDashboard.className = "side-btn-active btn-executive-dash-item";
      btnExecutiveDashboard.title = "Open Executive Dashboard";
      btnExecutiveDashboard.style.cursor = "pointer";
      if (badgeExecutiveDashboard) {
        badgeExecutiveDashboard.textContent = "Open";
        badgeExecutiveDashboard.style.backgroundColor = "#10b981";
        badgeExecutiveDashboard.style.color = "#ffffff";
        badgeExecutiveDashboard.style.fontWeight = "700";
      }
      const titleSpan = document.getElementById("dashboardBtnName");
      if (titleSpan && primaryTitle) {
        titleSpan.textContent = primaryTitle.endsWith("Dashboard") ? primaryTitle : `${primaryTitle} Dashboard`;
      }
      btnExecutiveDashboard.onclick = function (e) {
        if (e) e.preventDefault();
        window.open(primaryUrl || (dashboardUrlsMap ? Object.values(dashboardUrlsMap)[0] : "#"), "_blank");
      };
    }
  }

  function unlockExecutiveDashboard(url, dashTitle) {
    unlockAllExecutiveDashboards(null, url, dashTitle);
  }

  // Check URL params for explicit solution selection (indicates switching solution flow)
  const urlParams = new URLSearchParams(window.location.search);
  const isSolutionParam = urlParams.has("solution") || urlParams.has("solutions");

  if (!isSolutionParam && btnViewEDA && btnViewEDA.classList.contains("side-btn-active")) {
    const sessId = btnViewEDA.getAttribute("data-session-id") || "";
    unlockEdaButtons(sessId);
  } else {
    // Session is fresh without uploaded dataset - ensure no stale files from old sessions
    lockEdaButtons();
    localStorage.removeItem("quantellix_attached_files");
    localStorage.removeItem("quantellix_files_added");
    localStorage.removeItem("quantellix_eda_ready");
    localStorage.removeItem("quantellix_dashboard_url");
    localStorage.removeItem("quantellix_solution_urls");
    localStorage.removeItem("quantellix_dashboard_title");
  }

  const savedSolutionUrls = isSolutionParam ? null : localStorage.getItem("quantellix_solution_urls");
  const savedDashUrl = isSolutionParam ? null : localStorage.getItem("quantellix_dashboard_url");
  const savedDashTitle = isSolutionParam ? null : localStorage.getItem("quantellix_dashboard_title");
  if (savedSolutionUrls) {
    try {
      unlockAllExecutiveDashboards(JSON.parse(savedSolutionUrls), savedDashUrl, savedDashTitle);
    } catch (e) {
      unlockAllExecutiveDashboards(null, savedDashUrl, savedDashTitle);
    }
  } else if (savedDashUrl) {
    unlockAllExecutiveDashboards(null, savedDashUrl, savedDashTitle);
  }

  const savedMasterUrl = isSolutionParam ? null : localStorage.getItem("quantellix_master_url");
  if (savedMasterUrl) {
    unlockMasterButtons(savedMasterUrl);
  } else if (masterReadyBox && masterReadyBox.style.display === "block") {
    const href = btnDownloadMasterActive ? btnDownloadMasterActive.getAttribute("href") : null;
    if (href && href !== "#") {
      localStorage.setItem("quantellix_master_url", href);
    }
  }

  function renderRecentFiles() {
    let files = [];
    try {
      files = JSON.parse(localStorage.getItem("quantellix_attached_files") || "[]");
    } catch (e) {
      files = [];
    }

    if (!files || files.length === 0) {
      if (recentFilesBadge) recentFilesBadge.textContent = "0 Active";
      if (recentFilesContainer) {
        recentFilesContainer.innerHTML = `
          <div class="no-files-notice" id="noFilesNotice">
            No files added
          </div>
        `;
      }
    } else {
      if (recentFilesBadge) recentFilesBadge.textContent = `${files.length} Active`;
      if (recentFilesContainer) {
        recentFilesContainer.innerHTML = files.map(f => `
          <div class="file-item">
            <span class="file-icon">📊</span>
            <div class="file-details">
              <div class="file-name" title="${escapeHtml(f.name)}">${escapeHtml(f.name)}</div>
              <div class="file-meta">${escapeHtml(f.meta || "Ready for analysis")}</div>
            </div>
          </div>
        `).join("");
      }
    }
  }

  renderRecentFiles();



  if (btnAttachFile && fileAttachInput) {
    btnAttachFile.addEventListener("click", function () {
      fileAttachInput.click();
    });

    fileAttachInput.addEventListener("change", async function (e) {
      const file = e.target.files[0];
      if (!file) return;

      const sizeStr = file.size > 1024 * 1024
        ? (file.size / (1024 * 1024)).toFixed(1) + " MB"
        : (file.size / 1024).toFixed(0) + " KB";

      // Show attachment bubble in chat immediately
      appendUserMessage(`📎 Attached dataset: ${file.name} (${sizeStr})`);

      // 1. Show "Files uploaded successfully!" message FIRST
      const uploadSuccessRow = document.createElement("div");
      uploadSuccessRow.className = "msg-row bot";
      uploadSuccessRow.id = "uploadSuccessRow";
      uploadSuccessRow.innerHTML = `
        <div class="msg-bubble">
          <div id="uploadSuccessText">✅ <strong>Files uploaded successfully!</strong> Ingested <code>${escapeHtml(file.name)}</code> (${sizeStr}).</div>
        </div>
      `;
      chatMessages.appendChild(uploadSuccessRow);

      // 2. Show live pipeline progress card SECOND
      const progressRow = document.createElement("div");
      progressRow.className = "msg-row bot";
      progressRow.id = "pipelineProgressCard";
      progressRow.innerHTML = `
        <div class="msg-bubble" style="background: rgba(37, 99, 235, 0.04); border: 1px solid rgba(37, 99, 235, 0.18);">
          <div style="font-weight: 700; color: #1e40af; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
            <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#2563eb;"></span>
            <span>Data Pipeline Execution</span>
          </div>
          <div id="stepIngest" style="font-size: 13px; color: #1e40af; margin-bottom: 4px; font-weight: 600;">⏳ <b>Step 1/3:</b> Ingesting & validating dataset...</div>
          <div id="stepPreproc" style="font-size: 13px; color: #64748b; margin-bottom: 4px;">⏳ <b>Step 2/3:</b> Data preprocessing is going on...</div>
          <div id="stepEda" style="font-size: 13px; color: #64748b;">⏳ <b>Step 3/3:</b> Generating Automated EDA Report...</div>
        </div>
      `;
      chatMessages.appendChild(progressRow);
      scrollToBottom();

      const timer1 = setTimeout(() => {
        const s1 = document.getElementById("stepIngest");
        const s2 = document.getElementById("stepPreproc");
        if (s1) { s1.innerHTML = "✅ <b>Step 1/3:</b> Dataset ingested."; s1.style.color = "#16a34a"; s1.style.fontWeight = "normal"; }
        if (s2) { s2.innerHTML = "⚙️ <b>Step 2/3:</b> Data preprocessing is going on..."; s2.style.color = "#1e40af"; s2.style.fontWeight = "600"; }
      }, 700);

      const timer2 = setTimeout(() => {
        const s2 = document.getElementById("stepPreproc");
        const s3 = document.getElementById("stepEda");
        if (s2) { s2.innerHTML = "✅ <b>Step 2/3:</b> Data preprocessing is completed."; s2.style.color = "#16a34a"; s2.style.fontWeight = "normal"; }
        if (s3) { s3.innerHTML = "📊 <b>Step 3/3:</b> Generating Automated EDA Report..."; s3.style.color = "#1e40af"; s3.style.fontWeight = "600"; }
      }, 1700);

      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("/api/chat/upload", {
          method: "POST",
          body: formData
        });

        clearTimeout(timer1);
        clearTimeout(timer2);

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || `Upload failed (status ${response.status})`);
        }

        const data = await response.json();

        // Update upload success message with exact rows, columns and PK
        const ust = document.getElementById("uploadSuccessText");
        if (ust && data) {
          const pk = data.pk || "sku_id";
          const rows = data.row_count ? data.row_count.toLocaleString() : "";
          const cols = data.col_count || "";
          ust.innerHTML = `✅ <strong>Files uploaded successfully!</strong> Ingested <code>${escapeHtml(data.filename || file.name)}</code> (${rows} rows, ${cols} columns · PK: <code>${pk}</code>).`;
        }

        // Update progress card to completed
        const s1 = document.getElementById("stepIngest");
        const s2 = document.getElementById("stepPreproc");
        const s3 = document.getElementById("stepEda");
        if (s1) { s1.innerHTML = `✅ <b>Step 1/3:</b> Ingested <b>${data.row_count ? data.row_count.toLocaleString() : ""} rows</b> (${data.col_count || ""} cols)`; s1.style.color = "#16a34a"; s1.style.fontWeight = "normal"; }
        if (s2) { s2.innerHTML = "✅ <b>Step 2/3:</b> Data preprocessing is completed."; s2.style.color = "#16a34a"; s2.style.fontWeight = "normal"; }
        if (s3) { s3.innerHTML = "✅ <b>Step 3/3:</b> EDA report is ready, you can download it."; s3.style.color = "#16a34a"; s3.style.fontWeight = "normal"; }

        // Unlock View EDA button
        if (data.eda_ready) {
          unlockEdaButtons(data.session_id, data.eda_view_url);
        }

        // Unlock Master Dataset download button
        if (data.master_ready || data.master_download_url) {
          const mUrl = data.master_download_url || `/api/master/download/${data.session_id}`;
          unlockMasterButtons(mUrl);
          localStorage.setItem("quantellix_master_url", mUrl);
        }

        // Render any follow-up messages returned from server (such as Actionable Features prompt)
        if (data.messages && Array.isArray(data.messages) && data.messages.length > 0) {
          data.messages.forEach(msg => {
            appendBotMessage(msg);
          });
        }

        // Add to recent files storage
        let files = [];
        try {
          files = JSON.parse(localStorage.getItem("quantellix_attached_files") || "[]");
        } catch (err) {
          files = [];
        }

        files.unshift({
          name: file.name,
          meta: `${data.row_count ? data.row_count.toLocaleString() : ""} rows • ${data.col_count || ""} cols • ${sizeStr}`,
          timestamp: Date.now()
        });

        localStorage.setItem("quantellix_files_added", "true");
        localStorage.setItem("quantellix_attached_files", JSON.stringify(files));
        renderRecentFiles();

      } catch (err) {
        clearTimeout(timer1);
        clearTimeout(timer2);
        const pCard = document.getElementById("pipelineProgressCard");
        if (pCard) pCard.remove();
        console.error("Upload error:", err);
        appendBotMessage({
          sender: "bot",
          type: "text",
          text: `⚠️ **Upload Failed:** ${escapeHtml(err.message)}<br>Please ensure you are uploading a valid CSV or Excel file.`
        });
      } finally {
        fileAttachInput.value = "";
        scrollToBottom();
      }
    });
  }

  // Dashboards Menu Drawer Toggle
  window.toggleDashboardsMenu = function () {
    const drawer = document.getElementById("dashboardsMenuDrawer");
    const backdrop = document.getElementById("dashboardsBackdrop");
    if (!drawer || !backdrop) return;
    const isOpen = drawer.classList.contains("open");
    if (isOpen) {
      drawer.classList.remove("open");
      backdrop.classList.remove("open");
    } else {
      drawer.classList.add("open");
      backdrop.classList.add("open");
    }
  };

  // Close menu drawer with Escape key
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
      const drawer = document.getElementById("dashboardsMenuDrawer");
      const backdrop = document.getElementById("dashboardsBackdrop");
      if (drawer && drawer.classList.contains("open")) {
        drawer.classList.remove("open");
        backdrop.classList.remove("open");
      }
    }
  });

  // Escape HTML to prevent XSS
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // Format Markdown-like text safely
  function formatMessageText(text) {
    if (!text) return "";

    // Convert markdown bold **text** to <strong>text</strong>
    let formatted = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    // Convert markdown italic *text* to <em>text</em>
    formatted = formatted.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
    // Strip any remaining rogue asterisks
    formatted = formatted.replace(/\*/g, "");
    // Markdown inline code `text`
    formatted = formatted.replace(/`([^`\n]+)`/g, "<code>$1</code>");
    // Markdown headers ### Title
    formatted = formatted.replace(/^###\s*(.*)$/gm, "<strong>$1</strong>");
    // Markdown links [text](url)
    formatted = formatted.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" style="color:#2563eb;font-weight:700;text-decoration:underline;">$1</a>');
    // Unordered lists
    formatted = formatted.replace(/^\s*[-•]\s+(.*)$/gm, "<li>$1</li>");
    formatted = formatted.replace(/(?:<li>.*?<\/li>\s*)+/gs, function(match) {
      return '<ul style="margin:6px 0 6px 18px; padding:0;">' + match.trim() + '</ul>';
    });
    // Newlines to <br> if not already formatted with HTML breaks
    if (!/<br\s*[\/]?>/i.test(formatted)) {
      formatted = formatted.replace(/\n/g, "<br>");
    }
    return formatted;
  }

  // Append user message
  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "msg-row user";
    row.innerHTML = `
      <div class="msg-avatar">👤</div>
      <div class="msg-bubble">
        <p>${escapeHtml(text)}</p>
      </div>
    `;
    chatMessages.appendChild(row);
    scrollToBottom(true);
  }

  // Show typing indicator
  function showTypingIndicator() {
    const row = document.createElement("div");
    row.className = "msg-row bot";
    row.id = "typingIndicator";
    row.innerHTML = `
      <div class="msg-bubble">
        <div class="typing-dots">
          <span></span><span></span><span></span>
        </div>
      </div>
    `;
    chatMessages.appendChild(row);
    scrollToBottom(true);
  }

  function removeTypingIndicator() {
    const el = document.getElementById("typingIndicator");
    if (el) el.remove();
  }

  // Render Bot Message
  function appendBotMessage(msg) {
    if (!msg || (!msg.text && msg.type !== "license_card" && msg.type !== "pipeline_card" && msg.type !== "sxi_card")) {
      return;
    }

    if (msg.type === "pipeline_card") {
      const rows = msg.rows ? msg.rows.toLocaleString() : "30,688";
      const cols = msg.cols || "38";
      const pRow = document.createElement("div");
      pRow.className = "msg-row bot";
      pRow.innerHTML = `
        <div class="msg-bubble" style="background: rgba(37, 99, 235, 0.04); border: 1px solid rgba(37, 99, 235, 0.18);">
          <div style="font-weight: 700; color: #1e40af; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
            <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#2563eb;"></span>
            <span>Data Pipeline Execution</span>
          </div>
          <div style="font-size: 13px; color: #16a34a; margin-bottom: 4px;">✅ <b>Step 1/3:</b> Ingested <b>${rows} rows</b> (${cols} cols)</div>
          <div style="font-size: 13px; color: #16a34a; margin-bottom: 4px;">✅ <b>Step 2/3:</b> Data preprocessing is completed.</div>
          <div style="font-size: 13px; color: #16a34a;">✅ <b>Step 3/3:</b> EDA report is ready, you can download it.</div>
        </div>
      `;
      chatMessages.appendChild(pRow);
      scrollToBottom(true);
      return;
    }

    if (msg.type === "sxi_card") {
      const sxiRow = document.createElement("div");
      sxiRow.className = "msg-row bot";
      
      const solCode = escapeHtml(msg.solution_code || "E8");
      const solName = escapeHtml(msg.kpi_name || "Reduce Inventory Stockout Risk");
      const remainingCredits = msg.remaining_credits !== undefined ? msg.remaining_credits : 33;
      const curScore = msg.current_sxi_score || "5.8";
      const tgtScore = msg.target_sxi_score || "8.4";
      const baselineKpi = msg.baseline_kpi || "20.0%";
      const optimizedKpi = msg.optimized_kpi || "4.6%";
      const impPct = msg.improvement_pct || "-15.4%";
      const accPct = msg.accuracy_pct || "98.2";
      const modelType = msg.model_type || "XGBoost Ensemble";
      const execDashUrl = msg.executive_dashboard_url || `/portal/executive/${currentSessionId}?solutions=${(msg.solution_code || "e4").toLowerCase()}`;
      const solDashUrl = msg.solution_dashboard_url || msg.dashboard_url || `/portal/dashboard/${currentSessionId}?solution=${(msg.solution_code || "e4").toLowerCase()}`;
      const dashUrl = execDashUrl;
      const csvUrl = msg.download_csv_url || "#";

      let weightsHtml = "";
      if (Array.isArray(msg.top_weights) && msg.top_weights.length > 0) {
        weightsHtml = msg.top_weights.map(w => `
          <div class="sxi-weight-bar-row">
            <span class="sxi-weight-feat">${escapeHtml(w.Feature)}</span>
            <span class="sxi-weight-pct">${w.Composite_Weight}% consensus</span>
          </div>
        `).join("");
      }

      let rulesHtml = "";
      if (Array.isArray(msg.rules) && msg.rules.length > 0) {
        rulesHtml = msg.rules.map((r, i) => `
          <li><b>Rule ${i+1}:</b> ${escapeHtml(r.condition)} &rarr; <i>${escapeHtml(r.impact)}</i> <span style="color:#10b981;font-weight:700;">(${escapeHtml(r.confidence)})</span></li>
        `).join("");
      }

      const immText = Array.isArray(msg.interventions?.immediate) ? msg.interventions.immediate.join("<br>&bull; ") : (msg.interventions?.immediate || "Deploy targeted safety stock buffer on high-velocity SKUs.");
      const midText = Array.isArray(msg.interventions?.mid_term) ? msg.interventions.mid_term.join("<br>&bull; ") : (msg.interventions?.mid_term || "Implement dual-sourcing framework for critical suppliers.");
      const lngText = Array.isArray(msg.interventions?.long_term) ? msg.interventions.long_term.join("<br>&bull; ") : (msg.interventions?.long_term || "Automate EDI replenishment orders with dynamic lead times.");

      sxiRow.innerHTML = `
        <div class="sxi-completion-card">
          <div class="sxi-card-header">
            <div class="sxi-header-badge">
              <span class="sxi-pulse-dot"></span>
              <span>SXI Decision Intelligence Completed</span>
            </div>
            <div class="sxi-credits-pill">
              <span>💳</span> 1 Credit Used &bull; <strong>${remainingCredits} credits left</strong>
            </div>
          </div>

          <div class="sxi-sol-bar">
            <span class="sxi-sol-code">${solCode}</span>
            <span class="sxi-sol-name">${solName}</span>
          </div>

          <div class="sxi-kpi-grid">
            <div class="sxi-kpi-card">
              <div class="sxi-kpi-label">SXI Composite Score</div>
              <div class="sxi-kpi-value-row">
                <span class="sxi-kpi-main">${curScore} &rarr; ${tgtScore}</span>
                <span class="sxi-kpi-unit">/10</span>
              </div>
              <div class="sxi-kpi-sub positive">Optimized Target</div>
            </div>

            <div class="sxi-kpi-card">
              <div class="sxi-kpi-label">Target KPI Optimization</div>
              <div class="sxi-kpi-value-row">
                <span class="sxi-kpi-main">${baselineKpi} &rarr; ${optimizedKpi}</span>
              </div>
              <div class="sxi-kpi-sub positive">${impPct}% Improvement</div>
            </div>

            <div class="sxi-kpi-card">
              <div class="sxi-kpi-label">Holdout Test Accuracy</div>
              <div class="sxi-kpi-value-row">
                <span class="sxi-kpi-main">${accPct}%</span>
              </div>
              <div class="sxi-kpi-sub">${escapeHtml(modelType)}</div>
            </div>
          </div>

          <div class="sxi-actions-row">
            <a href="${execDashUrl}" target="_blank" class="sxi-btn-primary" id="btnOpenExecutiveDash">
              <span>🚀 Open Executive Dashboard</span>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
            </a>
            <a href="${solDashUrl}" target="_blank" class="sxi-btn-secondary" style="border:1px solid #cbd5e1; color:#0f172a; font-weight:600;">
              <span>📊 View Solution Dashboard</span>
            </a>
            <a href="${csvUrl}" download class="sxi-btn-secondary">
              <span>📥 Download CSV Results</span>
            </a>
          </div>

          <details class="sxi-details-accordion">
            <summary class="sxi-details-summary">
              <span>⚙️ View Ensemble Weights, Rules & Interventions</span>
              <span class="sxi-summary-chevron">▼</span>
            </summary>
            <div class="sxi-details-content">
              ${weightsHtml ? `
                <div class="sxi-detail-block">
                  <div class="sxi-detail-title">⚖️ Consensus Feature Weights (5-Model Consensus)</div>
                  <div style="display:flex; flex-direction:column; gap:4px;">${weightsHtml}</div>
                </div>
              ` : ''}

              ${rulesHtml ? `
                <div class="sxi-detail-block">
                  <div class="sxi-detail-title">🌲 Explainable Decision Tree Rules (A* Path Search)</div>
                  <ul class="sxi-rules-list">${rulesHtml}</ul>
                </div>
              ` : ''}

              <div class="sxi-detail-block">
                <div class="sxi-detail-title">🛠️ 3-Horizon Actionable Interventions</div>
                <div class="sxi-horizons-grid">
                  <div class="sxi-horizon-box">
                    <span class="sxi-horizon-tag h1">⚡ Horizon 1 (0–30 Days)</span>
                    <div class="sxi-horizon-text">${immText}</div>
                  </div>
                  <div class="sxi-horizon-box">
                    <span class="sxi-horizon-tag h2">🔄 Horizon 2 (30–90 Days)</span>
                    <div class="sxi-horizon-text">${midText}</div>
                  </div>
                  <div class="sxi-horizon-box">
                    <span class="sxi-horizon-tag h3">🏗️ Horizon 3 (90+ Days)</span>
                    <div class="sxi-horizon-text">${lngText}</div>
                  </div>
                </div>
              </div>
            </div>
          </details>
        </div>
      `;

      chatMessages.appendChild(sxiRow);
      scrollToBottom(true);
      setTimeout(() => {
        sxiRow.scrollIntoView({ behavior: "smooth", block: "end" });
      }, 100);
      return;
    }

    const row = document.createElement("div");
    row.className = "msg-row bot";

    if (msg.type === "license_card") {
      row.innerHTML = `
        <div class="license-card">
          <div class="license-header">
            <span class="license-icon">⚡</span>
            <span class="license-title">${escapeHtml(msg.title || "Signed in from your license.")}</span>
          </div>
          <div class="license-plan">${escapeHtml(msg.plan || "")}</div>
          <ul class="license-notices">
            ${msg.used_notice ? `<li>${escapeHtml(msg.used_notice)}</li>` : ""}
            ${msg.credits_notice ? `<li>${escapeHtml(msg.credits_notice)}</li>` : ""}
            ${msg.terms_notice ? `<li>${escapeHtml(msg.terms_notice)}</li>` : ""}
          </ul>
          <div class="license-footer">
            <span>🛡️</span>
            <span>${escapeHtml(msg.validity || "License active")}</span>
          </div>
        </div>
      `;
    } else {
      let buttonsHtml = '';
      if (msg.buttons && Array.isArray(msg.buttons) && msg.buttons.length > 0) {
        buttonsHtml = `<div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:10px;">` +
          msg.buttons.map(btn => `<button type="button" class="chat-choice-btn" onclick="sendChoiceMessage('${escapeHtml(btn)}')" style="padding:7px 16px; background:linear-gradient(135deg, #2563eb, #1d4ed8); color:#fff; border:1px solid rgba(255,255,255,0.2); border-radius:8px; font-weight:600; font-size:13px; cursor:pointer; transition:all 0.2s; box-shadow:0 2px 6px rgba(37,99,235,0.35);">${escapeHtml(btn)}</button>`).join('') +
          `</div>`;
      }

      let formHtml = '';
      if (msg.form_html) {
        formHtml = msg.form_html;
      }

      row.innerHTML = `
        <div class="msg-bubble">
          <div>${formatMessageText(msg.text || "")}</div>
          ${formHtml}
          ${buttonsHtml}
        </div>
      `;
    }

    chatMessages.appendChild(row);
    scrollToBottom(true);
    setTimeout(() => {
      row.scrollIntoView({ behavior: "smooth", block: "end" });
    }, 100);
  }

  // Send message handler
  async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    appendUserMessage(text);
    chatInput.value = "";
    chatInput.focus();

    btnSend.disabled = true;

    // Detect if this message is selecting Actionable Features (LLM or Manual)
    const cleanLower = text.toLowerCase().trim();
    
    function isActionableFeatureStage() {
      if (!chatMessages) return false;
      const botRows = chatMessages.querySelectorAll('.msg-row.bot');
      if (botRows.length === 0) return false;
      const lastBot = botRows[botRows.length - 1];
      const botTxt = (lastBot ? lastBot.textContent : "").toLowerCase();
      return botTxt.includes("actionable feature");
    }

    const inActionableStage = isActionableFeatureStage();
    const isLlmTrigger = inActionableStage && (
      cleanLower === '1' || cleanLower === '1.' || cleanLower === '1. llm based' || cleanLower === 'llm' || cleanLower.includes('llm based') || cleanLower === 'llm based'
    );
    const isManualTrigger = inActionableStage && (
      cleanLower === '2' || cleanLower === '2.' || cleanLower === '2. manual' || cleanLower === 'manual' || cleanLower.includes('manual select') || cleanLower.startsWith('manual_features:')
    );
    
    let stagedHandled = false;
    const requestStartTime = Date.now();

    if (isLlmTrigger || isManualTrigger) {
      stagedHandled = true;
      const targetCol = window.QUANT_TARGET_COL || "is_reorder_required";
      const modeText = isLlmTrigger
        ? "Actionable feature mode set to LLM Based."
        : "Actionable feature mode set to Manual (Selected Levers).";

      // Display initial staging messages responsively without artificial lag
      setTimeout(() => {
        appendBotMessage({ sender: "bot", type: "text", text: modeText });
      }, 50);

      setTimeout(() => {
        appendBotMessage({
          sender: "bot",
          type: "text",
          text: `Hi! 👋 I am **Agent-2 (Jaya)** — your modeling and interpretability engine.\n\n✅ Dataset ready (Modality: **TABULAR**).\n🎯 Prediction Goal: **predict ${targetCol}**\n🔍 Analyzing dataset and preparing model...`
        });
      }, 200);

      setTimeout(() => {
        appendBotMessage({
          sender: "bot",
          type: "text",
          text: "⚙️ Agent has started analysis... Preparing decision intelligence."
        });
        showTypingIndicator();
      }, 350);
    } else {
      showTypingIndicator();
    }

    try {
      const response = await fetch("/api/chat/interactive", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: text }),
      });

      removeTypingIndicator();

      if (!response.ok) {
        throw new Error("HTTP error " + response.status);
      }

      const data = await response.json();
      if (data && data.messages && data.messages.length > 0) {
        const botMsgs = data.messages.filter((msg) => msg.sender === "bot");

        if (stagedHandled) {
          // Immediately filter and render completion cards and messages without forced delay
          const completionMsgs = botMsgs.filter(m => m.type === "sxi_card" || (!m.text.includes("Actionable feature mode") && !m.text.includes("Agent-2") && !m.text.includes("Agent has started analysis")));
          
          for (const msg of completionMsgs) {
            appendBotMessage(msg);
          }
        } else {
          // Crisp, snappy rendering: instant for single messages, subtle 120ms micro-pause for multiple messages
          if (botMsgs.length <= 1) {
            botMsgs.forEach((msg) => appendBotMessage(msg));
          } else {
            for (let i = 0; i < botMsgs.length; i++) {
              const msg = botMsgs[i];
              if (i > 0) {
                showTypingIndicator();
                await new Promise((resolve) => setTimeout(resolve, 120));
                removeTypingIndicator();
              }
              appendBotMessage(msg);
            }
          }
        }
      }

      if (data && data.eda_ready) {
        unlockEdaButtons(data.session_id, data.eda_view_url);
        if (data.master_ready || data.master_download_url) {
          const mUrl = data.master_download_url || `/api/master/download/${data.session_id}`;
          unlockMasterButtons(mUrl);
          localStorage.setItem("quantellix_master_url", mUrl);
        }
        if (data.file_name) {
          const rows = data.row_count ? data.row_count.toLocaleString() : "35,000";
          const cols = data.col_count ? data.col_count : "24";
          const files = [{
            name: data.file_name,
            meta: `${rows} rows • ${cols} cols • Benchmark`,
            timestamp: Date.now()
          }];
          localStorage.setItem("quantellix_attached_files", JSON.stringify(files));
          renderRecentFiles();
        }
      }

      if (data && data.dashboard_urls) {
        localStorage.setItem("quantellix_solution_urls", JSON.stringify(data.dashboard_urls));
        unlockAllExecutiveDashboards(data.dashboard_urls, data.dashboard_url, data.dashboard_title);
      }
      if (data && data.dashboard_url) {
        const dTitle = data.dashboard_title || "Order Execution Dashboard";
        if (!data.dashboard_urls) {
          unlockAllExecutiveDashboards(null, data.dashboard_url, dTitle);
        }
        localStorage.setItem("quantellix_dashboard_url", data.dashboard_url);
        localStorage.setItem("quantellix_dashboard_title", dTitle);
      }

    } catch (err) {
      removeTypingIndicator();
      console.error("Chat error:", err);
      appendBotMessage({
        sender: "bot",
        type: "text",
        text: "⚠️ An error occurred while communicating with the assistant. Please try again.",
      });
    } finally {
      btnSend.disabled = false;
      scrollToBottom();
    }
  }

  if (chatForm) {
    chatForm.addEventListener("submit", function (e) {
      e.preventDefault();
      sendMessage();
    });
  }

  if (chatInput) {
    chatInput.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }
});
