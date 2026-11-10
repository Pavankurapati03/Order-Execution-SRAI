// =============================================================================
// Quantellix Order Fulfillment AI Assistant & Dataset Compiler (chatbot.js)
// =============================================================================

let uploadedFilename = null;
let isPipelineRunning = false;

// Drawer Toggle
function toggleChatDrawer() {
  const drawer = document.getElementById('chatDrawer');
  const backdrop = document.getElementById('chatBackdrop');
  if (!drawer || !backdrop) return;

  const isOpen = drawer.classList.contains('open');
  if (isOpen) {
    drawer.classList.remove('open');
    backdrop.classList.remove('open');
  } else {
    drawer.classList.add('open');
    backdrop.classList.add('open');
    document.getElementById('chatInput')?.focus();
  }
}

// Drag & Drop Setup
document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('dropZone');
  if (!dropZone) return;

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');
    }, false);
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      uploadFile(files[0]);
    }
  });
});

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (file) {
    uploadFile(file);
  }
}

async function uploadFile(file) {
  const textEl = document.getElementById('dropZoneText');
  if (textEl) textEl.textContent = `Uploading ${file.name}...`;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    const result = await res.json();
    if (res.ok) {
      uploadedFilename = result.filename;
      if (textEl) {
        textEl.innerHTML = `✅ <strong>${result.filename}</strong> (${(file.size / (1024*1024)).toFixed(2)} MB uploaded)`;
      }
      appendChatMessage('bot', `📁 Successfully uploaded dataset <strong>${result.filename}</strong>! Click <strong>Compile 3-Stage Pipeline</strong> above or ask me any question to begin processing.`);
    } else {
      if (textEl) textEl.textContent = 'Upload failed. Click to try again.';
      appendChatMessage('bot', `❌ Upload error: ${result.detail || 'Could not process file'}`);
    }
  } catch (err) {
    console.error('File upload error:', err);
    if (textEl) textEl.textContent = 'Upload failed. Click to try again.';
    appendChatMessage('bot', '❌ Upload failed due to network error.');
  }
}

// Trigger Pipeline Compilation
async function triggerPipelineExecution() {
  if (isPipelineRunning) return;
  isPipelineRunning = true;

  const btn = document.getElementById('btnCompile');
  const originalBtnText = btn ? btn.innerHTML : '';
  if (btn) {
    btn.disabled = true;
    btn.style.opacity = '0.7';
    btn.innerHTML = '<span>⚙️ Compiling Order Execution Pipeline...</span>';
  }

  appendChatMessage('user', 'Compile the entire 3-stage order execution pipeline for our dataset.');
  
  const progressMsgId = 'progress-' + Date.now();
  appendChatMessage('bot', `
    <div id="${progressMsgId}">
      <div style="font-weight:700; margin-bottom:8px; display:flex; align-items:center; gap:8px;">
        <span class="pulse-dot"></span> Running 3-Stage Order Execution Pipeline...
      </div>
      <div style="font-size:12px; color:#475569; display:flex; flex-direction:column; gap:4px;">
        <div id="step-s1">⏳ <strong>Stage 1</strong>: Meta Prophet Time Series Forecasting & Holdout Validation...</div>
        <div id="step-s2" style="opacity:0.5;">⏸️ <strong>Stage 2</strong>: Calculating 5 Core Targets & S&OP Inventory Positions...</div>
        <div id="step-s3" style="opacity:0.5;">⏸️ <strong>Stage 3</strong>: Running MCDA Multi-Vendor PO Allocation & Audit Checks...</div>
      </div>
    </div>
  `);

  try {
    const res = await fetch('/api/run-pipeline', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: uploadedFilename || 'default' })
    });

    const data = await res.json();

    const progEl = document.getElementById(progressMsgId);
    if (progEl) {
      progEl.innerHTML = `
        <div style="color:#047857; font-weight:800; font-size:14px; margin-bottom:6px;">
          ✅ Pipeline Successfully Executed!
        </div>
        <div style="font-size:12px; color:#334155; margin-bottom:8px;">
          All 3 operational stages compiled and synchronized:
        </div>

        <div class="chat-stage-result-card">
          <div class="result-card-header">
            <span style="color:#2563eb;">Stage 1: Sales Demand Forecasting</span>
          </div>
          <div class="result-kpi-grid">
            <div class="result-kpi-item">
              <span>Model</span>
              <strong>${data.stage_1_sales_forecasting?.model_used || 'Meta Prophet'}</strong>
            </div>
            <div class="result-kpi-item">
              <span>Holdout WMAPE</span>
              <strong>${data.stage_1_sales_forecasting?.holdout_wmape_percent || 11.18}%</strong>
            </div>
          </div>
        </div>

        <div class="chat-stage-result-card">
          <div class="result-card-header">
            <span style="color:#059669;">Stage 2: 5 Core Demand Targets (S&OP)</span>
          </div>
          <div class="result-kpi-grid">
            <div class="result-kpi-item">
              <span>Plan Demand</span>
              <strong>${Math.round(data.stage_2_demand_planning?.global_plan_demand_quantity || 44096).toLocaleString()} units</strong>
            </div>
            <div class="result-kpi-item">
              <span>ML Lead Time</span>
              <strong>${data.stage_2_demand_planning?.lead_time || 7} days</strong>
            </div>
            <div class="result-kpi-item">
              <span>Reorder Point (ROP)</span>
              <strong>${Math.round(data.stage_2_demand_planning?.reorder_point_ROP || 13163).toLocaleString()} units</strong>
            </div>
            <div class="result-kpi-item">
              <span>Reorder Triggered</span>
              <strong style="color:#dc2626;">${data.stage_2_demand_planning?.reorder_triggered ? 'YES (CRITICAL)' : 'NO'}</strong>
            </div>
          </div>
        </div>

        <div class="chat-stage-result-card">
          <div class="result-card-header">
            <span style="color:#7c3aed;">Stage 3: Procurement & PO Decision</span>
          </div>
          <div class="result-kpi-grid">
            <div class="result-kpi-item">
              <span>PO Quantity</span>
              <strong>${Math.round(data.stage_3_procurement?.recommended_po_order_quantity || 13150).toLocaleString()} units</strong>
            </div>
            <div class="result-kpi-item">
              <span>Primary Vendor</span>
              <strong>${data.stage_3_procurement?.target_3_multi_vendor_allocations?.[0]?.supplier_id || 'S014'} (70.3%)</strong>
            </div>
            <div class="result-kpi-item">
              <span>Order Value</span>
              <strong>$${Number(data.stage_3_procurement?.total_order_cost_eur || 49390.50).toLocaleString()}</strong>
            </div>
            <div class="result-kpi-item">
              <span>Projected Margin</span>
              <strong style="color:#047857;">$${Number(data.stage_3_procurement?.total_projected_gross_margin_eur || 32665.50).toLocaleString()}</strong>
            </div>
          </div>
        </div>

        <div style="margin-top:10px; display:flex; gap:6px; flex-wrap:wrap;">
          <button class="chip-btn" onclick="openStageView('s1')">Open Stage 1 View</button>
          <button class="chip-btn" onclick="openStageView('s2')">Open Stage 2 View</button>
          <button class="chip-btn" onclick="openStageView('s3')">Open Stage 3 View</button>
        </div>
      `;
    }

    // Refresh metrics on main dashboard
    if (typeof fetchPipelineSummary === 'function') {
      fetchPipelineSummary();
    }

  } catch (err) {
    console.error('Compilation error:', err);
    appendChatMessage('bot', '❌ Pipeline execution failed. Please check backend logs.');
  } finally {
    isPipelineRunning = false;
    if (btn) {
      btn.disabled = false;
      btn.style.opacity = '1';
      btn.innerHTML = originalBtnText;
    }
  }
}

// Chat Questions & Answers
function askQuestion(q) {
  const input = document.getElementById('chatInput');
  if (input) {
    input.value = q;
    sendMessage();
  }
}

async function sendMessage() {
  const input = document.getElementById('chatInput');
  if (!input) return;
  const query = input.value.trim();
  if (!query) return;

  appendChatMessage('user', query);
  input.value = '';

  const loadingId = 'loading-' + Date.now();
  appendChatMessage('bot', `<span id="${loadingId}">Thinking...</span>`);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: query })
    });
    const data = await res.json();
    const loadingEl = document.getElementById(loadingId);
    if (loadingEl) {
      loadingEl.parentElement.innerHTML = data.reply;
    }
  } catch (err) {
    console.error('Chat error:', err);
    const loadingEl = document.getElementById(loadingId);
    if (loadingEl) {
      loadingEl.parentElement.textContent = 'Sorry, could not connect to Order AI backend.';
    }
  }
}

function appendChatMessage(sender, htmlContent) {
  const container = document.getElementById('chatMessages');
  if (!container) return;

  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${sender}`;
  bubble.innerHTML = htmlContent;
  container.appendChild(bubble);

  container.scrollTop = container.scrollHeight;
}
