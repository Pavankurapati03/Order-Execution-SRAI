// =============================================================================
// Quantellix Order Fulfillment Dashboard - Client-side Logic (app.js)
// =============================================================================

const ALL_PROBLEM_IDS = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10'];
let selectedStages = new Set(['e1']); // Default recommended selection

document.addEventListener('DOMContentLoaded', () => {
  updateCardSelectionsUI();
  fetchPipelineSummary();
  if (window.location.hash) {
    const stageId = window.location.hash.replace('#stage-', '');
    if (ALL_PROBLEM_IDS.includes(stageId)) {
      setTimeout(() => openStageView(stageId), 150);
    }
  }
});

// Stage Card Selection Toggle
function toggleCardSelection(stageId) {
  if (selectedStages.has(stageId)) {
    selectedStages.delete(stageId);
  } else {
    selectedStages.add(stageId);
  }
  updateCardSelectionsUI();
}

function selectStageOnly(stageId) {
  selectedStages.clear();
  selectedStages.add(stageId);
  updateCardSelectionsUI();
}

function selectAllStages() {
  selectedStages = new Set(['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10']);
  updateCardSelectionsUI();
}

function selectAllSolutions() {
  selectAllStages();
}

function clearSelections() {
  selectedStages.clear();
  updateCardSelectionsUI();
}

function resetSelections() {
  selectedStages = new Set(['e1']);
  updateCardSelectionsUI();
  closeAllStageViews();
}

function updateCardSelectionsUI() {
  ALL_PROBLEM_IDS.forEach(id => {
    const card = document.getElementById(`card-${id}`);
    const checkbox = document.getElementById(`checkbox-${id}`);
    if (card && checkbox) {
      if (selectedStages.has(id)) {
        card.classList.add('selected');
      } else {
        card.classList.remove('selected');
      }
    }
  });

    const continueBtn = document.getElementById('btnContinueToAssistant');
  if (continueBtn) {
    const selectedList = Array.from(selectedStages);
    const activeStage = selectedList.length > 0 ? selectedList[0] : 'e3';
    const count = selectedList.length;
    if (count > 1) {
      continueBtn.href = `/portal/chatbot?solutions=${selectedList.join(',')}`;
      continueBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
        Continue to AI Assistant (${count} Solutions) &rarr;
      `;
    } else {
      continueBtn.href = `/portal/chatbot?solution=${activeStage}`;
      continueBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
        Continue to AI Assistant &rarr;
      `;
    }
  }

  const reviewText = document.getElementById('review-selected-text');
  if (reviewText) {
    const count = selectedStages.size;
    reviewText.innerHTML = `${count} Solution${count === 1 ? '' : 's'} selected &bull; Credits required: ${count}`;
  }

  const openBtn = document.getElementById('btnOpenSelected');
  if (openBtn) {
    if (selectedStages.size === 0) {
      openBtn.textContent = 'Select a Stage';
      openBtn.style.opacity = '0.6';
      openBtn.disabled = true;
    } else {
      const first = Array.from(selectedStages)[0].toUpperCase();
      openBtn.textContent = selectedStages.size === 1 ? `Open ${first} Dashboard` : `Open Selected (${selectedStages.size}) Dashboards`;
      openBtn.style.opacity = '1';
      openBtn.disabled = false;
    }
  }
}

// Open / Close Stage Drilldown Views
function openSelectedDashboard() {
  if (selectedStages.size === 0) {
    alert('Please select at least one stage (S1, S2, or S3) to view its dashboard.');
    return;
  }
  
  // Close any open ones first
  closeAllStageViews();

  // Open all selected stages
  selectedStages.forEach(stageId => {
    const panel = document.getElementById(`view-panel-${stageId}`);
    if (panel) {
      panel.classList.add('active');
    }
  });

  // Scroll to the first selected view
  const firstId = Array.from(selectedStages)[0];
  const firstPanel = document.getElementById(`view-panel-${firstId}`);
  if (firstPanel) {
    firstPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function openStageView(stageId) {
  closeAllStageViews();
  const panel = document.getElementById(`view-panel-${stageId}`);
  if (panel) {
    panel.classList.add('active');
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    window.location.hash = 'stage-' + stageId;
  }
  // Ensure the card is marked selected as well
  selectedStages.add(stageId);
  updateCardSelectionsUI();
}

function closeStageView(stageId) {
  const panel = document.getElementById(`view-panel-${stageId}`);
  if (panel) {
    panel.classList.remove('active');
    if (history.replaceState) {
      history.replaceState(null, null, ' ');
    }
  }
}

function closeAllStageViews() {
  ['s1', 's2', 's3'].forEach(id => {
    const panel = document.getElementById(`view-panel-${id}`);
    if (panel) panel.classList.remove('active');
  });
}

function toggleUserMenu(e) {
  if (e) e.stopPropagation();
  const menu = document.getElementById('userDropdown');
  if (menu) {
    menu.classList.toggle('show');
  }
}

function closeUserMenu() {
  const menu = document.getElementById('userDropdown');
  if (menu) {
    menu.classList.remove('show');
  }
}

function confirmSignOut() {
  if (confirm('Are you sure you want to sign out of your Quantellix account?')) {
    alert('You have been signed out.');
  }
}

// Close user dropdown if clicking outside
document.addEventListener('click', (e) => {
  const container = document.querySelector('.user-menu-container');
  if (container && !container.contains(e.target)) {
    closeUserMenu();
  }
});


// Fetch Pipeline Summary Metrics from Backend API
async function fetchPipelineSummary() {
  try {
    const res = await fetch('/api/summary');
    if (!res.ok) return;
    const data = await res.json();

    // Populate Stage 2 KPIs if present
    if (data.stage_2_demand_planning) {
      const s2 = data.stage_2_demand_planning;
      const elGlobal = document.getElementById('kpi-global-demand');
      if (elGlobal) elGlobal.innerHTML = `${Math.round(s2.global_plan_demand_quantity).toLocaleString()} <span style="font-size:14px; font-weight:600;">units</span>`;

      const elLT = document.getElementById('kpi-lead-time');
      if (elLT) elLT.innerHTML = `${s2.lead_time} <span style="font-size:14px; font-weight:600;">days</span>`;

      const elPrice = document.getElementById('kpi-price');
      if (elPrice) elPrice.innerHTML = `$${Number(s2.weightage_list_price).toFixed(2)} <span style="font-size:14px; font-weight:600;">/ unit</span>`;

      const elVendors = document.getElementById('kpi-vendors');
      if (elVendors) elVendors.innerHTML = `${s2.vendor_count} <span style="font-size:14px; font-weight:600;">suppliers</span>`;

      const elDefect = document.getElementById('kpi-defect-rate');
      if (elDefect) elDefect.innerHTML = `${Number(s2.vendor_defect_rate_percent).toFixed(2)}%`;
    }

    // Populate Stage 3 PO Table if present
    if (data.stage_3_procurement && data.stage_3_procurement.target_3_multi_vendor_allocations) {
      const tbody = document.getElementById('po-table-body');
      if (tbody) {
        tbody.innerHTML = '';
        data.stage_3_procurement.target_3_multi_vendor_allocations.forEach((alloc, idx) => {
          const tr = document.createElement('tr');
          const isPrimary = idx === 0;
          tr.innerHTML = `
            <td><span class="${isPrimary ? 'badge-pill-success' : 'badge-pill-primary'}">${alloc.role}</span></td>
            <td><strong>${alloc.supplier_id}</strong></td>
            <td><strong>${Number(alloc.composite_score).toFixed(4)}</strong></td>
            <td>$${Number(alloc.predicted_unit_cost).toFixed(2)}</td>
            <td>${Number(alloc.lead_time_days).toFixed(1)} days</td>
            <td>${(Number(alloc.reliability) * 100).toFixed(1)}%</td>
            <td><strong>${Number(alloc.allocation_share_percent).toFixed(1)}%</strong></td>
            <td><strong>${Number(alloc.allocated_quantity).toLocaleString()}</strong></td>
            <td>$${Number(alloc.order_cost_eur).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</td>
          `;
          tbody.appendChild(tr);
        });
      }
    }
  } catch (err) {
    console.warn('API summary fetch error, using statically rendered defaults:', err);
  }
}

// =============================================================================
// Dashboards Menu Drawer Toggle & Theme Handling (for Dashboard pages)
// =============================================================================
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

window.toggleDashboardTheme = function () {
  const current = document.documentElement.getAttribute("data-theme") || "light";
  const nextTheme = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", nextTheme);
  localStorage.setItem("quantellix_theme", nextTheme);
  const btn = document.getElementById("btnThemeToggle");
  if (btn) {
    btn.textContent = nextTheme === "dark" ? "☀️" : "🌙";
  }
};

// Initialize theme & listeners on load
document.addEventListener("DOMContentLoaded", () => {
  const savedTheme = localStorage.getItem("quantellix_theme") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);
  const btn = document.getElementById("btnThemeToggle");
  if (btn) {
    btn.textContent = savedTheme === "dark" ? "☀️" : "🌙";
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      const drawer = document.getElementById("dashboardsMenuDrawer");
      const backdrop = document.getElementById("dashboardsBackdrop");
      if (drawer && drawer.classList.contains("open")) {
        drawer.classList.remove("open");
        backdrop.classList.remove("open");
      }
    }
  });
});

