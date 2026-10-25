/**
 * Student Performance Prediction - Client-Side JavaScript
 * Pure Vanilla JavaScript & Fetch API implementation.
 */

// Toast notification helper
function showToast(message, type = 'info') {
  const toastContainer = document.getElementById('toastContainer');
  if (!toastContainer) return;

  const bgClasses = {
    success: 'text-bg-success',
    danger: 'text-bg-danger',
    warning: 'text-bg-warning',
    info: 'text-bg-dark'
  };

  const bgClass = bgClasses[type] || 'text-bg-dark';
  const toastId = 'toast_' + Date.now();

  const toastEl = document.createElement('div');
  toastEl.className = `toast align-items-center ${bgClass} border-0 shadow-sm`;
  toastEl.id = toastId;
  toastEl.setAttribute('role', 'alert');
  toastEl.setAttribute('aria-live', 'assertive');
  toastEl.setAttribute('aria-atomic', 'true');

  toastEl.innerHTML = `
    <div class="d-flex">
      <div class="toast-body">${message}</div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
    </div>
  `;

  toastContainer.appendChild(toastEl);
  const bsToast = new bootstrap.Toast(toastEl, { delay: 4500 });
  bsToast.show();

  toastEl.addEventListener('hidden.bs.toast', () => {
    toastEl.remove();
  });
}

// Risk category badge formatter
function getRiskBadgeHtml(category) {
  if (!category) return '<span class="badge bg-secondary">Unknown</span>';
  const lower = category.toLowerCase();
  if (lower.includes('high')) {
    return `<span class="badge-risk-high"><i class="bi bi-exclamation-triangle-fill"></i> ${category}</span>`;
  } else if (lower.includes('moderate') || lower.includes('medium')) {
    return `<span class="badge-risk-medium"><i class="bi bi-shield-exclamation"></i> ${category}</span>`;
  } else if (lower.includes('low')) {
    return `<span class="badge-risk-low"><i class="bi bi-check-circle-fill"></i> ${category}</span>`;
  }
  return `<span class="badge bg-secondary">${category}</span>`;
}

// Global initialization
document.addEventListener('DOMContentLoaded', () => {
  // Initialize Bootstrap Tooltips
  const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  tooltipTriggerList.map((el) => new bootstrap.Tooltip(el));
});
