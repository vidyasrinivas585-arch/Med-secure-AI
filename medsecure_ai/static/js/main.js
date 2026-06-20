/**
 * MedSecure AI — main.js
 * Global client-side utilities and interactions.
 */

// ── Smooth scroll for anchor links ────────────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// ── Auto-dismiss alerts after 5 s ─────────────────────────────────────────
setTimeout(() => {
  document.querySelectorAll('.alert.fade.show').forEach(el => {
    const alert = bootstrap.Alert.getOrCreateInstance(el);
    alert.close();
  });
}, 5000);

// ── Animate progress bars on result page ──────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.progress-bar[style]').forEach(bar => {
    const target = bar.style.width;
    bar.style.width = '0%';
    requestAnimationFrame(() => {
      bar.style.transition = 'width 1s ease-in-out';
      bar.style.width = target;
    });
  });
});

// ── Tooltip initialisation ─────────────────────────────────────────────────
const tooltipEls = document.querySelectorAll('[data-bs-toggle="tooltip"]');
tooltipEls.forEach(el => new bootstrap.Tooltip(el, { trigger: 'hover' }));

// ── Copy text helper ───────────────────────────────────────────────────────
function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => {
    showToast('Copied to clipboard!', 'success');
  });
}

// ── Toast notification helper ──────────────────────────────────────────────
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer') || (() => {
    const div = document.createElement('div');
    div.id = 'toastContainer';
    div.className = 'toast-container position-fixed bottom-0 end-0 p-3';
    div.style.zIndex = 9999;
    document.body.appendChild(div);
    return div;
  })();

  const id = 'toast-' + Date.now();
  const colours = { success: '#0F9D58', danger: '#D93025', info: '#1A73E8', warning: '#F59E0B' };
  const colour = colours[type] || colours.info;

  const html = `
    <div id="${id}" class="toast align-items-center text-white border-0 show"
         style="background:${colour}" role="alert" aria-live="assertive">
      <div class="d-flex">
        <div class="toast-body fw-semibold">${message}</div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto"
                data-bs-dismiss="toast"></button>
      </div>
    </div>`;
  container.insertAdjacentHTML('beforeend', html);

  const toastEl = document.getElementById(id);
  setTimeout(() => {
    toastEl.classList.remove('show');
    setTimeout(() => toastEl.remove(), 300);
  }, 3500);
}
