// ML Pipeline Header Component
function createMLHeader() {
  const path = window.location.pathname;
  let currentStep = 'home';

  if (path.includes('upload.html')) currentStep = 'upload';
  else if (path.includes('Analysis.html')) currentStep = 'analysis';
  else if (path.includes('preprocessing.html')) currentStep = 'preprocessing';
  else if (path.includes('training.html')) currentStep = 'training';
  else if (path.includes('export.html')) currentStep = 'export';

  const steps = [
    { key: 'upload', label: 'Upload Dataset', link: 'upload.html' },
    { key: 'analysis', label: 'Analysis', link: 'Analysis.html' },
    { key: 'preprocessing', label: 'Preprocessing', link: 'preprocessing.html' },
    { key: 'training', label: 'Training', link: 'training.html' },
    { key: 'export', label: 'Export Model', link: 'export.html' }
  ];

  const isHome = path === '/' || path.endsWith('index.html');
  const linkPrefix = isHome ? 'Frontend/Template/' : '';

  const currentIndex = steps.findIndex(step => step.key === currentStep);
  const hasSessionId = localStorage.getItem('session_id') !== null;

  const headerHtml = `
    <nav class="bg-white/90 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
      <div class="max-w-7xl mx-auto px-8 py-6">
        <div class="flex items-center justify-between mb-4">
          <a href="${isHome ? 'index.html' : '../../index.html'}" class="text-2xl font-bold text-indigo-600">ClickTrain</a>
          <button id="mobileMenuToggle" class="md:hidden text-gray-600">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path>
            </svg>
          </button>
        </div>
        <div class="flex flex-wrap items-center justify-center gap-2 md:gap-4">
          ${steps.map((step, index) => {
            let bgClass = 'bg-gray-100';
            let textClass = 'text-gray-600';
            let borderClass = 'border-gray-300';
            if (index < currentIndex) {
              bgClass = 'bg-indigo-200';
              textClass = 'text-indigo-800';
              borderClass = 'border-indigo-300';
            } else if (index === currentIndex) {
              bgClass = 'bg-indigo-600';
              textClass = 'text-white';
              borderClass = 'border-indigo-600';
            } else {
              bgClass = 'bg-gray-50';
              textClass = 'text-gray-400';
              borderClass = 'border-gray-200';
            }
            
            // Check if step is accessible
            const isCompleted = index < currentIndex;
            const isCurrent = index === currentIndex;
            const isUpload = step.key === 'upload';
            const isAccessible = isUpload || isCompleted || isCurrent || (hasSessionId && index <= currentIndex + 1);
            
            const linkOrSpan = isAccessible ? `a href="${linkPrefix}${step.link}"` : 'span';
            const cursorClass = isAccessible ? 'cursor-pointer hover:shadow-lg' : 'cursor-not-allowed';
            
            // Add click handler for non-accessible steps
            const clickHandler = isAccessible ? '' : `onclick="alert('Please upload a dataset first!'); return false;"`;
            
            return `
              <${linkOrSpan} class="flex items-center px-4 py-2 rounded-lg border ${bgClass} ${textClass} ${borderClass} transition-all duration-300 ${cursorClass} hover-lift" ${clickHandler}>
                ${step.label}
              </${linkOrSpan}>
              ${index < steps.length - 1 ? '<span class="text-gray-400 mx-2">→</span>' : ''}
            `;
          }).join('')}
        </div>
        <div id="mobileMenu" class="hidden md:hidden mt-4 pb-4 border-t border-gray-200">
          <div class="flex flex-col space-y-4 pt-4">
            <a href="${isHome ? '#features' : '../../index.html'}" class="text-gray-600 hover:text-indigo-600 transition-colors duration-300">${isHome ? 'Home' : 'Home'}</a>
            ${steps.map(step => {
              const isAccessible = step.key === 'upload' || hasSessionId;
              const clickHandler = isAccessible ? '' : `onclick="alert('Please upload a dataset first!'); return false;"`;
              return `<a href="${isAccessible ? linkPrefix + step.link : '#'}" class="text-gray-600 hover:text-indigo-600 transition-colors duration-300" ${clickHandler}>${step.label}</a>`;
            }).join('')}
          </div>
        </div>
      </div>
    </nav>
  `;

  document.getElementById('ml-header').innerHTML = headerHtml;

  // Mobile menu toggle
  const toggle = document.getElementById('mobileMenuToggle');
  const menu = document.getElementById('mobileMenu');
  if (toggle && menu) {
    toggle.addEventListener('click', () => {
      menu.classList.toggle('hidden');
    });
  }
}

// Initialize header when DOM is loaded
document.addEventListener('DOMContentLoaded', createMLHeader);