/*
 * Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
 * QuLab Infinite - Main JavaScript
 */

// Theme Management
const initTheme = () => {
  const savedTheme = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);
};

const toggleTheme = () => {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const newTheme = currentTheme === 'light' ? 'dark' : 'light';

  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
  updateThemeIcon(newTheme);
};

const updateThemeIcon = (theme) => {
  const themeToggle = document.querySelector('.theme-toggle');
  if (themeToggle) {
    themeToggle.innerHTML = theme === 'light'
      ? '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>'
      : '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>';
  }
};

// Navigation
const initNavigation = () => {
  const nav = document.querySelector('nav');
  const mobileToggle = document.querySelector('.mobile-toggle');
  const navMenu = document.querySelector('.nav-menu');

  // Scroll behavior
  let lastScroll = 0;
  window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;

    if (currentScroll > 50) {
      nav.classList.add('scrolled');
    } else {
      nav.classList.remove('scrolled');
    }

    lastScroll = currentScroll;
  });

  // Mobile menu
  if (mobileToggle) {
    mobileToggle.addEventListener('click', () => {
      navMenu.classList.toggle('active');
      mobileToggle.classList.toggle('active');
    });
  }

  // Close mobile menu on link click
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
      navMenu.classList.remove('active');
      mobileToggle.classList.remove('active');
    });
  });

  // Set active nav item
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-link').forEach(link => {
    if (link.getAttribute('href') === currentPage) {
      link.classList.add('active');
    }
  });
};

// Search Functionality
const initSearch = () => {
  const searchInput = document.querySelector('.search-input');
  const labCards = document.querySelectorAll('.lab-card');

  if (searchInput && labCards.length > 0) {
    searchInput.addEventListener('input', (e) => {
      const searchTerm = e.target.value.toLowerCase();

      labCards.forEach(card => {
        const title = card.querySelector('.lab-title').textContent.toLowerCase();
        const description = card.querySelector('.lab-description').textContent.toLowerCase();
        const category = card.querySelector('.lab-category')?.textContent.toLowerCase() || '';

        if (title.includes(searchTerm) || description.includes(searchTerm) || category.includes(searchTerm)) {
          card.style.display = '';
          card.style.animation = 'fadeInUp 0.5s ease-out';
        } else {
          card.style.display = 'none';
        }
      });

      // Show no results message if needed
      const visibleCards = document.querySelectorAll('.lab-card:not([style*="display: none"])');
      const noResults = document.querySelector('.no-results');

      if (visibleCards.length === 0 && noResults) {
        noResults.style.display = 'block';
      } else if (noResults) {
        noResults.style.display = 'none';
      }
    });
  }
};

// Filter Functionality
const initFilters = () => {
  const filterButtons = document.querySelectorAll('.filter-btn');
  const labCards = document.querySelectorAll('.lab-card');

  filterButtons.forEach(button => {
    button.addEventListener('click', () => {
      const filter = button.getAttribute('data-filter');

      // Update active state
      filterButtons.forEach(btn => btn.classList.remove('active'));
      button.classList.add('active');

      // Filter cards
      labCards.forEach(card => {
        const category = card.getAttribute('data-category');

        if (filter === 'all' || category === filter) {
          card.style.display = '';
          card.style.animation = 'fadeInUp 0.5s ease-out';
        } else {
          card.style.display = 'none';
        }
      });
    });
  });
};

// Smooth Scroll
const initSmoothScroll = () => {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });
};

// Intersection Observer for animations
const initAnimations = () => {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('animate-in');
      }
    });
  }, {
    threshold: 0.1
  });

  document.querySelectorAll('.card, .feature-card, .lab-card').forEach(el => {
    observer.observe(el);
  });
};

// Lab Demo Modal
const initLabDemos = () => {
  const demoButtons = document.querySelectorAll('.btn-demo');

  demoButtons.forEach(button => {
    button.addEventListener('click', (e) => {
      e.preventDefault();
      const labId = button.getAttribute('data-lab');
      openLabDemo(labId);
    });
  });
};

const openLabDemo = (labId) => {
  // Create modal
  const modal = document.createElement('div');
  modal.className = 'lab-modal';
  modal.innerHTML = `
    <div class="modal-content">
      <div class="modal-header">
        <h3>Loading ${labId} Demo...</h3>
        <button class="modal-close">&times;</button>
      </div>
      <div class="modal-body">
        <iframe src="../guis/${labId}/index.html" frameborder="0"></iframe>
      </div>
    </div>
  `;

  document.body.appendChild(modal);

  // Add styles
  const style = document.createElement('style');
  style.textContent = `
    .lab-modal {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0,0,0,0.8);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10000;
      animation: fadeIn 0.3s;
    }
    .modal-content {
      background: white;
      border-radius: 1rem;
      width: 90%;
      height: 90%;
      max-width: 1400px;
      display: flex;
      flex-direction: column;
    }
    .modal-header {
      padding: 1.5rem;
      border-bottom: 1px solid #e5e7eb;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .modal-close {
      background: none;
      border: none;
      font-size: 2rem;
      cursor: pointer;
      color: #6b7280;
      transition: color 0.3s;
    }
    .modal-close:hover {
      color: #1f2937;
    }
    .modal-body {
      flex: 1;
      padding: 1rem;
    }
    .modal-body iframe {
      width: 100%;
      height: 100%;
      border-radius: 0.5rem;
    }
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
  `;
  document.head.appendChild(style);

  // Close modal
  modal.querySelector('.modal-close').addEventListener('click', () => {
    modal.remove();
    style.remove();
  });

  modal.addEventListener('click', (e) => {
    if (e.target === modal) {
      modal.remove();
      style.remove();
    }
  });
};

// Copy Code Functionality
const initCodeCopy = () => {
  document.querySelectorAll('pre code').forEach(block => {
    const button = document.createElement('button');
    button.className = 'copy-code-btn';
    button.textContent = 'Copy';

    button.addEventListener('click', () => {
      navigator.clipboard.writeText(block.textContent).then(() => {
        button.textContent = 'Copied!';
        setTimeout(() => {
          button.textContent = 'Copy';
        }, 2000);
      });
    });

    const pre = block.parentElement;
    pre.style.position = 'relative';
    pre.appendChild(button);
  });
};

// Stats Counter Animation
const animateStats = () => {
  const stats = document.querySelectorAll('.stat-number');

  stats.forEach(stat => {
    const target = parseInt(stat.getAttribute('data-target'));
    const duration = 2000;
    const increment = target / (duration / 16);
    let current = 0;

    const updateCounter = () => {
      current += increment;
      if (current < target) {
        stat.textContent = Math.floor(current).toLocaleString();
        requestAnimationFrame(updateCounter);
      } else {
        stat.textContent = target.toLocaleString();
      }
    };

    // Start animation when in view
    const observer = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) {
        updateCounter();
        observer.disconnect();
      }
    });

    observer.observe(stat);
  });
};

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initNavigation();
  initSearch();
  initFilters();
  initSmoothScroll();
  initAnimations();
  initLabDemos();
  initCodeCopy();
  animateStats();

  // Add theme toggle listener
  const themeToggle = document.querySelector('.theme-toggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', toggleTheme);
  }

  // Add loading class removal
  document.body.classList.add('loaded');

  // Initialize tooltips
  const tooltips = document.querySelectorAll('[data-tooltip]');
  tooltips.forEach(el => {
    el.addEventListener('mouseenter', (e) => {
      const tooltip = document.createElement('div');
      tooltip.className = 'tooltip';
      tooltip.textContent = e.target.getAttribute('data-tooltip');
      document.body.appendChild(tooltip);

      const rect = e.target.getBoundingClientRect();
      tooltip.style.position = 'fixed';
      tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';
      tooltip.style.left = rect.left + rect.width / 2 - tooltip.offsetWidth / 2 + 'px';
    });

    el.addEventListener('mouseleave', () => {
      document.querySelectorAll('.tooltip').forEach(t => t.remove());
    });
  });
});

// Page-specific initializations
window.initLabsPage = () => {
  console.log('Labs page initialized');
  // Add any labs-specific functionality here
};

window.initQuPharmaPage = () => {
  console.log('QuPharma page initialized');
  // Add QuPharma-specific functionality here
  initMoleculeAnimation();
};

// Molecule animation for QuPharma
const initMoleculeAnimation = () => {
  const canvas = document.getElementById('molecule-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;

  const molecules = [];
  const numMolecules = 20;

  class Molecule {
    constructor() {
      this.x = Math.random() * canvas.width;
      this.y = Math.random() * canvas.height;
      this.vx = (Math.random() - 0.5) * 2;
      this.vy = (Math.random() - 0.5) * 2;
      this.radius = Math.random() * 3 + 2;
    }

    update() {
      this.x += this.vx;
      this.y += this.vy;

      if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
      if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
    }

    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(99, 102, 241, 0.6)';
      ctx.fill();

      // Draw connections
      molecules.forEach(other => {
        const dist = Math.hypot(this.x - other.x, this.y - other.y);
        if (dist < 100 && dist > 0) {
          ctx.beginPath();
          ctx.moveTo(this.x, this.y);
          ctx.lineTo(other.x, other.y);
          ctx.strokeStyle = `rgba(99, 102, 241, ${1 - dist / 100})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      });
    }
  }

  for (let i = 0; i < numMolecules; i++) {
    molecules.push(new Molecule());
  }

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    molecules.forEach(m => {
      m.update();
      m.draw();
    });
    requestAnimationFrame(animate);
  }

  animate();
};

// Export functions for use in other scripts
window.QuLabInfinite = {
  openLabDemo,
  initSearch,
  initFilters,
  toggleTheme
};