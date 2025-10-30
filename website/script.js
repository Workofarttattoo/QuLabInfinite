// Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

// Smooth scrolling for anchor links
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

// Copy code button functionality
document.querySelectorAll('.copy-btn').forEach(button => {
    button.addEventListener('click', async function() {
        const codeBlock = this.closest('.code-block').querySelector('code');
        const code = codeBlock.textContent;

        try {
            await navigator.clipboard.writeText(code);
            this.textContent = 'Copied!';
            setTimeout(() => {
                this.textContent = 'Copy';
            }, 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
            this.textContent = 'Error';
        }
    });
});

// Navbar transparency on scroll
const navbar = document.querySelector('.navbar');
let lastScroll = 0;

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;

    if (currentScroll > 100) {
        navbar.style.background = 'rgba(15, 23, 42, 0.98)';
    } else {
        navbar.style.background = 'rgba(15, 23, 42, 0.95)';
    }

    lastScroll = currentScroll;
});

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all feature cards and doc cards
document.querySelectorAll('.feature-card, .doc-card, .performance-card, .ech0-feature').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
});

// Stats counter animation
const animateValue = (element, start, end, duration) => {
    const range = end - start;
    const increment = end > start ? 1 : -1;
    const stepTime = Math.abs(Math.floor(duration / range));
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        element.textContent = current.toLocaleString();
        if (current === end) {
            clearInterval(timer);
        }
    }, stepTime);
};

// Trigger stat animations when in viewport
const statObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting && !entry.target.dataset.animated) {
            const number = entry.target.querySelector('.stat-number');
            const value = number.textContent;

            // Only animate if it's a pure number
            if (!isNaN(value.replace(/,/g, ''))) {
                const numValue = parseInt(value.replace(/,/g, ''));
                animateValue(number, 0, numValue, 2000);
                entry.target.dataset.animated = 'true';
            }
        }
    });
}, { threshold: 0.5 });

document.querySelectorAll('.stat').forEach(stat => {
    statObserver.observe(stat);
});

// Mobile menu toggle (if needed in future)
const navToggle = document.querySelector('.nav-toggle');
if (navToggle) {
    navToggle.addEventListener('click', () => {
        document.querySelector('.nav-menu').classList.toggle('active');
    });
}

// Add subtle parallax effect to hero
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const hero = document.querySelector('.hero-content');
    if (hero) {
        hero.style.transform = `translateY(${scrolled * 0.3}px)`;
        hero.style.opacity = 1 - (scrolled / 800);
    }
});

// Initialize
console.log('%cQuLabInfinite', 'font-size: 24px; font-weight: bold; color: #6366f1;');
console.log('%cProduction-Ready Scientific Computing Platform', 'font-size: 14px; color: #cbd5e1;');
console.log('%c© 2025 Corporation of Light. All Rights Reserved.', 'font-size: 12px; color: #64748b;');
