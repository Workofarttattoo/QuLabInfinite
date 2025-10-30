# QuLabInfinite Website

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

## Overview

Professional marketing website for QuLabInfinite, designed to be hosted at **https://www.QuLab.AioS.is**.

## Features

- 🎨 Modern, dark-themed design with gradient accents
- 📱 Fully responsive (mobile, tablet, desktop)
- ⚡ Performance optimized with smooth animations
- 🔒 Production API showcase
- 📊 Live metrics and performance stats
- 🧠 ECH0 14B integration highlights
- 📘 Comprehensive documentation links

## Structure

```
website/
├── index.html          # Main landing page
├── styles.css          # Complete styling
├── script.js           # Interactive features
└── README.md           # This file
```

## Deployment Options

### Option 1: GitHub Pages

1. Push to GitHub repository
2. Go to Settings → Pages
3. Source: Deploy from branch `main`
4. Folder: `/website` (or root if moved)
5. Configure custom domain: `www.QuLab.AioS.is`
6. Add CNAME record in DNS:
   ```
   www.QuLab.AioS.is CNAME [your-github-username].github.io
   ```

### Option 2: Vercel

```bash
cd website
vercel --prod

# Configure custom domain in Vercel dashboard:
# www.QuLab.AioS.is
```

### Option 3: Netlify

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Deploy
cd website
netlify deploy --prod

# Configure custom domain in Netlify dashboard
```

### Option 4: AWS S3 + CloudFront

```bash
# Install AWS CLI
aws s3 sync . s3://qulab-website --delete

# Configure CloudFront distribution with:
# - Origin: S3 bucket
# - SSL certificate: ACM
# - Custom domain: www.QuLab.AioS.is
```

### Option 5: Docker + Nginx

```dockerfile
# Dockerfile
FROM nginx:alpine
COPY . /usr/share/nginx/html
EXPOSE 80
```

```bash
docker build -t qulab-website .
docker run -d -p 80:80 qulab-website
```

## DNS Configuration

Since you've already set up the subdomain with CNAME, ensure:

```
Type: CNAME
Name: www.QuLab
Host: [your-hosting-provider]
TTL: 3600
```

## Local Development

```bash
# Simple HTTP server (Python)
cd website
python3 -m http.server 8080

# Or with Node.js
npx http-server . -p 8080

# Open browser
open http://localhost:8080
```

## Content Updates

### Updating Stats

Edit `index.html` line ~90 (hero-stats section):
```html
<div class="stat-number">1,059</div>  <!-- Update this -->
```

### Updating Features

Edit `index.html` starting line ~130 (features-grid section):
```html
<div class="feature-card">
  <div class="feature-icon">🧬</div>
  <h3>Your Feature Title</h3>
  <p>Your feature description</p>
</div>
```

### Updating API Examples

Edit `index.html` line ~250 (code-block section):
```html
<pre><code># Your API example here</code></pre>
```

## Performance Checklist

- ✅ Minified CSS/JS (optional, currently readable for development)
- ✅ Optimized images (when added)
- ✅ Lazy loading animations
- ✅ Smooth scrolling
- ✅ Mobile-responsive
- ✅ SEO meta tags included

## SEO

The website includes:
- Meta description
- Keywords
- Semantic HTML
- Fast load times
- Mobile-first design

Consider adding:
- `sitemap.xml`
- `robots.txt`
- Open Graph tags
- Twitter Card tags

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Analytics (Optional)

Add Google Analytics:
```html
<!-- Add before </head> -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_TRACKING_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_TRACKING_ID');
</script>
```

## Maintenance

Regular updates:
- API endpoint examples (as API evolves)
- Performance metrics (from actual production data)
- Feature highlights (new capabilities)
- Documentation links (keep in sync with docs)

## Support

For website-related issues:
- Email: joshua@corporationoflight.com
- GitHub: https://github.com/Workofarttattoo/QuLabInfinite

---

**Generated:** October 30, 2025
**Status:** Production Ready
**URL:** https://www.QuLab.AioS.is
