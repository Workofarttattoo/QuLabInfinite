# QuLab Infinite GUI Integration Plan

**Status:** In-progress (Figma design phase)
**Target:** Complete before Product Hunt launch
**Owner:** @designteam (Figma overhaul in progress)

---

## Overview

The GUI provides a beautiful, user-friendly interface for the three main gateways:

1. **Materials & R&D Dashboard** (MCP first)
2. **Unified API Browser** (REST + WebSocket)
3. **Medical Diagnostics Interface** (10 clinical labs)

**Design Philosophy:** Beautiful enough to impress on Product Hunt, but NOT distract from the powerful APIs underneath. MCP remains the primary programmatic entry point for serious users.

---

## Current State

### Figma Make (canonical)
- **File:** [Connect Frontend to Backend](https://www.figma.com/make/j9P7lJtC6OdLpVY0neIaRt/Connect-Frontend-to-Backend)
- **Local:** `qulab-gui/` (synced React app, 40+ routes)
- **Docs:** `docs/FIGMA_MAKE.md`

### Legacy Figma Design (wireframes only)
- **File:** [QuLab Infinite — Lab Console UX v1](https://www.figma.com/design/N9joP1YMYdWU1kWWIZbTBm)
- **Frames:**
  - `00 BACKEND_TOOL_CONTRACTS` — POST /tools/call examples
  - `01 Boot` — Landing screen
  - `02 Command center` — Main hub (3 wedges)
  - `03 Materials` — Material explorer
  - `04 Chemistry` — Molecular tools
  - `05 R&D orchestration` — Agent/tool playground
  - `06 Unlock synergy` — Integration showcase

### Existing Code
- `spaces/qulab-gui/` — Python backend (app.py, qulab_mcp_server.py)
- **Current state:** Stub implementation, needs frontend integration

---

## Integration Checklist (Before Product Hunt)

### Design Phase (In Progress)
- [x] Core Lab Console frames + backend contract frame (`00 BACKEND_TOOL_CONTRACTS`)
- [x] Collapsed-text publish glitch fixed (auto-resize + R&D section → frame)
- [x] Tactical Glass pass (dark `#131313`, cyan accent, panel chrome)
- [x] Basic prototype links (Boot → Command center → bundles)
- [x] Figma frames `07 Global dashboard`, `08 Medical directory` (aligned to React routes)
- [ ] Stitch HTML paste for pixel-perfect tactical layouts (optional polish)
- [x] Handoff: `docs/FIGMA_BACKEND_WIRING.md` + `qulab-gui/README.md`
- [ ] Accessibility review (WCAG 2.1 AA)

### Frontend Build
- [x] React 18 + TypeScript + Vite + Tailwind (`qulab-gui/`)
- [x] MCP HTTP gateway integration (`/mcp` proxy → :8102)
- [x] Medical directory links (8001–8010 `/docs`)
- [ ] Unified REST :8000 screens (optional; MCP is primary for PH)

### Implementation
- [x] Boot screen (`/`)
- [x] Command center + NL → `POST /tools/call`
- [x] Materials / Chemistry / R&D bundle pages
- [x] Medical directory tiles
- [x] Synergy unlock (`/unlock`)
- [x] Tool trace panel
- [ ] 3D structure viewer (Mol*)
- [ ] Settings & API key UI

### Integration with Startup Scripts
- [x] `LAUNCH_PRODUCT_HUNT.sh` builds/serves GUI on :3000
- [x] Auto-open browser when `open` is available
- [x] Graceful fallback if GUI build fails

### Testing
- [ ] Test all 3 gateways through GUI
- [ ] Test on Chrome, Firefox, Safari
- [ ] Test on mobile/tablet (responsive)
- [ ] Load test (many simultaneous requests)

### Documentation
- [ ] Update README with GUI section
- [ ] Add screenshot/GIF to landing
- [ ] Document GUI architecture in ARCHITECTURE.md
- [ ] Add GUI troubleshooting to GETTING_STARTED.md

---

## Technical Approach

### Recommended Stack
```
Frontend:        React 18 + TypeScript
UI Library:      shadcn/ui (uses Tailwind)
HTTP Client:     axios or fetch API
WebSocket:       native WebSocket API
State Mgmt:      Zustand or TanStack Query
Build Tool:      Vite
Package Manager: npm or pnpm
```

### Folder Structure
```
qulab-gui/
├── src/
│   ├── components/          # UI components (per Figma)
│   ├── pages/              # Route pages (boot, materials, medical, etc.)
│   ├── hooks/              # Custom hooks (useAPI, useMCP, etc.)
│   ├── services/           # API clients (materials, medical, tools)
│   ├── styles/             # Tailwind + custom CSS (Glass OS theme)
│   ├── App.tsx
│   └── main.tsx
├── public/
├── package.json
├── vite.config.ts
├── tailwind.config.ts
└── tsconfig.json
```

### API Client Pattern
```typescript
// src/services/mcp.ts
export class MCPClient {
  baseURL = process.env.REACT_APP_MCP_URL || 'http://localhost:8102'

  async callTool(tool: string, params: Record<string, any>) {
    const response = await fetch(`${this.baseURL}/tools/call`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tool, params })
    })
    return response.json()
  }

  async getFeatured() {
    return fetch(`${this.baseURL}/featured`).then(r => r.json())
  }

  async getTools(department?: string) {
    return fetch(`${this.baseURL}/tools?department=${department || 'materials_rd'}`)
      .then(r => r.json())
  }
}
```

### Color Scheme (Tactical Glass OS)
- **Background:** Near-black `#131313`
- **Accent cyan:** `#00dbe9` / `#00f0ff`
- **Secondary green:** `#13ff43` / `#00e639`
- **Text:** `#f0f0f0` on dark, `#131313` on light
- **Borders:** 0.5px light border, subtle backdrop blur
- **Optional:** Scanline overlay at very low opacity

### Typography
- **Headings:** Space Grotesk
- **Body:** Inter or system sans
- **Monospace:** JetBrains Mono (code, labels, data)

---

## Startup Integration

### Updated LAUNCH_PRODUCT_HUNT.sh
```bash
# Start MCP Server
python unified_mcp_server.py &

# Start Unified API
uvicorn api.unified_api:app --host 0.0.0.0 --port 8000 &

# Start Medical Labs
LAB_HOST=0.0.0.0 LAB_PORT_PREFIX=800 bash scripts/start_medical_labs.sh &

# Start GUI (if built)
if [ -d "qulab-gui/dist" ]; then
  cd qulab-gui
  npm run preview &  # Serve built GUI
  sleep 2
  echo "Opening GUI at http://localhost:3000"
  open http://localhost:3000
fi

# Wait for all services
wait
```

### Build Process
```bash
# Development
cd qulab-gui
npm run dev          # Start dev server + hot reload

# Production
npm run build        # Build optimized bundle
npm run preview      # Test production build locally
```

---

## Success Criteria

✅ **Visual:** Beautiful, matches Figma design
✅ **Functional:** All 3 gateways accessible through GUI
✅ **Responsive:** Works on desktop (Product Hunt demo focus)
✅ **Fast:** <2s page loads, <1s API responses
✅ **Accessible:** Keyboard navigation, screen reader support
✅ **Integrated:** Starts with `bash LAUNCH_PRODUCT_HUNT.sh`
✅ **Documented:** Clear setup + troubleshooting steps

---

## Timeline

| Phase | Tasks | Duration | Owner |
|-------|-------|----------|-------|
| **Design** | Figma overhaul + handoff | In progress | @you |
| **Setup** | Project scaffold, build config | 2-3 hours | @engineer |
| **Build** | Implement all 6 screens | 2-3 days | @engineer |
| **Integration** | Connect APIs, test flows | 1 day | @engineer |
| **Polish** | Performance, accessibility, docs | 1 day | @engineer + @you |
| **Launch Ready** | Final testing, deployment prep | 0.5 day | @engineer |

**Total:** ~5-6 days after Figma design is complete

---

## Fallback Plan

If GUI isn't ready before Product Hunt:
1. **Ship API-only** with Swagger docs
2. **Link to Figma design** in README ("Live prototype coming")
3. **Use Stitch** to demonstrate Figma → API wiring
4. **Ship GUI in v1.1** (week after launch)

This is actually fine—Product Hunt respects shipping quickly with a clear plan.

---

## Questions for Design Handoff

When your Figma overhaul is ready:

1. **Export format:** Do you want to use Stitch (Figma → Live prototype) or just design specs?
2. **Component library:** Will you provide a Figma component library for handoff?
3. **Interactions:** Are the Figma prototypes interactive, or design-only?
4. **Assets:** Any icons, illustrations, or brand assets to export?
5. **Accessibility:** Have you noted color contrast ratios, text sizes, etc.?

---

## Reference Links

- **Figma Design:** [Lab Console v1](https://www.figma.com/design/N9joP1YMYdWU1kWWIZbTBm)
- **Backend Wiring:** [docs/FIGMA_BACKEND_WIRING.md](docs/FIGMA_BACKEND_WIRING.md)
- **API Reference:** http://localhost:8000/docs (live)
- **MCP Tools:** http://localhost:8102/featured (live)

---

## Notes

- **MCP remains first-class:** The GUI is beautiful, but MCP is the power user entry point. Keep it that way.
- **API-first design:** Every GUI feature should expose the underlying API clearly (show request/response in dev tools).
- **No feature lock-in:** Users should never feel locked into the GUI; they can always drop to raw APIs.
- **Open source GUI:** Consider making the GUI code a separate repo (qulab-gui) so others can extend it.

---

**Status: Ready for design handoff. Once Figma is complete, engineer can begin build.**

