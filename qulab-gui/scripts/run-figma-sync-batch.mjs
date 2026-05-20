#!/usr/bin/env node
/**
 * Apply MCP FetchMcpResource bodies from a JSON file:
 *   { "relative/path": "Resource: ...\\n\\n<body>" }
 * Usage: node scripts/run-figma-sync-batch.mjs /path/to/batch.json
 */
import fs from "fs";
import { spawnSync } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const batchPath = process.argv[2];
if (!batchPath) {
  console.error("Usage: run-figma-sync-batch.mjs <batch.json>");
  process.exit(1);
}

const batch = JSON.parse(fs.readFileSync(batchPath, "utf8"));
let written = 0;
let skipped = 0;

for (const [relPath, raw] of Object.entries(batch)) {
  if (relPath.includes("src/imports/pasted_text/")) {
    skipped++;
    continue;
  }
  const r = spawnSync("node", ["scripts/sync-from-figma.mjs", relPath], {
    cwd: root,
    input: String(raw),
    encoding: "utf8",
  });
  if (r.stdout?.includes("WROTE")) written++;
  else if (r.stdout?.includes("SKIP") || r.stderr?.includes("SKIP")) skipped++;
  else if (r.status !== 0) {
    console.error(relPath, r.stderr || r.stdout);
  }
}

console.log(JSON.stringify({ written, skipped }));
