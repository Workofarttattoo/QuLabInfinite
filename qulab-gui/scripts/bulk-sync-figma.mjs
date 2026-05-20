#!/usr/bin/env node
/**
 * Fetch all missing paths from sync-figma-paths.txt via stdin-delimited MCP bodies.
 * Parent agent pipes FetchMcpResource results as:
 *   ---FILE:relative/path---
 *   <raw mcp body>
 *
 * Or apply pre-fetched raw files from .tmp-figma-raw/
 */
import fs from "fs";
import path from "path";
import { spawnSync } from "child_process";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const pathsFile = path.join(root, "scripts", "sync-figma-paths.txt");
const paths = fs
  .readFileSync(pathsFile, "utf8")
  .split("\n")
  .map((l) => l.trim())
  .filter(Boolean);

let written = 0;
let skipped = 0;
const skippedPaths = [];

for (const relPath of paths) {
  if (relPath.includes("src/imports/pasted_text/")) {
    skipped++;
    skippedPaths.push(relPath);
    continue;
  }
  if (fs.existsSync(path.join(root, relPath))) {
    skipped++;
    continue;
  }
  const rawPath = path.join(root, ".tmp-figma-raw", relPath);
  if (!fs.existsSync(rawPath)) {
    skipped++;
    skippedPaths.push(relPath);
    continue;
  }
  const raw = fs.readFileSync(rawPath, "utf8");
  const r = spawnSync("node", ["scripts/sync-from-figma.mjs", relPath], {
    cwd: root,
    input: raw,
    encoding: "utf8",
  });
  if (r.stdout?.includes("WROTE")) written++;
  else skipped++;
}

console.log(JSON.stringify({ written, skipped, skippedPaths: skippedPaths.slice(0, 20) }));
