#!/usr/bin/env node
/** Apply { "relPath": "raw MCP body with Resource header" } from stdin JSON */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");

function stripMcpHeader(text) {
  const m = text.match(/^Resource: [^\n]+\n\n([\s\S]*)$/);
  if (m) return m[1];
  if (text.includes("not found") && text.includes("Figma Debug UUID")) return null;
  return text;
}

const batch = JSON.parse(fs.readFileSync(0, "utf8"));
let written = 0;
let skipped = 0;

for (const [relPath, raw] of Object.entries(batch)) {
  if (relPath.includes("src/imports/pasted_text/")) {
    skipped++;
    continue;
  }
  const content = stripMcpHeader(String(raw).trimEnd() + "\n");
  if (content === null) {
    console.warn(`SKIP (not in Make): ${relPath}`);
    skipped++;
    continue;
  }
  const out = path.join(root, relPath);
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, content.endsWith("\n") ? content : content + "\n");
  written++;
}

console.log(JSON.stringify({ written, skipped }));
