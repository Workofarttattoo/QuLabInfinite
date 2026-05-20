#!/usr/bin/env node
/**
 * Write Figma Make file content (stdin) to qulab-gui/<relPath>.
 * Strips "Resource: ...\n\n" header from MCP FetchMcpResource output.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const relPath = process.argv[2];
if (!relPath) {
  console.error("Usage: sync-from-figma.mjs <relative-path>");
  process.exit(1);
}

let raw = "";
for await (const chunk of process.stdin) raw += chunk;

function stripMcpHeader(text) {
  const m = text.match(/^Resource: [^\n]+\n\n([\s\S]*)$/);
  if (m) return m[1];
  // Missing file from Figma
  if (text.includes("not found") && text.includes("Figma Debug UUID")) {
    return null;
  }
  return text;
}

const content = stripMcpHeader(raw.trimEnd() + "\n");
if (content === null) {
  console.warn(`SKIP (not in Make): ${relPath}`);
  process.exit(0);
}

const out = path.join(root, relPath);
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, content.endsWith("\n") ? content : content + "\n");
console.log(`WROTE ${relPath} (${content.length} bytes)`);
