import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline } from "node:stream/promises";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const defaultCsv = path.join(projectRoot, "datasets", "intake", "permitted_image_urls.csv");
const intakeRawDir = path.join(projectRoot, "datasets", "intake", "raw");
const sourcesCsv = path.join(projectRoot, "datasets", "intake", "sources.csv");

const allowedStates = new Set([
  "00_no_liner",
  "01_pen_near_eye",
  "02_root_mark",
  "03_tail_only",
  "04_return_line",
  "05_filled_wing",
  "06_too_long",
  "07_too_low",
  "08_too_thick",
  "09_occluded_hand"
]);

const extensionByType = new Map([
  ["image/jpeg", ".jpg"],
  ["image/jpg", ".jpg"],
  ["image/png", ".png"],
  ["image/webp", ".webp"],
  ["image/heic", ".heic"],
  ["image/heif", ".heif"]
]);

const dryRun = process.argv.includes("--dry-run");
const csvArg = process.argv.slice(2).find((arg) => !arg.startsWith("--"));
const csvPath = csvArg ? path.resolve(csvArg) : defaultCsv;

if (!fs.existsSync(csvPath)) {
  console.error(`Missing CSV: ${csvPath}`);
  process.exit(1);
}

const records = parseCsv(fs.readFileSync(csvPath, "utf8"))
  .filter((record) => record.url && record.url !== "https://example.com/permitted-tail-only.jpg");

if (!records.length) {
  console.log(`No image rows found in ${csvPath}`);
  process.exit(0);
}

ensureSourcesHeader();

let imported = 0;
let skipped = 0;

for (const [index, record] of records.entries()) {
  const state = normalizeState(record.state);
  if (!allowedStates.has(state)) {
    console.warn(`Skipping row ${index + 2}: unknown state "${record.state}"`);
    skipped += 1;
    continue;
  }

  try {
    const result = await importImage(record, state, index);
    if (result) imported += 1;
  } catch (error) {
    skipped += 1;
    console.warn(`Skipping row ${index + 2}: ${error.message}`);
  }
}

console.log(`Imported ${imported} image(s). Skipped ${skipped}.`);

async function importImage(record, state, index) {
  const url = record.url.trim();
  const response = await fetch(url, {
    headers: {
      "User-Agent": "swatch-dataset-intake/0.1"
    }
  });

  if (!response.ok) {
    throw new Error(`download failed ${response.status} ${response.statusText}`);
  }

  const contentType = response.headers.get("content-type")?.split(";")[0].toLowerCase() || "";
  const extension = getExtension(url, contentType);
  if (!extension) {
    throw new Error(`unsupported content type "${contentType || "unknown"}"`);
  }

  const stateDir = path.join(intakeRawDir, state);
  fs.mkdirSync(stateDir, { recursive: true });

  const filename = `${state}_${String(Date.now()).slice(-8)}_${String(index + 1).padStart(3, "0")}${extension}`;
  const outputPath = path.join(stateDir, filename);

  if (dryRun) {
    console.log(`[dry-run] ${url} -> ${path.relative(projectRoot, outputPath)}`);
    return true;
  }

  await pipeline(response.body, fs.createWriteStream(outputPath));
  appendSource({
    filename: path.relative(projectRoot, outputPath),
    state,
    sourceUrl: record.source_url || url,
    sourceType: record.source_type || "permitted_url",
    permissionNotes: record.permission_notes || "Permission confirmed by user",
    personConsent: record.person_consent || "",
    creator: record.creator || "",
    license: record.license || "",
    licenseUrl: record.license_url || ""
  });
  console.log(`Imported ${path.relative(projectRoot, outputPath)}`);
  return true;
}

function normalizeState(state) {
  const value = String(state || "").trim();
  if (allowedStates.has(value)) return value;

  const found = [...allowedStates].find((folder) => folder.endsWith(value));
  return found || value;
}

function getExtension(url, contentType) {
  const pathname = new URL(url).pathname.toLowerCase();
  const fromPath = path.extname(pathname);
  if ([".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"].includes(fromPath)) {
    return fromPath === ".jpeg" ? ".jpg" : fromPath;
  }

  return extensionByType.get(contentType) || "";
}

function ensureSourcesHeader() {
  if (fs.existsSync(sourcesCsv)) return;

  fs.writeFileSync(
    sourcesCsv,
    "filename,state,source_url,source_type,permission_notes,person_consent,creator,license,license_url\n"
  );
}

function appendSource({
  filename,
  state,
  sourceUrl,
  sourceType,
  permissionNotes,
  personConsent,
  creator,
  license,
  licenseUrl
}) {
  const row = [
    filename,
    state,
    sourceUrl,
    sourceType,
    permissionNotes,
    personConsent,
    creator,
    license,
    licenseUrl
  ].map(csvEscape).join(",");

  fs.appendFileSync(sourcesCsv, `${row}\n`);
}

function parseCsv(text) {
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = splitCsvLine(headerLine).map((header) => header.trim());

  return lines
    .filter((line) => line.trim() && !line.trim().startsWith("#"))
    .map((line) => {
      const values = splitCsvLine(line);
      return Object.fromEntries(headers.map((header, index) => [header, values[index]?.trim() || ""]));
    });
}

function splitCsvLine(line) {
  const values = [];
  let current = "";
  let quoted = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const next = line[i + 1];

    if (char === "\"" && quoted && next === "\"") {
      current += "\"";
      i += 1;
    } else if (char === "\"") {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }

  values.push(current);
  return values;
}

function csvEscape(value) {
  const text = String(value || "");
  if (!/[",\n]/.test(text)) return text;
  return `"${text.replaceAll("\"", "\"\"")}"`;
}
