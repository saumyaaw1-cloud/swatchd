import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const intakeDir = path.resolve(__dirname, "..", "datasets", "intake", "raw");
const imageExtensions = new Set([".jpg", ".jpeg", ".png", ".webp", ".heic"]);
const targets = {
  "00_no_liner": 80,
  "01_pen_near_eye": 80,
  "02_root_mark": 80,
  "03_tail_only": 80,
  "04_return_line": 80,
  "05_filled_wing": 80,
  "06_too_long": 40,
  "07_too_low": 40,
  "08_too_thick": 40,
  "09_occluded_hand": 40
};

function countImages(folder) {
  const fullPath = path.join(intakeDir, folder);
  if (!fs.existsSync(fullPath)) return 0;

  return fs.readdirSync(fullPath)
    .filter((file) => imageExtensions.has(path.extname(file).toLowerCase()))
    .length;
}

const rows = Object.entries(targets).map(([folder, target]) => {
  const count = countImages(folder);
  return {
    folder,
    count,
    target,
    remaining: Math.max(target - count, 0),
    ready: count >= target
  };
});

const total = rows.reduce((sum, row) => sum + row.count, 0);
const targetTotal = rows.reduce((sum, row) => sum + row.target, 0);

console.table(rows);
console.log(`Total images: ${total}/${targetTotal}`);

if (rows.some((row) => !row.ready)) {
  console.log("Next priority folders:");
  rows
    .filter((row) => !row.ready)
    .sort((a, b) => b.remaining - a.remaining)
    .slice(0, 4)
    .forEach((row) => {
      console.log(`- ${row.folder}: add ${row.remaining}`);
    });
}
