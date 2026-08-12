import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, "..");
const defaultSourceDirs = [
  path.join(projectRoot, "datasets", "intake", "labelme")
];
const outputDir = path.resolve(scriptDir, "..", "datasets", "eyeliner_seg");
const classNames = ["eyeliner"];
const imageExtensions = new Set([".jpg", ".jpeg", ".png", ".webp"]);

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function cleanDir(dir) {
  fs.rmSync(dir, { recursive: true, force: true });
  ensureDir(dir);
}

function normalizePoint(point, width, height) {
  const x = Math.max(0, Math.min(1, point[0] / width));
  const y = Math.max(0, Math.min(1, point[1] / height));
  return [x, y];
}

function toYoloSegLine(shape, width, height) {
  const classIndex = classNames.indexOf(shape.label);
  if (classIndex < 0 || shape.shape_type !== "polygon" || shape.points.length < 3) {
    return null;
  }

  const points = shape.points
    .map((point) => normalizePoint(point, width, height))
    .flat()
    .map((value) => value.toFixed(6));

  return `${classIndex} ${points.join(" ")}`;
}

function copySample(sourceDir, jsonFile, split, sourcePrefix) {
  const jsonPath = path.join(sourceDir, jsonFile);
  const labelme = JSON.parse(fs.readFileSync(jsonPath, "utf8"));
  const imagePath = path.join(sourceDir, labelme.imagePath);
  const imageExt = path.extname(imagePath).toLowerCase();

  if (!imageExtensions.has(imageExt) || !fs.existsSync(imagePath)) {
    throw new Error(`Missing image for ${jsonFile}: ${labelme.imagePath}`);
  }

  const baseName = `${sourcePrefix}__${path.basename(labelme.imagePath, imageExt)}`;
  const outImagePath = path.join(outputDir, "images", split, `${baseName}${imageExt}`);
  const outLabelPath = path.join(outputDir, "labels", split, `${baseName}.txt`);
  const lines = labelme.shapes
    .map((shape) => toYoloSegLine(shape, labelme.imageWidth, labelme.imageHeight))
    .filter(Boolean);

  fs.copyFileSync(imagePath, outImagePath);
  fs.writeFileSync(outLabelPath, `${lines.join("\n")}${lines.length ? "\n" : ""}`);

  return {
    image: outImagePath,
    label: outLabelPath,
    objects: lines.length
  };
}

function main() {
  const sourceDirs = getSourceDirs();
  cleanDir(outputDir);
  ["train", "val"].forEach((split) => {
    ensureDir(path.join(outputDir, "images", split));
    ensureDir(path.join(outputDir, "labels", split));
  });

  const samples = sourceDirs.flatMap((sourceDir) => {
    const sourcePrefix = path.basename(sourceDir).replace(/[^a-z0-9]+/gi, "_").toLowerCase();
    return fs.readdirSync(sourceDir)
      .filter((file) => file.endsWith(".json"))
      .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
      .map((jsonFile) => ({ sourceDir, sourcePrefix, jsonFile }));
  });

  const manifest = [];
  samples.forEach(({ sourceDir, sourcePrefix, jsonFile }, index) => {
    const split = index % 5 === 0 ? "val" : "train";
    manifest.push({
      sourceDir,
      jsonFile,
      split,
      ...copySample(sourceDir, jsonFile, split, sourcePrefix)
    });
  });

  const yaml = [
    "path: .",
    "train: images/train",
    "val: images/val",
    "names:",
    ...classNames.map((name, index) => `  ${index}: ${name}`),
    ""
  ].join("\n");

  fs.writeFileSync(path.join(outputDir, "data.yaml"), yaml);
  fs.writeFileSync(path.join(outputDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);

  const trainCount = manifest.filter((item) => item.split === "train").length;
  const valCount = manifest.filter((item) => item.split === "val").length;
  const objectCount = manifest.reduce((sum, item) => sum + item.objects, 0);

  console.log(JSON.stringify({
    outputDir,
    sourceDirs,
    jsonFiles: samples.length,
    trainCount,
    valCount,
    objectCount,
    classes: classNames
  }, null, 2));
}

function getSourceDirs() {
  const requested = [];
  process.argv.slice(2).forEach((arg, index, args) => {
    if (arg === "--source" && args[index + 1]) requested.push(path.resolve(args[index + 1]));
  });

  const sourceDirs = (requested.length ? requested : defaultSourceDirs)
    .filter((dir) => fs.existsSync(dir));

  if (!sourceDirs.length) {
    throw new Error("No LabelMe source directories found. Pass one or more --source paths.");
  }

  return sourceDirs;
}

main();
