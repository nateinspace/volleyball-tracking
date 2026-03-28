import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import ffmpegPath from "ffmpeg-static";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, "..");
const input = path.join(root, "public/videos/corrected_overlay.mp4");
const output = path.join(root, "public/videos/corrected_overlay.web.mp4");

if (!existsSync(input)) {
  console.error("Missing:", input);
  process.exit(1);
}
if (!ffmpegPath) {
  console.error("ffmpeg-static binary not found");
  process.exit(1);
}

const args = [
  "-y",
  "-i",
  input,
  "-c:v",
  "libx264",
  "-profile:v",
  "main",
  "-pix_fmt",
  "yuv420p",
  "-preset",
  "fast",
  "-crf",
  "23",
  "-an",
  "-movflags",
  "+faststart",
  output,
];

console.log("Transcoding to H.264 (web / <video>) …");
const ff = spawn(ffmpegPath, args, { stdio: "inherit" });
ff.on("exit", (code) => {
  if (code === 0) {
    console.log("Wrote:", output);
  } else {
    process.exit(code ?? 1);
  }
});
