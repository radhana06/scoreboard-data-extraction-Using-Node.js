'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const sharp = require('sharp');

// ============================================================
// CONFIGURATION
// ============================================================

const BASE_W = 1920;
const BASE_H = 1080;

const PLAYERS = ['J', 'V', 'P', 'T'];

// Scoreboard horizontal boundaries for the supplied video.
const X_EDGES = [
  250, 392, 532, 672, 812, 952,
  1092, 1232, 1372, 1512, 1652, 1880,
];

// Rows:
// notation row + cumulative score row
const ROWS = {
  J: [[126, 182], [184, 258]],
  V: [[263, 321], [322, 417]],
  P: [[421, 479], [480, 570]],
  T: [[572, 630], [631, 760]],
};

const TTL_BOX = [1510, 1880, 120, 760];

// ============================================================
// SAFE IMAGE CROPPING
// ============================================================
// A "frame" / "crop" is a raw pixel image: { data, width, height, channels }

function safeCrop(frame, x1, y1, x2, y2) {
  if (!frame || !frame.data || frame.data.length === 0) {
    return null;
  }

  const w = frame.width;
  const h = frame.height;

  const sx = w / BASE_W;
  const sy = h / BASE_H;

  x1 = Math.trunc(x1 * sx);
  x2 = Math.trunc(x2 * sx);
  y1 = Math.trunc(y1 * sy);
  y2 = Math.trunc(y2 * sy);

  x1 = Math.max(0, Math.min(x1, w - 1));
  x2 = Math.max(0, Math.min(x2, w));

  y1 = Math.max(0, Math.min(y1, h - 1));
  y2 = Math.max(0, Math.min(y2, h));

  if (x2 <= x1 || y2 <= y1) {
    return null;
  }

  const cw = x2 - x1;
  const ch = y2 - y1;
  const c = frame.channels;
  const out = Buffer.alloc(cw * ch * c);

  for (let row = 0; row < ch; row++) {
    const srcStart = ((y1 + row) * w + x1) * c;
    frame.data.copy(out, row * cw * c, srcStart, srcStart + cw * c);
  }

  return { data: out, width: cw, height: ch, channels: c };
}

// ============================================================
// TEXT NORMALIZATION
// ============================================================

function normalizeText(text) {
  if (!text) {
    return '';
  }

  text = text.toUpperCase().trim();

  const replacements = [
    ['×', 'X'],
    ['*', 'X'],
    ['O', '0'],
    ['I', '1'],
    ['L', '1'],
    ['—', '-'],
    ['–', '-'],
    ['_', '-'],
  ];

  for (const [oldValue, newValue] of replacements) {
    text = text.split(oldValue).join(newValue);
  }

  return text.replace(/[^0-9Xx/\-]/g, '');
}

function isValidCandidate(candidate, valid) {
  return (
    valid.has(candidate) ||
    /^[0-9]-$/.test(candidate) ||
    /^[0-9]\/$/.test(candidate) ||
    /^[0-9]{1,2}$/.test(candidate)
  );
}

function normalizeToken(text) {
  text = normalizeText(text);

  if (!text) {
    return '';
  }

  const valid = new Set(['X', '-', '--']);

  if (isValidCandidate(text, valid)) {
    return text;
  }

  // Search inside noisy OCR result
  for (const length of [2, 1]) {
    for (let i = 0; i < text.length - length + 1; i++) {
      const candidate = text.slice(i, i + length);

      if (isValidCandidate(candidate, valid)) {
        return candidate;
      }
    }
  }

  return '';
}

// ============================================================
// IMAGE PREPROCESSING
// ============================================================

// Otsu threshold on a single-channel 8-bit buffer (same behaviour as
// cv2.threshold(..., 0, 255, THRESH_BINARY + THRESH_OTSU)).
function otsuThreshold(gray) {
  const hist = new Array(256).fill(0);
  for (let i = 0; i < gray.length; i++) {
    hist[gray[i]]++;
  }

  const total = gray.length;
  let sum = 0;
  for (let t = 0; t < 256; t++) {
    sum += t * hist[t];
  }

  let sumB = 0;
  let wB = 0;
  let maxVar = -1;
  let threshold = 0;

  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;

    const wF = total - wB;
    if (wF === 0) break;

    sumB += t * hist[t];

    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);

    if (between > maxVar) {
      maxVar = between;
      threshold = t;
    }
  }

  const out = Buffer.alloc(gray.length);
  for (let i = 0; i < gray.length; i++) {
    out[i] = gray[i] > threshold ? 255 : 0;
  }
  return out;
}

// Returns an array of PNG buffers: [enhanced, otsu]
async function preprocessVariants(crop) {
  if (!crop || !crop.data || crop.data.length === 0) {
    return [];
  }

  const newW = crop.width * 6;
  const newH = crop.height * 6;

  // Enlarge 6x (bicubic)
  const enlarged = await sharp(crop.data, {
    raw: { width: crop.width, height: crop.height, channels: crop.channels },
  })
    .resize(newW, newH, { kernel: 'cubic' })
    .raw()
    .toBuffer({ resolveWithObject: true });

  // Grayscale
  const gray = await sharp(enlarged.data, {
    raw: { width: newW, height: newH, channels: enlarged.info.channels },
  })
    .grayscale()
    .raw()
    .toBuffer({ resolveWithObject: true });

  // CLAHE (clip limit 2.0, 8x8 tile grid)
  const enhancedRaw = await sharp(gray.data, {
    raw: { width: newW, height: newH, channels: 1 },
  })
    .clahe({
      width: Math.max(1, Math.ceil(newW / 8)),
      height: Math.max(1, Math.ceil(newH / 8)),
      maxSlope: 2,
    })
    .toColourspace('b-w')
    .raw()
    .toBuffer({ resolveWithObject: true });

  let enhanced = enhancedRaw.data;
  if (enhancedRaw.info.channels !== 1) {
    // Safety: collapse to the first channel if libvips returned >1 channel
    const c = enhancedRaw.info.channels;
    const single = Buffer.alloc(newW * newH);
    for (let i = 0; i < single.length; i++) single[i] = enhanced[i * c];
    enhanced = single;
  }

  const otsu = otsuThreshold(enhanced);

  const toPng = (buf) =>
    sharp(buf, { raw: { width: newW, height: newH, channels: 1 } })
      .png()
      .toBuffer();

  return [await toPng(enhanced), await toPng(otsu)];
}

// ============================================================
// OCR ENGINE
// ============================================================

class OCR {
  constructor(engine = 'auto') {
    this.engine = engine;
    this.engineName = null;
    this.worker = null;
  }

  // Async initialisation (JS constructors cannot be async)
  async init() {
    // EasyOCR exists only for Python. Node uses Tesseract (tesseract.js).
    if (this.engine === 'easyocr') {
      throw new Error(
        'EasyOCR is a Python library and is not available in Node.js. ' +
        'Use --engine auto or --engine tesseract.'
      );
    }

    console.log('[INFO] Using Tesseract OCR (tesseract.js).');

    const { createWorker } = require('tesseract.js');

    const cachePath = path.join(__dirname, '..', '.tesseract-cache');
    fs.mkdirSync(cachePath, { recursive: true });

    this.worker = await createWorker('eng', 1, { cachePath });

    await this.worker.setParameters({
      tessedit_char_whitelist: '0123456789Xx-/',
      tessedit_pageseg_mode: '7',
    });

    this.engineName = 'tesseract';
    return this;
  }

  async close() {
    if (this.worker) {
      await this.worker.terminate();
      this.worker = null;
    }
  }

  async read(crop) {
    const variants = await preprocessVariants(crop);

    if (!variants.length) {
      return '';
    }

    const candidates = [];

    for (const image of variants) {
      try {
        const { data } = await this.worker.recognize(image);
        const token = normalizeToken(data.text);

        if (token) {
          candidates.push([token, 0.5]);
        }
      } catch (e) {
        continue;
      }
    }

    if (!candidates.length) {
      return '';
    }

    const frequency = new Map();
    for (const [token] of candidates) {
      frequency.set(token, (frequency.get(token) || 0) + 1);
    }

    let best = null;
    let bestKey = null;

    for (const [token, count] of frequency) {
      const maxConf = Math.max(
        ...candidates.filter(([c]) => c === token).map(([, conf]) => conf)
      );
      const key = [count, maxConf];

      if (
        bestKey === null ||
        key[0] > bestKey[0] ||
        (key[0] === bestKey[0] && key[1] > bestKey[1])
      ) {
        best = token;
        bestKey = key;
      }
    }

    return best;
  }
}

// ============================================================
// SCOREBOARD EXTRACTION
// ============================================================

async function extractFrame(frame, ocr) {
  const result = {};

  for (const player of PLAYERS) {
    const [notationRow, cumulativeRow] = ROWS[player];

    const notationValues = [];
    const cumulativeValues = [];

    const [ny1, ny2] = notationRow;
    const [cy1, cy2] = cumulativeRow;

    // -----------------------------------------------
    // 10 bowling frames
    // -----------------------------------------------

    for (let i = 0; i < 10; i++) {
      const x1 = X_EDGES[i + 1];
      const x2 = X_EDGES[i + 2];

      // Remove scoreboard borders
      const cellWidth = x2 - x1;
      const marginX = Math.trunc(cellWidth * 0.12);

      const innerX1 = x1 + marginX;
      const innerX2 = x2 - marginX;

      // Notation cell
      const notationHeight = ny2 - ny1;
      const marginY = Math.trunc(notationHeight * 0.15);

      const innerNy1 = ny1 + marginY;
      const innerNy2 = ny2 - marginY;

      const notationCrop = safeCrop(frame, innerX1, innerNy1, innerX2, innerNy2);

      // Cumulative cell
      const cumulativeHeight = cy2 - cy1;
      const cumulativeMarginY = Math.trunc(cumulativeHeight * 0.15);

      const innerCy1 = cy1 + cumulativeMarginY;
      const innerCy2 = cy2 - cumulativeMarginY;

      const cumulativeCrop = safeCrop(frame, innerX1, innerCy1, innerX2, innerCy2);

      // -------------------------------------------
      // OCR notation
      // -------------------------------------------

      let notation = await ocr.read(notationCrop);

      // Only allow valid bowling notation
      if (notation) {
        notation = notation.toUpperCase();

        if (!['X', '-', '/'].includes(notation)) {
          // Keep only digits, X, / and -
          notation = Array.from(notation)
            .filter((ch) => /[0-9]/.test(ch) || 'X/-'.includes(ch))
            .join('');
        }
      }

      notationValues.push(notation);

      // -------------------------------------------
      // OCR cumulative score
      // -------------------------------------------

      const cumulativeText = await ocr.read(cumulativeCrop);

      // Keep digits only
      const cumulativeDigits = Array.from(cumulativeText)
        .filter((ch) => /[0-9]/.test(ch))
        .join('');

      if (cumulativeDigits) {
        const value = parseInt(cumulativeDigits, 10);

        // Bowling cumulative score should be reasonable
        if (!Number.isNaN(value) && value >= 0 && value <= 300) {
          cumulativeValues.push(value);
        } else {
          cumulativeValues.push(null);
        }
      } else {
        cumulativeValues.push(null);
      }
    }

    result[player] = {
      frames: notationValues,
      cumulative: cumulativeValues,
    };
  }

  return result;
}

// ============================================================
// TEMPORAL CONSENSUS
// ============================================================

function majority(values) {
  const valid = values.filter((v) => v !== '' && v !== null && v !== undefined);

  if (!valid.length) {
    return '';
  }

  // Same as Counter(valid).most_common(1)[0][0]: first-seen wins ties
  const counts = new Map();
  for (const v of valid) {
    counts.set(v, (counts.get(v) || 0) + 1);
  }

  let best = null;
  let bestCount = -1;
  for (const [v, c] of counts) {
    if (c > bestCount) {
      best = v;
      bestCount = c;
    }
  }
  return best;
}

function temporalConsensus(observations) {
  const result = {};

  for (const player of PLAYERS) {
    result[player] = {
      frames: [],
      cumulative: [],
    };

    for (let i = 0; i < 10; i++) {
      const values = observations.map((o) => o[player].frames[i]);

      result[player].frames.push(majority(values));

      const cumulativeValues = observations
        .map((o) => o[player].cumulative[i])
        .filter((v) => v !== null);

      if (cumulativeValues.length) {
        const value = majority(cumulativeValues);
        result[player].cumulative.push(parseInt(value, 10));
      } else {
        result[player].cumulative.push(null);
      }
    }
  }

  return result;
}

// ============================================================
// SCOREBOARD VISUALIZATION
// ============================================================

async function annotateScoreboard(frame, outputPath) {
  if (!frame) {
    return;
  }

  const w = frame.width;
  const h = frame.height;

  const x1 = Math.trunc((50 * w) / BASE_W);
  const y1 = Math.trunc((75 * h) / BASE_H);

  const x2 = Math.trunc((1880 * w) / BASE_W);
  const y2 = Math.trunc((855 * h) / BASE_H);

  const svg = Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">` +
      `<rect x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" ` +
      `fill="none" stroke="#00ff00" stroke-width="4"/>` +
      `<text x="${x1 + 15}" y="${y1 + 35}" font-family="Arial, Helvetica, sans-serif" ` +
      `font-size="32" fill="#00ff00">DETECTED SCOREBOARD</text>` +
      `</svg>`
  );

  await sharp(frame.data, {
    raw: { width: w, height: h, channels: frame.channels },
  })
    .composite([{ input: svg, top: 0, left: 0 }])
    .jpeg()
    .toFile(String(outputPath));
}

// ============================================================
// VIDEO HELPERS (ffmpeg / ffprobe)
// ============================================================

function run(cmd, args, options = {}) {
  const r = spawnSync(cmd, args, { maxBuffer: 1024 * 1024 * 64, ...options });
  if (r.error) {
    if (r.error.code === 'ENOENT') {
      throw new Error(
        `"${cmd}" was not found. Install FFmpeg (which includes ffprobe) and make sure it is in PATH.`
      );
    }
    throw r.error;
  }
  return r;
}

function parseRate(text) {
  if (!text) return 0;
  const [a, b] = String(text).split('/').map(Number);
  if (b === undefined) return a || 0;
  return b ? a / b : 0;
}

function probeVideo(videoPath) {
  const baseArgs = [
    '-v', 'error',
    '-select_streams', 'v:0',
    '-show_entries',
    'stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames',
    '-of', 'json',
  ];

  let r = run('ffprobe', [...baseArgs, videoPath]);
  if (r.status !== 0) {
    return null;
  }

  let stream = (JSON.parse(r.stdout.toString()).streams || [])[0];
  if (!stream) {
    return null;
  }

  let frameCount = parseInt(stream.nb_frames, 10);

  if (!Number.isFinite(frameCount) || frameCount <= 0) {
    // Container has no frame count: count by decoding
    r = run('ffprobe', [...baseArgs, '-count_frames', videoPath]);
    stream = (JSON.parse(r.stdout.toString()).streams || [])[0] || stream;
    frameCount = parseInt(stream.nb_read_frames, 10);
  }

  let fps = parseRate(stream.avg_frame_rate) || parseRate(stream.r_frame_rate);

  return {
    width: stream.width,
    height: stream.height,
    fps,
    frameCount: Number.isFinite(frameCount) ? frameCount : 0,
  };
}

// Same result as np.linspace(0, n - 1, k).astype(int)
function linspaceInt(start, stop, count) {
  if (count === 1) return [Math.trunc(start)];
  const out = [];
  const step = (stop - start) / (count - 1);
  for (let i = 0; i < count; i++) {
    out.push(Math.trunc(i === count - 1 ? stop : start + step * i));
  }
  return out;
}

// Extracts the requested frame numbers in a single decoding pass.
// Returns Map<frameIndex, rawFrame | null>
async function readFrames(videoPath, indices) {
  const unique = [...new Set(indices)].sort((a, b) => a - b);
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'scoreboard-frames-'));
  const frames = new Map();

  try {
    const expr = unique.map((n) => `eq(n,${n})`).join('+');

    run('ffmpeg', [
      '-v', 'error',
      '-i', videoPath,
      '-vf', `select='${expr}'`,
      '-vsync', '0',
      '-frames:v', String(unique.length),
      path.join(tmpDir, 'frame_%05d.png'),
    ]);

    for (let k = 0; k < unique.length; k++) {
      const file = path.join(tmpDir, `frame_${String(k + 1).padStart(5, '0')}.png`);

      if (!fs.existsSync(file)) {
        frames.set(unique[k], null);
        continue;
      }

      const { data, info } = await sharp(file)
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

      frames.set(unique[k], {
        data,
        width: info.width,
        height: info.height,
        channels: info.channels,
      });
    }
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }

  return frames;
}

// ============================================================
// CLI
// ============================================================

function parseArgs(argv) {
  const args = {
    video: null,
    output: 'outputs',
    samples: 12,
    engine: 'auto',
  };

  const usage =
    'usage: node src/scoreboardExtractor.js [-h] [--output OUTPUT] ' +
    '[--samples SAMPLES] [--engine {auto,easyocr,tesseract}] video';

  for (let i = 0; i < argv.length; i++) {
    let arg = argv[i];
    let inlineValue = null;

    if (arg.startsWith('--') && arg.includes('=')) {
      const idx = arg.indexOf('=');
      inlineValue = arg.slice(idx + 1);
      arg = arg.slice(0, idx);
    }

    const next = () => (inlineValue !== null ? inlineValue : argv[++i]);

    if (arg === '-h' || arg === '--help') {
      console.log(usage);
      console.log('\nBowling Scoreboard Data Extraction');
      console.log('\npositional arguments:\n  video              Input video path');
      console.log('\noptions:');
      console.log('  --output OUTPUT    Output directory (default: outputs)');
      console.log('  --samples SAMPLES  Number of frames to process (default: 12)');
      console.log('  --engine ENGINE    auto | easyocr | tesseract (default: auto)');
      process.exit(0);
    } else if (arg === '--output') {
      args.output = next();
    } else if (arg === '--samples') {
      args.samples = parseInt(next(), 10);
      if (Number.isNaN(args.samples)) {
        console.error(usage + '\nerror: argument --samples: invalid int value');
        process.exit(2);
      }
    } else if (arg === '--engine') {
      args.engine = next();
      if (!['auto', 'easyocr', 'tesseract'].includes(args.engine)) {
        console.error(
          usage + `\nerror: argument --engine: invalid choice: '${args.engine}' ` +
          "(choose from 'auto', 'easyocr', 'tesseract')"
        );
        process.exit(2);
      }
    } else if (!arg.startsWith('-') && args.video === null) {
      args.video = arg;
    } else {
      console.error(usage + `\nerror: unrecognized arguments: ${argv[i]}`);
      process.exit(2);
    }
  }

  if (args.video === null) {
    console.error(usage + '\nerror: the following arguments are required: video');
    process.exit(2);
  }

  return args;
}

// ============================================================
// MAIN
// ============================================================

async function main() {
  const args = parseArgs(process.argv.slice(2));

  // --------------------------------------------------------
  // Output folder
  // --------------------------------------------------------

  const outputDir = path.resolve(args.output);
  fs.mkdirSync(outputDir, { recursive: true });

  // --------------------------------------------------------
  // Open video
  // --------------------------------------------------------

  console.log();
  console.log('='.repeat(60));
  console.log('BOWLING SCOREBOARD EXTRACTION');
  console.log('='.repeat(60));

  console.log();
  console.log('[INFO] Opening video:');
  console.log(args.video);

  if (!fs.existsSync(args.video)) {
    console.error('[ERROR] Could not open video.');
    process.exit(1);
  }

  const info = probeVideo(args.video);

  if (!info || !info.frameCount) {
    console.error('[ERROR] Could not open video.');
    process.exit(1);
  }

  // --------------------------------------------------------
  // Video information
  // --------------------------------------------------------

  const frameCount = info.frameCount;
  let fps = info.fps;
  if (fps <= 0) {
    fps = 30;
  }

  const width = info.width;
  const height = info.height;
  const duration = frameCount / fps;

  console.log();
  console.log(`[INFO] Resolution : ${width} x ${height}`);
  console.log(`[INFO] FPS        : ${fps.toFixed(2)}`);
  console.log(`[INFO] Frames     : ${frameCount}`);
  console.log(`[INFO] Duration   : ${duration.toFixed(2)} seconds`);

  // --------------------------------------------------------
  // OCR
  // --------------------------------------------------------

  const ocr = new OCR(args.engine);
  await ocr.init();

  // --------------------------------------------------------
  // Select frames
  // --------------------------------------------------------

  const sampleCount = Math.max(1, Math.min(args.samples, frameCount));

  const indices = linspaceInt(0, frameCount - 1, sampleCount);

  const observations = [];

  let bestFrame = null;
  let bestQuality = -1;
  let bestIndex = 0;

  console.log();
  console.log('[INFO] Processing video frames...');

  const frames = await readFrames(args.video, indices);

  // --------------------------------------------------------
  // Process frames
  // --------------------------------------------------------

  for (let n = 0; n < indices.length; n++) {
    const index = indices[n];
    const number = n + 1;

    const frame = frames.get(index);

    if (!frame) {
      console.log(`[WARNING] Could not read frame ${index}`);
      continue;
    }

    console.log(
      `[INFO] Processing sample ${number}/${sampleCount} (frame ${index})`
    );

    const observation = await extractFrame(frame, ocr);

    observations.push(observation);

    // Quality score
    let quality = 0;

    for (const player of PLAYERS) {
      quality += observation[player].frames.filter((v) => Boolean(v)).length;
      quality += observation[player].cumulative.filter((v) => v !== null).length;
    }

    if (quality > bestQuality) {
      bestQuality = quality;
      bestFrame = frame;
      bestIndex = index;
    }
  }

  await ocr.close();

  // --------------------------------------------------------
  // Validate
  // --------------------------------------------------------

  if (!observations.length) {
    console.error('[ERROR] No frames were successfully processed.');
    process.exit(1);
  }

  // --------------------------------------------------------
  // Stabilize
  // --------------------------------------------------------

  const finalResult = temporalConsensus(observations);

  // --------------------------------------------------------
  // JSON output
  // --------------------------------------------------------

  const payload = {
    video: path.basename(args.video),

    video_info: {
      width,
      height,
      fps: Math.round(fps * 100) / 100,
      frame_count: frameCount,
      duration_seconds: Math.round(duration * 100) / 100,
    },

    ocr_engine: ocr.engineName,

    sampled_frames: indices.map((i) => Math.trunc(i)),

    players: finalResult,

    best_frame_index: bestIndex,

    notes: [
      'Scoreboard extracted using OpenCV and OCR.',
      'Multiple video frames are sampled.',
      'Temporal majority voting is used to reduce OCR noise.',
      'The supplied video uses a fixed scoreboard layout.',
    ],
  };

  const jsonPath = path.join(outputDir, 'extracted_scoreboard.json');

  fs.writeFileSync(jsonPath, JSON.stringify(payload, null, 4), 'utf-8');

  // --------------------------------------------------------
  // Detected scoreboard image
  // --------------------------------------------------------

  const imagePath = path.join(outputDir, 'detected_scoreboard.jpg');

  await annotateScoreboard(bestFrame, imagePath);

  // --------------------------------------------------------
  // Finished
  // --------------------------------------------------------

  console.log();
  console.log('='.repeat(60));
  console.log('EXTRACTION COMPLETED');
  console.log('='.repeat(60));

  console.log();
  console.log('[OK] JSON output:');
  console.log(jsonPath);

  console.log();
  console.log('[OK] Detected scoreboard image:');
  console.log(imagePath);

  console.log();
  console.log('[OK] OCR engine:');
  console.log(ocr.engineName);

  console.log();
  console.log('Final extracted data:');
  console.log(JSON.stringify(finalResult, null, 4));

  console.log();
  console.log('='.repeat(60));
}

module.exports = {
  safeCrop,
  normalizeText,
  normalizeToken,
  preprocessVariants,
  OCR,
  extractFrame,
  majority,
  temporalConsensus,
  annotateScoreboard,
  TTL_BOX,
};

if (require.main === module) {
  main().catch((err) => {
    console.error(err && err.message ? err.message : err);
    process.exit(1);
  });
}
