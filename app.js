'use strict';

// Node.js replacement for the Streamlit app (app.py).
// Upload a video, run the extractor, view the table / image, download JSON.

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const express = require('express');
const multer = require('multer');

const app = express();
const PORT = process.env.PORT || 3000;

const dataDir = path.join(__dirname, 'runtime');
const outDir = path.join(dataDir, 'outputs');
fs.mkdirSync(dataDir, { recursive: true });

const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => cb(null, dataDir),
    filename: (req, file, cb) => cb(null, path.basename(file.originalname)),
  }),
  fileFilter: (req, file, cb) => {
    const ok = /\.(mp4|mov|avi)$/i.test(file.originalname);
    cb(ok ? null : new Error('Only mp4, mov or avi files are allowed.'), ok);
  },
});

app.use('/runtime-outputs', express.static(outDir));
app.use(express.static(path.join(__dirname, 'public')));

app.post('/extract', upload.single('video'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No video uploaded.' });
  }

  const args = [
    path.join(__dirname, 'src', 'scoreboardExtractor.js'),
    req.file.path,
    '--output', outDir,
    '--samples', '8',
    '--engine', 'auto',
  ];

  const child = spawn(process.execPath, args);
  let stdout = '';
  let stderr = '';

  child.stdout.on('data', (d) => (stdout += d));
  child.stderr.on('data', (d) => (stderr += d));

  child.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({ error: stderr || stdout });
    }

    try {
      const result = JSON.parse(
        fs.readFileSync(path.join(outDir, 'extracted_scoreboard.json'), 'utf-8')
      );
      res.json({ result, image: '/runtime-outputs/detected_scoreboard.jpg?t=' + Date.now() });
    } catch (e) {
      res.status(500).json({ error: String(e) });
    }
  });
});

app.get('/download/json', (req, res) => {
  res.download(path.join(outDir, 'extracted_scoreboard.json'), 'extracted_scoreboard.json');
});

app.use((err, req, res, next) => {
  res.status(400).json({ error: err.message });
});

app.listen(PORT, () => {
  console.log(`Bowling Scoreboard OCR running at http://localhost:${PORT}`);
});
