# Bowling Scoreboard Data Extraction from Video (Node.js)

**Round 1 – Computer Vision Engineer assessment**

Node.js port of the original Python project. The scoreboard grid coordinates,
OCR token rules, validation, temporal voting and output format are the same as
the Python version. It extracts scoreboard information from the supplied bowling
video using image processing + OCR, for the fixed scoreboard layout in the
assessment clip (coordinates are scaled to the input resolution).

## Architecture

```
Video
  ↓
Frame sampling (evenly spaced frames, FFmpeg)
  ↓
Scoreboard ROI / grid localization
  ↓
Cell-wise preprocessing (sharp: 6x upscale, grayscale, CLAHE, Otsu)
  ↓
Tesseract OCR (tesseract.js)
  ↓
Token normalization + validation
  ↓
Temporal majority voting
  ↓
JSON + annotated scoreboard image
```

## Python → Node.js mapping

| Python                      | Node.js                                  |
| --------------------------- | ---------------------------------------- |
| `src/scoreboard_extractor.py` | `src/scoreboardExtractor.js`           |
| `app.py` (Streamlit)        | `app.js` + `public/index.html` (Express) |
| OpenCV (`cv2`)              | `sharp` + FFmpeg                         |
| EasyOCR / pytesseract       | `tesseract.js`                           |
| `requirements.txt`          | `package.json`                           |

Note: EasyOCR is Python-only, so the Node version always uses Tesseract.
`--engine easyocr` prints an error; `auto` and `tesseract` both use Tesseract.

## Requirements

- Node.js 18 or newer
- FFmpeg (includes `ffprobe`) installed and available in PATH
  - Windows: `winget install Gyan.FFmpeg` (then reopen the terminal)
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`

## Run in VS Code / Windows

### 1. Open the project folder

```
cd scoreboard-data-extraction
```

### 2. Install dependencies

```
npm install
```

tesseract.js downloads its English language data on the first run (cached in
`.tesseract-cache/`). Internet access is required for that first download.

### 3. Put the assessment video in the project root

```
input_video.mp4
```

### 4. Run the extractor

```
node src/scoreboardExtractor.js input_video.mp4 --output outputs --samples 8 --engine auto
```

(or `npm run extract`)

### 5. Open the results

```
outputs/extracted_scoreboard.json
outputs/detected_scoreboard.jpg
```

## Optional web UI (replaces Streamlit)

```
npm start
```

Open http://localhost:3000, upload the video and click **Extract Scoreboard**.

## Visible result in the supplied clip

The stable end frame visibly shows the scoreboard title **VISHAL**.

| Player | Frame notation    | Cumulative     | Final visible total |
| ------ | ----------------- | -------------- | ------------------- |
| J      | X, 5-, -7, 4-, X  | 15, 20, 27, 31 | 41                  |
| V      | 8-, 3-, 71, 81, 9 | 8, 11, 19, 28  | 37                  |
| P      | X, 4/, 9-, 6-     | 20, 39, 48, 54 | 54                  |
| T      | 61, 1/, 8-, 34    | 7, 25, 33, 40  | 40                  |

A copy of this visually verified expected result is included in
`sample_data/EXPECTED_VISIBLE_RESULT.json` for testing/documentation; it is
**not used by the extractor**.

## Accuracy strategy

1. **Cell-wise OCR:** the scoreboard is divided into individual frame cells instead of OCRing the whole screen.
2. **Multiple preprocessing variants:** CLAHE-enhanced grayscale and Otsu-thresholded variants are tested.
3. **Restricted OCR alphabet:** only digits and bowling symbols are accepted.
4. **Token validation:** malformed OCR output is discarded.
5. **Temporal consensus:** the same field is read from multiple frames and the most frequent valid value is selected.
6. **Resolution scaling:** ROI coordinates are normalized to the 1920×1080 reference and scaled for another resolution.

## Important scope

The assessment video uses a stable, fixed scoreboard layout. Therefore the
ROI/grid is intentionally calibrated to that video. For arbitrary videos from
different bowling systems, replace the fixed grid with a trained scoreboard
detector (for example YOLO) and keep the OCR + temporal-consensus stages.

## Project layout

```
scoreboard-data-extraction/
├── src/
│   └── scoreboardExtractor.js
├── public/
│   └── index.html
├── outputs/
│   ├── extracted_scoreboard.json
│   └── detected_scoreboard.jpg
├── sample_data/
│   └── EXPECTED_VISIBLE_RESULT.json
├── app.js
├── package.json
├── README.md
└── .gitignore
```

Do not commit the 141 MB video directly to a normal GitHub repository. If the
evaluator requires the video inside GitHub, use Git LFS; otherwise keep the
video as the assessment attachment and put the code/results in GitHub.
