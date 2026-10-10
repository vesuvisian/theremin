# Virtual Theremin

A multi-player virtual theremin that runs entirely in the browser. No build step, no server—just static files suitable for GitHub Pages.

It uses the webcam, [MediaPipe Tasks Vision](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/web_js) (pose + hands), and the Web Audio API so each person in frame gets their own synthesizer voice.

## How to play

1. Open the page and click **Start Audio**. Allow camera access when prompted.
2. Stand so your torso and both hands are visible.
3. **Pitch (right hand):** Move your right hand toward an invisible vertical antenna on your right. Closer = higher pitch. (The video is mirrored like a selfie, so “your right” matches what you see.)
4. **Volume (left hand):** Raise your left hand above an invisible horizontal antenna. Higher = louder. Close that hand into a fist to mute.
5. **Silence:** Drop both hands below the instrument, or move them out of frame, to mute *your* voice. Other performers keep playing independently.
6. **Expression:** Open/close the right fingers for fine pitch and vibrato; twist the right wrist to change brightness (filter cutoff).

Multiple people can play at once—each tracked body gets its own oscillator chain. Mute and gesture mapping are always per player.

### On-page controls

While running, a bar at the bottom offers:

| Control | Default | Description |
| --- | --- | --- |
| **Pose overlay** | Off | Shows subtle skeleton / hand landmarks |
| **Theremin overlay** | Off | Draws the vertical pitch rod and horizontal volume loop relative to each player |
| **Reverse L/R** | Off | Swaps pitch and volume hands (and antenna sides) |
| **Stop** | — | Stops audio, tracking, and the camera; returns to the start screen |

## Run locally

Webcam access needs a secure context (`localhost` or HTTPS):

```bash
python3 -m http.server 8080
```

Open [http://localhost:8080](http://localhost:8080).

Any other static file server works (`npx serve .`, etc.).

## Deploy (GitHub Pages)

Publish the repository root (or `docs/`) as a Pages site from the `main` branch. Keep `index.html` at the served root. MediaPipe WASM and model files load from absolute CDN / Google Storage URLs, so they resolve correctly on Pages.

## Project layout

```
index.html   # Page shell, directions, controls
styles.css   # Layout and HUD
script.js    # Tracking, gesture mapping, Web Audio
README.md    # This file
```

## Tech notes

- **Stack:** Vanilla JS, HTML5 Canvas, Web Audio API
- **Vision:** `@mediapipe/tasks-vision@0.10.18` via jsDelivr; GPU delegate with CPU fallback
- **Audio graph (per performer):** `OscillatorNode` → `BiquadFilterNode` (low-pass) → `GainNode` → destination
- Voices are created when a person enters frame and torn down shortly after they leave

## Requirements

- A modern browser with WebGL (or WebGPU) preferred for model inference
- Webcam permission
- Network access on first load to fetch MediaPipe WASM and `.task` models
