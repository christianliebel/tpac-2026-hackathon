# Web Platform Planet

A planet whose mountains are web platform features from [caniuse.com](https://caniuse.com).
Every feature gets its own mountain the moment the first browser ships it, and the mountain grows as
browser support increases. Scrub the timeline from 2000 to today, or let it play.

Built for the TPAC 2026 hackathon. Rendered with three.js on WebGPU (falls back to WebGL 2).

## How support is computed

For every month since January 2000, each browser lineage (IE/Edge, Chrome, Firefox, Safari, Safari on iOS,
Opera, Samsung Internet) is represented by its most recent release at that time. The support share of a
feature is the average over the lineages that exist at that time: full support counts 1, partial support 0.5.

Features are grouped on the planet by caniuse category, so categories form continents.

## Steer with your face

The "Steer with your face" button turns on the webcam and lets your head control the planet. Turning your head
turns the planet, leaning closer zooms in. Press C to recalibrate the neutral pose. All inference runs locally.

- **WebNN (NPU)**: when the browser exposes the W3C Web Neural Network API (`navigator.ml`), the YuNet face
  detector (OpenCV Zoo) runs through ONNX Runtime Web's WebNN execution provider, trying the NPU first, then GPU,
  then CPU. Yaw, pitch and roll are estimated from its five landmarks. In Chrome, enable
  `chrome://flags/#web-machine-learning-neural-network`.
- **MediaPipe Face Landmarker**: fallback without WebNN, using the facial transformation matrix on the GPU.
- `?engine=onnx` forces the ONNX path with a WASM fallback, for testing the decoder without WebNN.

## Talk to the planet

"Talk to the planet" opens a voice session with the OpenAI Realtime API (model `gpt-realtime`) over WebRTC,
straight from the browser. Paste your OpenAI API key into the field next to the button; it is kept in
`localStorage` only and used to mint an ephemeral client secret. The assistant has tools to look up caniuse
features ("When did ArrayBuffer ship?"), fly to a feature ("Go to ES6 classes"), jump to a year and control playback.

## Development

```sh
npm install
npm run data   # regenerates public/data from caniuse-db
npm run dev
```

Keyboard: space toggles playback, arrow keys step by one month.
