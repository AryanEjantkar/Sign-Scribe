# SignScribe – AI-Powered ASL Interpreter
![HTML](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)
![CSS](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-FF6F00?logo=tensorflow&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-4285F4?logo=google&logoColor=white)

A real-time, browser-based American Sign Language (ASL) interpreter that uses AI-powered hand pose detection to bridge the communication gap between the hearing and the Deaf community — no installation required.

---

## 📘 Table of Contents
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Workflow](#-workflow)
- [Tech Stack](#-tech-stack)
- [Pages](#-pages)
- [How It Works](#-how-it-works)
- [Installation & Usage](#-installation--usage)
- [Future Enhancements](#-future-enhancements)
- [LLM Prompts Used](#-llm-prompts-used-to-create-this-project)
- [Author](#-author)

---

## Features
- Real-time ASL hand gesture recognition via webcam
- Runs entirely in the browser — no backend, no installs
- AI-powered hand landmark detection using MediaPipe Hands
- Geometric gesture classifier for letters: A, B, C, D, E, F, I, K, L, M, N, O, R, S, U, V, W, Y
- Temporal smoothing (buffer voting) for stable, noise-free predictions
- Live confidence score bar for prediction reliability
- Sentence builder with Space, Backspace, and Clear controls
- Text-to-Speech output for the built sentence
- Glassmorphism dark-mode UI with Outfit font

---

## System Architecture

> The system uses the browser's camera feed, processes each frame through TensorFlow.js + MediaPipe to extract 21 hand landmarks, classifies the gesture using a custom geometric algorithm, and displays the result in a real-time UI.
**Architecture Images:**
 

![Architecture Diagram](/archi5.png)

---

## Workflow
1. Open **Interpreter Pro** page — webcam and AI engine initialize automatically
2. Show an ASL hand sign to the camera
3. The system detects 21 hand landmarks in real time
4. The classifier analyses finger curl angles and positions
5. A 10-frame buffer votes for the most stable letter
6. The current sign and confidence bar update live
7. Click the detected letter to add it to the sentence
8. Use **Add Space**, **Backspace**, or **Clear All** to edit
9. Hit **Speak Text** for audio output via the Web Speech API

---

## Tech Stack

| Layer | Technology |
|---|---|
| Structure | HTML5 |
| Styling | CSS3 (Custom Properties, Glassmorphism) |
| Logic | Vanilla JavaScript (ES6+) |
| Hand Detection | MediaPipe Hands |
| ML Runtime | TensorFlow.js (WebGL backend) |
| Fonts | Google Fonts – Outfit |
| Speech Output | Web Speech API (SpeechSynthesis) |
| Deployment | Static hosting (GitHub Pages / Netlify) |

---

## Pages

| Page | File | Description |
|---|---|---|
| Home | `index.html` | Landing page with hero section and CTA |
| Interpreter Pro | `scribe_pro.html` | Live ASL detection interface |
| About | `about.html` | Project mission and description |
| Contact | `contact.html` | Contact information |

---

## How It Works

### Hand Landmark Detection
MediaPipe Hands provides 21 3D keypoints per hand, covering all joints from wrist to fingertips.

### Gesture Classification (`asl_engine.js`)
A custom geometric classifier runs on every frame:
- **Curl Analysis** – Calculates the angle between MCP→PIP and PIP→TIP vectors for each finger to determine if it is straight or curled.
- **Distance Ratios** – Measures normalised distances between landmarks (e.g., thumb tip to index tip) to distinguish similar signs.
- **Thumb Position** – Checks whether the thumb is extended, across the palm, or tucked to differentiate letters like A, S, M, N.

### Temporal Smoothing
A 10-frame rolling buffer collects predictions. The most frequently voted letter wins, eliminating flickering and improving accuracy. Confidence is computed as `votes / buffer_size × 100%`.

---

## Installation & Usage

Since SignScribe is a fully client-side web app, no build step is needed.

```bash
# Clone the repository
git clone https://github.com/your-username/signscribe.git
cd signscribe

# Option 1: Open directly in browser
start index.html

# Option 2: Serve locally (recommended for camera permissions)
npx serve .
# or
python -m http.server 8080
```

> **Note:** Camera access requires the page to be served over `http://localhost` or `https://`. Opening `index.html` directly as a `file://` URL may block webcam access in some browsers.

---

## Future Enhancements
- Full ASL alphabet coverage (J, X, Z require motion)
- Dynamic gesture support (words like "Hello", "Thank You")
- Continuous sign-to-word translation (not just letter-by-letter)
- Multi-hand detection for two-handed signs
- Sign language learning / quiz mode
- Support for ISL (Indian Sign Language) and BSL

---

## LLM Prompts Used to Create This Project
This project was developed with assistance from Large Language Models (LLMs) such as Gemini and GPT for system design, classifier logic, and UI generation.

---

### 1. Project Ideation
> Design a browser-based real-time ASL sign language interpreter that requires no backend, using only the user's webcam and AI hand-pose detection.

---

### 2. Architecture Design
> Design a pipeline for real-time gesture recognition: webcam → hand landmark detection → geometric classifier → temporal smoothing → UI output.

---

### 3. Core Classifier Logic
> Write a JavaScript function that takes 21 MediaPipe hand landmarks and classifies them into ASL letters using finger curl angles and normalised landmark distances.

---

### 4. UI/UX Design
> Design a modern glassmorphism dark-mode UI for a real-time sign language interpreter, with a webcam feed, detected letter display, confidence bar, and sentence builder.

---

### 5. Stability & Accuracy
> Implement temporal smoothing using a prediction buffer and voting system to reduce flickering in real-time gesture classification.

---

### 6. Evaluation Design
> Suggest evaluation metrics for a hand gesture classifier, including per-letter accuracy, false positive rate, confidence calibration, and latency.

---

## 👨‍💻 Author

**Aryan Vimal Ejantkar**
🎓 B.Tech (AIML) – VIT Bhopal
💼 Passionate about AI, ML, and accessibility technology
📧 aryanvimalejantkar@gmail.com

---
