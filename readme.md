📄 Product Requirements Document (PRD)
Project Name

ATLAS (Working Title)
A Vision-Driven Autonomous Desktop Agent using Local AI Models

1. Overview
   1.1 Problem Statement

Current automation tools rely on:

brittle scripts

APIs and app-specific hooks

browser extensions

predefined workflows

These approaches break when:

UI layouts change

APIs are unavailable

the user is remote (mobile control)

apps are closed-source or uninstrumented

Humans, however, can operate any computer using only visual perception and reasoning.

1.2 Solution

ATLAS is a vision-driven autonomous desktop agent that:

perceives the screen visually

reasons about UI state and user intent

plans actions step-by-step

executes mouse/keyboard actions

verifies outcomes visually

works entirely using local models

requires no APIs, plugins, or integrations

The screen is the API.

2. Goals & Non-Goals
   2.1 Goals

Operate any desktop application via visual understanding

Execute multi-step tasks from a single natural language prompt

Support remote control (phone → desktop)

Run fully offline using local models

Recover from UI errors and ambiguity

2.2 Non-Goals

No app-specific integrations

No end-to-end RL training

No reliance on OS accessibility APIs (optional later)

No guaranteed perfection on first click (verification handles this)

3. High-Level Architecture
   User Prompt (Mobile / Desktop)
   ↓
   Intent Interpreter (LLM)
   ↓
   High-Level Task Planner (LLM)
   ↓
   ┌─────────────────────────────┐
   │ Perception–Action Loop │
   │ │
   │ Screen Capture │
   │ ↓ │
   │ OCR + VLM Perception │
   │ ↓ │
   │ Screen State Graph │
   │ ↓ │
   │ Action Planner (LLM) │
   │ ↓ │
   │ Mouse / Keyboard Executor │
   │ ↓ │
   │ Visual Verification │
   │ ↺ (loop) │
   └─────────────────────────────┘
   ↓
   Task Completion / Failure

4. Core User Stories
   4.1 Example 1

“Open Spotify, search song ABC, and add it to my playlist.”

4.2 Example 2

“Open file manager, find report.pdf, and send it to Aarav on WhatsApp.”

4.3 Example 3

“Find the research paper I opened last Tuesday and email a summary.”

5. ML Stack (Local Models Only)
   5.1 OCR (Text + Bounding Boxes)

Recommended

PaddleOCR (best balance of accuracy + speed)

Supports:

text detection

text recognition

bounding box polygons

Why

Precise text localization

Stable bounding boxes

Works offline

Mature ecosystem

5.2 Vision-Language Model (UI Understanding)

Primary Choice

LLaVA 1.6 / LLaVA-Next (quantized)

Alternatives

BLIP-2 (lighter, less reasoning)

InternVL (if GPU available)

Role

Identify UI regions (input fields, buttons, icons)

Understand layout and relationships

Provide semantic descriptions of screen regions

Why LLaVA

Strong spatial reasoning

Can consume OCR outputs

Good grounding for UI tasks

Proven in “computer use” agents

5.3 Language Model (Planning & Reasoning)

Primary Choice

Mistral 7B / Mixtral 8x7B (quantized)

Lightweight Option

Phi-3 (for CPU-only setups)

Role

Intent extraction

Task decomposition

Action planning

Candidate ranking

Error recovery

Why

Strong reasoning per parameter

Excellent local inference support

Good tool-use and planning behavior

6. Detailed ML / Agent Pipeline
   6.1 Step 1: Intent Interpretation (LLM)

Trigger

User submits a prompt

Input

"Open Spotify, search song ABC and add it to playlist"

Output (Structured Intent)

{
"goal": "music_action",
"app": "Spotify",
"actions": [
{ "type": "search", "query": "ABC" },
{ "type": "add_to_playlist" }
]
}

Notes

No UI assumptions

No coordinates

Pure semantic understanding

6.2 Step 2: High-Level Task Planning (LLM)

Input

Structured intent

Output

1. Ensure Spotify is open
2. Locate search functionality
3. Search for song "ABC"
4. Select correct result
5. Open options menu
6. Add song to playlist

This plan is abstract, not UI-specific.

6.3 Step 3: Screen Capture

Trigger

Before every action

Output

Screenshot (RGB image)

Screen resolution (W, H)

This defines the coordinate system for the current loop.

6.4 Step 4: OCR Processing

Input

Screenshot

Output

[
{
"text": "Search",
"bbox": [[120,45],[260,45],[260,75],[120,75]]
},
{
"text": "Your Library",
"bbox": [...]
}
]

Purpose

Provide semantic anchors

Provide pixel-accurate bounding boxes

6.5 Step 5: VLM UI Region Detection

Input

Screenshot

OCR results

Prompt:

“Identify interactive UI regions and their roles.”

Output

[
{
"type": "input_field",
"bbox": [95, 40, 620, 92],
"confidence": 0.87
},
{
"type": "button",
"bbox": [...],
"label": "Play"
}
]

6.6 Step 6: Bounding Box Fusion Logic
Sources of Bounding Boxes

OCR text boxes

VLM UI region boxes

Geometry-based rectangles (heuristics)

Fusion Steps

Normalize all boxes → relative coordinates

Expand OCR boxes into functional regions

Merge overlapping boxes (IoU threshold)

Assign semantic roles using LLM reasoning

Result

{
"role": "search_bar",
"bbox_norm": [0.05, 0.04, 0.32, 0.08],
"confidence": 0.92
}

6.7 Step 7: Action Planning (LLM)

Input

Current screen state graph

Next abstract task step

Output

{
"action": "click",
"target_role": "search_bar"
}

6.8 Step 8: Coordinate Resolution

Convert

x = x_norm _ screen_width
y = y_norm _ screen_height

Apply DPI scaling calibration if needed

6.9 Step 9: Action Execution

Actions Supported

mouse_move(x, y)

mouse_click()

type(text)

key_press(key)

Executor

OS-level input injection

No app hooks

6.10 Step 10: Visual Verification Loop

After every action

Check:

Did the expected UI change occur?

Did cursor appear?

Did text render?

Did results update?

If yes

Advance to next step

If no

Re-run perception

Try next ranked candidate

Re-plan if needed

This is closed-loop control.

7. Learning & Memory (Non-Training)
   7.1 Experience Memory

Store:

successful UI patterns

element locations (relative)

common failures

Used for:

speed

ranking

confidence

No gradient updates.

8. Performance Requirements

Perception loop: ≤ 500 ms

Action latency: ≤ 100 ms

FPS target: 2–5 (sufficient for UI)

GPU optional, CPU fallback supported

9. Risks & Mitigations
   Risk Mitigation
   UI ambiguity Candidate ranking + verification
   Layout changes Re-perception every step
   DPI mismatch Runtime calibration
   OCR failure VLM fallback
   Wrong click Verification + retry
10. MVP Scope (Hackathon-Ready)
    Supported Tasks

App launching

Search & text input

File navigation

Media playback actions

Messaging attachment sending

Out of Scope (MVP)

Drag-and-drop

Multi-user OS

Complex CAPTCHAs

11. Key Insight (Core Principle)

Training teaches the system how to see.
Reasoning decides what to do.
Verification ensures correctness.
