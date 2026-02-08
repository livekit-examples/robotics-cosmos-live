# cosmos-live

An intelligent video feed management system that orchestrates multiple live camera streams through a multiplexed architecture with 1-second heartbeat intervals, enabling both manual control and AI-driven automated feed selection.

## Core Capabilities

### Manual Feed Control

Direct operator commands for explicit feed selection (e.g., "put feed 1 up" switches the output to camera feed 1).

### Autonomous Content Selection

AI agent analyzes available feeds in real-time and selects the most engaging or relevant content based on configurable criteria.

### Subject Tracking

Multi-camera person tracking that automatically switches between feeds to maintain continuous coverage of a specific subject (e.g., "follow binh through cameras" maintains visual tracking across camera boundaries).

### Narrative Mode

Intelligent feed sequencing that creates cohesive storylines by selecting and transitioning between feeds to form compelling visual narratives.

## Architecture

Two flows: **Ingestion** (video in) and **Control** (user interaction), connected by shared stores.

```
                         INGESTION                              CONTROL

Clients ──► LiveKit ──► Orchestrator                  User Prompt
                            │                              │
               ┌────────────┼──────────┐                   ▼
               ▼            ▼          ▼              ControlAgent
         VideoWorker   VideoWorker  AudioWorker      /           \
               │            │          │        VectorStore   GraphStore
          FrameBuffer  FrameBuffer    STT            \           /
               │            │          │              ▼         ▼
          VisionModel  VisionModel     │          StreamOperator
               │            │          │              │
               └─────┬──────┘─────────┘              ▼
                     ▼                          Live Output
              VectorStore + GraphStore        (YouTube, RTMP)
```

Ingestion writes to the stores, control reads from them. The `ControlAgent` can also drive the `StreamOperator` to switch feeds, apply overlays, etc.

## Project Structure

```
cosmos_live/
├── app.py                  ← entry point, wires everything together
└── config.py               ← pydantic config models, loads config.yaml

packages/
├── core/                   ABCs and types
│   └── cosmos_core/
│       ├── types.py            Frame, AudioSegment, Document, etc.
│       ├── buffer.py           FrameBuffer ABC
│       ├── vision.py           VisionModel ABC
│       ├── storage.py          VectorStore / GraphStore ABCs
│       ├── transcription.py    Transcriber ABC
│       └── streaming.py        StreamOperator ABC
│
├── storage/                Store implementations
│   └── cosmos_storage/
│       └── memory.py           In-memory stores (dev)
│
├── ingestion/              Ingestion pipeline (LiveKit)
│   └── cosmos_ingestion/
│       ├── orchestrator.py     Connects to LiveKit room, spawns workers
│       ├── publisher.py        Captures and publishes to LiveKit room
│       ├── video_worker.py     push_frame → buffer → sample → vision → store
│       ├── audio_worker.py     push_audio → transcribe → store
│       └── vllm.py             vLLM vision implementation
│
└── control/                Control pipeline
    └── cosmos_control/
        ├── agent.py            LLM loop with RAG access
        └── operator.py         FFmpeg stream operator implementation
```

Swappable backends (vision models, stores, stream operators) implement ABCs from `core`. LiveKit is used directly for transport. Config is centralized in `config.yaml` (see `config.example.yaml`).

## Getting Started

```bash
uv sync
uv run python -m cosmos_live.app
```

## Development

```bash
uv add --package cosmos-ingestion <dep>
uv run pytest
```
