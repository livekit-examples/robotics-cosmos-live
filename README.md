# cosmos-live

An intelligent video feed management system that orchestrates multiple live camera streams through a multiplexed architecture with 1-second heartbeat intervals, enabling both manual control and AI-driven automated feed selection.

## Core Capabilities

### Manual Feed Control

Direct voice commands for explicit feed selection.

> **User:** "put feed 1 up"
>
> **System:** The voice agent calls its `switch_feed` tool with `feed_id="feed-1"`. The stream operator immediately switches the live output to that camera's video and audio track. FFmpeg cuts to the new feed on the next frame.

### Autonomous Content Selection

AI agent analyzes available feeds in real-time and selects the most engaging or relevant content based on configurable criteria.

> **User:** "auto-direct this stream, pick the most interesting camera"
>
> **System:** The agent calls `query` with `text="most activity"` to semantic-search the latest vision analyses across all feeds. It compares what's happening on each camera and calls `switch_feed` to cut to the one with the most action — e.g., a group discussion on cam-3 instead of an empty hallway on cam-1.

### Subject Tracking

Multi-camera person tracking that automatically switches between feeds to maintain continuous coverage of a specific subject across camera boundaries.

> **User:** "follow binh through cameras"
>
> **System:** The agent calls `query` with `text="binh"` to find recent vision chunks mentioning binh across all track IDs. When binh leaves cam-1's field of view and appears in cam-3's analysis chunks, the agent calls `switch_feed` to cut to cam-3.

### Narrative Mode

Intelligent feed sequencing that creates cohesive storylines by selecting and transitioning between feeds to form compelling visual narratives.

> **User:** "create a narrative of the office tour"
>
> **System:** The agent calls `search` to retrieve vision and transcript chunks chronologically, identifies a sequence of related events (lobby entrance on cam-1, hallway walk on cam-2, conference room arrival on cam-4), and sequences `switch_feed` calls in that order to assemble a coherent storyline.

### Security Monitoring

Real-time surveillance analysis powered by semantic search over continuous video feeds.

> **User:** "is there any suspicious activity in the last 10 minutes?"
>
> **System:** The agent calls `query` with `text="suspicious activity"` and filters to recent timestamps. It retrieves the most relevant vision analysis chunks — e.g., _"a person is trying to open a locked door repeatedly"_ from cam-2 or _"someone left an unattended bag near the entrance"_ from cam-5 — and speaks back the camera ID, time, and description of each flagged event.

## Architecture

The system runs two separate processes that share a Milvus vector store:

- **Ingestion** (`ingestion_app.py`) — subscribes to camera feeds, runs vision analysis, writes text chunks to the store.
- **Retrieval** (`retrieval_app.py`) — voice agent that reads from the store, answers questions, and controls the live stream output.

```
            INGESTION                                  RETRIEVAL

  Cameras ──► LiveKit ──► Orchestrator            User (voice)
                              │                        │
                 ┌────────────┼────────────┐       STT (Deepgram)
                 │            │            │           │
                 ▼            ▼            ▼           ▼
           VideoWorker  VideoWorker  AudioWorker  CosmosAgent ◄── LLM (GPT-4.1-mini)
                 │            │            │       │        │
            FrameBuffer  FrameBuffer     STT      │        │         TTS (Cartesia)
                 │            │            │   queries   commands         │
            VisionModel  VisionModel      │      │        │              │
                 │            │            │      │        ▼              ▼
                 └────────────┴────────────┘      │   StreamOperator   User
                              │                   │        │          (voice)
                           writes                 │        ▼
                              │                   │   Live Output
                              ▼                   │  (YouTube / RTMP)
                         ┌─────────┐              │
                         │ Vector  │ ◄────────────┘
                         │ Store   │
                         └─────────┘
```

## How It Works

### Ingestion: turning video into searchable text

1. **Camera clients** publish live video and audio to a [LiveKit](https://livekit.io/) room.
2. The **Orchestrator** joins the room and subscribes to every track. For each video track it spawns a **VideoWorker**; for each audio track an **AudioWorker**.
3. Each VideoWorker has a **FrameBuffer** (a fixed-capacity ring buffer). Incoming frames are pushed into the buffer at the configured `target_fps`.
4. When the buffer is full, the worker **samples** N evenly-spaced frames from it, JPEG-encodes them as base64, and sends them along with a text prompt to a **VisionModel** (vLLM running `nvidia/Cosmos-Reason2-2B`).
5. The vision model returns a short natural-language description of what it sees — for example: _"Two people are standing at a whiteboard, one is drawing a diagram while the other watches."_
6. That description is wrapped in a **Document** and inserted into the vector store. The store embeds the text using `all-MiniLM-L6-v2` (384-dim vectors) and indexes it for semantic search.

AudioWorkers do the same thing for speech: transcribe audio segments and store the transcript as a Document.

The result is a continuously growing collection of timestamped, camera-tagged text chunks that describe everything happening across all feeds.

### Retrieval: voice agent that answers questions and drives the stream

The retrieval side is a [LiveKit Agents](https://docs.livekit.io/agents/) voice pipeline. A user talks to the agent, and the agent talks back.

1. User speech is transcribed by **STT** (Deepgram).
2. The transcript goes to the **LLM** (GPT-4.1-mini) which runs the **CosmosAgent**.
3. The agent has access to these tools:

   | Tool                               | What it does                                                             |
   | ---------------------------------- | ------------------------------------------------------------------------ |
   | `query(text, top_k)`               | Semantic search — finds documents by meaning (e.g., "someone cooking")   |
   | `search(expr, top_k)`              | Field-based filter — Milvus WHERE clause (e.g., `'track_id == "cam-1"'`) |
   | `switch_feed(feed_id)`             | Switch the active stream to a different camera                           |
   | `list_feeds()`                     | List all available camera participant identities                         |
   | `set_overlay(slot, text)`          | Place a text overlay on the stream (`lower_third`, `title`, `banner`)    |
   | `clear_overlay(slot)`              | Remove an overlay                                                        |
   | `start_stream()` / `stop_stream()` | Start or stop the stream operator                                        |

4. The agent's text response is converted to speech by **TTS** (Cartesia) and played back to the user.
5. When the agent calls `switch_feed` or `set_overlay`, the **StreamOperator** composites the selected video feed with overlays and pushes the result to an RTMP endpoint via FFmpeg.

### Document schema

Every chunk stored in Milvus follows the `Document` schema:

```python
@dataclass
class Document:
    content: str                          # text payload (vision description or transcript)
    track_id: str = ""                    # source camera (e.g. "cam-1")
    kind: str = ""                        # "transcript" or "analysis"
    timestamp: float = 0.0                # capture time (epoch seconds)
    embedding: list[float] | None = None  # 384-dim vector, populated at insert time
```

Stored in Milvus with the following schema:

| Field       | Type           | Description                                     |
| ----------- | -------------- | ----------------------------------------------- |
| `id`        | VARCHAR(64)    | Auto-generated UUID v4, primary key             |
| `content`   | VARCHAR(65535) | Text content from the document                  |
| `track_id`  | VARCHAR(256)   | Source track identifier (e.g. `"cam-1"`)        |
| `kind`      | VARCHAR(64)    | Chunk type: `"transcript"`, `"analysis"`        |
| `timestamp` | DOUBLE         | Capture time in epoch seconds                   |
| `embedding` | FLOAT_VECTOR   | 384-dim normalized embedding (all-MiniLM-L6-v2) |

Index: `IVF_FLAT` with `COSINE` metric, `nlist=128`.

## Project Structure

```
cosmos_live/
├── ingestion_app.py        ← entry point: ingestion pipeline
├── retrieval_app.py        ← entry point: voice agent + stream operator
└── config.py               ← pydantic config models, loads config.yaml

packages/
├── core/                   ABCs and types
│   └── cosmos_core/
│       ├── types.py            Frame, AudioSegment, Document, Overlay, etc.
│       ├── buffer.py           FrameBuffer ABC
│       ├── ring_buffer.py      RingFrameBuffer implementation
│       ├── vision.py           VisionModel ABC
│       ├── storage.py          VectorStore ABC
│       ├── transcription.py    Transcriber ABC
│       └── streaming.py        StreamOperator ABC
│
├── storage/                Store implementations
│   └── cosmos_storage/
│       ├── memory.py           In-memory stores (dev)
│       └── milvus.py           Milvus vector store (query + search)
│
├── ingestion/              Ingestion pipeline
│   └── cosmos_ingestion/
│       ├── orchestrator.py     Connects to LiveKit room, spawns workers
│       ├── publisher.py        Captures and publishes to LiveKit room
│       ├── video_worker.py     push_frame → buffer → sample → vision → store
│       ├── audio_worker.py     push_audio → transcribe → store
│       └── vllm.py             vLLM vision implementation
│
├── control/                Voice agent + stream operator
│   └── cosmos_control/
│       ├── agent.py            CosmosAgent — LiveKit voice agent with tools
│       └── operator.py         FFmpegStreamOperator — composites and streams
│
└── utils/                  Shared utilities
    └── cosmos_utils/
        └── token.py            LiveKit JWT token generation
```

Swappable backends (vision models, stores, stream operators) implement ABCs from `core`. LiveKit is used directly for transport. Config is centralized in `config.yaml` (see `config.example.yaml`).

## Dependencies

### Python (>= 3.12)

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. All Python dependencies are installed automatically via `uv sync`.

| Package                          | Used By            | Purpose                                                                     |
| -------------------------------- | ------------------ | --------------------------------------------------------------------------- |
| `livekit` >= 1.0                 | ingestion, control | WebRTC SDK for real-time video/audio transport                              |
| `livekit-api` >= 1.0             | ingestion, utils   | LiveKit server API and JWT token generation                                 |
| `livekit-agents` >= 1.0          | root, control      | LiveKit Agents framework (voice pipeline, `AgentSession`, `@function_tool`) |
| `livekit-plugins-silero` >= 1.0  | root               | Voice Activity Detection (VAD) for the voice agent                          |
| `openai` >= 1.0                  | ingestion          | AsyncOpenAI client to talk to vLLM (OpenAI-compatible API)                  |
| `opencv-python-headless` >= 4.10 | ingestion          | Video frame encoding (JPEG for base64 payloads)                             |
| `pymilvus` >= 2.4                | storage            | Milvus vector database client                                               |
| `milvus-lite` >= 2.4             | storage            | Serverless embedded Milvus (runs locally, no server needed)                 |
| `sentence-transformers` >= 3.0   | storage            | Text embedding model (`all-MiniLM-L6-v2`, 384-dim vectors)                  |
| `Pillow` >= 10.0                 | control            | Image processing and text overlay rendering on stream frames                |
| `numpy` >= 2.0                   | core, control      | Numerical computing, frame data representation                              |
| `pydantic` >= 2.12.5             | root, utils        | Config validation and type-safe YAML parsing                                |
| `pyyaml` >= 6.0.3                | root               | YAML config file loading                                                    |

Dev dependencies: `pytest` >= 8, `pytest-asyncio` >= 1.

### System Tools

| Tool       | Required            | Purpose                                                                                                                                   |
| ---------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **FFmpeg** | Yes (for streaming) | Encodes and muxes the composited video+audio stream and pushes it to an RTMP endpoint (e.g., YouTube Live). Must be available on `$PATH`. |
| **uv**     | Yes                 | Python package manager used to install dependencies and run the project. Install from [docs.astral.sh/uv](https://docs.astral.sh/uv/).    |

### External Services

#### LiveKit

Real-time WebRTC server used for video/audio transport between camera clients, the orchestrator, and the stream operator. You need a [LiveKit Cloud](https://livekit.io/) project or a self-hosted LiveKit server. Set your URL, API key, and API secret in `config.yaml`.

#### vLLM (Vision Model Inference)

The ingestion pipeline sends sampled video frames to a vLLM instance running the **`nvidia/Cosmos-Reason2-2B`** model for visual understanding. The vLLM server exposes an OpenAI-compatible API that the `openai` Python client talks to.

**Our setup:** The team hosts a vLLM instance on [Nebius](https://nebius.com/) with **1x NVIDIA H100 80GB GPU** serving `nvidia/Cosmos-Reason2-2B`. The `base_url` in `config.yaml` points to this instance. If you are running your own, start vLLM with:

```bash
vllm serve nvidia/Cosmos-Reason2-2B
```

and set `vllm.base_url` in your `config.yaml` to the address of your vLLM server (e.g., `http://localhost:8000`).

#### LLM, STT, TTS (Voice Agent)

The voice agent uses hosted APIs for speech and language:

| Service | Default                 | Used for                                           |
| ------- | ----------------------- | -------------------------------------------------- |
| **STT** | `deepgram/nova-3:multi` | Transcribing user speech to text                   |
| **LLM** | `openai/gpt-4.1-mini`   | Agent reasoning, tool calling, response generation |
| **TTS** | `cartesia/sonic-3`      | Converting agent responses back to speech          |

These are configured in the `agent` section of `config.yaml`. API keys for Deepgram, OpenAI, and Cartesia must be set as environment variables (see LiveKit Agents plugin docs for details).

#### RTMP Endpoint (YouTube Live, etc.)

The stream operator pushes the composited output to an RTMP URL. By default this is configured for YouTube Live (`rtmp://a.rtmp.youtube.com/live2`). Set your `stream.rtmp_url` and `stream.stream_key` in `config.yaml`.

## Getting Started

1. Install [uv](https://docs.astral.sh/uv/) if you don't have it:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

3. Copy the example config and fill in your credentials:

   ```bash
   cp config.example.yaml config.yaml
   ```

   Edit `config.yaml` with your LiveKit credentials, vLLM endpoint, and stream key.

4. Make sure FFmpeg is installed:

   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt install ffmpeg
   ```

5. Run the ingestion pipeline (subscribes to camera feeds, runs vision analysis, stores chunks):

   ```bash
   uv run python -m cosmos_live.ingestion_app
   ```

6. Run the voice agent (connects to the LiveKit room, answers questions, controls the stream):

   ```bash
   uv run python -m cosmos_live.retrieval_app console
   ```

   These are separate processes. Run them in two terminals or as background services.

## Configuration

See `config.example.yaml` for all available options. Key sections:

- **`livekit`** -- LiveKit server URL, API key, API secret, and room name.
- **`vllm`** -- vLLM server base URL, model name (`nvidia/Cosmos-Reason2-2B`), and optional API key.
- **`milvus`** -- Local DB path, collection name, and embedding model.
- **`video_worker`** -- Frame buffer size, sample count, target FPS, and the vision analysis prompt.
- **`agent`** -- Voice agent STT, LLM, and TTS model identifiers, plus the system instructions.
- **`stream`** -- RTMP URL, stream key, and operator settings (resolution, bitrate, encoding preset, etc.).

## Development

```bash
# Add a dependency to a specific package
uv add --package cosmos-ingestion <dep>

# Run tests
uv run pytest

# Run tests excluding integration tests (which require a running vLLM server)
uv run pytest -m "not integration"
```
