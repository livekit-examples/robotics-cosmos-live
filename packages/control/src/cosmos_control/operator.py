from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from livekit import rtc
from PIL import Image, ImageDraw, ImageFont

from cosmos_core import Overlay, StreamOperator
from cosmos_utils import generate_livekit_token

if TYPE_CHECKING:
    from cosmos_live.config import LiveKitConfig, StreamConfig

logger = logging.getLogger(__name__)

# Overlay slot layout constants
_SLOT_LAYOUTS: dict[str, dict[str, Any]] = {
    "lower_third": {
        "anchor": "bottom_left",
        "y_offset": 80,
        "x_offset": 40,
        "bg_color": (0, 0, 0, 180),
        "text_color": (255, 255, 255),
        "padding": (20, 12),
    },
    "title": {
        "anchor": "top_left",
        "y_offset": 40,
        "x_offset": 40,
        "bg_color": (0, 0, 0, 200),
        "text_color": (255, 255, 255),
        "padding": (16, 8),
    },
    "banner": {
        "anchor": "top_full",
        "y_offset": 0,
        "x_offset": 0,
        "bg_color": (200, 30, 30, 230),
        "text_color": (255, 255, 255),
        "padding": (20, 10),
    },
}


class _TrackInfo:
    """Tracks available for a participant."""

    def __init__(self) -> None:
        self.video: rtc.Track | None = None
        self.audio: rtc.Track | None = None


class FFmpegStreamOperator(StreamOperator):
    """Stream operator backed by FFmpeg for RTMP/YouTube output.

    Connects to a LiveKit room as a separate participant, subscribes to
    tracks, composites a single active feed with text overlays, and
    streams video+audio to an RTMP endpoint via FFmpeg.
    """

    def __init__(
        self,
        stream_config: StreamConfig,
        livekit_config: LiveKitConfig,
    ) -> None:
        self._stream_config = stream_config
        self._livekit_config = livekit_config
        self._op = stream_config.operator

        # LiveKit state
        self._room: rtc.Room | None = None
        self._available_tracks: dict[str, _TrackInfo] = {}

        # Active feed
        self._active_feed: str | None = None
        self._feed_lock = asyncio.Lock()
        self._latest_frame: np.ndarray | None = None
        self._consume_video_task: asyncio.Task[None] | None = None
        self._consume_audio_task: asyncio.Task[None] | None = None

        # Overlays — threading.Lock because the render thread reads them
        self._overlays: dict[str, Overlay] = {}
        self._overlay_lock = threading.Lock()

        # Frame lock — protects _latest_frame between asyncio and render thread
        self._frame_lock = threading.Lock()

        # FFmpeg
        self._ffmpeg_proc: subprocess.Popen[bytes] | None = None
        self._audio_fifo_path: str | None = None
        self._audio_fifo_fd: int | None = None

        # Audio write lock — prevents interleaved writes from consume + silence
        self._audio_write_lock = threading.Lock()

        # Threads
        self._stop_event = threading.Event()
        self._render_thread: threading.Thread | None = None
        self._silence_thread: threading.Thread | None = None
        self._audio_active = threading.Event()

        # Font cache
        self._font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None

        # Placeholder cache
        self._placeholder: np.ndarray | None = None

    # --- Public API ---

    @property
    def available_feeds(self) -> list[str]:
        """Return participant identities that have at least one subscribed track."""
        return list(self._available_tracks.keys())

    async def start(self) -> None:
        """Create audio FIFO, spawn FFmpeg, connect to LiveKit, start render loop."""
        logger.info("Starting FFmpegStreamOperator")

        # Create audio FIFO
        fifo_dir = tempfile.mkdtemp(prefix="cosmos_")
        self._audio_fifo_path = os.path.join(fifo_dir, "audio_fifo")
        os.mkfifo(self._audio_fifo_path)

        # Spawn FFmpeg
        self._spawn_ffmpeg()

        # Open audio FIFO for writing (blocks until FFmpeg opens the read end).
        # Run in a thread to avoid blocking the event loop.
        self._audio_fifo_fd = await asyncio.to_thread(
            os.open, self._audio_fifo_path, os.O_WRONLY
        )

        # Connect to LiveKit
        await self._connect_livekit()

        # Start render and silence-fill threads (dedicated for precise timing)
        self._stop_event.clear()
        self._render_thread = threading.Thread(
            target=self._render_loop, name="render", daemon=True
        )
        self._silence_thread = threading.Thread(
            target=self._silence_fill, name="silence-fill", daemon=True
        )
        self._render_thread.start()
        self._silence_thread.start()

        logger.info("FFmpegStreamOperator started")

    async def stop(self) -> None:
        """Disconnect from LiveKit, stop tasks, close FFmpeg, clean up."""
        logger.info("Stopping FFmpegStreamOperator")

        # Signal threads to exit
        self._stop_event.set()

        # Cancel async consume tasks
        for task in (self._consume_video_task, self._consume_audio_task):
            if task is not None:
                task.cancel()

        # Wait for consume tasks to finish
        tasks = [
            t
            for t in (self._consume_video_task, self._consume_audio_task)
            if t is not None
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Join threads (they check _stop_event, should exit promptly)
        if self._render_thread is not None:
            self._render_thread.join(timeout=5)
            self._render_thread = None
        if self._silence_thread is not None:
            self._silence_thread.join(timeout=5)
            self._silence_thread = None

        # Disconnect from LiveKit
        if self._room is not None:
            await self._room.disconnect()
            self._room = None

        # Close FFmpeg stdin to signal EOF
        if self._ffmpeg_proc is not None and self._ffmpeg_proc.stdin:
            self._ffmpeg_proc.stdin.close()

        # Close audio FIFO
        if self._audio_fifo_fd is not None:
            os.close(self._audio_fifo_fd)
            self._audio_fifo_fd = None

        # Wait for FFmpeg to exit
        if self._ffmpeg_proc is not None:
            self._ffmpeg_proc.wait(timeout=10)
            self._ffmpeg_proc = None

        # Clean up FIFO file
        if self._audio_fifo_path is not None:
            try:
                os.unlink(self._audio_fifo_path)
                os.rmdir(os.path.dirname(self._audio_fifo_path))
            except OSError:
                pass
            self._audio_fifo_path = None

        logger.info("FFmpegStreamOperator stopped")

    async def set_feed(self, feed_id: str) -> None:
        """Switch the active video/audio feed by participant identity."""
        async with self._feed_lock:
            if feed_id == self._active_feed:
                return

            old_feed = self._active_feed
            logger.info("Switching feed: %s -> %s", old_feed, feed_id)

            # Cancel old consume tasks
            await self._cancel_consume_tasks()

            self._active_feed = feed_id
            with self._frame_lock:
                self._latest_frame = None

            # Start consuming new feed if tracks are available
            track_info = self._available_tracks.get(feed_id)
            if track_info is not None:
                self._start_consume_tasks(track_info)
            else:
                logger.warning(
                    "Feed %s not found in available tracks; "
                    "will start consuming when tracks arrive",
                    feed_id,
                )

    async def set_overlay(self, slot: str, overlay: Overlay) -> None:
        """Set a named overlay slot."""
        with self._overlay_lock:
            self._overlays[slot] = overlay
        logger.info("Overlay set: slot=%s kind=%s", slot, overlay.kind)

    async def clear_overlay(self, slot: str) -> None:
        """Remove the overlay from a named slot."""
        with self._overlay_lock:
            removed = self._overlays.pop(slot, None)
        if removed is not None:
            logger.info("Overlay cleared: slot=%s", slot)

    # --- LiveKit connection ---

    async def _connect_livekit(self) -> None:
        """Connect to the LiveKit room as a subscribe-only participant."""
        self._room = rtc.Room()
        token = generate_livekit_token(
            self._livekit_config, "stream-operator", self._livekit_config.room
        )

        @self._room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            identity = participant.identity
            info = self._available_tracks.setdefault(identity, _TrackInfo())

            if track.kind == rtc.TrackKind.KIND_VIDEO:
                info.video = track
                logger.info("Video track available from %s", identity)
            elif track.kind == rtc.TrackKind.KIND_AUDIO:
                info.audio = track
                logger.info("Audio track available from %s", identity)

            # If this participant is the active feed and we're not consuming yet,
            # start consuming.
            if identity == self._active_feed:
                if (
                    track.kind == rtc.TrackKind.KIND_VIDEO
                    and self._consume_video_task is None
                ):
                    self._consume_video_task = asyncio.create_task(
                        self._consume_video(track)
                    )
                elif (
                    track.kind == rtc.TrackKind.KIND_AUDIO
                    and self._consume_audio_task is None
                ):
                    self._consume_audio_task = asyncio.create_task(
                        self._consume_audio(track)
                    )

        @self._room.on("track_unsubscribed")
        def on_track_unsubscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            identity = participant.identity
            info = self._available_tracks.get(identity)
            if info is None:
                return

            if track.kind == rtc.TrackKind.KIND_VIDEO:
                info.video = None
                if identity == self._active_feed and self._consume_video_task is not None:
                    self._consume_video_task.cancel()
                    self._consume_video_task = None
                    with self._frame_lock:
                        self._latest_frame = None
            elif track.kind == rtc.TrackKind.KIND_AUDIO:
                info.audio = None
                if identity == self._active_feed and self._consume_audio_task is not None:
                    self._consume_audio_task.cancel()
                    self._consume_audio_task = None

            # Remove participant entry if no tracks remain
            if info.video is None and info.audio is None:
                self._available_tracks.pop(identity, None)

            logger.info(
                "Track unsubscribed: kind=%s from %s", track.kind, identity
            )

        logger.info(
            "Connecting to LiveKit %s room=%s as stream-operator",
            self._livekit_config.url,
            self._livekit_config.room,
        )
        await self._room.connect(self._livekit_config.url, token)
        logger.info("Connected to LiveKit room")

    # --- Feed consumption ---

    async def _cancel_consume_tasks(self) -> None:
        """Cancel running video/audio consume tasks."""
        tasks: list[asyncio.Task[None]] = []
        if self._consume_video_task is not None:
            self._consume_video_task.cancel()
            tasks.append(self._consume_video_task)
            self._consume_video_task = None
        if self._consume_audio_task is not None:
            self._consume_audio_task.cancel()
            tasks.append(self._consume_audio_task)
            self._consume_audio_task = None
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _start_consume_tasks(self, track_info: _TrackInfo) -> None:
        """Spawn consume tasks for the given participant's tracks."""
        if track_info.video is not None:
            self._consume_video_task = asyncio.create_task(
                self._consume_video(track_info.video)
            )
        if track_info.audio is not None:
            self._consume_audio_task = asyncio.create_task(
                self._consume_audio(track_info.audio)
            )

    async def _consume_video(self, track: rtc.Track) -> None:
        """Read frames from a LiveKit video track and store the latest."""
        frame_interval = 1.0 / self._op.fps
        last_frame_time = 0.0

        try:
            video_stream = rtc.VideoStream(track)
            async for frame_event in video_stream:
                now = time.monotonic()
                if now - last_frame_time < frame_interval:
                    continue
                last_frame_time = now

                frame = frame_event.frame
                rgb_frame = frame.convert(rtc.VideoBufferType.RGB24)
                # .copy() to own the data — livekit may reuse the buffer
                arr = np.frombuffer(rgb_frame.data, dtype=np.uint8).reshape(
                    (rgb_frame.height, rgb_frame.width, 3)
                ).copy()
                with self._frame_lock:
                    self._latest_frame = arr
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Error in video consume loop")

    async def _consume_audio(self, track: rtc.Track) -> None:
        """Read audio from a LiveKit audio track and write PCM to the FIFO."""
        try:
            audio_stream = rtc.AudioStream(
                track,
                sample_rate=self._op.audio_sample_rate,
                num_channels=self._op.audio_channels,
            )
            async for audio_event in audio_stream:
                audio_frame = audio_event.frame
                pcm_data = audio_frame.data.tobytes()
                self._audio_active.set()
                await asyncio.to_thread(self._write_audio_fifo, pcm_data)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Error in audio consume loop")

    def _write_audio_fifo(self, data: bytes) -> None:
        """Write audio data to the FIFO, serialised by lock."""
        with self._audio_write_lock:
            fd = self._audio_fifo_fd
            if fd is not None:
                os.write(fd, data)

    # --- Render loop (dedicated thread) ---

    def _render_loop(self) -> None:
        """Composite frames at the target FPS and write to FFmpeg stdin.

        Runs in its own thread with absolute-time tracking to avoid drift.
        """
        frame_interval = 1.0 / self._op.fps
        frame_size = self._op.width * self._op.height * 3
        next_frame_time = time.monotonic()

        while not self._stop_event.is_set():
            # Grab latest frame under lock
            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                frame = self._make_placeholder()
            else:
                frame = self._resize_frame(frame)

            # Snapshot overlays under lock
            with self._overlay_lock:
                overlays = dict(self._overlays) if self._overlays else {}

            if overlays:
                frame = self._apply_overlays(frame, overlays)

            # Write to FFmpeg
            rgb_bytes = frame.tobytes()
            if len(rgb_bytes) == frame_size and self._ffmpeg_proc is not None:
                stdin = self._ffmpeg_proc.stdin
                if stdin is not None:
                    try:
                        stdin.write(rgb_bytes)
                    except BrokenPipeError:
                        logger.error("FFmpeg stdin pipe broken, stopping render")
                        break

            # Advance to next absolute frame time to avoid cumulative drift
            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.monotonic()
            if sleep_time > 0:
                # wait() returns immediately if stop_event is set
                if self._stop_event.wait(timeout=sleep_time):
                    break
            elif sleep_time < -frame_interval:
                # More than one frame behind — reset to avoid death spiral
                next_frame_time = time.monotonic()

    # --- Silence fill (dedicated thread) ---

    def _silence_fill(self) -> None:
        """Write silence to the audio FIFO when no real audio is flowing."""
        # 10ms of silence: sample_rate * channels * 2 bytes/sample * 0.01s
        chunk_size = int(
            self._op.audio_sample_rate * self._op.audio_channels * 2 * 0.01
        )
        silence = bytes(chunk_size)
        interval = 0.01  # 10ms

        while not self._stop_event.is_set():
            self._audio_active.clear()
            # Sleep 10ms; returns True immediately if stop requested
            if self._stop_event.wait(timeout=interval):
                break
            if not self._audio_active.is_set():
                try:
                    self._write_audio_fifo(silence)
                except OSError:
                    pass

    # --- FFmpeg management ---

    def _spawn_ffmpeg(self) -> None:
        """Start the FFmpeg subprocess for RTMP streaming."""
        rtmp_url = (
            f"{self._stream_config.rtmp_url}/"
            f"{self._stream_config.stream_key.get_secret_value()}"
        )

        cmd = [
            "ffmpeg",
            "-y",
            # Video input: raw RGB from stdin
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self._op.width}x{self._op.height}",
            "-r", str(self._op.fps),
            "-i", "pipe:0",
            # Audio input: raw PCM from FIFO
            "-f", "s16le",
            "-ar", str(self._op.audio_sample_rate),
            "-ac", str(self._op.audio_channels),
            "-i", self._audio_fifo_path,
            # Video encoding
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", self._op.preset,
            "-b:v", self._op.video_bitrate,
            "-g", str(self._op.gop_size),
            "-keyint_min", str(self._op.gop_size),
            # Audio encoding
            "-c:a", "aac",
            "-b:a", self._op.audio_bitrate,
            # Output
            "-f", "flv",
            rtmp_url,
        ]

        logger.info("Spawning FFmpeg: %s", " ".join(cmd[:6]) + " ...")
        self._ffmpeg_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    # --- Frame processing ---

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a frame to the output resolution if needed."""
        h, w = frame.shape[:2]
        if w == self._op.width and h == self._op.height:
            return frame
        img = Image.fromarray(frame)
        img = img.resize((self._op.width, self._op.height), Image.LANCZOS)
        return np.array(img)

    def _make_placeholder(self) -> np.ndarray:
        """Return a solid dark placeholder frame at output resolution."""
        if self._placeholder is None:
            self._placeholder = np.full(
                (self._op.height, self._op.width, 3),
                self._op.placeholder_color,
                dtype=np.uint8,
            )
        return self._placeholder

    # --- Overlay rendering ---

    def _apply_overlays(
        self, frame: np.ndarray, overlays: dict[str, Overlay]
    ) -> np.ndarray:
        """Draw text overlays onto the frame using Pillow."""
        # Work with RGBA for semi-transparent backgrounds
        img = Image.fromarray(frame).convert("RGBA")
        overlay_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_layer)
        font = self._get_font()

        for slot, overlay in overlays.items():
            text = overlay.data.get("text", "")
            if not text:
                continue
            self._draw_text_overlay(draw, font, slot, text)

        img = Image.alpha_composite(img, overlay_layer)
        return np.array(img.convert("RGB"))

    def _draw_text_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        slot: str,
        text: str,
    ) -> None:
        """Position and draw a text overlay based on the slot layout."""
        layout = _SLOT_LAYOUTS.get(slot)
        if layout is None:
            # Unknown slot — draw at top-left as fallback
            layout = _SLOT_LAYOUTS["title"]

        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad_x, pad_y = layout["padding"]
        bg_w = text_w + pad_x * 2
        bg_h = text_h + pad_y * 2

        anchor = layout["anchor"]
        if anchor == "bottom_left":
            x = layout["x_offset"]
            y = self._op.height - layout["y_offset"] - bg_h
        elif anchor == "top_left":
            x = layout["x_offset"]
            y = layout["y_offset"]
        elif anchor == "top_full":
            x = 0
            y = layout["y_offset"]
            bg_w = self._op.width
        else:
            x = layout["x_offset"]
            y = layout["y_offset"]

        # Draw background rectangle
        draw.rectangle(
            [x, y, x + bg_w, y + bg_h],
            fill=layout["bg_color"],
        )

        # Draw text centered in the background
        text_x = x + (bg_w - text_w) // 2
        text_y = y + (bg_h - text_h) // 2
        draw.text(
            (text_x, text_y),
            text,
            fill=layout["text_color"],
            font=font,
        )

    def _get_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Lazy-load the configured font."""
        if self._font is None:
            if self._op.font_path:
                self._font = ImageFont.truetype(
                    self._op.font_path, self._op.font_size
                )
            else:
                try:
                    self._font = ImageFont.truetype(
                        "DejaVuSans.ttf", self._op.font_size
                    )
                except OSError:
                    self._font = ImageFont.load_default()
        return self._font
