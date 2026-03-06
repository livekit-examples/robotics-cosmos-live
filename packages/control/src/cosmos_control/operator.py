from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

import numpy as np
from livekit import rtc
from PIL import Image, ImageDraw, ImageFont

from cosmos_core import Overlay, StreamOperator
from cosmos_utils import generate_livekit_token

if TYPE_CHECKING:
    from cosmos_live.config import LiveKitConfig, OperatorConfig

logger = logging.getLogger(__name__)

# Overlay slot layout constants
_SLOT_LAYOUTS: dict[str, dict[str, Any]] = {
    "lower_third": {
        "anchor": "bottom_left",
        "y_offset": 80,
        "x_offset": 40,
        "bg_color": (0, 0, 0),
        "bg_alpha": 180,  # 0–255
        "text_color": (255, 255, 255),
        "padding": (20, 12),
    },
    "title": {
        "anchor": "top_left",
        "y_offset": 40,
        "x_offset": 40,
        "bg_color": (0, 0, 0),
        "bg_alpha": 204,
        "text_color": (255, 255, 255),
        "padding": (16, 8),
    },
    "banner": {
        "anchor": "top_full",
        "y_offset": 0,
        "x_offset": 0,
        "bg_color": (30, 30, 200),
        "bg_alpha": 230,
        "text_color": (255, 255, 255),
        "padding": (20, 10),
    },
}

_FONT_SIZE = 24
_JPEG_QUALITY = 80


class _TrackInfo:
    """Tracks available for a participant."""

    def __init__(self) -> None:
        self.video: rtc.Track | None = None


# ---------------------------------------------------------------------------
# MJPEG HTTP handler
# ---------------------------------------------------------------------------

class _MJPEGHandler(BaseHTTPRequestHandler):
    """Serves an MJPEG stream or a simple index page."""

    server: _MJPEGServer  # type: ignore[assignment]

    def do_GET(self) -> None:
        if self.path == "/stream":
            self._stream_mjpeg()
        else:
            self._serve_index()

    def _serve_index(self) -> None:
        port = self.server.server_address[1]
        html = (
            "<!DOCTYPE html><html><head>"
            "<title>Cosmos Live</title>"
            "<style>body{margin:0;background:#111;display:flex;"
            "justify-content:center;align-items:center;height:100vh}</style>"
            "</head><body>"
            f'<img src="http://localhost:{port}/stream" '
            f'style="max-width:100%;max-height:100vh">'
            "</body></html>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _stream_mjpeg(self) -> None:
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        op = self.server.operator
        last_seq = -1

        while not op._stop_event.is_set():
            with op._jpeg_cond:
                # Wait until a new frame is available
                op._jpeg_cond.wait(timeout=1.0)
                seq = op._jpeg_seq
                if seq == last_seq:
                    continue
                data = op._jpeg_data
                last_seq = seq

            if not data:
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Silence per-request logs
        pass


class _MJPEGServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that holds a back-reference to the operator."""

    daemon_threads = True

    def __init__(self, addr: tuple[str, int], operator: CVDisplayOperator) -> None:
        self.operator = operator
        super().__init__(addr, _MJPEGHandler)


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class CVDisplayOperator(StreamOperator):
    """Stream operator that serves composited frames as an MJPEG HTTP stream.

    Connects to a LiveKit room as a separate participant, subscribes to
    video tracks, composites a single active feed with text overlays, and
    serves the result at ``http://localhost:<port>/stream``.
    """

    def __init__(
        self,
        operator_config: OperatorConfig,
        livekit_config: LiveKitConfig,
    ) -> None:
        self._cfg = operator_config
        self._livekit_config = livekit_config

        # LiveKit state
        self._room: rtc.Room | None = None
        self._available_tracks: dict[str, _TrackInfo] = {}

        # Active feed
        self._active_feed: str | None = None
        self._feed_lock = asyncio.Lock()
        self._latest_frame: np.ndarray | None = None
        self._consume_video_task: asyncio.Task[None] | None = None

        # Overlays — threading.Lock because the display thread reads them
        self._overlays: dict[str, Overlay] = {}
        self._overlay_lock = threading.Lock()

        # Frame lock — protects _latest_frame between asyncio and display thread
        self._frame_lock = threading.Lock()

        # Display / compositing thread
        self._stop_event = threading.Event()
        self._display_thread: threading.Thread | None = None

        # MJPEG output shared with HTTP handlers
        self._jpeg_cond = threading.Condition()
        self._jpeg_data: bytes = b""
        self._jpeg_seq: int = 0

        # HTTP server
        self._http_server: _MJPEGServer | None = None
        self._http_thread: threading.Thread | None = None

        # Placeholder cache
        self._placeholder: Image.Image | None = None

        # Font
        try:
            self._font = ImageFont.load_default(size=_FONT_SIZE)
        except TypeError:
            # Pillow < 10.1 fallback
            self._font = ImageFont.load_default()

    # --- Public API ---

    @property
    def available_feeds(self) -> list[str]:
        """Return participant identities that have at least one subscribed track."""
        return list(self._available_tracks.keys())

    @property
    def active_feed(self) -> str | None:
        """Return the identity of the currently active feed, if any."""
        return self._active_feed

    async def start(self) -> None:
        """Connect to LiveKit and start the display loop.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._display_thread is not None:
            return
        logger.info("Starting CVDisplayOperator")

        # Connect to LiveKit
        await self._connect_livekit()

        # Start compositing thread
        self._stop_event.clear()
        self._display_thread = threading.Thread(
            target=self._display_loop, name="cv-display", daemon=True
        )
        self._display_thread.start()

        # Start MJPEG HTTP server
        port = self._cfg.display_port
        self._http_server = _MJPEGServer(("0.0.0.0", port), self)
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            name="mjpeg-http",
            daemon=True,
        )
        self._http_thread.start()

        logger.info(
            "CVDisplayOperator started — stream at http://localhost:%d/stream",
            port,
        )

    async def stop(self) -> None:
        """Disconnect from LiveKit, stop tasks, close display."""
        logger.info("Stopping CVDisplayOperator")

        # Signal thread to exit
        self._stop_event.set()

        # Wake any MJPEG handlers waiting on the condition
        with self._jpeg_cond:
            self._jpeg_cond.notify_all()

        # Shut down HTTP server
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=5)
            self._http_thread = None

        # Cancel async consume task
        if self._consume_video_task is not None:
            self._consume_video_task.cancel()
            await asyncio.gather(self._consume_video_task, return_exceptions=True)
            self._consume_video_task = None

        # Join display thread
        if self._display_thread is not None:
            self._display_thread.join(timeout=5)
            self._display_thread = None

        # Disconnect from LiveKit
        if self._room is not None:
            await self._room.disconnect()
            self._room = None

        logger.info("CVDisplayOperator stopped")

    async def set_feed(self, feed_id: str) -> None:
        """Switch the active video feed by participant identity."""
        async with self._feed_lock:
            if feed_id == self._active_feed:
                return

            old_feed = self._active_feed
            logger.info("Switching feed: %s -> %s", old_feed, feed_id)

            # Cancel old consume task
            await self._cancel_consume_task()

            self._active_feed = feed_id
            with self._frame_lock:
                self._latest_frame = None

            # Start consuming new feed if tracks are available
            track_info = self._available_tracks.get(feed_id)
            if track_info is not None and track_info.video is not None:
                self._consume_video_task = asyncio.create_task(
                    self._consume_video(track_info.video)
                )
                
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

                # If this participant is the active feed and we're not
                # consuming yet, start consuming.
                if identity == self._active_feed and self._consume_video_task is None:
                    self._consume_video_task = asyncio.create_task(
                        self._consume_video(track)
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

                # Remove participant entry if no video track remains
                if info.video is None:
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

    async def _cancel_consume_task(self) -> None:
        """Cancel the running video consume task."""
        if self._consume_video_task is not None:
            self._consume_video_task.cancel()
            await asyncio.gather(self._consume_video_task, return_exceptions=True)
            self._consume_video_task = None

    async def _consume_video(self, track: rtc.Track) -> None:
        """Read frames from a LiveKit video track and store the latest."""
        frame_interval = 1.0 / self._cfg.fps
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

    # --- Compositing loop (dedicated thread) ---

    def _display_loop(self) -> None:
        """Composite frames at the target FPS and encode to JPEG.

        Runs in its own thread with absolute-time tracking to avoid drift.
        """
        frame_interval = 1.0 / self._cfg.fps
        next_frame_time = time.monotonic()

        while not self._stop_event.is_set():
            # Grab latest frame under lock
            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                img = self._make_placeholder()
            else:
                img = self._resize_frame(frame)

            # Snapshot overlays under lock
            with self._overlay_lock:
                overlays = dict(self._overlays) if self._overlays else {}

            if overlays:
                img = self._apply_overlays(img, overlays)

            # Encode to JPEG and notify MJPEG clients
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
            jpeg = buf.getvalue()

            with self._jpeg_cond:
                self._jpeg_data = jpeg
                self._jpeg_seq += 1
                self._jpeg_cond.notify_all()

            # Advance to next absolute frame time to avoid cumulative drift
            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.monotonic()
            if sleep_time > 0:
                if self._stop_event.wait(timeout=sleep_time):
                    break
            elif sleep_time < -frame_interval:
                # More than one frame behind — reset to avoid death spiral
                next_frame_time = time.monotonic()

    # --- Frame processing ---

    def _resize_frame(self, frame: np.ndarray) -> Image.Image:
        """Convert numpy RGB array to a PIL Image, resizing if needed."""
        img = Image.fromarray(frame)
        if img.width != self._cfg.width or img.height != self._cfg.height:
            img = img.resize(
                (self._cfg.width, self._cfg.height), Image.LANCZOS
            )
        return img

    def _make_placeholder(self) -> Image.Image:
        """Return a solid dark placeholder frame at output resolution."""
        if self._placeholder is None:
            self._placeholder = Image.new(
                "RGB",
                (self._cfg.width, self._cfg.height),
                self._cfg.placeholder_color,
            )
        return self._placeholder

    # --- Overlay rendering ---

    def _apply_overlays(
        self, img: Image.Image, overlays: dict[str, Overlay]
    ) -> Image.Image:
        """Draw text overlays onto the image using Pillow."""
        # Work in RGBA so we can alpha-composite
        base = img.convert("RGBA")
        overlay_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_layer)

        for slot, overlay in overlays.items():
            text = overlay.data.get("text", "")
            if not text:
                continue
            self._draw_text_overlay(draw, slot, text)

        img = Image.alpha_composite(base, overlay_layer).convert("RGB")
        return img

    def _draw_text_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        slot: str,
        text: str,
    ) -> None:
        """Position and draw a text overlay based on the slot layout."""
        layout = _SLOT_LAYOUTS.get(slot, _SLOT_LAYOUTS["title"])

        bbox = draw.textbbox((0, 0), text, font=self._font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pad_x, pad_y = layout["padding"]
        bg_w = text_w + pad_x * 2
        bg_h = text_h + pad_y * 2

        anchor = layout["anchor"]
        if anchor == "bottom_left":
            x = layout["x_offset"]
            y = self._cfg.height - layout["y_offset"] - bg_h
        elif anchor == "top_left":
            x = layout["x_offset"]
            y = layout["y_offset"]
        elif anchor == "top_full":
            x = 0
            y = layout["y_offset"]
            bg_w = self._cfg.width
        else:
            x = layout["x_offset"]
            y = layout["y_offset"]

        bg_color = layout["bg_color"]
        bg_alpha = layout["bg_alpha"]

        # Semi-transparent background rectangle
        draw.rectangle(
            [x, y, x + bg_w, y + bg_h],
            fill=(*bg_color, bg_alpha),
        )

        # Centered text
        text_x = x + (bg_w - text_w) // 2
        text_y = y + pad_y
        draw.text(
            (text_x, text_y),
            text,
            fill=layout["text_color"],
            font=self._font,
        )
