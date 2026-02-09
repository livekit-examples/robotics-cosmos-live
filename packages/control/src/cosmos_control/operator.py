from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from livekit import rtc

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
        "bg_alpha": 0.7,
        "text_color": (255, 255, 255),
        "padding": (20, 12),
    },
    "title": {
        "anchor": "top_left",
        "y_offset": 40,
        "x_offset": 40,
        "bg_color": (0, 0, 0),
        "bg_alpha": 0.8,
        "text_color": (255, 255, 255),
        "padding": (16, 8),
    },
    "banner": {
        "anchor": "top_full",
        "y_offset": 0,
        "x_offset": 0,
        "bg_color": (30, 30, 200),
        "bg_alpha": 0.9,
        "text_color": (255, 255, 255),
        "padding": (20, 10),
    },
}

_FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 1.0
_FONT_THICKNESS = 2
_WINDOW_NAME = "Cosmos Live"


class _TrackInfo:
    """Tracks available for a participant."""

    def __init__(self) -> None:
        self.video: rtc.Track | None = None


class CVDisplayOperator(StreamOperator):
    """Stream operator that displays frames in an OpenCV window.

    Connects to a LiveKit room as a separate participant, subscribes to
    video tracks, composites a single active feed with text overlays, and
    displays the result using ``cv2.imshow``.
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

        # Display thread
        self._stop_event = threading.Event()
        self._display_thread: threading.Thread | None = None

        # Placeholder cache
        self._placeholder: np.ndarray | None = None

    # --- Public API ---

    @property
    def available_feeds(self) -> list[str]:
        """Return participant identities that have at least one subscribed track."""
        return list(self._available_tracks.keys())

    async def start(self) -> None:
        """Connect to LiveKit and start the display loop.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._display_thread is not None:
            return
        logger.info("Starting CVDisplayOperator")

        # Connect to LiveKit
        await self._connect_livekit()

        # Start display thread
        self._stop_event.clear()
        self._display_thread = threading.Thread(
            target=self._display_loop, name="cv-display", daemon=True
        )
        self._display_thread.start()

        logger.info("CVDisplayOperator started")

    async def stop(self) -> None:
        """Disconnect from LiveKit, stop tasks, close display."""
        logger.info("Stopping CVDisplayOperator")

        # Signal thread to exit
        self._stop_event.set()

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

    # --- Display loop (dedicated thread) ---

    def _display_loop(self) -> None:
        """Composite frames at the target FPS and show via cv2.imshow.

        Runs in its own thread with absolute-time tracking to avoid drift.
        """
        frame_interval = 1.0 / self._cfg.fps
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

            # Convert RGB -> BGR for OpenCV display
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(_WINDOW_NAME, bgr)

            # waitKey is required for the window to refresh; also lets us
            # detect if the user closed the window (press 'q' to quit).
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("User pressed 'q' — stopping display")
                self._stop_event.set()
                break

            # Advance to next absolute frame time to avoid cumulative drift
            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.monotonic()
            if sleep_time > 0:
                if self._stop_event.wait(timeout=sleep_time):
                    break
            elif sleep_time < -frame_interval:
                # More than one frame behind — reset to avoid death spiral
                next_frame_time = time.monotonic()

        cv2.destroyAllWindows()

    # --- Frame processing ---

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a frame to the output resolution if needed."""
        h, w = frame.shape[:2]
        if w == self._cfg.width and h == self._cfg.height:
            return frame
        return cv2.resize(
            frame, (self._cfg.width, self._cfg.height), interpolation=cv2.INTER_LINEAR
        )

    def _make_placeholder(self) -> np.ndarray:
        """Return a solid dark placeholder frame at output resolution."""
        if self._placeholder is None:
            self._placeholder = np.full(
                (self._cfg.height, self._cfg.width, 3),
                self._cfg.placeholder_color,
                dtype=np.uint8,
            )
        return self._placeholder

    # --- Overlay rendering ---

    def _apply_overlays(
        self, frame: np.ndarray, overlays: dict[str, Overlay]
    ) -> np.ndarray:
        """Draw text overlays onto the frame using OpenCV."""
        frame = frame.copy()
        for slot, overlay in overlays.items():
            text = overlay.data.get("text", "")
            if not text:
                continue
            self._draw_text_overlay(frame, slot, text)
        return frame

    def _draw_text_overlay(
        self,
        frame: np.ndarray,
        slot: str,
        text: str,
    ) -> None:
        """Position and draw a text overlay based on the slot layout."""
        layout = _SLOT_LAYOUTS.get(slot)
        if layout is None:
            layout = _SLOT_LAYOUTS["title"]

        (text_w, text_h), baseline = cv2.getTextSize(
            text, _FONT_FACE, _FONT_SCALE, _FONT_THICKNESS
        )
        pad_x, pad_y = layout["padding"]
        bg_w = text_w + pad_x * 2
        bg_h = text_h + baseline + pad_y * 2

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

        # Draw semi-transparent background rectangle
        overlay_img = frame[y : y + bg_h, x : x + bg_w].copy()
        cv2.rectangle(overlay_img, (0, 0), (bg_w, bg_h), layout["bg_color"], -1)
        alpha = layout["bg_alpha"]
        frame[y : y + bg_h, x : x + bg_w] = cv2.addWeighted(
            overlay_img, alpha, frame[y : y + bg_h, x : x + bg_w], 1 - alpha, 0
        )

        # Draw text centered in the background
        text_x = x + (bg_w - text_w) // 2
        text_y = y + pad_y + text_h
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            _FONT_FACE,
            _FONT_SCALE,
            layout["text_color"],
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )
