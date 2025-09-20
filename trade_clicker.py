"""
Trade Clicker - Time-based auto clicker with image recognition and Sri Lanka virtual time.

Features:
- Beautiful Tkinter UI (ttk styling)
- Select Buy/Sell image files (used by pyautogui to locate buttons on screen)
- Enter a URL to open in the default external browser on Start
- Uses internet time for Asia/Colombo (GMT+5:30) via worldtimeapi.org (does not rely on system time)
- Displays live Sri Lanka time clock in UI
- Paste a schedule like:
    01:00 S
    01:05 B
    01:10 B
  The app will:
    * Find the next signal time greater than current time
    * Wait until that time, then wait 5 seconds and click the correct image/button
    * If the signal is more than 10 seconds late (>= HH:MM:10), it skips the trade
    * Optional interval lockout (e.g., 15 min) to avoid opening another trade too soon

Notes:
- pyautogui image search with "confidence" requires opencv-python to be installed.
  Without it, only exact image matches will work.
- Ensure your Buy/Sell images clearly match the on-screen buttons (same resolution/scale).
"""

import threading
import time
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple
import json
import urllib.request
import urllib.error
import importlib
import subprocess
import os
import sys
import traceback

# Optional global for delayed import (auto-installed on Start)
pyautogui = None  # type: ignore

# --------------- Configuration ---------------

WORLD_TIME_API = "http://worldtimeapi.org/api/timezone/Asia/Colombo"
TIME_RESYNC_SECONDS = 30  # periodically re-sync virtual time from internet
IMAGE_SEARCH_INTERVAL = 1.5  # seconds between image search attempts
IMAGE_SEARCH_TIMEOUT = 300  # max seconds to wait for locating both buttons (5 minutes)
CLICK_CONFIDENCE = 0.8  # used if OpenCV is available
LOG_MAX_LINES = 500

# Auto-install helper
def ensure_package(import_name: str, pip_name: Optional[str] = None, log=None):
    """
    Ensure a Python package is importable. If not, install via pip and import it.
    Tries several strategies to improve success on Windows and newer Python versions.
    """
    try:
        return importlib.import_module(import_name)
    except ImportError:
        pkg = pip_name or import_name
        if log:
            try:
                log(f"Installing dependency: {pkg} ...")
            except Exception:
                pass

        def run_pip(args):
            cmd = [sys.executable, "-m", "pip"] + args
            return subprocess.call(cmd)  # return code only

        # 1) Try plain install
        rc = run_pip(["install", pkg])
        if rc == 0:
            return importlib.import_module(import_name)

        # 2) Upgrade pip/setuptools/wheel then retry
        if log:
            log("Upgrading pip/setuptools/wheel and retrying...")
        run_pip(["install", "--upgrade", "pip", "setuptools", "wheel"])
        rc = run_pip(["install", pkg])
        if rc == 0:
            return importlib.import_module(import_name)

        # 3) Try --user
        if log:
            log("Retrying with --user ...")
        rc = run_pip(["install", "--user", pkg])
        if rc == 0:
            return importlib.import_module(import_name)

        # 4) Special handling for pyautogui: install its dependencies individually then retry
        if pkg.lower() in {"pyautogui", "pyautoGUI".lower()}:
            if log:
                log("Installing PyAutoGUI dependencies individually...")
            deps = ["pillow", "pyscreeze", "pygetwindow", "pymsgbox", "mouseinfo", "pyrect"]
            for d in deps:
                run_pip(["install", d])
            # Retry pyautogui once more
            rc = run_pip(["install", pkg])
            if rc == 0:
                return importlib.import_module(import_name)

        # 5) Try pinning a common stable version (best-effort)
        pinned_versions = {
            "pyautogui": ["0.9.54", "0.9.53"],
        }
        if pkg.lower() in pinned_versions:
            for ver in pinned_versions[pkg.lower()]:
                if log:
                    log(f"Retrying {pkg}=={ver} ...")
                rc = run_pip(["install", f"{pkg}=={ver}"])
                if rc == 0:
                    return importlib.import_module(import_name)

        raise RuntimeError(f"Failed to install {pkg}")

# --------------- Utilities ---------------

class InternetTimeProvider:
    """Provides virtual time for Sri Lanka (Asia/Colombo) using internet, not system time.
    Tries multiple sources and gracefully falls back to last-known offset or fixed +05:30.
    """

    def __init__(self):
        self._offset = timedelta(0)  # offset to apply to system time to map to Colombo internet time
        self._last_sync = 0.0
        self._tz_fallback = timezone(timedelta(hours=5, minutes=30))

    # --------- Internet time sources ---------

    def _fetch_worldtimeapi(self) -> datetime:
        req = urllib.request.Request(WORLD_TIME_API, headers={"User-Agent": "trade-clicker/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        dt_str = data.get("datetime")
        if not dt_str:
            raise RuntimeError("worldtimeapi: missing 'datetime'")
        return datetime.fromisoformat(dt_str)  # tz-aware

    def _fetch_timeapi_io(self) -> datetime:
        # Example: https://timeapi.io/api/Time/current/zone?timeZone=Asia/Colombo
        url = "https://timeapi.io/api/Time/current/zone?timeZone=Asia/Colombo"
        req = urllib.request.Request(url, headers={"User-Agent": "trade-clicker/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # They provide "dateTime" like "2023-09-20T12:34:56.0000000"
        # and "timeZone" and "currentLocalTime"
        dt_str = data.get("dateTime") or data.get("currentLocalTime") or data.get("dateTimeISO")
        if not dt_str:
            raise RuntimeError("timeapi.io: missing 'dateTime'")
        # Returned string may be naive; attach Colombo tz
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self._tz_fallback)
        return dt.astimezone(self._tz_fallback)

    def _fetch_from_date_header(self) -> datetime:
        # Use Date header from Google (UTC), then convert to Colombo TZ
        url = "https://www.google.com/generate_204"
        req = urllib.request.Request(url, headers={"User-Agent": "trade-clicker/1.0"}, method="HEAD")
        with urllib.request.urlopen(req, timeout=8) as resp:
            date_hdr = resp.headers.get("Date")
        if not date_hdr:
            raise RuntimeError("google: missing Date header")
        # Parse RFC 7231 date, e.g., "Tue, 15 Nov 1994 08:12:31 GMT"
        dt = datetime.strptime(date_hdr, "%a, %d %b %Y %H:%M:%S GMT")
        dt = dt.replace(tzinfo=timezone.utc).astimezone(self._tz_fallback)
        return dt

    def _fetch_colombo_time(self) -> datetime:
        """Fetch current Colombo time using multiple providers."""
        errors = []
        for fetcher in (self._fetch_worldtimeapi, self._fetch_timeapi_io, self._fetch_from_date_header):
            try:
                return fetcher()
            except Exception as e:
                errors.append(str(e))
                continue
        raise RuntimeError("All time sources failed: " + " | ".join(errors))

    # --------- Public API ---------

    def sync(self):
        """Sync offset between system time and true Colombo time."""
        colombo_now = self._fetch_colombo_time()
        system_now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
        # Use detected offset if available; otherwise fallback +05:30
        tz = colombo_now.tzinfo or self._tz_fallback
        system_as_colombo = system_now_utc.astimezone(tz)
        self._offset = colombo_now - system_as_colombo
        self._last_sync = time.time()

    def now(self) -> datetime:
        """Return current virtual Colombo time. Re-sync periodically."""
        try:
            if (time.time() - self._last_sync) > TIME_RESYNC_SECONDS or self._last_sync == 0.0:
                self.sync()
        except Exception:
            # If sync fails, keep previous offset and continue ticking locally
            pass
        # Use system time as base, apply offset to emulate Colombo internet time
        system_now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
        tz = self._tz_fallback
        system_as_colombo = system_now_utc.astimezone(tz)
        return system_as_colombo + self._offset

# --------------- Parsing and Scheduling ---------------

def parse_schedule(text: str) -> List[Tuple[int, int, str]]:
    """
    Parse schedule lines like "01:10 B" or "23:45 S" (case-insensitive B/S).
    Also accepts synonyms: BUY/CALL => B, SELL/PUT => S.

    Returns: list of tuples (hour, minute, side) where side in {"B", "S"} sorted by time.
    """
    entries = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        hhmm, side = parts
        if ":" not in hhmm:
            continue
        try:
            hh, mm = hhmm.split(":")
            h = int(hh)
            m = int(mm)
            if not (0 <= h <= 23 and 0 <= m <= 59):
                continue
        except ValueError:
            continue
        side_up = side.upper()
        # Map synonyms to canonical B/S
        if side_up in {"B", "BUY", "CALL"}:
            canon = "B"
        elif side_up in {"S", "SELL", "PUT"}:
            canon = "S"
        else:
            continue
        entries.append((h, m, canon))
    # Sort by hour, then minute
    entries.sort(key=lambda x: (x[0], x[1]))
    return entries


def find_next_signal(now_dt: datetime, schedule: List[Tuple[int, int, str]]) -> Tuple[datetime, str]:
    """
    Given current datetime (Colombo) and schedule list, find the next signal datetime (today or next day) and side.
    """
    if not schedule:
        raise ValueError("Schedule is empty.")
    today = now_dt.date()
    # Search today first
    for h, m, side in schedule:
        candidate = datetime(now_dt.year, now_dt.month, now_dt.day, h, m, 0, tzinfo=now_dt.tzinfo)
        if candidate > now_dt:
            return candidate, side
    # If none left today, roll to tomorrow with first schedule item
    h, m, side = schedule[0]
    tomorrow = today + timedelta(days=1)
    candidate = datetime(tomorrow.year, tomorrow.month, tomorrow.day, h, m, 0, tzinfo=now_dt.tzinfo)
    return candidate, side

# --------------- Image Recognition and Clicking ---------------

class ScreenAutomation:
    """
    Abstraction over screen search and clicking.
    Tries to use pyautogui if available; otherwise falls back to pyscreeze + pydirectinput.
    """

    def __init__(self, log=None):
        self.log = log or (lambda *a, **k: None)
        self.backend = None  # "pyautogui" or "fallback"
        self.pg = None
        self.pyscreeze = None
        self.direct = None
        self.has_confidence = False  # True if OpenCV is available

        # Detect OpenCV availability
        try:
            importlib.import_module("cv2")
            self.has_confidence = True
        except ImportError:
            self.has_confidence = False

        # Try pyautogui first WITHOUT installing, to avoid unnecessary installs
        try:
            self.pg = importlib.import_module("pyautogui")
            ver = getattr(self.pg, "__version__", "?")
            self.backend = "pyautogui"
            self.log(f"Found pyautogui {ver}. Using pyautogui backend.")
        except ImportError:
            # Not installed; DO NOT auto-install. Try fallback imports only.
            # pyscreeze
            try:
                self.pyscreeze = importlib.import_module("pyscreeze")
                self.log("Found pyscreeze.")
            except ImportError:
                raise RuntimeError("pyscreeze not installed. Please install it or install pyautogui.")

            # Pillow (PIL)
            try:
                importlib.import_module("PIL")
                self.log("Found pillow.")
            except ImportError:
                raise RuntimeError("pillow not installed. Please install it or install pyautogui.")

            # pydirectinput
            try:
                self.direct = importlib.import_module("pydirectinput")
                self.log("Found pydirectinput.")
            except ImportError:
                raise RuntimeError("pydirectinput not installed. Please install it or install pyautogui.")

            # Optional: OpenCV to allow confidence param in locateOnScreen (already detected)

            self.backend = "fallback"
            self.log("Using fallback backend (pyscreeze + pydirectinput).")

    def screen_size(self) -> Tuple[int, int]:
        """Return (width, height) of the primary screen."""
        if self.backend == "pyautogui":
            try:
                size = self.pg.size()
                return int(size[0]), int(size[1])
            except Exception:
                pass
        # fallback using pyscreeze screenshot
        try:
            im = self.pyscreeze.screenshot() if self.pyscreeze else self.pg.screenshot()
            return im.size[0], im.size[1]
        except Exception:
            # Conservative default full HD
            return 1920, 1080

    def locate_on_screen(self, image_path: str, confidence: Optional[float] = None, region: Optional[Tuple[int, int, int, int]] = None):
        # Only pass confidence if cv2 is available
        if not self.has_confidence:
            confidence = None

        # Helper to decide if an exception is the "not found" case
        def _is_not_found(exc: Exception) -> bool:
            name = exc.__class__.__name__
            return name == "ImageNotFoundException"

        if self.backend == "pyautogui":
            try:
                kwargs = {}
                if confidence is not None:
                    kwargs["confidence"] = confidence
                if region is not None:
                    kwargs["region"] = region
                return self.pg.locateOnScreen(image_path, **kwargs)
            except Exception as e:
                # If it's the "not found" case, return None instead of raising
                if _is_not_found(e):
                    return None
                # Retry once without confidence (in case confidence unsupported), then handle not-found again
                try:
                    kwargs.pop("confidence", None)
                    return self.pg.locateOnScreen(image_path, **kwargs)
                except Exception as e2:
                    if _is_not_found(e2):
                        return None
                    raise
        else:
            # pyscreeze.locateOnScreen supports confidence if OpenCV is available
            try:
                kwargs = {}
                if confidence is not None:
                    kwargs["confidence"] = confidence
                if region is not None:
                    kwargs["region"] = region
                return self.pyscreeze.locateOnScreen(image_path, **kwargs)
            except Exception as e:
                if _is_not_found(e):
                    return None
                try:
                    kwargs.pop("confidence", None)
                    return self.pyscreeze.locateOnScreen(image_path, **kwargs)
                except Exception as e2:
                    if _is_not_found(e2):
                        return None
                    raise

    @staticmethod
    def center_of(box):
        # box is either a pyscreeze.Box or tuple (left, top, width, height)
        left, top, width, height = box
        cx = int(left + width / 2)
        cy = int(top + height / 2)
        return cx, cy

    def move_to(self, x: int, y: int, duration: float = 0.2):
        if self.backend == "pyautogui":
            self.pg.moveTo(x, y, duration=duration)
        else:
            # pydirectinput has moveTo but no smooth duration; emulate with sleep
            self.direct.moveTo(x, y)
            if duration:
                time.sleep(duration)

    def click(self, x: int, y: int):
        if self.backend == "pyautogui":
            self.pg.click(x, y)
        else:
            self.direct.click(x=x, y=y)

    # ---------- Color-aware helpers (used when OpenCV is available) ----------

    def _screenshot_bgr(self):
        """
        Return current screen as a NumPy array in BGR order, or None if unavailable.
        Requires OpenCV (cv2) and NumPy; only used when has_confidence is True.
        """
        if not self.has_confidence:
            return None
        try:
            import numpy as np  # cv2 depends on numpy; should be present with cv2
            from PIL import Image
        except Exception:
            return None

        try:
            pil_img = None
            if self.backend == "pyautogui" and hasattr(self.pg, "screenshot"):
                pil_img = self.pg.screenshot()
            elif self.pyscreeze and hasattr(self.pyscreeze, "screenshot"):
                pil_img = self.pyscreeze.screenshot()
            if pil_img is None:
                return None
            rgb = np.array(pil_img)  # RGB
            # Convert to BGR
            bgr = rgb[:, :, ::-1].copy()
            return bgr
        except Exception:
            return None

    def classify_box_color(self, box) -> str:
        """
        Classify the dominant color in the given box as 'green', 'red', or 'other'.
        Uses HSV thresholds on a screenshot. Returns 'other' if unavailable.
        Only used when OpenCV is available.
        """
        if not self.has_confidence:
            return "other"
        try:
            import cv2
            import numpy as np
        except Exception:
            return "other"

        bgr = self._screenshot_bgr()
        if bgr is None:
            return "other"

        left, top, width, height = box
        h, w = bgr.shape[:2]
        x1 = max(int(left), 0)
        y1 = max(int(top), 0)
        x2 = min(int(left + width), w)
        y2 = min(int(top + height), h)
        if x2 <= x1 or y2 <= y1:
            return "other"

        roi = bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return "other"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Thresholds
        # Green: H in [35, 85], S and V not too low
        lower_green = (35, 80, 60)
        upper_green = (85, 255, 255)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Red wraps around: [0,10] and [170,180]
        lower_red1 = (0, 80, 60)
        upper_red1 = (10, 255, 255)
        lower_red2 = (170, 80, 60)
        upper_red2 = (180, 255, 255)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Compute ratios
        total = roi.shape[0] * roi.shape[1]
        if total == 0:
            return "other"
        green_ratio = float(np.count_nonzero(mask_green)) / float(total)
        red_ratio = float(np.count_nonzero(mask_red)) / float(total)

        # Heuristic thresholds
        if green_ratio >= 0.12 and green_ratio > red_ratio * 1.2:
            return "green"
        if red_ratio >= 0.12 and red_ratio > green_ratio * 1.2:
            return "red"
        return "other"

def locate_image_center(screen: "ScreenAutomation", path: str, confidence: Optional[float] = None, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
    """
    Locate image on screen. Returns a Box (left, top, width, height) or None.
    If confidence is provided and OpenCV is available, uses that confidence.
    Optionally restrict search to a region=(left, top, width, height).
    """
    return screen.locate_on_screen(path, confidence=confidence, region=region)


def wait_for_images(screen: "ScreenAutomation", buy_img: str, sell_img: str, log, stop_event: threading.Event) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Wait until both Buy and Sell images are located on the screen.
    Ensures the two matches are different on-screen locations. If both initially
    resolve to the same area, retries the second search excluding a small region
    around the first match. If OpenCV is available, validate color (BUY=green, SELL=red)
    and discard mismatched color hits to reduce confusion from nearly identical shapes.
    Returns center points for both as ((x,y)_buy, (x,y)_sell).
    """
    start = time.time()
    buy_box = None
    sell_box = None
    buy_center = None
    sell_center = None

    while not stop_event.is_set():
        if time.time() - start > IMAGE_SEARCH_TIMEOUT:
            raise TimeoutError("Timed out waiting for Buy/Sell buttons on screen.")

        # Try find BUY if missing
        if buy_center is None:
            box = locate_image_center(screen, buy_img, CLICK_CONFIDENCE)
            if box:
                # Accept the first reasonable match. We'll use color later as a hint,
                # but do not block progress here to avoid infinite waiting.
                bx, by = screen.center_of(box)
                buy_box = box
                buy_center = (bx, by)
                log(f"Identified BUY button at {buy_center}")

        # Try find SELL if missing
        if sell_center is None:
            box = locate_image_center(screen, sell_img, CLICK_CONFIDENCE)
            if box:
                sx, sy = screen.center_of(box)

                # If SELL initially lands on the same center as BUY, try to disambiguate immediately
                if buy_center and (sx, sy) == buy_center:
                    width, height = screen.screen_size()
                    ex_w = 80
                    ex_h = 80
                    bx, by = buy_center
                    ex_left = max(bx - ex_w // 2, 0)
                    ex_top = max(by - ex_h // 2, 0)
                    ex_right = min(bx + ex_w // 2, width)
                    ex_bottom = min(by + ex_h // 2, height)

                    regions = []
                    if ex_top > 0:
                        regions.append((0, 0, width, ex_top))
                    if ex_botto <m height:
                        regions.append((0, ex_bottom, width, height - ex_bottom))
                    if ex_left > 0:
                        regions.append((0, ex_top, ex_left, ex_bottom - ex_top))
                    if ex_righ <t width:
                        regions.append((ex_right, ex_top, width - ex_right, ex_bottom - ex_top))

                    found_alt = None
                    for r in regions:
                        alt_box = locate_image_center(screen, sell_img, CLICK_CONFIDENCE, region=r)
                        if alt_box:
                            # Optional: quick color sanity for SELL
                            color = screen.classify_box_color(alt_box)
                           ")

        # If both found but centers are identical, attempt a disambiguation pass
        if buy_center and sell_center and buy_center == sell_center:
            # Exclude a small rectangle around the first (BUY) and re-search SELL
            width, height = screen.screen_size()
            # Exclusion margin around the first center
            ex_w = 80
            ex_h = 80
            bx, by = buy_center
            ex_left = max(bx - ex_w // 2, 0)
            ex_top = max(by - ex_h // 2, 0)
            ex_right = min(bx + ex_w // 2, width)
            ex_bottom = min(by + ex_h // 2, height)

            # Build up to 4 regions around the exclusion zone
            regions = []
            # Top region
            if ex_top > 0:
                regions.append((0, 0, width, ex_top))
            # Bottom region
            if ex_bottom < height:
                regions.append((0, ex_bottom, width, height - ex_bottom))
            # Left region
            if ex_left > 0:
                regions.append((0, ex_top, ex_left, ex_bottom - ex_top))
            # Right region
            if ex_right < width:
                regions.append((ex_right, ex_top, width - ex_right, ex_bottom - ex_top))

            found_alt = None
            for r in regions:
                alt_box = locate_image_center(screen, sell_img, CLICK_CONFIDENCE, region=r)
                if alt_box:
                    # Validate color for SELL
                    color = screen.classify_box_color(alt_box)
                    if color == "green":
                        continue
                    found_alt = alt_box
                    break

            if found_alt:
                sx, sy = screen.center_of(found_alt)
                sell_box = found_alt
                sell_center = (sx, sy)
                log(f"Adjusted SELL match to {sell_center}")
            else:
                # Could not disambiguate now; clear SELL and retry in next loop
                sell_center = None
                sell_box = None

        # If both found, as a final color sanity check (when available), correct or swap if needed
        if buy_center and sell_center and screen.has_confidence:
            buy_color = screen.classify_box_color(buy_box)
            sell_color = screen.classify_box_color(sell_box)
            if buy_color == "red" and sell_color == "green":
                # Likely swapped; swap the assignments
                buy_center, sell_center = sell_center, buy_center
                buy_box, sell_box = sell_box, buy_box
                log("Swapped BUY/SELL matches based on color validation.")
            elif buy_color == "red":
                # Discard BUY and retry
                buy_center, buy_box = None, None
            elif sell_color == "green":
                # Discard SELL and retry
                sell_center, sell_box = None, None

        if buy_center and sell_center:
            return buy_center, sell_center

        time.sleep(IMAGE_SEARCH_INTERVAL)

    raise RuntimeError("Stopped while waiting for images.")

# --------------- UI App ---------------

class TradeClickerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Trade Clicker - Sri Lanka Time")
        self.root.geometry("900x650")
        self.root.minsize(860, 600)

        self.time_provider = InternetTimeProvider()

        # State
        self.buy_image_path: Optional[str] = None
        self.sell_image_path: Optional[str] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.last_trade_at: Optional[datetime] = None

        # UI
        self._build_ui()
        self._start_clock_updater()

    # ---------- UI Builders ----------

    def _build_ui(self):
        self.style = ttk.Style()
        # Try a nicer theme if available
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")
        # Style tweaks
        self.style.configure("TButton", padding=6)
        self.style.configure("TLabel", padding=4)
        self.style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"))
        self.style.configure("Clock.TLabel", font=("Segoe UI", 14, "bold"), foreground="#0A8754")
        self.style.configure("Log.TText", font=("Consolas", 10))

        header = ttk.Frame(self.root, padding=10)
        header.pack(fill="x")
        ttk.Label(header, text="Trade Clicker", style="Header.TLabel").pack(side="left")
        self.clock_var = tk.StringVar(value="--:--:--")
        self.clock_label = ttk.Label(header, textvariable=self.clock_var, style="Clock.TLabel")
        self.clock_label.pack(side="right")

        # Settings frame
        settings = ttk.LabelFrame(self.root, text="Settings", padding=10)
        settings.pack(fill="x", padx=12, pady=8)

        # URL input
        ttk.Label(settings, text="URL:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
        self.url_var = tk.StringVar(value="https://binomo.com/trading")
        self.url_entry = ttk.Entry(settings, textvariable=self.url_var, width=60)
        self.url_entry.grid(row=0, column=1, sticky="we", pady=4)
        settings.grid_columnconfigure(1, weight=1)

        # Interval
        ttk.Label(settings, text="Interval lockout (minutes):").grid(row=0, column=2, sticky="e", padx=(12, 6))
        self.interval_var = tk.StringVar(value="0")
        self.interval_entry = ttk.Entry(settings, textvariable=self.interval_var, width=8)
        self.interval_entry.grid(row=0, column=3, sticky="w")

        # Image selectors
        img_frame = ttk.Frame(settings)
        img_frame.grid(row=1, column=0, columnspan=4, sticky="we", pady=(8, 0))
        img_frame.grid_columnconfigure(1, weight=1)
        img_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(img_frame, text="Buy image:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.buy_img_var = tk.StringVar(value="Not selected")
        self.buy_img_label = ttk.Label(img_frame, textvariable=self.buy_img_var)
        self.buy_img_label.grid(row=0, column=1, sticky="we")
        ttk.Button(img_frame, text="Choose...", command=self.choose_buy_image).grid(row=0, column=2, padx=6)

        ttk.Label(img_frame, text="Sell image:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(6, 0))
        self.sell_img_var = tk.StringVar(value="Not selected")
        self.sell_img_label = ttk.Label(img_frame, textvariable=self.sell_img_var)
        self.sell_img_label.grid(row=1, column=1, sticky="we", pady=(6, 0))
        ttk.Button(img_frame, text="Choose...", command=self.choose_sell_image).grid(row=1, column=2, padx=6, pady=(6, 0))

        # Schedule
        schedule_frame = ttk.LabelFrame(self.root, text="Schedule (HH:MM B/S, one per line)", padding=8)
        schedule_frame.pack(fill="both", expand=True, padx=12, pady=8)

        self.schedule_text = tk.Text(schedule_frame, height=12, wrap="none", font=("Consolas", 11))
        self.schedule_text.pack(fill="both", expand=True)
        self.schedule_text.insert("1.0", "01:00 S\n01:05 B\n01:10 B\n01:15 S\n01:20 S\n01:25 B\n01:30 S\n01:35 S\n")

        # Controls
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill="x", padx=12, pady=(0, 8))
        self.start_btn = ttk.Button(controls, text="Start", command=self.on_start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(controls, text="Stop", command=self.on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))

        # Log
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=8)
        log_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.log_text = tk.Text(log_frame, height=10, wrap="word", state="disabled", font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True)

    # ---------- UI Handlers ----------

    def choose_buy_image(self):
        path = filedialog.askopenfilename(
            title="Choose Buy image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")]
        )
        if path:
            self.buy_image_path = path
            self.buy_img_var.set(os.path.basename(path))

    def choose_sell_image(self):
        path = filedialog.askopenfilename(
            title="Choose Sell image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")]
        )
        if path:
            self.sell_image_path = path
            self.sell_img_var.set(os.path.basename(path))

    def log(self, message: str):
        ts = self.time_provider.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        # Trim log
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > LOG_MAX_LINES:
            self.log_text.delete("1.0", f"{lines-LOG_MAX_LINES}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _start_clock_updater(self):
        def tick():
            try:
                now = self.time_provider.now()
                self.clock_var.set(now.strftime("Sri Lanka Time: %Y-%m-%d %H:%M:%S"))
            finally:
                self.root.after(500, tick)
        tick()

    # ---------- Start/Stop ----------

    def on_start(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning("Missing URL", "Please enter a URL.")
            return
        if not self.buy_image_path or not self.sell_image_path:
            messagebox.showwarning("Missing Images", "Please select both Buy and Sell images.")
            return
        try:
            interval_min = int(self.interval_var.get() or "0")
            if interval_min < 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid Interval", "Interval lockout must be a non-negative integer (minutes).")
            return

        schedule_text = self.schedule_text.get("1.0", "end")
        schedule = parse_schedule(schedule_text)
        if not schedule:
            messagebox.showwarning("Invalid Schedule", "Please provide at least one valid schedule line (HH:MM B/S).")
            return

        # Disable start, enable stop
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.stop_event.clear()
        self.last_trade_at = None

        self.worker_thread = threading.Thread(
            target=self.worker_main,
            args=(url, schedule, interval_min),
            daemon=True
        )
        self.worker_thread.start()

    def on_stop(self):
        self.stop_event.set()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.log("Stopped by user.")

    # ---------- Worker Thread ----------

    def worker_main(self, url: str, schedule: List[Tuple[int, int, str]], interval_min: int):
        def log(msg: str):
            self.root.after(0, self.log, msg)

        # Prepare screen automation backend (auto-installs required packages)
        try:
            screen = ScreenAutomation(log=log)
        except Exception as e:
            log(f"Failed to prepare screen automation backend: {e}")
            self.root.after(0, self.on_stop)
            return

        # Ensure internet time available
        log("Connecting to internet time service...")
        while not self.stop_event.is_set():
            try:
                self.time_provider.sync()
                break
            except Exception as e:
                log(f"Time sync failed: {e}. Retrying in 3s...")
                time.sleep(3)

        if self.stop_event.is_set():
            return

        # Open URL in default browser
        log(f"Opening URL: {url}")
        try:
            webbrowser.open(url, new=1, autoraise=True)
        except Exception as e:
            log(f"Failed to open browser: {e}")

        # Identify Buy/Sell buttons
        log("Waiting for BUY and SELL buttons to appear on screen...")
        try:
            # Extra validation to catch common issues early
            if not os.path.isfile(self.buy_image_path):
                raise FileNotFoundError(f"Buy image not found: {self.buy_image_path}")
            if not os.path.isfile(self.sell_image_path):
                raise FileNotFoundError(f"Sell image not found: {self.sell_image_path}")
            # Prevent using the same file for both buttons
            if os.path.samefile(self.buy_image_path, self.sell_image_path):
                raise ValueError("Buy and Sell images are the same file. Please select two different images.")

            buy_center, sell_center = wait_for_images(screen, self.buy_image_path, self.sell_image_path, log, self.stop_event)
            log("Both buttons identified.")
        except Exception as e:
            tb = traceback.format_exc()
            log(f"Error identifying buttons: {e!r}\n{tb}")
            self.root.after(0, self.on_stop)
            return

        # Main loop
        next_signal_dt, next_side = find_next_signal(self.time_provider.now(), schedule)
        exec_dt = next_signal_dt - timedelta(minutes=1)
        log(f"Next signal at {next_signal_dt.strftime('%H:%M')} {next_side} (execute at {exec_dt.strftime('%H:%M')})")

        while not self.stop_event.is_set():
            now = self.time_provider.now()

            # Interval lockout check
            if self.last_trade_at and interval_min > 0:
                until = self.last_trade_at + timedelta(minutes=interval_min)
                if now < until:
                    # Still in lockout; skip signals that occur during lockout
                    # But we continue to advance next_signal_dt as time passes
                    pass

            # Advance next execution if it's in the past (missed) by >= 10s or we are beyond it
            # If we are before it, we can sleep a bit
            if now < exec_dt:
                time.sleep(0.2)
                continue

            # We are at or after the execution minute (signal time minus one minute)
            delta_sec = (now - exec_dt).total_seconds()

            if delta_sec < 5:
                # Wait until we reach +5s window start
                time.sleep(0.2)
                continue
            elif 5 <= delta_sec < 10:
                # Eligible window to execute, but ensure lockout
                in_lockout = False
                if self.last_trade_at and interval_min > 0:
                    if now < (self.last_trade_at + timedelta(minutes=interval_min)):
                        in_lockout = True

                if in_lockout:
                    log(f"In lockout window, skipping signal {next_signal_dt.strftime('%H:%M')} {next_side}")
                else:
                    # Re-locate before clicking in case layout moved
                    try:
                        if next_side == "B":
                            box = locate_image_center(screen, self.buy_image_path, CLICK_CONFIDENCE)
                            center = screen.center_of(box) if box else None
                            if center:
                                x, y = center
                                screen.move_to(x, y, duration=0.2)
                                screen.click(x, y)
                                self.last_trade_at = now
                                log(f"Executed BUY at {now.strftime('%H:%M:%S')} (clicked at {center})")
                            else:
                                log("BUY button not found at execution time. Skipping.")
                        else:
                            box = locate_image_center(screen, self.sell_image_path, CLICK_CONFIDENCE)
                            center = screen.center_of(box) if box else None
                            if center:
                                x, y = center
                                screen.move_to(x, y, duration=0.2)
                                screen.click(x, y)
                                self.last_trade_at = now
                                log(f"Executed SELL at {now.strftime('%H:%M:%S')} (clicked at {center})")
                            else:
                                log("SELL button not found at execution time. Skipping.")
                    except Exception as e:
                        log(f"Error during click: {e}")

                # Regardless, schedule next signal
                next_signal_dt, next_side = find_next_signal(now, schedule)
                exec_dt = next_signal_dt - timedelta(minutes=1)
                log(f"Next signal at {next_signal_dt.strftime('%H:%M')} {next_side} (execute at {exec_dt.strftime('%H:%M')})")

            else:
                # >= 10s late relative to execution time, skip and schedule next
                log(f"Missed execution for signal {next_signal_dt.strftime('%H:%M')} {next_side} (>{int(delta_sec)}s late). Skipping.")
                next_signal_dt, next_side = find_next_signal(now, schedule)
                exec_dt = next_signal_dt - timedelta(minutes=1)
                log(f"Next signal at {next_signal_dt.strftime('%H:%M')} {next_side} (execute at {exec_dt.strftime('%H:%M')})")

            time.sleep(0.05)

    # ---------- Run ----------

def main():
    # Note: pyautogui will be prepared after Start; skip pre-check to allow auto-install.
    root = tk.Tk()
    app = TradeClickerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()