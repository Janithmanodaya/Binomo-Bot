import os
import re
import sys
import time
import subprocess
from typing import Optional, Tuple


# This helper launches the existing Streamlit UI and exposes it publicly in Google Colab.
# It prefers ngrok when a valid auth token is available; otherwise it falls back to Cloudflare
# (trycloudflare) via pycloudflared which does not require authentication.


def _install_if_missing(pkg: str):
    try:
        __import__(pkg)
    except Exception:
        print(f"+ Installing missing dependency: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def _open_ngrok(port: int) -> Tuple[Optional[str], Optional[object]]:
    """
    Try to open an ngrok tunnel on port. Returns (public_url, ngrok_module_or_process).
    public_url is None on failure.
    """
    try:
        _install_if_missing("pyngrok")
        from pyngrok import ngrok  # type: ignore
    except Exception as e:
        print(f"! pyngrok unavailable: {e}")
        return None, None

    token = os.environ.get("NGROK_AUTH_TOKEN") or os.environ.get("NGROK_TOKEN")
    if not token:
        print("! No NGROK_AUTH_TOKEN detected; skipping ngrok.")
        return None, ngrok  # type: ignore[name-defined]

    try:
        ngrok.set_auth_token(token)
        print("+ ngrok auth token configured")
    except Exception as e:
        print(f"! Failed to set ngrok token: {e}")

    try:
        http_tunnel = ngrok.connect(port, "http")  # type: ignore[name-defined]
        public_url = getattr(http_tunnel, "public_url", None)
        if public_url:
            return str(public_url), ngrok  # type: ignore[name-defined]
    except Exception as e:
        print(f"! Failed to open ngrok tunnel: {e}")

    return None, ngrok  # type: ignore[name-defined]


def _open_cloudflared(port: int) -> Tuple[Optional[str], Optional[subprocess.Popen]]:
    """
    Open a Cloudflare (trycloudflare) tunnel to the given port.
    Returns (public_url, process). public_url is None if it couldn't be determined yet.
    """
    # Ensure helper/binary is available
    try:
        _install_if_missing("pycloudflared")
    except Exception as e:
        print(f"! Failed to install pycloudflared: {e}")

    # Try Python helper first
    try:
        # Some versions expose try_cloudflare, others expose start_cloudflared
        try:
            from pycloudflared import try_cloudflare  # type: ignore
            url = try_cloudflare(port=port)  # type: ignore[call-arg]
            if isinstance(url, (list, tuple)) and url:
                url = url[0]
            if isinstance(url, str) and "trycloudflare.com" in url:
                print("+ Cloudflare tunnel established via helper")
                return url, None
        except Exception:
            from pycloudflared import start_cloudflared  # type: ignore
            url = start_cloudflared(port)  # type: ignore[call-arg]
            if isinstance(url, (list, tuple)) and url:
                url = url[0]
            if isinstance(url, str) and "trycloudflare.com" in url:
                print("+ Cloudflare tunnel established via helper")
                return url, None
    except Exception as e:
        print(f"! pycloudflared helper import failed, falling back to binary: {e}")

    # Fallback: run cloudflared binary directly and parse URL from stdout
    cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate", "--metrics", "127.0.0.1:0"]
    print("+ Starting cloudflared:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    public_url: Optional[str] = None

    # Try to read a few lines quickly to capture the URL
    try:
        assert proc.stdout is not None
        start = time.time()
        while time.time() - start < 20.0:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.2)
                continue
            print("[cloudflared]", line, end="")
            m = re.search(r"https?://[\\w\\-\\.]+\\.trycloudflare\\.com", line)
            if m:
                public_url = m.group(0)
                break
    except Exception:
        pass

    if public_url:
        print("+ Cloudflare tunnel established")
    else:
        print("! Could not detect Cloudflare public URL yet; it may still appear in logs above.")

    return public_url, proc


def main():
    # Determine port and UI script
    port = int(os.environ.get("UI_PORT", "8501"))
    ui_script = os.path.join("src", "ui_app.py")
    if not os.path.exists(ui_script):
        print(f"{ui_script} not found.")
        raise SystemExit(1)

    # Launch Streamlit
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", ui_script,
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    print("+ Starting Streamlit UI...")
    print("+ Command:", " ".join(streamlit_cmd))
    st_proc = subprocess.Popen(streamlit_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Open public tunnel: ngrok (with token) first, else cloudflared (no auth needed)
    print(f"+ Opening public tunnel on port {port} ...")
    public_url = None
    tunnel_handle = None

    # Try ngrok first if token is present
    ngrok_url, ngrok_mod = _open_ngrok(port)
    if ngrok_url:
        public_url = ngrok_url
        tunnel_handle = ("ngrok", ngrok_mod)
    else:
        # Fallback to Cloudflare
        cf_url, cf_proc = _open_cloudflared(port)
        public_url = cf_url
        tunnel_handle = ("cloudflared", cf_proc)

    if public_url:
        print("\n================= PUBLIC URL =================")
        print(public_url)
        print("==============================================\n")
        print("Tip: In Google Colab, click the URL above to open the web UI.\n")
    else:
        print("! Failed to establish an authenticated ngrok tunnel and Cloudflare URL not detected yet.")
        print("  The Streamlit UI is still running locally; watch logs for a trycloudflare URL.")

    # Stream logs to console for visibility in Colab
    try:
        assert st_proc.stdout is not None
        for line in st_proc.stdout:
            print(line, end="")
            if public_url and ("Network URL" in line or "Local URL" in line):
                print(f"\n[Public URL] {public_url}\n")
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up processes
        try:
            st_proc.terminate()
        except Exception:
            pass
        try:
            if isinstance(tunnel_handle, tuple) and tunnel_handle[0] == "ngrok":
                mod = tunnel_handle[1]
                if mod:
                    try:
                        mod.kill()  # type: ignore[attr-defined]
                    except Exception:
                        pass
            elif isinstance(tunnel_handle, tuple) and tunnel_handle[0] == "cloudflared":
                proc = tunnel_handle[1]
                if isinstance(proc, subprocess.Popen):
                    try:
                        proc.terminate()
                    except Exception:
                        pass
        except Exception:
            pass


if __name__ == "__main__":
    main()