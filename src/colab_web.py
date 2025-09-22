import os
import sys
import time
import subprocess
from typing import Optional


# This helper launches the existing Streamlit UI.
# In Google Colab, it prints a direct Colab proxy URL (no tunneling required),
# with a fallback to Cloudflare (or ngrok) tunneling if the proxy URL isn't available.
# Outside Colab, it just runs Streamlit locally.


def _in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def _colab_proxy_url(port: int, retries: int = 30, delay: float = 1.0) -> Optional[str]:
    """
    Ask Colab to proxy the given local port and return a direct URL on the colab domain.
    Retries for a short period until the server responds.
    """
    try:
        from google.colab import output  # type: ignore
    except Exception:
        return None

    url: Optional[str] = None
    for _ in range(max(1, retries)):
        try:
            # This returns a fully qualified URL on the Colab domain
            url = output.eval_js(f"google.colab.kernel.proxyPort({int(port)})")  # type: ignore
            if isinstance(url, str) and url.startswith("http"):
                return url
        except Exception:
            pass
        time.sleep(delay)
    return url


def _try_cloudflare_tunnel(port: int) -> Optional[str]:
    """
    Try to open a Cloudflare quick tunnel using pycloudflared.
    Returns the public URL if successful.
    """
    try:
        # Common helper used in many Colab notebooks
        from pycloudflared import try_cloudflare  # type: ignore
        # Some versions accept port positional, others keyword
        url = try_cloudflare(port=port)  # type: ignore
        if isinstance(url, str) and url.startswith("http"):
            return url
    except Exception:
        pass
    # Fallback: attempt direct cloudflared invocation if library helper not available
    try:
        # Spawn cloudflared tunnel --url http://localhost:port and parse the printed URL
        proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{int(port)}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Give it a few seconds to print the assigned URL
        assert proc.stdout is not None
        start = time.time()
        while time.time() - start < 20:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.2)
                continue
            # Cloudflared usually prints "trycloudflare.com" URL
            if "trycloudflare.com" in line or "https://" in line:
                tokens = [t for t in line.strip().split() if t.startswith("http")]
                if tokens:
                    return tokens[0]
        # If we couldn't parse, leave the tunnel running but return None
    except Exception:
        pass
    return None


def _try_ngrok_tunnel(port: int) -> Optional[str]:
    """
    Try to open an ngrok tunnel if available. Uses NGROK_AUTH_TOKEN if set.
    """
    try:
        from pyngrok import ngrok  # type: ignore
    except Exception:
        return None
    try:
        token = os.environ.get("NGROK_AUTH_TOKEN")
        if token:
            ngrok.set_auth_token(token)
        tunnel = ngrok.connect(int(port), "http")
        url = getattr(tunnel, "public_url", None)
        if isinstance(url, str) and url.startswith("http"):
            return url
    except Exception:
        return None
    return None


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

    public_url: Optional[str] = None

    if _in_colab():
        print("+ Detected Google Colab environment. Preparing Colab proxy URL ...")
        # Give Streamlit a moment to bind, then ask Colab for the proxy URL
        time.sleep(2.5)
        public_url = _colab_proxy_url(port)
        if public_url:
            print("\n================= COLAB URL =================")
            print(public_url)
            print("=============================================\n")
            print("Tip: Click the URL above to open the Streamlit UI in Colab.\n")
        else:
            print("! Could not get Colab proxy URL yet. Trying a tunnel (Cloudflare/ngrok)...")
            # First try Cloudflare (no auth required)
            public_url = _try_cloudflare_tunnel(port)
            if public_url:
                print("\n================= TUNNEL URL (Cloudflare) =================")
                print(public_url)
                print("===========================================================\n")
            else:
                # Then try ngrok if available
                public_url = _try_ngrok_tunnel(port)
                if public_url:
                    print("\n================= TUNNEL URL (ngrok) =================")
                    print(public_url)
                    print("======================================================\n")
                else:
                    print("! Tunneling failed; falling back to logs. You can still use the Colab 'Public URL' if it appears below.")
    else:
        print(f"+ Streamlit running locally at http://0.0.0.0:{port}")

    # Stream logs to console for visibility
    try:
        assert st_proc.stdout is not None
        for line in st_proc.stdout:
            print(line, end="")
            # When running in Colab, try again to fetch the proxy URL once Streamlit reports readiness
            if _in_colab() and public_url is None and ("Network URL" in line or "Local URL" in line or "You can now view your Streamlit app" in line):
                # Retry Colab proxy briefly
                url = _colab_proxy_url(port, retries=5, delay=1.0)
                if not url:
                    # Retry tunnels briefly
                    url = _try_cloudflare_tunnel(port) or _try_ngrok_tunnel(port)
                if url:
                    public_url = url
                    print(f"\n[Public URL] {public_url}\n")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            st_proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()