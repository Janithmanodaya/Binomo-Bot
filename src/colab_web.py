import os
import sys
import time
import subprocess
from typing import Optional

# This helper launches the existing Streamlit UI and exposes it publicly in Google Colab
# using an ngrok tunnel. The Tkinter app remains available via `python run.py --tk` locally.

def _install_if_missing(pkg: str):
    try:
        __import__(pkg)
    except Exception:
        print(f"+ Installing missing dependency: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def main():
    # Ensure pyngrok is available
    _install_if_missing("pyngrok")

    from pyngrok import ngrok  # type: ignore

    # Configure auth token if provided
    token = os.environ.get("NGROK_AUTH_TOKEN") or os.environ.get("NGROK_TOKEN")
    if token:
        try:
            ngrok.set_auth_token(token)
            print("+ ngrok auth token configured")
        except Exception as e:
            print(f"! Failed to set ngrok token: {e}")

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
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    print("+ Starting Streamlit UI...")
    print("+ Command:", " ".join(streamlit_cmd))
    st_proc = subprocess.Popen(streamlit_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Open public tunnel
    print(f"+ Opening ngrok tunnel on port {port} ...")
    public_url = None
    try:
        http_tunnel = ngrok.connect(port, "http")
        public_url = http_tunnel.public_url
        print("\n================= PUBLIC URL =================")
        print(public_url)
        print("==============================================\n")
        print("Tip: In Google Colab, click the URL above to open the web UI.\n")
    except Exception as e:
        print(f"! Failed to open ngrok tunnel: {e}")
        print("You can still access via the notebook proxy or local port forwarding if available.")

    # Stream logs to console for visibility in Colab
    try:
        assert st_proc.stdout is not None
        for line in st_proc.stdout:
            print(line, end="")
            # Print the public URL periodically to keep it visible
            if public_url and ("Network URL" in line or "Local URL" in line):
                print(f"\n[Public URL] {public_url}\n")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            st_proc.terminate()
        except Exception:
            pass
        try:
            ngrok.kill()
        except Exception:
            pass

if __name__ == "__main__":
    main()