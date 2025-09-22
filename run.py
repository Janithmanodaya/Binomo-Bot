import os
import sys
import subprocess


def run(cmd, check=True):
    print(f"+ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)


def _in_colab() -> bool:
    # Multiple heuristics to detect Google Colab
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        pass
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    try:
        return os.path.exists("/content") and os.getcwd().startswith("/content")
    except Exception:
        return False


def main():
    # 1) Install requirements using the current interpreter
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print("requirements.txt not found.")
    else:
        run(f'"{sys.executable}" -m pip install -r "{req_file}"')

    args = set(sys.argv[1:])

    # Default behavior:
    # - In desktop/servers: open Tkinter UI (Windows double-click friendly)
    # - In Google Colab: open Streamlit UI with a public ngrok URL
    if not args:
        if _in_colab():
            args.add("--colab")
        else:
            args.add("--tk")

    # Tkinter UI
    if "--tk" in args or "tk" in args:
        tk_script = os.path.join("src", "tk_app.py")
        if not os.path.exists(tk_script):
            print(f"{tk_script} not found.")
            raise SystemExit(1)
        run(f'"{sys.executable}" "{tk_script}"')
        return

    # Streamlit UI (local)
    if "--ui" in args or "ui" in args or "--streamlit" in args:
        ui_script = os.path.join("src", "ui_app.py")
        if not os.path.exists(ui_script):
            print(f"{ui_script} not found.")
            raise SystemExit(1)
        # Allow overriding port via env var UI_PORT
        port = os.environ.get("UI_PORT", "8501")
        # Bind to all interfaces to support containers/VMs and remote port forwarding
        run(
            f'"{sys.executable}" -m streamlit run "{ui_script}" '
            f'--server.port {port} --server.address 0.0.0.0 --server.headless true'
        )
        return

    # Colab-friendly web launcher (Streamlit + ngrok)
    if "--colab" in args or "colab" in args:
        colab_script = os.path.join("src", "colab_web.py")
        if not os.path.exists(colab_script):
            print(f"{colab_script} not found.")
            raise SystemExit(1)
        run(f'"{sys.executable}" "{colab_script}"')
        return

    # Otherwise forward all args to the pipeline
    extra_args = [a for a in sys.argv[1:] if a not in {"--ui", "ui", "--streamlit", "--tk", "tk", "--colab", "colab"}]
    extra = " ".join([f'"{a}"' if " " in a else a for a in extra_args])
    script = os.path.join("src", "run_pipeline.py")
    if not os.path.exists(script):
        print(f"{script} not found.")
        raise SystemExit(1)

    run(f'"{sys.executable}" "{script}" {extra}'.strip())


if __name__ == "__main__":
    main()