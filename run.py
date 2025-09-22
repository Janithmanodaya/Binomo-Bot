import os
import sys
import subprocess


def run(cmd, check=True):
    print(f"+ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)


def _get_arg_value(name: str) -> str | None:
    """
    Fetch value for flags like --port 9000 or --port=9000 from sys.argv.
    """
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def main():
    # 1) Install requirements using the current interpreter
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print("requirements.txt not found.")
    else:
        run(f'"{sys.executable}" -m pip install -r "{req_file}"')

    args = set(sys.argv[1:])

    # Default behavior: if no args, open Tkinter UI (Windows double-click friendly)
    if not args or "--tk" in args or "tk" in args:
        tk_script = os.path.join("src", "tk_app.py")
        if not os.path.exists(tk_script):
            print(f"{tk_script} not found.")
            raise SystemExit(1)
        run(f'"{sys.executable}" "{tk_script}"')
        return

    # If user requested the Streamlit UI, launch it
    if "--ui" in args or "ui" in args or "--streamlit" in args:
        ui_script = os.path.join("src", "ui_app.py")
        if not os.path.exists(ui_script):
            print(f"{ui_script} not found.")
            raise SystemExit(1)
        # Allow overriding port via env var UI_PORT or via --port flag
        port = _get_arg_value("--port") or os.environ.get("UI_PORT", "8501")
        run(f'"{sys.executable}" -m streamlit run "{ui_script}" --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --server.port {port}')
        return

    # Colab-friendly web launcher (Streamlit + ngrok)
    if "--colab" in args or "colab" in args:
        colab_script = os.path.join("src", "colab_web.py")
        if not os.path.exists(colab_script):
            print(f"{colab_script} not found.")
            raise SystemExit(1)
        # Forward --port if provided
        port = _get_arg_value("--port")
        if port:
            run(f'"{sys.executable}" "{colab_script}" --port {port}')
        else:
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