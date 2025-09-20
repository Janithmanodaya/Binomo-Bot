import os
import sys
import subprocess


def run(cmd, check=True):
    print(f"+ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    # 1) Install requirements using the current interpreter
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print("requirements.txt not found.")
    else:
        run(f'"{sys.executable}" -m pip install -r "{req_file}"')

    # If user requested the UI, launch Streamlit
    if "--ui" in sys.argv or "ui" in sys.argv:
        ui_script = os.path.join("src", "ui_app.py")
        if not os.path.exists(ui_script):
            print(f"{ui_script} not found.")
            raise SystemExit(1)
        # Allow overriding port via env var UI_PORT
        port = os.environ.get("UI_PORT", "8501")
        run(f'"{sys.executable}" -m streamlit run "{ui_script}" --server.port {port}')
        return

    # 2) Forward all extra args to the pipeline
    extra_args = [a for a in sys.argv[1:] if a not in {"--ui", "ui"}]
    extra = " ".join([f'"{a}"' if " " in a else a for a in extra_args])
    script = os.path.join("src", "run_pipeline.py")
    if not os.path.exists(script):
        print(f"{script} not found.")
        raise SystemExit(1)

    run(f'"{sys.executable}" "{script}" {extra}'.strip())


if __name__ == "__main__":
    main()