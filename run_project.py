"""One-command setup and execution of the complete credit-scoring project."""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "cleaned" / "Du_lieu_sach_Chu_de_1.csv"
REQUIREMENTS = ROOT / "requirements.txt"
VENV = ROOT / ".venv"
STEPS = (
    "credit_scoring.analysis",
    "credit_scoring.robustness_supplement",
    "credit_scoring.finalize_outputs",
)


def environment_python():
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run_pipeline(python):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    for module in STEPS:
        print(f"\n=== Running {module} ===", flush=True)
        subprocess.run([str(python), "-m", module], cwd=ROOT, env=env, check=True)
    print("\nComplete. Results are in results/, figures/, data/generated/, and models/.")


def main():
    if sys.version_info[:2] != (3, 13):
        raise SystemExit("Python 3.13 is required; see README.md.")
    if not SOURCE.is_file():
        raise SystemExit(f"Missing cleaned data: {SOURCE}")
    if not REQUIREMENTS.is_file():
        raise SystemExit(f"Missing requirements: {REQUIREMENTS}")
    if sys.argv[1:] == ["--run-in-env"]:
        if Path(sys.executable).resolve() != environment_python().resolve():
            raise SystemExit("The internal run must use this project's .venv")
        run_pipeline(Path(sys.executable))
        return
    if len(sys.argv) != 1:
        raise SystemExit("Usage: python run_project.py")

    python = environment_python()
    if not python.is_file():
        print("Creating the project environment...", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], cwd=ROOT, check=True)
    print("Checking/installing pinned dependencies...", flush=True)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)], cwd=ROOT, check=True)
    subprocess.run([str(python), str(ROOT / "run_project.py"), "--run-in-env"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
