"""Read-only preflight check before running 01/02 on a new machine.

Run this first on the final machine (for example Windows + PyCharm) with the
same interpreter that will run the experiments. It installs nothing, downloads
nothing and does not modify data. The only file operation is a temporary write
test in the results location, which is deleted immediately.

Exit code 0 means no FAIL items were found; WARN items should still be read.
See Readme.txt for the current numbered entry points and environment notes.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = Path(__file__).resolve().parent
REQUIRED = (("numpy", "numpy"), ("pandas", "pandas"), ("PIL", "Pillow"), ("cv2", "OpenCV"),
            ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib"))
OPENCV_DISTRIBUTIONS = ("opencv-python", "opencv-python-headless",
                        "opencv-contrib-python", "opencv-contrib-python-headless")
RESNET18_WEIGHTS = "resnet18-f37072fd.pth"
# Longest path below results_dir measured in a handcrafted_padding_audit smoke run
# was 117 characters (.../SupCon_ContentHSV_HOG192/tau_0.05/inner_2/training_loss.json);
# extended temperatures such as tau_0.0125 add a few more.
LONGEST_RESULT_SUFFIX = 125
WINDOWS_PATH_LIMIT = 260
MINIMUM_FREE_GB_BEFORE_PREPROCESSING = 10.0
MINIMUM_FREE_GB_AFTER_PREPROCESSING = 3.0


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status, item, detail):
        self.rows.append((status, item, detail))
        # ASCII-only output avoids console encoding failures on Windows.
        print(f"[{status:4}] {item}: {detail}".encode("ascii", "replace").decode("ascii"), flush=True)

    def count(self, status):
        return sum(row[0] == status for row in self.rows)


def check_python(report):
    version = sys.version_info
    detail = f"{platform.python_version()} at {sys.executable}"
    if version < (3, 9):
        report.add("FAIL", "Python", detail + " (3.9+ required; 3.11 was used for validation)")
    elif (version.major, version.minor) != (3, 11):
        report.add("WARN", "Python", detail + " (other versions are untested; validation used 3.11)")
    else:
        report.add("PASS", "Python", detail)
    report.add("INFO", "System", f"{platform.platform()}; {os.cpu_count()} logical CPUs")


def check_packages(report):
    for module, label in REQUIRED:
        try:
            imported = importlib.import_module(module)
            report.add("PASS", label, getattr(imported, "__version__", "version unknown"))
        except Exception as exc:  # report every broken import, not only ImportError
            report.add("FAIL", label, f"cannot import {module}: {exc}")
    installed = []
    for name in OPENCV_DISTRIBUTIONS:
        try:
            installed.append(f"{name} {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    if len(installed) > 1:
        report.add("WARN", "OpenCV builds", "several pip OpenCV builds are installed and can "
                   "conflict: " + ", ".join(installed) + ". Keep only one.")


def nvidia_gpu_name():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run([executable, "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=15, check=True)
        return result.stdout.strip().splitlines()[0] if result.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def check_torch(report, settings):
    extractors = {spec["extractor"] for spec in settings["feature_sets"].values()} if settings else set()
    needs_torch_for_features = "resnet18" in extractors
    gpu = nvidia_gpu_name()
    try:
        import torch
        import torchvision
    except Exception as exc:
        status = "FAIL" if needs_torch_for_features else "WARN"
        report.add(status, "PyTorch", f"cannot import torch/torchvision: {exc}. "
                   "Required for --method supcon and ResNet feature profiles.")
        return
    report.add("PASS", "PyTorch", f"torch {torch.__version__}, torchvision {torchvision.__version__}")
    cuda_build = torch.version.cuda
    if torch.cuda.is_available():
        report.add("PASS", "CUDA", f"available on {torch.cuda.get_device_name(0)} (build {cuda_build})")
    elif gpu and cuda_build is None:
        report.add("WARN", "CUDA", f"NVIDIA GPU '{gpu}' found, but this torch build is CPU-only; "
                   "experiments will use the CPU. Install a CUDA build only if GPU use is wanted.")
    elif gpu:
        report.add("WARN", "CUDA", f"NVIDIA GPU '{gpu}' found but CUDA is unavailable to torch "
                   f"(build {cuda_build}); check the driver.")
    else:
        report.add("INFO", "CUDA", "not available; device 'auto' will use the CPU")
    if settings:
        report.add("INFO", "SupCon device", f"configured '{settings['supcon']['device']}', "
                   f"cpu_threads={settings['supcon']['cpu_threads']}")
    if needs_torch_for_features:
        weights = Path(torch.hub.get_dir()) / "checkpoints" / RESNET18_WEIGHTS
        if weights.is_file():
            report.add("PASS", "ResNet18 weights", f"cached at {weights}")
        else:
            report.add("WARN", "ResNet18 weights", "not cached; torchvision downloads about 45 MB "
                       "on the first ResNet run, so internet access is needed once.")


def load_settings(report, config_path):
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from utils.sop_config import load_config
        settings = load_config(config_path)
    except Exception as exc:
        report.add("FAIL", "Configuration", f"{config_path}: {exc}")
        return None
    report.add("PASS", "Configuration", f"{config_path.name}: profile={settings['feature_profile']}, "
               f"run_name={settings['run_name']}, official_test={settings['run_official_test']}")
    return settings


def find_zip(zip_path):
    if zip_path is not None:
        return zip_path if zip_path.is_file() else None
    for folder in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        candidate = folder / "Stanford_Online_Products.zip"
        if candidate.is_file():
            return candidate
    return None


def check_data(report, settings, zip_path):
    """Return True when preprocessing has already completed."""
    data_dir = Path(settings["data_dir"])
    summary_path = data_dir / "audit_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            report.add("FAIL", "Preprocessed data", f"unreadable {summary_path}: {exc}")
            return False
        complete = summary.get("complete", True) is True
        manifest = data_dir / "clean_manifest.csv"
        images = data_dir / "images"
        if complete and manifest.is_file() and images.is_dir():
            report.add("PASS", "Preprocessed data", f"{data_dir}: {summary.get('clean_images')} clean "
                       f"images, {summary.get('clean_products')} products; 01 can be skipped")
            return True
        report.add("FAIL", "Preprocessed data", f"{data_dir} is incomplete (summary/manifest/images). "
                   "Keep it for inspection and point data_dir or --output-dir to a new folder.")
        return False
    if data_dir.is_dir() and any(data_dir.iterdir()):
        report.add("FAIL", "Preprocessed data", f"{data_dir} is nonempty but has no audit_summary.json; "
                   "01 refuses to write there. Use a new output folder.")
        return False
    found = find_zip(zip_path)
    if found:
        report.add("WARN", "Preprocessed data", f"not found at {data_dir}. Run 01 first; ZIP found at {found}")
    elif zip_path is not None:
        report.add("FAIL", "SOP ZIP", f"--zip-path does not exist: {zip_path}")
    else:
        report.add("FAIL", "SOP ZIP", "no preprocessed data and no Stanford_Online_Products.zip beside "
                   "PythonCode or its parent folders. Pass --zip-path to 01, or place the ZIP there.")
    return False


def existing_ancestor(path):
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def check_storage(report, settings, preprocessed):
    results_root = existing_ancestor(settings["results_dir"])
    free_gb = shutil.disk_usage(results_root).free / 1024 ** 3
    needed = MINIMUM_FREE_GB_AFTER_PREPROCESSING if preprocessed else MINIMUM_FREE_GB_BEFORE_PREPROCESSING
    status = "PASS" if free_gb >= needed else "WARN"
    report.add(status, "Free disk space", f"{free_gb:.1f} GB at {results_root} "
               f"(recommended >= {needed:.0f} GB {'for caches/results' if preprocessed else 'for 01 output, caches and results'})")
    try:
        with tempfile.NamedTemporaryFile(dir=results_root, prefix=".write_test_", delete=True):
            pass
        report.add("PASS", "Write access", str(results_root))
    except OSError as exc:
        report.add("FAIL", "Write access", f"cannot write in {results_root}: {exc}")


def windows_long_paths_enabled():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            return winreg.QueryValueEx(key, "LongPathsEnabled")[0] == 1
    except OSError:
        return False


def check_windows(report, settings):
    if os.name != "nt":
        report.add("INFO", "Windows checks", "skipped on this operating system")
        return
    longest = max(len(str(Path(settings[field]))) for field in ("results_dir", "cache_dir", "protocol_dir"))
    estimated = longest + LONGEST_RESULT_SUFFIX
    if estimated < WINDOWS_PATH_LIMIT:
        report.add("PASS", "Path length", f"longest expected output path about {estimated} characters")
    elif windows_long_paths_enabled():
        report.add("PASS", "Path length", f"about {estimated} characters; Windows long paths are enabled")
    else:
        report.add("WARN", "Path length", f"expected output paths reach about {estimated} characters "
                   f"(limit {WINDOWS_PATH_LIMIT}). Move the project to a short folder such as D:\\sml\\.")
    encoding = (sys.stdout.encoding or "unknown").lower()
    if os.environ.get("PYTHONUTF8") == "1" or encoding.replace("-", "") == "utf8":
        report.add("PASS", "Console encoding", encoding)
    else:
        report.add("WARN", "Console encoding", f"{encoding}; set PYTHONUTF8=1 (the shared PyCharm "
                   "run configurations already do) if non-ASCII paths are printed.")


def main():
    parser = argparse.ArgumentParser(description="Read-only environment and data preflight for 01/02.")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument("--zip-path", type=Path, help="SOP ZIP to check when data is not yet preprocessed.")
    args = parser.parse_args()
    report = Report()
    print("SOP preflight check (read-only; installs and downloads nothing)", flush=True)
    check_python(report)
    check_packages(report)
    settings = load_settings(report, args.config.expanduser().resolve())
    check_torch(report, settings)
    if settings:
        preprocessed = check_data(report, settings, args.zip_path.expanduser().resolve() if args.zip_path else None)
        check_storage(report, settings, preprocessed)
        check_windows(report, settings)
    failures, warnings = report.count("FAIL"), report.count("WARN")
    print(f"\nSummary: {failures} FAIL, {warnings} WARN. "
          + ("Fix FAIL items before running 01/02." if failures else "Ready to run the next step."), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
