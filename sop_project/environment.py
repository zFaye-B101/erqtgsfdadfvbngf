"""Read-only environment diagnostics; no package installation or downloads."""
import importlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys


def check_environment():
    report = {
        "python": sys.version,
        "interpreter": sys.executable,
        "system": platform.platform(),
        "packages": {},
    }
    for name in ("numpy", "PIL", "torch", "torchvision", "sklearn", "ipykernel"):
        try:
            if importlib.util.find_spec(name) is None:
                report["packages"][name] = {"status": "not_installed"}
                continue
            module = importlib.import_module(name)
            report["packages"][name] = {
                "status": "import_ok", "version": getattr(module, "__version__", "unknown")
            }
            if name == "torch":
                cuda = module.cuda.is_available()
                report["torch_cuda"] = {"available": cuda, "build_version": module.version.cuda}
                if cuda:
                    device = module.cuda.get_device_properties(0)
                    report["torch_cuda"].update(
                        device=device.name, memory_gb=round(device.total_memory / 1024 ** 3, 2)
                    )
        except Exception as exc:
            report["packages"][name] = {"status": "import_error", "error": str(exc)}
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run(
                [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15, check=True,
            )
            report["nvidia_smi"] = result.stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            report["nvidia_smi"] = "Could not query driver: " + str(exc)
    else:
        report["nvidia_smi"] = "Not on PATH; this alone does not establish whether an NVIDIA GPU exists."
    return report


if __name__ == "__main__":
    print(json.dumps(check_environment(), ensure_ascii=False, indent=2))
