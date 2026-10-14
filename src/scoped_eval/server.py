"""Manage only the local vLLM process launched by this evaluation run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def api_url(base_url):
    return base_url.rstrip("/").removesuffix("/v1")


def check_server(base_url, model_name, expected_model=None):
    with urllib.request.urlopen(api_url(base_url) + "/v1/models", timeout=5) as response:
        models = json.load(response)["data"]
    entry = next((model for model in models if model["id"] == model_name), None)
    if entry is None:
        raise RuntimeError(f"API server does not serve requested model name {model_name!r}")
    if expected_model and entry.get("root"):
        actual = entry["root"]
        if Path(expected_model).exists():
            matches = Path(actual).resolve() == Path(expected_model).resolve()
        else:
            matches = actual == expected_model
        if not matches:
            raise RuntimeError(f"Existing server serves {actual!r}, not requested model {expected_model!r}; choose another port")


def stop_server(process):
    if process is None or process.poll() is not None:
        return
    import signal
    # Let the API server gracefully shut down its engine workers first.
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def server_command(model, model_name, port, tp, dp, settings):
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
               "--served-model-name", model_name, "--host", "127.0.0.1", "--port", str(port),
               "--tensor-parallel-size", str(tp), "--max-model-len", str(settings["max_model_len"]),
               "--gpu-memory-utilization", str(settings["gpu_memory"]),
               "--max-num-seqs", str(settings["max_num_seqs"]), "--dtype", settings["dtype"],
               "--seed", str(settings["seed"]), "--generation-config", "vllm",
               "--trust-remote-code", "--limit-mm-per-prompt", json.dumps({"image": settings["limit_images"], "video": 0})]
    if settings["max_num_batched_tokens"] is not None:
        command += ["--max-num-batched-tokens", str(settings["max_num_batched_tokens"])]
    if dp > 1:
        command += ["--data-parallel-size", str(dp)]
    return command


def start_server(model, model_name, port, tp, dp, gpus, settings, log_path, startup_timeout):
    base_url = f"http://127.0.0.1:{port}"
    try:
        check_server(base_url, model_name, expected_model=model)
        raise RuntimeError(f"Port {port} already serves this model; use --base-url {base_url} to explicitly reuse its server settings")
    except (OSError, urllib.error.URLError):
        pass
    # A responding but mismatched server fails above rather than being reused.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["VLLM_USE_V1"] = "1"
    command = server_command(model, model_name, port, tp, dp, settings)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM server exited; inspect {log_path}")
            try:
                check_server(base_url, model_name, expected_model=model)
                print(f"vLLM ready: {base_url}", flush=True)
                return process, base_url
            except (OSError, urllib.error.URLError):
                time.sleep(2)
        raise TimeoutError(f"vLLM startup timed out; inspect {log_path}")
    except BaseException:
        stop_server(process)
        raise
