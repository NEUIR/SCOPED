"""Own the extraction server's lifetime; release GPUs before returning."""
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from .engine import Engine


@contextmanager
def expert_server(model, capability, work_dir, config, port, base_url=None, revision=None):
    if base_url:
        engine = Engine(capability, base_url, config, os.environ.get("SCOPED_SKILL_API_KEY", "EMPTY"))
        try:
            engine.check_model(model)
            yield engine
        finally:
            engine.client.close()
        return
    # Do not attach to or shut down someone else's service on an occupied port.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    log = Path(work_dir) / "logs" / f"{capability}_server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
               "--served-model-name", capability, "--host", "127.0.0.1", "--port", str(port),
               "--tensor-parallel-size", str(config["server_tp"]),
               "--max-model-len", str(config["server_max_model_len"]),
               "--gpu-memory-utilization", str(config["server_gpu_memory_utilization"]),
               "--max-num-seqs", str(config["server_max_num_seqs"]),
               "--max-num-batched-tokens", str(config["server_max_num_batched_tokens"]),
               "--seed", str(config["seed"]),
               "--limit-mm-per-prompt", '{"image":' + str(config["server_max_images"]) + '}',
               "--generation-config", "vllm"]
    if revision:
        command.extend(["--revision", revision, "--tokenizer-revision", revision])
    env = os.environ.copy()
    env["NO_PROXY"] = env.get("NO_PROXY", "") + ",localhost,127.0.0.1,::1"
    env["no_proxy"] = env["NO_PROXY"]
    engine = None
    with log.open("w") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            # The Python client also needs the loopback proxy bypass.
            os.environ["NO_PROXY"] = env["NO_PROXY"]
            os.environ["no_proxy"] = env["NO_PROXY"]
            engine = Engine(capability, f"http://127.0.0.1:{port}/v1", config)
            deadline = time.monotonic() + config["server_start_timeout"]
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Expert server exited; inspect {log}")
                # Use a short health probe instead of the long generation timeout.
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        pass
                    engine.check_model(model)
                    break
                except Exception as exc:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"Expert server startup timed out; inspect {log}") from exc
                    time.sleep(1)
            yield engine
        finally:
            if engine is not None:
                engine.client.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
