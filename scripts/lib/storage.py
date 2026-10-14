import os
from pathlib import Path


def storage_root():
    if os.environ.get("SCOPED_STORAGE"):
        return Path(os.environ["SCOPED_STORAGE"]).expanduser().resolve()
    if Path("/data").is_dir() and os.access("/data", os.W_OK):
        return Path("/data/SCOPED")
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "scoped"


def data_root():
    return Path(os.environ.get("DATA_DIR", storage_root() / "data/hf_dataset"))


def output_root():
    return Path(os.environ.get("OUTPUT_DIR", storage_root() / "outputs"))
