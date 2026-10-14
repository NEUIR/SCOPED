#!/usr/bin/env python3
import argparse
from pathlib import Path
from huggingface_hub.errors import HfHubHTTPError
from scoped_data import CAPABILITIES, prepare
from storage import data_root


def main():
    parser = argparse.ArgumentParser(description="Prepare the pinned HF dataset and verify local shards")
    parser.add_argument("--repo-id")
    parser.add_argument("--revision")
    parser.add_argument("--capability", choices=(*CAPABILITIES, "all"), default="all")
    parser.add_argument("--output", type=Path, default=data_root())
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        prepare(args.output, CAPABILITIES if args.capability == "all" else (args.capability,), args.repo_id, args.revision, args.offline)
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code in (401, 403, 404):
            raise SystemExit("Private HF dataset: set HF_TOKEN or run `hf auth login` with an authorized account.") from None
        raise


if __name__ == "__main__":
    main()
