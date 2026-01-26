import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

# Load config directly to avoid triggering heavy package initialization (which might require R)
config_path = Path(__file__).resolve().parent.parent / "src/ttstl/config.py"
spec = importlib.util.spec_from_file_location("ttstl.config", config_path)
ttstl_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ttstl_config)

DATA_DIR = ttstl_config.DATA_DIR
MANIFEST_PATH = ttstl_config.MANIFEST_PATH
DATASETS = ttstl_config.DATASETS


def calculate_sha256(filepath):
    """Calculates the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def validate_data(dataset_filter):
    """Validates data files against the manifest."""
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    manifest_datasets = manifest.keys()
    files_to_check = {}

    for dataset_name in manifest_datasets:
        # Check if we should process this dataset
        if dataset_filter != "all" and dataset_filter != dataset_name:
            continue

        if dataset_name not in DATASETS:
            print(
                f"Warning: Dataset '{dataset_name}' found in manifest but not configured in DATASETS. Skipping."
            )
            continue

        dataset_dir = DATASETS[dataset_name]
        dataset_files = manifest[dataset_name]

        for manifest_path, expected_hash in dataset_files.items():
            # The manifest paths might be different from our local structure.
            # We assume the filenames match and are located in the configured dataset directory.
            filename = Path(manifest_path).name
            full_path = dataset_dir / filename
            files_to_check[full_path] = expected_hash

    if not files_to_check:
        print(f"No files found in manifest matching dataset '{dataset_filter}'.")
        return

    print(f"Verifying {len(files_to_check)} files for dataset: {dataset_filter}...")

    corrupted_files = []
    missing_files = []

    for full_path, expected_hash in tqdm(files_to_check.items(), unit="file"):
        if not full_path.exists():
            missing_files.append(full_path.name)
            continue

        computed_hash = calculate_sha256(full_path)
        if computed_hash != expected_hash:
            corrupted_files.append(full_path.name)

    # Report results
    if not corrupted_files and not missing_files:
        print("\nVerification Successful: All files matched the manifest.")
    else:
        print("\nVerification Completed with Issues:")
        if missing_files:
            print(f"  - {len(missing_files)} files missing.")
        if corrupted_files:
            print(f"  - {len(corrupted_files)} files corrupted or mismatched.")
            print("\nCorruption Detected.")
            print(
                "Note: This is expected if you have generated fresh synthetic data, as it will differ from the official manifest."
            )

        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Validate downloaded data against the official manifest."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "synthetic", "intc"],
        default="all",
        help="Which dataset to validate. Defaults to 'all'.",
    )

    args = parser.parse_args()
    validate_data(args.dataset)


if __name__ == "__main__":
    main()
