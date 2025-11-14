#!/usr/bin/env python3
"""
Capsule creation script for CapsuleOS game evolution.
Creates CBOR-serialized λgame capsules with CID generation.
"""
import argparse
import json
import hashlib
import cbor2
import os
import sys
from pathlib import Path
import requests
def generate_cid(data: bytes) -> str:
    """Generate CID for capsule data."""
    hash_digest = hashlib.sha256(data).digest()
    # CID format: CIDv1 with dag-cbor codec
    cid_version = 1
    codec = 0x71  # dag-cbor
    hash_function = 0x12  # SHA256
    hash_length = 32
    cid_bytes = bytes([cid_version]) + bytes([codec]) + bytes([hash_function]) + bytes([hash_length]) + hash_digest
    import base64
    cid_base32 = base64.b32encode(cid_bytes).decode('utf-8').rstrip('=')
    return f"bafy{cid_base32}"
def create_capsule(input_dir: str, output_file: str, genesis_api: str = None):
    """Create game capsule from build artifacts."""
    input_path = Path(input_dir)
    # Gather all relevant files
    capsule_data = {
        "version": "1.0",
        "type": "lambda-game",
        "timestamp": int(os.environ.get("CI_JOB_STARTED_AT", "0")),
        "commit": os.environ.get("GITHUB_SHA", "unknown"),
        "artifacts": {}
    }
    # Include JS files
    for js_file in input_path.glob("**/*.js"):
        with open(js_file, 'rb') as f:
            js_content = f.read()
            capsule_data["artifacts"][str(js_file.relative_to(input_path))] = {
                "type": "javascript",
                "hash": hashlib.sha256(js_content).hexdigest(),
                "size": len(js_content)
            }
    # Include WASM files
    wasm_dir = Path("pkg/wasm")
    if wasm_dir.exists():
        for wasm_file in wasm_dir.glob("**/*.wasm"):
            with open(wasm_file, 'rb') as f:
                wasm_content = f.read()
                capsule_data["artifacts"][f"wasm/{wasm_file.name}"] = {
                    "type": "wasm",
                    "hash": hashlib.sha256(wasm_content).hexdigest(),
                    "size": len(wasm_content)
                }
    # Serialize as CBOR
    cbor_data = cbor2.dumps(capsule_data)
    # Generate CID
    cid = generate_cid(cbor_data)
    capsule_data["cid"] = cid
    # Write capsule metadata
    with open(output_file, 'w') as f:
        json.dump(capsule_data, f, indent=2)
    # Write CBOR capsule
    cbor_file = output_file.replace('.json', '.cbor')
    with open(cbor_file, 'wb') as f:
        f.write(cbor_data)
    print(f"🎮 Capsule created: {cid}")
    print(f"📄 Metadata: {output_file}")
    print(f"📦 CBOR data: {cbor_file}")
    # Register with Genesis API if provided
    if genesis_api:
        try:
            response = requests.post(
                f"{genesis_api}/genesis/capsules",
                json=capsule_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            print(f"✅ Capsule registered in Genesis Graph")
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Failed to register capsule: {e}")
    return cid
def main():
    parser = argparse.ArgumentParser(description="Create CapsuleOS game capsule")
    parser.add_argument("--input", required=True, help="Input directory with build artifacts")
    parser.add_argument("--output", required=True, help="Output metadata file (JSON)")
    parser.add_argument("--genesis-api", help="Genesis API URL for registration")
    parser.add_argument("--wasm", help="WASM directory to include")
    args = parser.parse_args()
    create_capsule(args.input, args.output, args.genesis_api)
if __name__ == "__main__":
    main()
