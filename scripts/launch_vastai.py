#!/usr/bin/env python3
"""VAST AI instance launcher for infinigen vispos rendering.
Uses the VAST AI API directly with the user's API key.
"""
import json
import subprocess
import sys
import time

import os
VASTAI_KEY = os.environ.get("VASTAI_API_KEY", "")
REPO_URL = "https://github.com/vlordier/infinigen.git"
SETUP_SCRIPT = "scripts/vastai_deploy.sh"

HEADERS = {
    "Authorization": f"Bearer {VASTAI_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}


def api(method, path, data=None):
    cmd = ["curl", "-s", "-X", method, f"https://console.vast.ai/api/v0{path}"]
    for k, v in HEADERS.items():
        cmd += ["-H", f"{k}: {v}"]
    if data:
        cmd += ["-d", json.dumps(data)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout) if result.stdout else {}


def find_gpu_instance():
    """Find an available RTX 4090 or A6000 instance under $0.60/hr."""
    print("Searching for GPU instances...")
    offers = api("GET", "/bundles?q%5Bgpu_name%5D=RTX+4090&q%5Btype%5D=on-demand&q%5Brentable%5D=true&q%5Bmin_gpu_ram%5D=20&q%5Blimit%5D=5&q%5Border%5D=lowest_price")
    
    if isinstance(offers, dict) and "offers" in offers:
        for offer in offers["offers"]:
            price = offer.get("dph_total", 999)
            if price < 0.60:
                print(f"  Found: {offer['id']} — {offer.get('gpu_name','?')} at ${price:.3f}/hr ({offer.get('gpu_ram','?')}GB VRAM)")
                return offer
    print("  No suitable instances found")
    return None


def launch_instance(offer):
    """Launch an instance with the setup script."""
    print(f"Launching instance {offer['id']}...")
    
    launch_data = {
        "client_id": "me",
        "image": "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        "disk": 50,
        "runtype": "ssh",
        "env": {
            "TZ": "UTC",
        },
        "onstart": f"""#!/bin/bash
apt-get update && apt-get install -y git awscli wget python3-pip
cd /workspace
git clone {REPO_URL} -b main
cd infinigen
bash {SETUP_SCRIPT}
""",
        "price": offer["dph_total"],
    }
    
    result = api("PUT", f"/asks/{offer['id']}/", launch_data)
    if isinstance(result, dict):
        instance_id = result.get("new_contract") or result.get("contract_id") or result.get("id")
        print(f"  Instance launched: {instance_id}")
        return instance_id
    else:
        print(f"  Launch result: {result}")
        return None


def main():
    offer = find_gpu_instance()
    if not offer:
        print("No suitable GPU found. Check available instances at https://console.vast.ai/")
        return
    
    instance_id = launch_instance(offer)
    if instance_id:
        print(f"\nInstance {instance_id} launched!")
        print(f"Monitor at: https://console.vast.ai/instances/")
        print(f"Output will appear in S3 bucket after completion (~10-15 minutes)")


if __name__ == "__main__":
    main()
