#!/usr/bin/env python3
"""
Govee Device Discovery
======================
Queries your Govee account via the cloud API to list all devices,
their device IDs, and model (SKU) numbers — everything you need for config.yaml.

Usage:
    source ster/bin/activate
    python scripts/discover_govee.py

    # Or pass the key directly:
    python scripts/discover_govee.py --api-key YOUR_KEY_HERE

    # Local LAN scan instead (no API key needed, but requires Local Control enabled):
    python scripts/discover_govee.py --local
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cloud_discovery(api_key: str):
    from core.govee import GoveeCloud

    print(f"\nQuerying Govee cloud API...")
    devices = GoveeCloud.list_devices(api_key)

    if not devices:
        print("\nNo devices returned. Check your API key and that devices are added to your account.")
        return

    print(f"\nFound {len(devices)} device(s):\n")
    print("─" * 50)
    for d in devices:
        print(f"  Name:        {d.get('deviceName', 'Unknown')}")
        print(f"  Device ID:   {d.get('device', '')}")
        print(f"  Model (SKU): {d.get('model', '')}")
        support = d.get('supportCmds', [])
        print(f"  Commands:    {', '.join(support)}")
        print()

    print("─" * 50)
    print("\nAdd to config.yaml:\n")
    print("govee:")
    print(f'  enabled: true')
    print(f'  api_key: "{api_key}"')
    print(f'  devices:')
    for d in devices:
        print(f'    - name: "{d.get("deviceName", "Light")}"')
        print(f'      device_id: "{d.get("device", "")}"')
        print(f'      model: "{d.get("model", "")}"')
    print()


def local_discovery():
    from core.govee import GoveeLocal

    print("\nBroadcasting local LAN discovery (3 seconds)...")
    print("Make sure Local Control is ON in the Govee Home app.\n")

    govee = GoveeLocal(devices=[], discovery_timeout=3.0)
    found = govee.discover()
    govee.close()

    if not found:
        print("No devices found.")
        print("\nTroubleshooting:")
        print("  1. Govee Home app → Device → '...' → More Settings → Local Control → ON")
        print("  2. Mac and device must be on the same WiFi")
        print("  3. Check UDP ports 4001-4003 aren't blocked")
        return

    print(f"Found {len(found)} device(s):\n")
    print("─" * 50)
    for d in found:
        print(f"  Model (SKU): {d.model or 'Unknown'}")
        print(f"  IP:          {d.ip}")
        print(f"  Device ID:   {d.device_id or 'N/A'}")
        print()

    print("─" * 50)
    print("\nAdd to config.yaml (local LAN mode — no api_key field):\n")
    print("govee:")
    print("  enabled: true")
    print("  devices:")
    for d in found:
        print(f'    - name: "{d.model or "Light"}"')
        print(f'      ip: "{d.ip}"')
        if d.device_id:
            print(f'      device_id: "{d.device_id}"')
        if d.model:
            print(f'      model: "{d.model}"')
    print()


def main():
    parser = argparse.ArgumentParser(description="Discover Govee devices")
    parser.add_argument("--api-key", help="Govee cloud API key (overrides config)")
    parser.add_argument("--local", action="store_true", help="Use local LAN discovery instead of cloud API")
    args = parser.parse_args()

    if args.local:
        local_discovery()
        return

    api_key = args.api_key
    if not api_key:
        # Try loading from config.yaml
        try:
            import yaml
            cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            api_key = cfg.get("govee", {}).get("api_key", "").strip()
        except Exception:
            pass

    if not api_key:
        print("No API key found. Provide one with --api-key or set govee.api_key in config.yaml.")
        print("Use --local for local LAN discovery instead.")
        sys.exit(1)

    cloud_discovery(api_key)


if __name__ == "__main__":
    main()
