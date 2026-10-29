"""Run a single self-contained ARTON demo (server + simulator + AI loop).

Unlike the previous script, this does not start a second simulator process, which
caused duplicate WebSocket clients and ambiguous command/telemetry behavior.
"""
import argparse
import asyncio
import logging
from main import run_system

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Run ARTON hardware-free demo")
    parser.add_argument("--duration", type=float, default=30.0, help="Run duration in seconds")
    parser.add_argument("--hardware", action="store_true", help="Start server and policy loop without built-in simulator")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("\nARTON — AUTONOMOUS RAILWAY TRANSIT OPTIMIZATION NETWORK")
    print("Dashboard: http://127.0.0.1:8000/dashboard")
    print("Mode:", "hardware bridge (no simulator)" if args.hardware else "software-only simulator")
    print(f"Duration: {args.duration:.1f} seconds\n")
    asyncio.run(run_system(use_simulator=not args.hardware, duration=args.duration))
    print("\nDemo stopped cleanly.")
