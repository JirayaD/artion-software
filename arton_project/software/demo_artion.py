"""
ARTION System Demonstration Script
Shows the complete system working with hardware simulator
"""
import asyncio
import subprocess
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_demo():
    """Run a complete demonstration of the ARTION system"""
    print("=" * 60)
    print("🚂 ARTION SYSTEM DEMONSTRATION 🚂")
    print("Autonomous Railway Transit Optimization Network")
    print("=" * 60)
    print()
    print("This demo shows:")
    print("1. Hardware Simulator (mimicking ESP32)")
    print("2. ARTION Software (GraphSAGE + DQN + WebSocket)")
    print("3. Real-time command and telemetry exchange")
    print()

    # Start hardware simulator in background
    print("🔧 Starting Hardware Simulator...")
    simulator_proc = subprocess.Popen([
        sys.executable, "hardware_simulator.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Give simulator time to start
    await asyncio.sleep(2)

    # Start ARTION system with simulator mode
    print("🚂 Starting ARTION System (with simulator)...")
    artion_proc = subprocess.Popen([
        sys.executable, "main.py", "--simulator"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    print()
    print("✅ Both systems started! Watching for interaction...")
    print("✅ Look for:")
    print("   - Hardware simulator connecting to WebSocket server")
    print("   - ARTION sending optimization commands")
    print("   - Hardware simulator responding with telemetry")
    print("   - Dynamic track switch and signal updates")
    print()
    print("Press Ctrl+C to stop the demo")
    print("-" * 60)

    try:
        # Let the systems run and interact for 30 seconds
        await asyncio.sleep(30)

    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal...")

    finally:
        # Clean up processes
        print("\n🧹 Shutting down systems...")

        simulator_proc.terminate()
        artion_proc.terminate()

        # Wait for graceful shutdown
        try:
            simulator_proc.wait(timeout=3)
            artion_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            simulator_proc.kill()
            artion_proc.kill()

        # Show final output
        print("\n📋 Hardware Simulator Output:")
        sim_out, sim_err = simulator_proc.communicate()
        if sim_out:
            print(sim_out[-500:])  # Last 500 chars
        if sim_err:
            print("SIM ERROR:", sim_err[-200:])

        print("\n📋 ARTION System Output:")
        art_out, art_err = artion_proc.communicate()
        if art_out:
            print(art_out[-500:])  # Last 500 chars
        if art_err:
            print("ARTION ERROR:", art_err[-200:])

        print("\n✅ Demo completed!")

if __name__ == "__main__":
    asyncio.run(run_demo())