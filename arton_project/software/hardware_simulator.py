"""
Hardware Simulator for ARTION System
Mimics ESP32 hardware for demonstration purposes
"""
import asyncio
import json
import logging
import random
import time
import websockets
from enum import Enum
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrackSwitch:
    """Simulates a track switch (points)"""
    def __init__(self, switch_id: int):
        self.switch_id = switch_id
        self.position = "NORMAL"  # NORMAL or REVERSE
        self.powered = True
        self.last_move_time = 0

    def set_position(self, position: str):
        """Set switch position"""
        if position in ["NORMAL", "REVERSE"]:
            old_position = self.position
            self.position = position
            self.last_move_time = time.time()
            logger.info(f"Switch {self.switch_id}: {old_position} -> {position}")
            return True
        return False

    def get_status(self) -> dict:
        """Get current switch status"""
        return {
            "switch_id": self.switch_id,
            "position": self.position,
            "powered": self.powered,
            "last_move": time.time() - self.last_move_time
        }

class OccupancySensor:
    """Simulates a track occupancy sensor"""
    def __init__(self, sensor_id: int, track_section: str):
        self.sensor_id = sensor_id
        self.track_section = track_section
        self.occupied = False
        self.occupancy_start = 0
        self.failure_rate = 0.01  # 1% chance of false reading

    def update_occupancy(self, occupied: bool):
        """Update occupancy state"""
        if occupied != self.occupied:
            old_state = self.occupied
            self.occupied = occupied
            if occupied:
                self.occupancy_start = time.time()
            logger.info(f"Sensor {self.sensor_id} ({self.track_section}): {old_state} -> {occupied}")

    def get_status(self) -> dict:
        """Get sensor status with occasional noise"""
        # Simulate occasional sensor noise
        actual_occupied = self.occupied
        if random.random() < self.failure_rate:
            reported_occupied = not self.occupied
        else:
            reported_occupied = self.occupied

        return {
            "sensor_id": self.sensor_id,
            "track_section": self.track_section,
            "occupied": reported_occupied,
            "actual_occupied": self.occupied,  # For debugging
            "occupancy_duration": time.time() - self.occupancy_start if self.occupied else 0
        }

class Signal:
    """Simulates a railway signal"""
    def __init__(self, signal_id: int):
        self.signal_id = signal_id
        self.aspect = "GREEN"  # GREEN, YELLOW, RED
        self.last_change = time.time()

    def set_aspect(self, aspect: str):
        """Set signal aspect"""
        if aspect in ["GREEN", "YELLOW", "RED"]:
            old_aspect = self.aspect
            self.aspect = aspect
            self.last_change = time.time()
            logger.info(f"Signal {self.signal_id}: {old_aspect} -> {aspect}")
            return True
        return False

    def get_status(self) -> dict:
        """Get signal status"""
        return {
            "signal_id": self.signal_id,
            "aspect": self.aspect,
            "time_since_change": time.time() - self.last_change
        }

class HardwareSimulator:
    """Main hardware simulator coordinating all simulated components"""
    def __init__(self, uri: str = "ws://localhost:8000/ws/hardware"):
        self.uri = uri
        self.websocket = None

        # Initialize simulated hardware
        self.switches = {i: TrackSwitch(i) for i in range(1, 5)}  # 4 switches
        self.sensors = {
            1: OccupancySensor(1, "SECTION_A"),
            2: OccupancySensor(2, "SECTION_B"),
            3: OccupancySensor(3, "SECTION_C"),
            4: OccupancySensor(4, "SECTION_D")
        }
        self.signals = {i: Signal(i) for i in range(1, 4)}  # 3 signals

        # Simulation state
        self.running = False
        self.command_count = 0

    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.uri)
            logger.info(f"Connected to hardware server at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.uri}: {e}")
            return False

    async def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info("Disconnected from hardware server")

    async def listen_for_commands(self):
        """Listen for commands from the software"""
        try:
            async for message in self.websocket:
                await self.handle_command(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error listening for commands: {e}")

    async def handle_command(self, message: str):
        """Handle incoming command from software"""
        try:
            command = json.loads(message)
            self.command_count += 1

            logger.info(f"Received command #{self.command_count}: {command}")

            # Process different command types
            if command.get("type") == "switch_control":
                await self.handle_switch_command(command)
            elif command.get("type") == "signal_control":
                await self.handle_signal_command(command)
            elif command.get("type") == "set_occupancy":
                await self.handle_occupancy_command(command)
            elif command.get("type") == "heartbeat":
                # Just acknowledge heartbeat
                pass
            else:
                logger.warning(f"Unknown command type: {command.get('type')}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message}")
        except Exception as e:
            logger.error(f"Error handling command: {e}")

    async def handle_switch_command(self, command: dict):
        """Handle track switch commands"""
        switch_id = command.get("switch_id")
        position = command.get("position")

        if switch_id in self.switches and position:
            success = self.switches[switch_id].set_position(position)
            if success:
                # Simulate some random occupancy changes based on switch position
                await self.simulate_traffic_flow()

    async def handle_signal_command(self, command: dict):
        """Handle signal commands"""
        signal_id = command.get("signal_id")
        aspect = command.get("aspect")

        if signal_id in self.signals and aspect:
            self.signals[signal_id].set_aspect(aspect)

    async def handle_occupancy_command(self, command: dict):
        """Handle direct occupancy setting (for testing)"""
        sensor_id = command.get("sensor_id")
        occupied = command.get("occupied")

        if sensor_id in self.sensors and occupied is not None:
            self.sensors[sensor_id].update_occupancy(occupied)

    async def simulate_traffic_flow(self):
        """Simulate realistic train traffic based on switch positions"""
        # Simple simulation: trains flow based on switch alignments
        for sensor_id, sensor in self.sensors.items():
            # Random chance of train entering/leaving sections
            if random.random() < 0.1:  # 10% chance per cycle
                new_state = not sensor.occupied
                sensor.update_occupancy(new_state)

                # Possibly affect adjacent sections
                if sensor_id < len(self.sensors) and random.random() < 0.3:
                    adjacent_id = sensor_id + 1
                    if adjacent_id in self.sensors:
                        adjacent_sensor = self.sensors[adjacent_id]
                        # Sometimes trains propagate to next section
                        if random.random() < 0.4:
                            adjacent_sensor.update_occupancy(not adjacent_sensor.occupied)

    async def broadcast_telemetry(self):
        """Periodically broadcast hardware telemetry"""
        while self.running:
            try:
                # Collect status from all components
                telemetry = {
                    "timestamp": time.time(),
                    "simulator": "ARTION_Hardware_Simulator",
                    "switches": [sw.get_status() for sw in self.switches.values()],
                    "sensors": [sen.get_status() for sen in self.sensors.values()],
                    "signals": [sig.get_status() for sig in self.signals.values()],
                    "stats": {
                        "commands_received": self.command_count,
                        "uptime": time.time() - getattr(self, '_start_time', time.time())
                    }
                }

                # Send telemetry to software
                if self.websocket:
                    await self.websocket.send(json.dumps(telemetry))
                    logger.debug(f"Sent telemetry: {len(telemetry['switches'])} switches, "
                               f"{len(telemetry['sensors'])} sensors, {len(telemetry['signals'])} signals")

                # Wait before next transmission (10Hz update rate for demo)
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error broadcasting telemetry: {e}")
                await asyncio.sleep(1)

    async def run_simulation(self):
        """Main simulation loop"""
        self._start_time = time.time()
        self.running = True

        logger.info("Starting ARTION Hardware Simulator...")
        logger.info("Will connect to WebSocket server and simulate ESP32 hardware")

        # Connect to server
        if not await self.connect():
            logger.error("Failed to connect to hardware server. Exiting.")
            return

        try:
            # Start listening for commands and broadcasting telemetry concurrently
            listen_task = asyncio.create_task(self.listen_for_commands())
            telemetry_task = asyncio.create_task(self.broadcast_telemetry())

            # Wait for either to finish
            await asyncio.gather(listen_task, telemetry_task)

        except asyncio.CancelledError:
            logger.info("Simulation cancelled")
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        finally:
            self.running = False
            await self.disconnect()

def print_banner():
    """Print demonstration banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║        🚂 ARTION Hardware Simulator - Demonstration Mode 🚂                ║
    ║                                                                              ║
    ║  This simulator mimics ESP32 hardware for the ARTION system:               ║
    ║  • 4 Track Switches (Points)                                               ║
    ║  • 4 Occupancy Sensors                                                     ║
    ║  • 3 Railway Signals                                                       ║
    ║                                                                              ║
    ║  Watch for dynamic updates as software sends commands!                     ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def main():
    """Main entry point"""
    print_banner()

    simulator = HardwareSimulator()

    try:
        await simulator.run_simulation()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        simulator.running = False
        await simulator.disconnect()
        logger.info("Hardware simulator stopped")

if __name__ == "__main__":
    asyncio.run(main())