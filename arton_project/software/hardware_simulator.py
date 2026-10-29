"""Hardware-free ESP32-style simulator for ARTON (4 switches, 4 sensors, 3 signals)."""
from __future__ import annotations
import asyncio
import json
import logging
import random
import time
from typing import Any
import websockets

logger = logging.getLogger(__name__)

class TrackSwitch:
    def __init__(self, switch_id: int):
        self.switch_id, self.position, self.powered = switch_id, "NORMAL", True
        self.last_move_time = time.time()
    def set_position(self, position: str) -> bool:
        if position not in {"NORMAL", "REVERSE"}: return False
        old = self.position; self.position = position; self.last_move_time = time.time()
        logger.info("Switch %s: %s -> %s", self.switch_id, old, position); return True
    def get_status(self) -> dict[str, Any]:
        return {"switch_id": self.switch_id, "position": self.position, "powered": self.powered,
                "last_move": round(time.time()-self.last_move_time, 2)}

class OccupancySensor:
    def __init__(self, sensor_id: int, track_section: str, failure_rate: float = 0.01):
        self.sensor_id, self.track_section = sensor_id, track_section
        self.occupied, self.occupancy_start, self.failure_rate = False, 0.0, failure_rate
    def update_occupancy(self, occupied: bool):
        if bool(occupied) != self.occupied:
            self.occupied = bool(occupied)
            self.occupancy_start = time.time() if self.occupied else 0.0
            logger.info("Sensor %s (%s): occupied=%s", self.sensor_id, self.track_section, self.occupied)
    def get_status(self) -> dict[str, Any]:
        reported = not self.occupied if random.random() < self.failure_rate else self.occupied
        return {"sensor_id": self.sensor_id, "track_section": self.track_section,
                "occupied": reported, "actual_occupied": self.occupied,
                "occupancy_duration": round(time.time()-self.occupancy_start, 2) if self.occupied else 0.0}

class Signal:
    def __init__(self, signal_id: int):
        self.signal_id, self.aspect, self.last_change = signal_id, "GREEN", time.time()
    def set_aspect(self, aspect: str) -> bool:
        if aspect not in {"GREEN", "YELLOW", "RED"}: return False
        old = self.aspect; self.aspect = aspect; self.last_change = time.time()
        logger.info("Signal %s: %s -> %s", self.signal_id, old, aspect); return True
    def get_status(self) -> dict[str, Any]:
        return {"signal_id": self.signal_id, "aspect": self.aspect,
                "time_since_change": round(time.time()-self.last_change, 2)}

class HardwareSimulator:
    def __init__(self, uri: str = "ws://127.0.0.1:8000/ws/hardware", seed: int = 7):
        self.uri, self.websocket, self.running = uri, None, False
        random.seed(seed)
        self.switches = {i: TrackSwitch(i) for i in range(1, 5)}
        self.sensors = {i: OccupancySensor(i, f"SECTION_{chr(64+i)}") for i in range(1, 5)}
        self.signals = {i: Signal(i) for i in range(1, 4)}
        self.command_count, self._start_time = 0, time.time()

    async def connect(self, attempts: int = 30) -> bool:
        for attempt in range(1, attempts+1):
            try:
                self.websocket = await websockets.connect(self.uri, open_timeout=1.5, close_timeout=1.0)
                logger.info("Connected to ARTON WebSocket server: %s", self.uri)
                return True
            except Exception as exc:
                if attempt == attempts:
                    logger.error("Could not connect after %d attempts: %s", attempts, exc)
                    return False
                await asyncio.sleep(0.25)
        return False

    async def disconnect(self):
        if self.websocket:
            try: await self.websocket.close()
            except Exception: pass
            self.websocket = None

    async def handle_command(self, raw: str):
        try: command = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Ignoring malformed command JSON"); return
        self.command_count += 1
        ctype = command.get("type")
        if ctype == "switch_control":
            switch_id = command.get("switch_id"); pos = command.get("position")
            if switch_id in self.switches and self.switches[switch_id].set_position(pos):
                await self.simulate_traffic_flow()
        elif ctype == "signal_control":
            signal_id = command.get("signal_id"); aspect = command.get("aspect")
            if signal_id in self.signals: self.signals[signal_id].set_aspect(aspect)
        elif ctype == "set_occupancy":
            sensor_id = command.get("sensor_id")
            if sensor_id in self.sensors and "occupied" in command:
                self.sensors[sensor_id].update_occupancy(command["occupied"])
        elif ctype not in {"heartbeat", "no-op"}:
            logger.warning("Unknown command type: %s", ctype)

    async def listen_for_commands(self):
        assert self.websocket is not None
        try:
            async for message in self.websocket:
                await self.handle_command(message)
        except asyncio.CancelledError: raise
        except Exception as exc:
            if self.running: logger.warning("Command listener stopped: %s", exc)

    async def simulate_traffic_flow(self):
        # Deterministic-ish state changes keep the visual demo active without hardware.
        for sensor in self.sensors.values():
            if random.random() < 0.22:
                sensor.update_occupancy(not sensor.occupied)

    def telemetry(self) -> dict[str, Any]:
        return {"timestamp": time.time(), "simulator": "ARTON_Hardware_Simulator",
                "switches": [s.get_status() for s in self.switches.values()],
                "sensors": [s.get_status() for s in self.sensors.values()],
                "signals": [s.get_status() for s in self.signals.values()],
                "stats": {"commands_received": self.command_count,
                          "uptime": round(time.time()-self._start_time, 2)}}

    async def broadcast_telemetry(self, interval: float = 0.25):
        while self.running and self.websocket:
            try:
                await self.websocket.send(json.dumps(self.telemetry()))
                # Add natural sensor movement when no route changes occur.
                if random.random() < 0.16:
                    sensor = self.sensors[random.randint(1, 4)]
                    sensor.update_occupancy(not sensor.occupied)
                await asyncio.sleep(interval)
            except asyncio.CancelledError: raise
            except Exception as exc:
                logger.warning("Telemetry sender stopped: %s", exc); break

    async def run_simulation(self):
        self._start_time, self.running = time.time(), True
        logger.info("Starting ARTON software-only hardware simulator (4 switches, 4 sensors, 3 signals)")
        if not await self.connect():
            self.running = False
            raise ConnectionError(f"Unable to connect to {self.uri}; ensure the FastAPI server is running")
        tasks = [asyncio.create_task(self.listen_for_commands()),
                 asyncio.create_task(self.broadcast_telemetry())]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Simulator cancellation requested")
            raise
        finally:
            self.running = False
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.disconnect()

async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sim = HardwareSimulator()
    try: await sim.run_simulation()
    except KeyboardInterrupt: pass
    finally: sim.running = False; await sim.disconnect()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main())
