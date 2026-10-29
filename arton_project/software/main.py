"""ARTON coordinator. Run `python main.py --simulator` for a no-hardware demo."""
from __future__ import annotations
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
import networkx as nx
import numpy as np
import torch

# Small graph inference is faster and more predictable with fewer CPU worker threads.
# This also avoids OpenMP oversubscription on laptops used for project demos.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # Another library may already have initialized inter-op workers.

# Support both `cd arton_project/software && python main.py` and repo-root invocation.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from software.graphsage_model import SpatiotemporalGraphSAGE, extract_spatiotemporal_features
from software.dqn_agent import DQNRoutingAgent
from software.fastapi_server import app, manager, add_event, set_runtime_state
from software.hardware_simulator import HardwareSimulator
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("arton.main")

class ARTONSystem:
    def __init__(self, use_simulator: bool = False, interval: float = 0.35):
        self.graphsage_model = None
        self.dqn_agent = None
        self.use_simulator = use_simulator
        self.interval = interval
        self.server = None
        self.simulator = None
        self.timestep = 0
        self.running = True
        self.graph = nx.grid_graph(dim=[5, 4])
        self.rng = np.random.default_rng(2026)

    async def initialize(self):
        self.graphsage_model = SpatiotemporalGraphSAGE(input_dim=10, hidden_dim=64, output_dim=5, num_layers=3)
        # Exploration remains deliberately low for readable demo behavior; weights are not pretrained.
        self.dqn_agent = DQNRoutingAgent(state_dim=20, action_dim=10, learning_rate=0.001, epsilon_start=0.20)
        set_runtime_state(mode="SIMULATOR" if self.use_simulator else "HARDWARE BRIDGE",
                          model_note="Synthetic features · randomly initialized model weights")
        add_event("GraphSAGE and DQN components initialized", "OK")
        logger.info("Initialized GraphSAGE (10→64→5) and DQN (state=20, actions=10)")

    async def start_server(self):
        config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning", access_log=False)
        self.server = uvicorn.Server(config)
        await self.server.serve()

    async def wait_for_server(self, timeout: float = 12.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.server and self.server.started:
                return
            await asyncio.sleep(0.1)
        raise TimeoutError("FastAPI server did not start on port 8000")

    async def get_railway_state(self):
        # A synthetic 5x4 network: safe for demo; replace with timetable and telemetry data later.
        features = extract_spatiotemporal_features(current_time=self.timestep, num_nodes=len(self.graph.nodes()),
                                                   seed=2026 + self.timestep)
        hw_sensors = manager.hardware_status.get("sensors", [])
        occupied = sum(1 for s in hw_sensors if s.get("occupied"))
        # Include occupancy as a weak synthetic feature; other fields remain illustrative inputs.
        if len(features): features[:, 5] = float(occupied) / max(len(hw_sensors), 1) if hw_sensors else 0.0
        return {"graph": self.graph, "features": features, "timestep": self.timestep,
                "occupied_sections": occupied, "source": "synthetic simulator"}

    async def send_hardware_commands(self, actions: dict):
        commands = actions.get("commands", [])
        set_runtime_state(last_action=actions)
        add_event(f"DQN action {actions.get('action_id')}: {actions.get('description', 'routing action')}", "INFO")
        for command in commands:
            sent = await manager.broadcast(__import__("json").dumps(command))
            add_event(f"{command.get('type')} broadcast to {sent} simulator connection(s)", "OK" if sent else "WARN")
        logger.info("Action %s | %s | broadcast=%d", actions.get("action_id"), actions.get("description"), len(manager.active_connections))

    async def run_prediction_loop(self):
        assert self.graphsage_model is not None and self.dqn_agent is not None
        while self.running:
            start = time.perf_counter()
            try:
                railway_state = await self.get_railway_state()
                predictions = self.graphsage_model.predict_delay(railway_state["graph"], railway_state["features"])
                action = self.dqn_agent.optimize_routing(railway_state, predictions)
                cycle_ms = (time.perf_counter() - start) * 1000.0
                self.timestep += 1
                set_runtime_state(loop_count=self.timestep, last_cycle_ms=cycle_ms,
                                  prediction_summary={"mean": float(np.mean(predictions)), "min": float(np.min(predictions)),
                                                      "max": float(np.max(predictions)), "nodes": int(predictions.shape[0]),
                                                      "horizon": int(predictions.shape[1])})
                await self.send_hardware_commands(action)
                logger.info("Loop %d complete in %.1f ms | prediction shape=%s", self.timestep, cycle_ms, predictions.shape)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Prediction loop error: %s", exc)
                add_event(f"Prediction loop error: {type(exc).__name__}", "WARN")
            await asyncio.sleep(self.interval)

    async def shutdown(self):
        self.running = False
        if self.server: self.server.should_exit = True

async def run_system(use_simulator: bool = False, duration: float | None = None):
    system = ARTONSystem(use_simulator=use_simulator)
    await system.initialize()
    server_task = asyncio.create_task(system.start_server(), name="arton-fastapi")
    try:
        await system.wait_for_server()
        add_event("FastAPI WebSocket server is listening on 127.0.0.1:8000", "OK")
        tasks = [asyncio.create_task(system.run_prediction_loop(), name="arton-prediction-loop")]
        if use_simulator:
            system.simulator = HardwareSimulator()
            tasks.append(asyncio.create_task(system.simulator.run_simulation(), name="arton-hardware-simulator"))
            add_event("No-hardware simulator mode enabled", "OK")
        if duration is not None:
            await asyncio.sleep(duration)
            system.running = False
            for task in tasks: task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            await asyncio.gather(*tasks)
    finally:
        await system.shutdown()
        if system.simulator:
            system.simulator.running = False
            await system.simulator.disconnect()
        # Ask Uvicorn to exit through its normal lifespan shutdown path first.
        if not server_task.done():
            try:
                await asyncio.wait_for(server_task, timeout=5.0)
            except asyncio.TimeoutError:
                server_task.cancel()
                await asyncio.gather(server_task, return_exceptions=True)

async def main():
    use_simulator = "--simulator" in sys.argv
    duration = None
    if "--duration" in sys.argv:
        idx = sys.argv.index("--duration")
        try: duration = float(sys.argv[idx + 1])
        except (IndexError, ValueError): raise SystemExit("Usage: python main.py --simulator [--duration seconds]")
    logger.info("Starting ARTON in %s mode", "SIMULATOR" if use_simulator else "HARDWARE BRIDGE")
    try: await run_system(use_simulator=use_simulator, duration=duration)
    except KeyboardInterrupt: logger.info("Shutdown requested")

if __name__ == "__main__":
    asyncio.run(main())
