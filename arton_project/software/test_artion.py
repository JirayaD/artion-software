"""Fast, deterministic component and integration tests for ARTON."""
import asyncio
import sys
from pathlib import Path
import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from software.graphsage_model import SpatiotemporalGraphSAGE, HAS_TORCH_GEOMETRIC
from software.dqn_agent import DQNRoutingAgent
from software.hardware_simulator import HardwareSimulator

def test_graphsage_model():
    model = SpatiotemporalGraphSAGE(input_dim=10, hidden_dim=32, output_dim=5, num_layers=2)
    graph = nx.grid_graph(dim=[4, 3])
    features = torch.randn(len(graph.nodes()), 10)
    predictions = model.predict_delay(graph, features)
    assert predictions.shape == (len(graph.nodes()), 5), predictions.shape
    assert np.isfinite(predictions).all()
    print(f"PASS GraphSAGE output={predictions.shape} backend={'PyG' if HAS_TORCH_GEOMETRIC else 'pure-PyTorch fallback'}")

def test_dqn_agent():
    agent = DQNRoutingAgent(state_dim=20, action_dim=10, epsilon_start=0.0, batch_size=4)
    state = np.zeros(20, dtype=np.float32)
    action = agent.act(state)
    assert 0 <= action < 10
    # Exercise replay training path and catch shape / missing-import errors.
    for i in range(4): agent.remember(state, i % 10, float(i), state + 0.1, i == 3)
    loss = agent.replay()
    assert loss is not None and np.isfinite(loss)
    command = agent._action_to_commands(4)
    assert command["commands"][0] == {"type": "switch_control", "switch_id": 1, "position": "REVERSE"}
    print(f"PASS DQN action={action}, replay_loss={loss:.4f}, action mapping={command['description']}")

def test_integration():
    graph = nx.grid_graph(dim=[5, 4])
    features = torch.randn(len(graph.nodes()), 10)
    model = SpatiotemporalGraphSAGE(input_dim=10, hidden_dim=32, output_dim=5, num_layers=2)
    predictions = model.predict_delay(graph, features)
    agent = DQNRoutingAgent(state_dim=20, action_dim=10, epsilon_start=0.2)
    action = agent.optimize_routing({"graph": graph, "features": features}, predictions)
    assert predictions.shape == (20, 5)
    assert 0 <= action["action_id"] < 10
    assert action["commands"] and action["commands"][0]["type"] in {"switch_control", "signal_control"}
    print(f"PASS integration nodes={len(graph.nodes())}, edges={len(graph.edges())}, pred={predictions.shape}, command={action['commands'][0]}")

def test_hardware_simulator_controls():
    sim = HardwareSimulator(seed=1)
    async def check():
        await sim.handle_command('{"type":"switch_control","switch_id":2,"position":"REVERSE"}')
        await sim.handle_command('{"type":"signal_control","signal_id":1,"aspect":"RED"}')
    asyncio.run(check())
    assert sim.switches[2].position == "REVERSE"
    assert sim.signals[1].aspect == "RED"
    telem = sim.telemetry()
    assert len(telem["switches"]) == 4 and len(telem["sensors"]) == 4 and len(telem["signals"]) == 3
    print("PASS simulator control + telemetry (4 switches, 4 sensors, 3 signals)")

def main():
    tests = [test_graphsage_model, test_dqn_agent, test_integration, test_hardware_simulator_controls]
    for test in tests: test()
    print(f"\nAll {len(tests)} ARTON tests passed.")

if __name__ == "__main__": main()
