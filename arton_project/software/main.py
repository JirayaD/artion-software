"""
ARTON - Autonomous Railway Transit Optimization Network
Main entry point for the software system
"""
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from software.graphsage_model import SpatiotemporalGraphSAGE
from software.dqn_agent import DQNRoutingAgent
from software.fastapi_server import app
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARTONSystem:
    def __init__(self, use_simulator=False):
        self.graphsage_model = None
        self.dqn_agent = None
        self.server_task = None
        self.use_simulator = use_simulator
        self.simulator_task = None

    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing ARTON system...")

        # Initialize GraphSAGE model for delay prediction
        self.graphsage_model = SpatiotemporalGraphSAGE(
            input_dim=10,  # Features per node (time, day, etc.)
            hidden_dim=64,
            output_dim=5,  # Delay prediction for next 5 time steps
            num_layers=3
        )

        # Initialize DQN agent for routing optimization
        self.dqn_agent = DQNRoutingAgent(
            state_dim=20,  # Combined graph state + delay predictions
            action_dim=10,  # Possible routing actions
            learning_rate=0.001
        )

        logger.info("ARTON system initialized successfully")

    async def start_server(self):
        """Start the FastAPI WebSocket server"""
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def start_simulator(self):
        """Start the hardware simulator"""
        if self.use_simulator:
            logger.info("Starting hardware simulator...")
            from hardware_simulator import HardwareSimulator
            simulator = HardwareSimulator()
            self.simulator_task = asyncio.create_task(simulator.run_simulation())

    async def run_prediction_loop(self):
        """Main prediction and optimization loop"""
        logger.info("Starting prediction loop...")
        while True:
            try:
                # 1. Get current railway network state (simulated)
                # In real system, this would come from sensors/schedule data
                current_state = await self.get_railway_state()

                # 2. Predict delay propagation using GraphSAGE
                delay_predictions = self.graphsage_model.predict_delay(
                    current_state['graph'],
                    current_state['features']
                )

                # 3. Optimize routing using DQN
                optimal_actions = self.dqn_agent.optimize_routing(
                    current_state,
                    delay_predictions
                )

                # 4. Send commands to hardware via WebSocket
                await self.send_hardware_commands(optimal_actions)

                # Sleep for next iteration (real-time constraint: <100ms)
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(1)

    async def get_railway_state(self):
        """Get current state of railway network (placeholder)"""
        # This would interface with actual railway data/sensors
        # For now, return simulated data
        import torch
        import networkx as nx

        # Create a sample railway graph
        G = nx.grid_graph(dim=[5, 4])  # 5x4 grid representing stations/tracks

        # Generate random node features (time, day, weather, etc.)
        num_nodes = len(G.nodes())
        node_features = torch.randn(num_nodes, 10)

        return {
            'graph': G,
            'features': node_features,
            'timestep': 0
        }

    async def send_hardware_commands(self, actions):
        """Send optimized actions to hardware (or simulator)"""
        # In real system with hardware, this would send commands via WebSocket to ESP32
        # With simulator, commands are received by the simulator through the WebSocket server
        logger.info(f"Sending hardware commands: {actions}")

        # Also log a simplified version for cleaner output
        if isinstance(actions, dict) and 'action_id' in actions:
            logger.info(f"Executing action: {actions.get('description', 'Unknown action')}")

    async def shutdown(self):
        """Shutdown all components gracefully"""
        logger.info("Shutting down ARTON system...")
        if self.server_task:
            self.server_task.cancel()
        if self.simulator_task:
            self.simulator_task.cancel()
        # Save models if needed
        logger.info("ARTON system shutdown complete")

async def main():
    """Main entry point"""
    # Check if simulator mode is requested
    use_simulator = "--simulator" in sys.argv
    if use_simulator:
        logger.info("Running in SIMULATOR mode")
        # Remove simulator flag from argv so it doesn't interfere with other processing
        sys.argv = [arg for arg in sys.argv if arg != "--simulator"]
    else:
        logger.info("Running in HARDWARE mode (expecting actual ESP32 hardware)")

    system = ARTONSystem(use_simulator=use_simulator)

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal...")
        asyncio.create_task(system.shutdown())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize system
        await system.initialize()

        # Start server, simulator (if enabled), and prediction loop concurrently
        tasks = [
            asyncio.create_task(system.start_server()),
            asyncio.create_task(system.run_prediction_loop())
        ]

        if system.use_simulator:
            tasks.append(asyncio.create_task(system.start_simulator()))

        # Wait for either to complete
        await asyncio.gather(*tasks)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
