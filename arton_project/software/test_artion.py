"""
Test script for ARTON system components
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from graphsage_model import SpatiotemporalGraphSAGE
    from dqn_agent import DQNRoutingAgent
except ImportError:
    # Fallback for when run from parent directory
    from software.graphsage_model import SpatiotemporalGraphSAGE
    from software.dqn_agent import DQNRoutingAgent

import torch
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_graphsage_model():
    """Test the GraphSAGE model"""
    logger.info("Testing GraphSAGE model...")

    # Create model
    model = SpatiotemporalGraphSAGE(
        input_dim=10,
        hidden_dim=32,
        output_dim=5,
        num_layers=2
    )

    # Create sample graph
    G = nx.grid_graph(dim=[4, 3])
    node_features = torch.randn(len(G.nodes()), 10)

    # Test prediction - predict_delay handles NetworkX graph conversion internally
    with torch.no_grad():
        delay_pred = model.predict_delay(G, node_features)

    logger.info(f"GraphSAGE output shape: {delay_pred.shape}")
    assert delay_pred.shape == (len(G.nodes()), 5), f"Expected shape ({len(G.nodes())}, 5), got {delay_pred.shape}"
    logger.info("GraphSAGE model test passed!")

def test_dqn_agent():
    """Test the DQN agent"""
    logger.info("Testing DQN agent...")

    # Create agent
    agent = DQNRoutingAgent(
        state_dim=15,
        action_dim=5,
        learning_rate=0.001
    )

    # Test action selection
    state = torch.randn(15)
    action = agent.act(state.numpy())  # Greedy action

    logger.info(f"Selected action: {action}")
    assert 0 <= action < 5, f"Action {action} out of bounds [0, 5)"
    logger.info("DQN agent test passed!")

async def test_integration():
    """Test integration of components"""
    logger.info("Testing component integration...")

    # Initialize models
    graphsage_model = SpatiotemporalGraphSAGE(input_dim=10, hidden_dim=32, output_dim=5, num_layers=2)
    dqn_agent = DQNRoutingAgent(state_dim=20, action_dim=8, learning_rate=0.001)

    # Simulate railway state
    G = nx.grid_graph(dim=[5, 4])
    node_features = torch.randn(len(G.nodes()), 10)

    # Run prediction - predict_delay handles NetworkX graph conversion internally
    with torch.no_grad():
        delay_predictions = graphsage_model.predict_delay(G, node_features)

    logger.info(f"Delay predictions shape: {delay_predictions.shape}")

    # Simulate combined state for DQN
    # Combine graph features with delay predictions (simplified)
    # Convert delay_predictions to torch tensor for consistency
    delay_predictions_tensor = torch.FloatTensor(delay_predictions)
    combined_state = torch.cat([
        node_features.mean(dim=0),  # Average node features
        delay_predictions_tensor.mean(dim=0)  # Average delay predictions
    ])

    # Pad or trim to expected state dimension
    if len(combined_state) < 20:
        combined_state = torch.cat([combined_state, torch.zeros(20 - len(combined_state))])
    else:
        combined_state = combined_state[:20]

    # Get routing action
    # Convert to numpy array for the act method
    action = dqn_agent.act(combined_state.numpy())
    logger.info(f"Selected routing action: {action}")

    logger.info("Integration test passed!")

def main():
    """Run all tests"""
    logger.info("Starting ARTON system tests...")

    try:
        test_graphsage_model()
        test_dqn_agent()
        asyncio.run(test_integration())
        logger.info("All tests passed!")
        return 0
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())