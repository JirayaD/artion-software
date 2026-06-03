# ARTION Software System

## Overview
This is the software component of the ARTION (Edge-Connected Spatiotemporal Graph Neural Framework for Dynamic Train Scheduling & Track Switch Actuation) system. The software implements:

1. **Spatiotemporal GraphSAGE Neural Network** for delay propagation prediction
2. **Deep Q-Network (DQN) Agent** for routing optimization  
3. **FastAPI WebSocket Server** for hardware communication with ESP32 controllers
4. **Main System Coordinator** integrating all components

## Components

### 1. GraphSAGE Model (`graphsage_model.py`)
- Spatiotemporal GraphSAGE for predicting train delays
- Takes railway network topology and features as input
- Outputs delay predictions for multiple time steps
- Uses PyTorch Geometric for efficient graph convolutions

### 2. DQN Routing Agent (`dqn_agent.py`)
- Deep Q-Network for optimizing train routing decisions
- Combines current railway state with delay predictions
- Uses epsilon-greedy exploration and experience replay
- Outputs actionable routing commands for track switches

### 3. FastAPI WebSocket Server (`fastapi_server.py`)
- Real-time communication bridge to hardware (ESP32, sensors, actuators)
- WebSocket endpoints for bidirectional hardware communication
- Background loops for sending optimization commands
- Status endpoints for monitoring system health

### 4. Main System (`main.py`)
- Entry point initializing all components
- Concurrent async execution of prediction loop and WebSocket server
- Graceful shutdown handling
- Simulated railway state generation (for testing)

### 5. Test Suite (`test_artion.py`)
- Unit tests for GraphSAGE model
- Unit tests for DQN agent
- Integration test verifying component interaction
- All tests passing

## Dependencies
See `requirements.txt`:
- torch>=2.0.0
- torch-geometric>=2.0.0
- networkx>=3.0
- stable-baselines3[extra]>=2.0.0
- gymnasium>=0.26.0
- fastapi>=0.100.0
- uvicorn[standard]>=0.23.0
- websockets>=12.0
- pandas>=2.0.0
- numpy>=1.24.0

## Installation
```bash
cd ~/arton_project/software
pip3 install -r requirements.txt
```

## Usage
```bash
cd ~/arton_project/software
python3 main.py
```

## System Architecture
```
[delay prediction]     [routing optimization]
     GraphSAGE Model  ---->  DQN Agent
           ^                       ^
           |                       |
[railway state]  <---------------- [hardware feedback]
          (WebSocket Server)
```

## Real-Time Constraints
- Designed for <100ms cycle times
- Async processing for non-blocking operations
- WebSocket communication for low-latency hardware interface

## Testing
Run tests with:
```bash
python3 test_artion.py
```

All tests should pass, verifying:
- GraphSAGE model correctness
- DQN agent functionality
- Component integration