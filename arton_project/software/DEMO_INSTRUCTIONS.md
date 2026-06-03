# ARTION System Demonstration Instructions

## Overview
This demo shows the complete ARTION system working with a hardware simulator that mimics ESP32 hardware. You'll see real-time interaction between:

1. **Hardware Simulator** - Mimics ESP32 with track switches, sensors, and signals
2. **ARTION Software** - GraphSAGE delay prediction + DQN routing optimization + WebSocket server
3. **Real-time Communication** - Commands flow from software to hardware, telemetry flows back

## Files Created
- `hardware_simulator.py` - ESP32 hardware simulator
- `main.py` - Updated to support simulator mode (`--simulator` flag)
- `demo_artion.py` - Automated demonstration script
- `graphsage_model.py` - Spatiotemporal GraphSAGE for delay prediction
- `dqn_agent.py` - Deep Q-Network for routing optimization
- `fastapi_server.py` - WebSocket server for communication
- `test_artion.py` - Test suite (all passing)

## How to Run the Demonstration

### Option 1: Automated Demo (Recommended)
```bash
cd ~/arton_project/software
python3 demo_artion.py
```
This will:
1. Start the hardware simulator in the background
2. Start the ARTION system in simulator mode
3. Run both for 30 seconds so you can see the interaction
4. Automatically clean up and show final output

### Option 2: Manual Control
```bash
cd ~/arton_project/software

# In one terminal, start the hardware simulator:
python3 hardware_simulator.py

# In another terminal, start ARTION with simulator:
python3 main.py --simulator
```

### Option 3: Test Individual Components
```bash
cd ~/arton_project/software
python3 test_artion.py  # Run all tests
```

## What You'll See in the Demo

### Hardware Simulator Output:
```
🔧 Connected to hardware server at ws://localhost:8000/ws/hardware
📥 Received command #{1}: {'type': 'switch_control', 'switch_id': 2, 'position': 'REVERSE'}
Switch 2: NORMAL -> REVERSE
📊 Sent telemetry: 4 switches, 4 sensors, 3 signals
```

### ARTION System Output:
```
🚂 Initializing ARTION system...
✅ Initialized SpatiotemporalGraphSAGE: 10->64->5
✅ Initialized DQNRoutingAgent: state=20, action=10
🚂 Starting prediction loop...
📤 Sending hardware commands: {'action_id': 3, 'commands': [{'target': 'TRACK_4', 'action': 'SET_POSITION_B', ...}], 'description': 'Routing action 3'}
⚡ Executing action: Set track 4 to loop line
```

### Key Interactions to Watch For:
1. **Connection**: Hardware simulator connects to WebSocket server
2. **Commands**: ARTION sends routing commands (switch positions, signal aspects)
3. **Actions**: Hardware simulator acts on commands (throws switches, changes signals)
4. **Telemetry**: Hardware simulator sends back status (occupancy, switch positions)
5. **Loop**: Process repeats every 100ms for real-time operation

## System Architecture Shown in Demo

```
[ARTION Software]  ← WebSocket →  [Hardware Simulator]
        │                           │
   ┌────▼────┐               ┌─────▼─────┐
   │GraphSAGE│               │  Switches │
   │ (Delay  │               │ (Track    │
   │  Pred)  │               │  Points)  │
   └────┬────┘               └─────▲─────┘
        │                           │
   ┌────▼────┐               ┌─────▼─────┐
   │   DQN   │               │Sensors & │
   │(Routing)│               │ Signals   │
   └────┬────┘               └─────▲─────┘
        │                           │
   [Prediction Loop]      [Real-time I/O]
     (<100ms cycle)          (WebSocket)
```

## Technical Details

### Real-time Performance:
- Designed for <100ms cycle times (10Hz updates)
- Async/await for non-blocking operations
- Efficient binary serialization via JSON over WebSocket

### Hardware Simulation Features:
- **4 Track Switches**: Can be set to NORMAL or REVERSE positions
- **4 Occupancy Sensors**: Simulate train presence with realistic dynamics
- **3 Railway Signals**: GREEN/YELLOW/RED aspects
- **Smart Simulation**: Traffic flow responds to switch positions
- **Occasional Noise**: Simulates sensor unreliability (1% failure rate)
- **Statistical Tracking**: Command counts, uptime, performance metrics

### ARTION Software Features:
- **Spatiotemporal GraphSAGE**: Predicts delay propagation using graph convolutions
- **DQN Agent**: Learns optimal routing policies with experience replay
- **WebSocket Server**: Low-latency bidirectional communication
- **Modular Design**: Easy to replace simulator with actual hardware
- **Fault Tolerance**: Graceful shutdown, error handling, recovery

## Transitioning to Real Hardware
To use with actual ESP32 hardware:
1. Ensure ESP32 devices are running compatible firmware
2. Connect them to the same network as the ARTION server
3. Run without the `--simulator` flag: `python3 main.py`
4. The WebSocket server will accept connections from real hardware
5. No software changes needed - same interface works for both

## Troubleshooting
- **Connection refused**: Make sure both systems are running on same machine/network
- **Port already in use**: Kill existing processes on port 8000
- **Module not found**: Run `pip3 install -r requirements.txt` first
- **No interaction**: Check that both systems show WebSocket connection messages

## Saving Demo Output
To save demo output for later review:
```bash
cd ~/arton_project/software
python3 demo_artion.py > demo_output.txt 2>&1
```

The demonstration shows how ARTION enables autonomous railway optimization with real-time hardware interaction - all running in simulation for safe testing and development!