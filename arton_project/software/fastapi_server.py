"""
FastAPI WebSocket Server for ARTON system
Handles communication between software and hardware components
"""
import asyncio
import json
import logging
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ARTON Hardware Communication Server")

# Enable CORS for hardware communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to hardware IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.hardware_status: Dict = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Hardware connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Hardware disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message to hardware: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to hardware: {e}")
                self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws/hardware")
async def hardware_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for hardware communication"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive data from hardware (sensor readings, status updates)
            data = await websocket.receive_text()
            logger.info(f"Received from hardware: {data}")

            # Process hardware data (update status, store in database, etc.)
            try:
                hardware_data = json.loads(data)
                manager.hardware_status.update(hardware_data)

                # Example: Process sensor data for feedback loop
                await process_hardware_feedback(hardware_data)

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received from hardware: {data}")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Hardware WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def process_hardware_feedback(hardware_data: dict):
    """Process feedback from hardware components"""
    # This would integrate with the ML models for closed-loop control
    # For example: update track occupancy based on sensor readings
    logger.info(f"Processing hardware feedback: {hardware_data}")

    # Example processing:
    # - Update track occupancy status
    # - Detect hardware faults
    # - Adjust ML model inputs based on real-world feedback

@app.get("/")
async def root():
    return {"message": "ARTON Hardware Communication Server", "status": "running"}

@app.get("/status")
async def get_status():
    return {
        "server": "running",
        "hardware_connections": len(manager.active_connections),
        "hardware_status": manager.hardware_status
    }

@app.post("/commands")
async def send_command_to_hardware(command: dict):
    """Send command to all connected hardware"""
    message = json.dumps(command)
    await manager.broadcast(message)
    return {"status": "command sent", "command": command}

# Background task to simulate sending optimization results to hardware
async def hardware_communication_loop():
    """Background loop to send optimization commands to hardware"""
    while True:
        try:
            # In real system, this would get commands from the DQN agent
            # For now, send a heartbeat/status command
            if manager.active_connections:
                command = {
                    "type": "heartbeat",
                    "timestamp": asyncio.get_event_loop().time(),
                    "status": "software_alive"
                }
                await manager.broadcast(json.dumps(command))

            # Wait before next communication (adjust based on real-time requirements)
            await asyncio.sleep(0.1)  # 10Hz update rate

        except Exception as e:
            logger.error(f"Error in hardware communication loop: {e}")
            await asyncio.sleep(1)

# Start background task when app starts
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(hardware_communication_loop())
    logger.info("ARTON server started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("ARTON server shutting down")
    # Close all WebSocket connections
    for connection in manager.active_connections:
        await connection.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)