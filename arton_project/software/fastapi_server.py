"""FastAPI + WebSocket bridge and live dashboard for ARTON simulation."""
from __future__ import annotations
import asyncio
import json
import logging
import time
from typing import Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)
app = FastAPI(title="ARTON Railway Optimization Simulator", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["*"])

runtime_state: dict[str, Any] = {
    "started_at": time.time(), "loop_count": 0, "last_cycle_ms": None,
    "last_action": None, "prediction_summary": None, "recent_events": [],
    "mode": "SIMULATOR", "model_note": "Synthetic state • untrained prototype model"
}

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.hardware_status: dict[str, Any] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("Hardware simulator connected (%d active)", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("Hardware simulator disconnected (%d active)", len(self.active_connections))

    async def broadcast(self, message: str) -> int:
        sent = 0
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
                sent += 1
            except Exception as exc:
                logger.warning("Removing stale WebSocket: %s", exc)
                self.disconnect(connection)
        return sent

manager = ConnectionManager()

def add_event(message: str, level: str = "INFO") -> None:
    events = runtime_state.setdefault("recent_events", [])
    events.insert(0, {"time": time.strftime("%H:%M:%S"), "message": str(message), "level": level})
    del events[16:]

def set_runtime_state(**values) -> None:
    runtime_state.update(values)

@app.websocket("/ws/hardware")
async def hardware_websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    add_event("Simulator connected to WebSocket bridge", "OK")
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                add_event("Rejected malformed telemetry JSON", "WARN")
                continue
            if not isinstance(data, dict):
                continue
            manager.hardware_status.update(data)
            runtime_state["telemetry_received_at"] = time.time()
    except WebSocketDisconnect:
        add_event("Simulator WebSocket disconnected", "WARN")
    except Exception as exc:
        logger.exception("WebSocket endpoint failure: %s", exc)
        add_event(f"WebSocket error: {type(exc).__name__}", "WARN")
    finally:
        manager.disconnect(websocket)

@app.get("/")
async def root():
    return {"message": "ARTON Railway Optimization Simulator", "status": "running", "dashboard": "/dashboard"}

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/status")
async def get_status():
    return {"server": "running", "hardware_connections": len(manager.active_connections),
            "hardware_status": manager.hardware_status, "runtime": runtime_state}

@app.get("/api/state")
async def get_api_state():
    hw = manager.hardware_status
    switches = hw.get("switches", []) or []
    sensors = hw.get("sensors", []) or []
    signals = hw.get("signals", []) or []
    return {
        "status": "LIVE" if manager.active_connections else "WAITING FOR SIMULATOR",
        "uptime_s": round(time.time() - runtime_state["started_at"], 1),
        "connections": len(manager.active_connections),
        "loop_count": runtime_state.get("loop_count", 0),
        "last_cycle_ms": runtime_state.get("last_cycle_ms"),
        "last_action": runtime_state.get("last_action"),
        "prediction_summary": runtime_state.get("prediction_summary"),
        "recent_events": runtime_state.get("recent_events", []),
        "mode": runtime_state.get("mode", "SIMULATOR"),
        "model_note": runtime_state.get("model_note", "Synthetic data"),
        "switches": switches, "sensors": sensors, "signals": signals,
        "commands_received": hw.get("stats", {}).get("commands_received", 0),
        "telemetry_timestamp": hw.get("timestamp"),
    }

@app.post("/commands")
async def send_command_to_hardware(command: dict):
    sent = await manager.broadcast(json.dumps(command))
    add_event(f"Manual command broadcast to {sent} device(s)", "INFO")
    return {"status": "command sent" if sent else "no simulator connected", "connections": sent, "command": command}

DASHBOARD_HTML = r'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ARTON Live Simulator</title>
<style>
:root{--bg:#07111f;--panel:#0c1d32;--panel2:#102844;--line:#1b4564;--cyan:#44d9f2;--blue:#5b9cff;--amber:#ffc45c;--green:#52e2b0;--red:#ff687f;--muted:#91a9c5;--text:#ecf5ff}*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 75% 4%,#12365a 0%,var(--bg) 48%);font-family:Inter,Segoe UI,Arial,sans-serif;color:var(--text);padding:26px}header{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}.brand{font-weight:800;letter-spacing:2px;font-size:25px}.brand span{color:var(--cyan)}.subtitle{color:var(--muted);font-size:12px;margin-top:6px;letter-spacing:1px}.pill{border:1px solid #28637a;color:var(--cyan);background:#0d3046;border-radius:30px;padding:9px 15px;font-size:11px;font-weight:700;letter-spacing:1px}.notice{padding:10px 14px;background:#142b40;border:1px solid #2c5772;border-radius:10px;color:#bcd3e9;font-size:11px;margin-bottom:17px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-bottom:14px}.metric,.panel{background:linear-gradient(145deg,rgba(16,40,68,.95),rgba(9,26,45,.97));border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 25px #0002}.metric{padding:17px 18px;min-height:101px}.label{font-size:10px;color:var(--muted);letter-spacing:1.2px;font-weight:700}.value{font-size:27px;font-weight:800;margin-top:8px}.small{font-size:11px;color:var(--muted);margin-top:5px}.main{display:grid;grid-template-columns:1.55fr 1fr;gap:14px}.panel{padding:17px 18px}.panel h2{font-size:13px;margin:0 0 13px;letter-spacing:1px;text-transform:uppercase}.cyan{color:var(--cyan)}.railmap{background:#061523;border:1px solid #21435f;border-radius:10px;overflow:hidden}.railmap svg{display:block;width:100%;height:254px}.bottom{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}.device-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.device{border:1px solid #254863;background:#0a1b2e;border-radius:9px;padding:10px;text-align:center}.device b{font-size:10px;display:block;margin-bottom:9px}.state{font-size:9px;font-weight:800;padding:4px;border-radius:14px;display:inline-block}.normal{background:#123d39;color:var(--green)}.reverse{background:#49391e;color:var(--amber)}.busy{background:#4a2030;color:var(--red)}.free{background:#123d39;color:var(--green)}.signal-row{display:flex;gap:10px}.signal{flex:1;display:flex;align-items:center;gap:8px;background:#0a1b2e;border:1px solid #254863;border-radius:9px;padding:10px;font-size:10px}.light{width:11px;height:11px;border-radius:50%;display:inline-block;background:#355069;box-shadow:0 0 9px currentColor}.light.GREEN{background:var(--green);color:var(--green)}.light.YELLOW{background:var(--amber);color:var(--amber)}.light.RED{background:var(--red);color:var(--red)}.action-card{background:#071827;border:1px solid #254863;border-radius:10px;padding:13px;margin-bottom:10px}.action-id{font-size:10px;color:var(--cyan);font-weight:800;letter-spacing:1px}.action-title{font-size:16px;font-weight:800;margin-top:7px}.action-detail{font-size:11px;color:var(--muted);margin-top:5px}.event{display:flex;gap:10px;border-bottom:1px solid #19374f;padding:8px 0;font-size:10px}.event:last-child{border-bottom:0}.eventtime{color:var(--muted);min-width:58px}.OK{color:var(--green)}.WARN{color:var(--amber)}.INFO{color:var(--cyan)}.footer{font-size:9px;color:#637e9a;margin-top:16px;display:flex;justify-content:space-between}@media(max-width:1000px){body{padding:13px}.metrics{grid-template-columns:repeat(2,1fr)}.main,.bottom{grid-template-columns:1fr}.device-grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<header><div><div class="brand">◈ <span>ARTON</span> / CONTROL ROOM</div><div class="subtitle">AUTONOMOUS RAILWAY TRANSIT OPTIMIZATION NETWORK · LOCAL SIMULATION</div></div><div class="pill" id="status">● CONNECTING</div></header>
<div class="notice">SIMULATOR MODE · Synthetic railway state and an untrained prototype policy · No physical railway hardware connected</div>
<div class="metrics"><div class="metric"><div class="label">PREDICTION CYCLES</div><div class="value" id="loops">—</div><div class="small">Completed coordinator loops</div></div><div class="metric"><div class="label">TRACK SWITCHES</div><div class="value"><span id="swCount">4</span><span style="font-size:13px;color:var(--muted)"> / 4 online</span></div><div class="small">NORMAL / REVERSE actuation</div></div><div class="metric"><div class="label">OCCUPIED SECTIONS</div><div class="value" id="occupied">—</div><div class="small">Live simulator sensor telemetry</div></div><div class="metric"><div class="label">LAST LOOP TIME</div><div class="value" id="cycle">—</div><div class="small">Observed local cycle duration · ms</div></div></div>
<div class="main"><section class="panel"><h2><span class="cyan">01</span> · Network topology / live telemetry</h2><div class="railmap"><svg viewBox="0 0 700 270" aria-label="Illustrative railway topology"><defs><pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M24 0H0V24" fill="none" stroke="#17405b" stroke-width=".6"/></pattern><filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><rect width="700" height="270" fill="url(#grid)"/><g stroke="#236081" stroke-width="5" fill="none" opacity=".9"><path d="M50 210 L150 140 L250 180 L355 95 L475 130 L600 55"/><path d="M150 140 L170 55 L355 95 L385 225 L500 210 L600 55"/><path d="M250 180 L300 230 L385 225"/></g><g stroke="#64dff4" stroke-width="1.4" fill="none" stroke-dasharray="7 7"><path d="M50 210 L150 140 L250 180 L355 95 L475 130 L600 55"/><path d="M150 140 L170 55 L355 95 L385 225 L500 210 L600 55"/></g><g id="nodes" fill="#43d8ef" stroke="#d6fbff" stroke-width="1.5"><circle cx="50" cy="210" r="7"/><circle cx="150" cy="140" r="7"/><circle cx="250" cy="180" r="7"/><circle cx="355" cy="95" r="7"/><circle cx="475" cy="130" r="7"/><circle cx="600" cy="55" r="7"/><circle cx="170" cy="55" r="7"/><circle cx="385" cy="225" r="7"/><circle cx="500" cy="210" r="7"/><circle cx="300" cy="230" r="7"/></g><g fill="#a9c2d9" font-size="10" font-family="Segoe UI,Arial"><text x="32" y="235">ST-01</text><text x="126" y="123">J-02</text><text x="226" y="201">ST-03</text><text x="332" y="78">J-04</text><text x="452" y="114">ST-05</text><text x="580" y="37">ST-06</text><text x="143" y="37">J-07</text><text x="366" y="247">ST-08</text><text x="479" y="234">J-09</text><text x="280" y="253">ST-10</text></g><g id="train" filter="url(#glow)"><circle cx="250" cy="180" r="7" fill="#ffc45c" stroke="#fff3c6" stroke-width="2"/></g><text x="18" y="20" fill="#44d9f2" font-size="10" font-weight="700">SYNTHETIC GRID · MODEL OUTPUTS ARE NOT CALIBRATED DELAY MINUTES</text></svg></div><div class="small" style="display:flex;justify-content:space-between;margin-top:9px"><span>● <b style="color:var(--cyan)">Node</b> / track adjacency</span><span>● <b style="color:var(--amber)">Active action indicator</b></span><span id="telem">Telemetry waiting</span></div></section>
<section class="panel"><h2><span class="cyan">02</span> · Policy output</h2><div class="action-card"><div class="action-id" id="actionId">ACTION —</div><div class="action-title" id="actionTitle">Waiting for first prediction…</div><div class="action-detail" id="actionDetail">A DQN action will appear when the loop starts.</div></div><div class="label">LATEST MODEL OUTPUT SUMMARY</div><div class="value" style="font-size:18px" id="pred">—</div><div class="small" id="predNote">Five raw output values per synthetic graph node; no real delay labels.</div><div style="margin-top:18px"><div class="label" style="margin-bottom:10px">RAILWAY SIGNALS</div><div class="signal-row" id="signals"><div class="signal">Awaiting telemetry</div></div></div></section></div>
<div class="bottom"><section class="panel"><h2><span class="cyan">03</span> · Trackside components</h2><div class="label" style="margin-bottom:11px">SWITCH POSITIONS</div><div class="device-grid" id="switches"><div class="device">Switch 1</div><div class="device">Switch 2</div><div class="device">Switch 3</div><div class="device">Switch 4</div></div><div class="label" style="margin:17px 0 11px">OCCUPANCY SENSORS</div><div class="device-grid" id="sensors"><div class="device">Sensor 1</div><div class="device">Sensor 2</div><div class="device">Sensor 3</div><div class="device">Sensor 4</div></div></section><section class="panel"><h2><span class="cyan">04</span> · Event stream</h2><div id="events"><div class="event"><span class="eventtime">—</span><span>Waiting for server events…</span></div></div></section></div>
<div class="footer"><span>ARTON / FINAL YEAR PROJECT REVIEW · REAL LOCAL SIMULATOR TELEMETRY</span><span id="uptime">UPTIME —</span></div>
<script>
const el=id=>document.getElementById(id);function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function renderDevices(items,kind){if(!items?.length)return '<div class="device">Waiting for telemetry</div>';return items.map(v=>{let id=kind==='switch'?v.switch_id:kind==='sensor'?v.sensor_id:v.signal_id;let state=kind==='switch'?v.position:kind==='sensor'?(v.occupied?'OCCUPIED':'CLEAR'):v.aspect;let cls=kind==='sensor'?(v.occupied?'busy':'free'):String(state).toLowerCase();return `<div class="device"><b>${kind.toUpperCase()} ${id}</b><span class="state ${esc(cls)}">${esc(state)}</span></div>`}).join('')}
async function refresh(){try{const r=await fetch('/api/state',{cache:'no-store'});const d=await r.json();el('status').textContent='● '+d.status;el('status').style.color=d.connections?'var(--green)':'var(--amber)';el('loops').textContent=d.loop_count;el('swCount').textContent=(d.switches||[]).length||4;el('occupied').textContent=(d.sensors||[]).filter(s=>s.occupied).length+' / '+((d.sensors||[]).length||4);el('cycle').textContent=d.last_cycle_ms==null?'—':Number(d.last_cycle_ms).toFixed(1);el('uptime').textContent='UPTIME '+d.uptime_s+' s';el('telem').textContent=d.telemetry_timestamp?'TELEMETRY LIVE':'Waiting on first telemetry';el('switches').innerHTML=renderDevices(d.switches,'switch');el('sensors').innerHTML=renderDevices(d.sensors,'sensor');el('signals').innerHTML=(d.signals||[]).map(s=>`<div class="signal"><span class="light ${esc(s.aspect)}"></span><span>Signal ${esc(s.signal_id)} · ${esc(s.aspect)}</span></div>`).join('')||'<div class="signal">Awaiting telemetry</div>';if(d.last_action){el('actionId').textContent='ACTION '+String(d.last_action.action_id).padStart(2,'0');el('actionTitle').textContent=d.last_action.description||'Routing action';el('actionDetail').textContent=(d.last_action.commands||[]).map(c=>c.type==='switch_control'?`Switch ${c.switch_id} → ${c.position}`:`Signal ${c.signal_id} → ${c.aspect}`).join(' · ')+' · Broadcast to simulator'}if(d.prediction_summary){el('pred').textContent='mean '+Number(d.prediction_summary.mean).toFixed(3)+' · range '+Number(d.prediction_summary.min).toFixed(3)+' to '+Number(d.prediction_summary.max).toFixed(3);el('predNote').textContent=`${d.prediction_summary.nodes} nodes × ${d.prediction_summary.horizon} raw outputs · synthetic features`;}el('events').innerHTML=(d.recent_events||[]).slice(0,9).map(e=>`<div class="event"><span class="eventtime">${esc(e.time)}</span><span class="${esc(e.level)}">${esc(e.message)}</span></div>`).join('')||'<div class="event">No events recorded</div>';const nodes=el('nodes').querySelectorAll('circle');const train=el('train').firstElementChild; if(train&&nodes.length){const p=nodes[(d.loop_count||0)%nodes.length];train.setAttribute('cx',p.getAttribute('cx'));train.setAttribute('cy',p.getAttribute('cy'));}}catch(e){el('status').textContent='● SERVER OFFLINE'}}refresh();setInterval(refresh,650);
</script></body></html>'''

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)

@app.on_event("startup")
async def startup_event():
    logger.info("ARTON API started; dashboard available at /dashboard")

@app.on_event("shutdown")
async def shutdown_event():
    for connection in list(manager.active_connections):
        try:
            await connection.close()
        except Exception:
            pass
        manager.disconnect(connection)
