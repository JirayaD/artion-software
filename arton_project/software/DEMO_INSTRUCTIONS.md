# ARTON software-only simulation: run guide

## Prerequisites

- Python 3.10+ (Python 3.11/3.12 recommended; smoke-tested on Python 3.13.5).
- `pip` and the packages listed in `requirements.txt`.
- A browser for the local dashboard.
- No physical ESP32 or other railway hardware is needed for simulator mode.

## Install

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r arton_project/software/requirements.txt
```

## Run the live simulation

```bash
cd arton_project/software
python main.py --simulator
```

Open <http://127.0.0.1:8000/dashboard>. The machine runs a synthetic 20-node railway graph and a software-only simulator for 4 switches, 4 sensors and 3 signals. Stop with `Ctrl+C`.

## Run a self-terminating demo

```bash
python demo_artion.py --duration 30
```

This launches one in-process simulator and shuts down after the specified number of seconds.

## Run tests

```bash
python test_artion.py
```

The current component suite checks GraphSAGE prediction shape/finite values, DQN action selection and replay, end-to-end prediction-to-command mapping, simulator switch/signal control and telemetry counts.

## Dashboard and API

- `GET /dashboard` — live visual dashboard.
- `GET /api/state` — current loop, prediction summary, last action, switch/sensor/signal telemetry and event stream.
- `GET /` — service status.
- `WS /ws/hardware` — local bidirectional simulator connection.

## Troubleshooting

- Install requirements inside the same Python environment used to run the demo.
- If port `8000` is occupied, stop the old ARTON process before starting another.
- Use `demo_artion.py` for a complete one-command run; do not launch another simulator at the same time.
- PyTorch Geometric is optional. The pure-PyTorch GraphSAGE fallback should work without it.

## Important limitations

The inputs are synthetic and model weights are randomly initialized. Raw predictions do not represent calibrated delay minutes. The DQN is a demonstration policy, not a trained railway controller. Observed local loop durations are not a guaranteed performance target, and this prototype must not be used to control real railway infrastructure.
