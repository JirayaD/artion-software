# ARTON v1.1.0 — Local validation report

**Validation date:** 29 October 2026  
**Mode:** CPU-only software simulator; no hardware connected

## Environment

- Python 3.13.5
- PyTorch 2.10.0+cpu
- NetworkX 3.6.1
- NumPy 2.3.5
- FastAPI 0.128.2
- Uvicorn 0.48.0
- WebSockets 16.0
- PyTorch Geometric not installed; the pure-PyTorch GraphSAGE fallback was exercised.

## Results

- Component/integration tests: **4 passed**.
- Timed software-only demonstration: **30 seconds**, stopped cleanly.
- During the screenshot capture snapshot: **51 prediction cycles**, **50 simulator commands**, **1 active WebSocket connection**.
- The API showed a 20-node synthetic graph and a `(20, 5)` prediction output shape.
- Dashboard and runtime-log images are saved in `docs/screenshots/` and embedded in the project-review PowerPoint.

Runtime duration varied across cycles (the run included occasional slower cycles), so the snapshot is not a performance guarantee. Inputs are synthetic, weights are randomly initialized and output values are not calibrated delays. This validation checks the prototype wiring, not railway safety or forecasting accuracy.
