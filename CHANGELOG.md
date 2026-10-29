# Changelog

Notable changes are recorded here. The project follows Semantic Versioning (`MAJOR.MINOR.PATCH`).

## [1.1.0] - 2026-10-29

### Fixed
- Added a pure-PyTorch GraphSAGE convolution fallback when PyTorch Geometric is unavailable.
- Ensured NetworkX edges are represented in both directions for GraphSAGE message passing.
- Returned CPU NumPy predictions safely from model inference.
- Standardized DQN command payloads to match the simulator's switch and signal command schema.
- Wired the prediction loop to broadcast actions to the local simulator and track telemetry in the API state.
- Consolidated timed demo lifecycle to avoid duplicate simulator processes and improve clean shutdown.

### Added
- Live dashboard and API state endpoints for the software-only simulator.
- Component/integration coverage for GraphSAGE output shape, DQN replay, policy integration and simulator telemetry.
- Dashboard and runtime-log screenshots under `docs/screenshots/`.
- Documented prerequisites, quick start, troubleshooting and Semantic Versioning.

### Notes
- Inputs remain synthetic and model weights are randomly initialized. This release verifies the software demonstration loop; it does not establish railway-grade safety, forecast accuracy or guaranteed runtime performance.

## [1.0.0] - 2026-06

- Initial documented ARTON prototype with GraphSAGE-style delay prediction, DQN routing, FastAPI/WebSocket communication, and ESP32-style simulation concept.
