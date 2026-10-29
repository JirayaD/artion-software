"""Deep Q-Network routing policy for ARTON's simulator.

The agent is an algorithm scaffold; meaningful deployment requires training against
validated railway data, a reward function, and safety constraints.
"""
from __future__ import annotations
import logging
import random
from collections import deque
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class DQNRoutingAgent:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001,
                 gamma: float = 0.99, epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, memory_size: int = 10000, batch_size: int = 64):
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        self.state_dim, self.action_dim = state_dim, action_dim
        self.gamma, self.epsilon = gamma, epsilon_start
        self.epsilon_end, self.epsilon_decay = epsilon_end, epsilon_decay
        self.batch_size = batch_size
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        logger.info("Initialized DQN agent state=%s action=%s", state_dim, action_dim)

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((np.asarray(state, dtype=np.float32), int(action), float(reward),
                            np.asarray(next_state, dtype=np.float32), bool(done)))

    def act(self, state) -> int:
        state_np = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_np.size != self.state_dim:
            raise ValueError(f"Expected state length {self.state_dim}, got {state_np.size}")
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(torch.from_numpy(state_np).unsqueeze(0))
        return int(q_values.argmax(dim=1).item())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states = torch.as_tensor(np.stack([e[0] for e in batch]), dtype=torch.float32)
        actions = torch.as_tensor([e[1] for e in batch], dtype=torch.long)
        rewards = torch.as_tensor([e[2] for e in batch], dtype=torch.float32)
        next_states = torch.as_tensor(np.stack([e[3] for e in batch]), dtype=torch.float32)
        dones = torch.as_tensor([e[4] for e in batch], dtype=torch.bool)
        # `act()` switches the online network to eval mode. Switch back before
        # replay updates so the training path has an explicit, predictable mode.
        self.q_network.train()
        self.target_network.eval()
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1).values
            target_q = rewards + self.gamma * next_q * (~dones)
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def _create_state_vector(self, railway_state: dict, delay_predictions) -> np.ndarray:
        pred = np.asarray(delay_predictions, dtype=np.float32).reshape(-1)[:10]
        features = railway_state.get("features")
        if isinstance(features, torch.Tensor):
            feature_summary = features.detach().cpu().numpy().mean(axis=0)[:7]
        elif features is not None:
            feature_summary = np.asarray(features, dtype=np.float32).mean(axis=0)[:7]
        else:
            feature_summary = np.zeros(7, dtype=np.float32)
        vector = np.concatenate([pred, feature_summary]).astype(np.float32)
        if vector.size < self.state_dim:
            vector = np.pad(vector, (0, self.state_dim-vector.size))
        return vector[:self.state_dim]

    def _action_to_commands(self, action_index: int, railway_state: dict | None = None) -> dict[str, Any]:
        """Map each discrete action to a simulator-compatible command."""
        if 0 <= action_index <= 3:
            command = {"type": "switch_control", "switch_id": action_index + 1, "position": "NORMAL"}
            description = f"Set switch {action_index + 1} to NORMAL"
        elif 4 <= action_index <= 7:
            switch_id = action_index - 3
            command = {"type": "switch_control", "switch_id": switch_id, "position": "REVERSE"}
            description = f"Set switch {switch_id} to REVERSE"
        elif action_index == 8:
            command = {"type": "signal_control", "signal_id": 1, "aspect": "YELLOW"}
            description = "Set signal 1 to YELLOW"
        elif action_index == 9:
            command = {"type": "signal_control", "signal_id": 1, "aspect": "RED"}
            description = "Set signal 1 to RED"
        else:
            # Generalize beyond the default 10-action configuration safely.
            command = {"type": "heartbeat", "status": "no-op", "action_id": action_index}
            description = f"No-op action {action_index}"
        return {"action_id": int(action_index), "commands": [command], "description": description}

    def optimize_routing(self, railway_state: dict, delay_predictions) -> dict[str, Any]:
        state_vector = self._create_state_vector(railway_state, delay_predictions)
        action_index = self.act(state_vector)
        return self._action_to_commands(action_index, railway_state)
