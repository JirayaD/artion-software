"""
Deep Q-Network Agent for optimal train routing
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Deep Q-Network for routing decisions
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.dropout = nn.Dropout(0.2)
        
        logger.info(f"Initialized DQN Network: {state_dim}->{hidden_dim}->{action_dim}")
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class DQNRoutingAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, 
                 epsilon_decay=0.995, memory_size=10000, batch_size=64):
        """
        DQN Agent for train routing optimization
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible routing actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            memory_size: Size of replay memory
            batch_size: Batch size for training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Q-Networks
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Update target network
        self.update_target_network()
        
        logger.info(f"Initialized DQNRoutingAgent: state={state_dim}, action={action_dim}")
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action index
        """
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def optimize_routing(self, railway_state, delay_predictions):
        """
        Optimize train routing based on current state and delay predictions
        
        Args:
            railway_state: Current state of railway network
            delay_predictions: Predicted delays from GraphSAGE model
            
        Returns:
            actions: Optimal routing actions for each train/junction
        """
        # Combine railway state and delay predictions into agent state
        # This is a simplified version - real implementation would be more sophisticated
        state_vector = self._create_state_vector(railway_state, delay_predictions)
        
        # Get action from DQN
        action_index = self.act(state_vector)
        
        # Convert action index to meaningful routing commands
        actions = self._action_to_commands(action_index, railway_state)
        
        return actions
    
    def _create_state_vector(self, railway_state, delay_predictions):
        """Create state vector for DQN from railway state and predictions"""
        # Flatten and combine relevant information
        # This is a placeholder - real implementation would extract meaningful features
        
        # Example: combine graph statistics, delay predictions, current train positions
        state_parts = []
        
        # Add delay predictions (flattened)
        if hasattr(delay_predictions, 'flatten'):
            state_parts.append(delay_predictions.flatten()[:10])  # Take first 10 elements
        else:
            state_parts.append(np.array(delay_predictions).flatten()[:10])
        
        # Add some basic graph statistics (placeholder)
        state_parts.append(np.array([0.5, 0.3, 0.2]))  # Example: network utilization metrics
        
        # Concatenate and pad/truncate to fixed size
        state_vector = np.concatenate(state_parts)
        if len(state_vector) < self.state_dim:
            # Pad with zeros
            state_vector = np.pad(state_vector, (0, self.state_dim - len(state_vector)))
        else:
            # Truncate
            state_vector = state_vector[:self.state_dim]
            
        return state_vector
    
    def _action_to_commands(self, action_index, railway_state):
        """Convert action index to specific routing commands"""
        # This would map discrete actions to specific track switch commands
        # For now, return a placeholder
        
        # Example mapping:
        # 0: No action
        # 1-4: Switch track 1-4 to position A
        # 5-8: Switch track 1-4 to position B
        # etc.
        
        commands = {
            'action_id': action_index,
            'commands': [],
            'description': f'Routing action {action_index}'
        }
        
        # Simple example: based on action index, suggest some track switches
        if action_index < 4:
            commands['commands'].append({
                'target': f'TRACK_{action_index+1}',
                'action': 'SET_POSITION_A',
                'description': f'Set track {action_index+1} to main line'
            })
        elif action_index < 8:
            track_num = action_index - 3
            commands['commands'].append({
                'target': f'TRACK_{track_num}',
                'action': 'SET_POSITION_B',
                'description': f'Set track {track_num} to loop line'
            })
        else:
            commands['commands'].append({
                'target': 'SYSTEM',
                'action': 'HOLD_CURRENT_ROUTING',
                'description': 'Maintain current routing configuration'
            })
        
        return commands
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a pre-trained model"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        logger.info(f"Model loaded from {path}")

# Import F here since it's used in the class definition
import torch.nn.functional as F
