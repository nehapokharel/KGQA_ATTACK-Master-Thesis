import pickle
import keras
from keras import layers
import numpy as np
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class KGQARLAgent:
    def __init__(self, state_size, action_size, model_path, memory_file="experience_replay.pkl"):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.memory_file = memory_file
        self.memory_capacity = 2000
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.batch_size = 16
        self.model_path = model_path

        self.model = self._build_model()
        self.load_experience()

        # load pre-trained model
        self.load_model()

        if os.path.exists(self.model_path):
            print(f"[INFO] Loading pre-trained model from {self.model_path}")
            self.model = keras.models.load_model(self.model_path)

    def _build_model(self):
        """Build a feed-forward Q-learning network."""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def save_experience(self):
        """Save the replay memory to a file."""
        with open(self.memory_file, "wb") as f:
            pickle.dump(self.memory, f)

    def load_experience(self):
        """Load replay memory from a file, if it exists."""
        try:
            with open(self.memory_file, "rb") as f:
                self.memory = pickle.load(f)
            print(f"Loaded {len(self.memory)} past experiences from {self.memory_file}.")
        except FileNotFoundError:
            print("No previous experience found. Starting with an empty memory.")

    def act(self, state, similarity_scores):
        """Choose an action using an epsilon-greedy strategy with similarity-based guidance."""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(len(similarity_scores))

        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        q_values = q_values[:len(similarity_scores)]
        contains_true_answer = state[-2]

        combined_values = 0.7 * q_values + 0.3 * similarity_scores if contains_true_answer else 0.5 * q_values + 0.5 * similarity_scores
        return int(np.argmax(combined_values))

    def replay(self):
        """Train the model using experience replay."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))
        states = np.vstack(states)
        next_states = np.vstack(next_states)

        target_qs = self.model.predict(states, verbose=0)
        next_qs = self.model.predict(next_states, verbose=0)

        target_qs[np.arange(self.batch_size), actions] = rewards + (self.gamma * np.max(next_qs, axis=1) * (1 - dones))

        self.model.fit(states, target_qs, epochs=1, verbose=0, batch_size=self.batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        """Save the trained model."""
        self.model.save(self.model_path)
        print(f"[INFO] Model saved at {self.model_path}")

    def load_model(self):
        """Load a trained model from a file."""
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            print(f"[INFO] Loaded model from {self.model_path}")
        else:
            print(f"[WARNING] Model file {self.model_path} not found. Using a new model.")
