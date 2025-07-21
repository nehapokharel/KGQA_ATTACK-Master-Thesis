import os
import pickle
import numpy as np
import random
import keras
from keras import layers, Model

class KGQARLAgent:
    """
        A Reinforcement Learning Agent for attacking Knowledge Graph Question Answering systems.

        This agent uses a Deep Q-Network (DQN) with an attention mechanism to learn
        optimal strategies for modifying a knowledge graph to make a QA system fail.
        It employs an epsilon-greedy strategy for action selection, balancing exploration
        and exploitation, and uses an experience replay buffer to learn from past actions.

        Attributes:
            action_size (int): The maximum number of possible actions (triples).
            model (keras.Model): The neural network model for predicting Q-values.
            memory (list): A replay buffer storing past experiences.
            epsilon (float): The current probability of choosing a random action (exploration).
            gamma (float): The discount factor for future rewards.
    """
    def __init__(self, state_shapes, action_size, model_path, memory_file="experience_replay/experience_replay_gnn.pkl"):
        """
            Initializes the KGQARLAgent.

            Args:
                state_shapes (dict): A dictionary defining the shapes of the different
                                     parts of the state representation.
                action_size (int): The total number of possible actions (max_triples).
                model_path (str): The file path for saving or loading the trained model.
                memory_file (str, optional): The file path for saving or loading the
                                             experience replay memory.
        """
        self.state_shapes = state_shapes
        self.action_size = action_size
        self.memory = []
        self.memory_capacity = 2000
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.batch_size = 16
        self.model_path = model_path
        self.memory_file = memory_file
        self.model = self._build_model()
        self.load_experience()
        self.load_model()

    def _build_model(self):
        """
            Builds and compiles the Deep Q-Network model using the Keras Functional API.

            The model uses an attention mechanism to weigh the importance of different
            triples in the subgraph based on the question. The resulting context vector
            is combined with other features to predict the Q-value for each possible action.

            Returns:
                keras.Model: The compiled Keras model.
        """
        # Define Inputs for the different parts of the state
        question_input = layers.Input(shape=(self.state_shapes["question_embedding"],), name="question_input")
        triples_input = layers.Input(shape=(self.action_size, self.state_shapes["padded_triples"]), name="triples_input")
        action_mask_input = layers.Input(shape=(self.action_size,), name="action_mask_input")
        other_features_input = layers.Input(shape=(self.state_shapes["other_features"],), name="other_features_input")

        # Attention Mechanism to create a context-aware representation of the subgraph
        # The question embedding acts as the query
        attention_query = layers.Dense(64, activation='relu')(question_input)
        attention_query = layers.RepeatVector(self.action_size)(attention_query)

        # The triple embeddings act as the keys
        attention_keys = layers.Dense(64, activation='relu')(triples_input)

        # Calculate attention scores
        multiplied = layers.Multiply()([attention_query, attention_keys])
        attention_scores = layers.Lambda(
            lambda x: keras.ops.sum(x, axis=-1),
            output_shape=(self.action_size,)
        )(multiplied)

        # Apply the action mask to ignore padded, non-existent triples
        masking_layer = layers.Lambda(lambda x: (1 - x) * -1e9)(action_mask_input)
        attention_scores = layers.Add()([attention_scores, masking_layer])

        # Normalize scores to get weights
        attention_weights = layers.Activation('softmax')(attention_scores)

        # Compute the context vector by taking a weighted average of triple embeddings
        context_vector = layers.Dot(axes=[1, 1])([attention_weights, triples_input])

        # Combine the context vector with the original question and other features
        combined_features = layers.Concatenate()([context_vector, question_input, other_features_input])

        # Final Dense Layers to predict Q-values for each action
        dense1 = layers.Dense(128, activation='relu')(combined_features)
        dense2 = layers.Dense(64, activation='relu')(dense1)
        output_q_values = layers.Dense(self.action_size, activation='linear', name='q_values')(dense2)

        # Build and compile the Model
        model = Model(
            inputs=[question_input, triples_input, action_mask_input, other_features_input],
            outputs=output_q_values
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        model.summary()
        return model


    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in the replay memory.

        An experience consists of the state, action taken, reward received,
        the resulting next state, and a boolean indicating if the episode ended.

        Args:
            state (dict): The state from which the action was taken.
            action (int): The action that was taken.
            reward (float): The reward received for the action.
            next_state (dict): The state transitioned to after the action.
            done (bool): True if the episode terminated, otherwise False.
        """
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))


    # def act(self, state, similarity_scores):
    #     """
    #         Chooses an action based on the current state using an epsilon-greedy strategy.
    #
    #         With probability epsilon, it chooses a random valid action (exploration).
    #         Otherwise, it uses the neural network to predict the best action (exploitation),
    #         using similarity scores as a heuristic bonus.
    #
    #         Args:
    #             state (dict): The current state of the environment.
    #             similarity_scores (np.ndarray): Pre-calculated similarity scores to use
    #                                             as a heuristic bonus.
    #
    #         Returns:
    #             int: The index of the chosen action.
    #     """
    #     action_mask = state["action_mask_input"]
    #     valid_actions_indices = np.where(action_mask == 1.0)[0]
    #
    #     if len(valid_actions_indices) == 0:
    #         return 0
    #
    #     # Exploration: Choose a random valid action
    #     if np.random.rand() <= self.epsilon:
    #         return np.random.choice(valid_actions_indices)
    #
    #     # Exploitation: Predict the best action using the model
    #     # Add a batch dimension to the state for the model's predict method
    #     state_for_pred = {key: np.expand_dims(val, axis=0) for key, val in state.items()}
    #
    #     q_values = self.model.predict(state_for_pred, verbose=0)[0]
    #
    #     # Use similarity scores as a heuristic bonus to guide the decision
    #     bonus = np.zeros_like(q_values)
    #     if similarity_scores.size > 0:
    #         num_valid = len(similarity_scores)
    #         bonus[:num_valid] = 0.3 * similarity_scores
    #
    #     combined_values = q_values + bonus
    #
    #     # Mask out invalid actions by setting their value to a very low number
    #     combined_values[action_mask == 0.0] = -np.inf
    #
    #     return int(np.argmax(combined_values))

    def act(self, state):
        """Choose an action using an epsilon-greedy strategy."""
        action_mask = state["action_mask_input"]
        valid_actions_indices = np.where(action_mask == 1.0)[0]

        if len(valid_actions_indices) == 0:
            return 0

        # Exploration
        if np.random.rand() <= self.epsilon:
            return np.random.choice(valid_actions_indices)

        # Exploitation
        state_for_pred = {key: np.expand_dims(val, axis=0) for key, val in state.items()}
        q_values = self.model.predict(state_for_pred, verbose=0)[0]

        # Mask out invalid actions
        q_values[action_mask == 0.0] = -np.inf

        return int(np.argmax(q_values))


    def replay(self):
        """
        Trains the neural network using a random minibatch of experiences from memory.

        This method implements the core learning step of the DQN algorithm. It uses
        the Bellman equation to calculate the target Q-values and trains the model
        to predict these targets.
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        # Unzip and prepare batch data
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Batch states and next_states dictionaries
        def batch_states_dict(states_list):
            batch = {}
            for key in states_list[0].keys():
                batch[key] = np.array([s[key] for s in states_list])
            return batch

        states_batch = batch_states_dict(states)
        next_states_batch = batch_states_dict(next_states)

        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Predict Q-values for current and next states
        target_qs = self.model.predict(states_batch, verbose=0)
        next_qs = self.model.predict(next_states_batch, verbose=0)

        # Mask invalid actions in the next states before finding the max Q-value
        next_action_masks = next_states_batch["action_mask_input"]
        next_qs[next_action_masks == 0.0] = -np.inf

        # Calculate the target Q-value using the Bellman equation
        target_qs[np.arange(self.batch_size), actions] = rewards + (self.gamma * np.max(next_qs, axis=1) * (1 - dones))

        # Train the model on the calculated targets
        self.model.fit(states_batch, target_qs, epochs=1, verbose=0, batch_size=self.batch_size)

        # Decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        """Save the trained model."""
        self.model.save(self.model_path)
        print(f"[INFO] Model saved at {self.model_path}")

    def load_model(self):
        """Load a trained model from a file."""
        if os.path.exists(self.model_path):
            # Add safe_mode=False to allow deserialization of Lambda layers
            self.model = keras.models.load_model(self.model_path, safe_mode=False)
            print(f"[INFO] Loaded model from {self.model_path}")
        else:
            print(f"[WARNING] Model file {self.model_path} not found. Using a new model.")

    def save_experience(self):
        """Save the replay memory to a file."""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, "wb") as f:
            pickle.dump(self.memory, f)

    def load_experience(self):
        """Load replay memory from a file, if it exists."""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        try:
            with open(self.memory_file, "rb") as f:
                self.memory = pickle.load(f)
            print(f"Loaded {len(self.memory)} past experiences from {self.memory_file}.")
        except FileNotFoundError:
            print("No previous experience found. Starting with an empty memory.")