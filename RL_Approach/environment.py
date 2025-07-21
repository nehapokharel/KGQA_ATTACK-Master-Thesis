import gym
import numpy as np
import random
import networkx as nx
from SPARQLWrapper import SPARQLWrapper, JSON
from utils import execute_sparql_update, format_triple_for_sparql


class KGQARLEnvironment(gym.Env):
    """
        A Reinforcement Learning Environment for attacking Knowledge Graph Question Answering.

        This environment simulates  an adversarial setting where an RL agent learns to
        modify a knowledge graph (KG) subgraph to make a QA system produce an incorrect
        answer. The agent's actions consist of selecting triples to either remove from
        the subgraph or use as a basis for adding new, potentially misleading triples.

        The state is composed of the question embedding and the embeddings of the triples
        in the current subgraph.

        The reward is determined by whether the attack successfully changes the answer to the
        original question when queried against a live SPARQL endpoint.

        Attributes:
            question_text (str): The natural language question.
            question_embedding (np.ndarray): The embedding of the question.
            answer_embedding (np.ndarray): The embedding of the ground truth answer.
            subgraph_data (list): A list of tuples, each containing embeddings and URIs for a triple.
            attack_mode (str): The attack strategy, either "remove" or "add".
            sparql_endpoint (str): The SPARQL endpoint for querying.
            sparql_update_endpoint (str): The SPARQL endpoint for INSERT/DELETE operations.
            action_space (gym.spaces.Discrete): The space of possible actions (triple indices).
            observation_space (gym.spaces.Dict): The structure of the state representation.
    """
    def __init__(self, question_text, question_embedding, answer_embedding,
                 subgraph_embeddings, ground_truth_answer,
                 expected_entity_dim: int,
                 expected_relation_dim: int,
                 max_triples: int,
                 attack_mode="remove", add_mode="semantic",
                 support_threshold_factor=0.5,
                 sparql_endpoint=None, sparql_update_endpoint=None, sparql_query_gt=None):
        """
            Initializes the KGQARLEnvironment.

            Args:
                question_text (str): The natural language text of the question.
                question_embedding (np.ndarray): The vector embedding of the question.
                answer_embedding (np.ndarray): The vector embedding of the correct answer.
                subgraph_embeddings (list): A list of tuples, where each tuple contains
                                            (h_emb, r_emb, t_emb, h_uri, r_uri, t_uri) for a triple.
                ground_truth_answer (str): The URI of the correct answer.
                expected_entity_dim (int): The expected dimension of entity embeddings.
                expected_relation_dim (int): The expected dimension of relation embeddings.
                max_triples (int): The maximum number of triples the environment can handle (for padding).
                attack_mode (str): The type of attack, either "remove" or "add".
                add_mode (str): If attack_mode is "add", the strategy for generating new triples.
                support_threshold_factor (float): A factor to determine the minimum required support for the true answer.
                sparql_endpoint (str): The URL of the SPARQL query endpoint.
                sparql_update_endpoint (str): The URL of the SPARQL update endpoint.
                sparql_query_gt (str): The ground truth SPARQL query.
        """
        super(KGQARLEnvironment, self).__init__()
        self.question_text = question_text
        self.question_embedding = np.asarray(question_embedding, dtype=np.float32)
        self.answer_embedding = np.asarray(answer_embedding, dtype=np.float32)
        self.ground_truth_answer = ground_truth_answer
        self.subgraph_data = subgraph_embeddings
        self.attack_mode = attack_mode
        self.add_mode = add_mode
        self.support_threshold_factor = support_threshold_factor
        self.sparql_endpoint = sparql_endpoint
        self.sparql_query = sparql_query_gt
        self.sparql_update_endpoint = sparql_update_endpoint

        # Dimensionality setup
        self.entity_embedding_dim = expected_entity_dim
        self.relation_embedding_dim = expected_relation_dim
        self.question_embedding_dim = self.question_embedding.shape[0]
        self.max_triples = max_triples
        self.triple_embedding_dim = self.entity_embedding_dim + self.relation_embedding_dim + self.entity_embedding_dim

        # Process and validate initial subgraph embeddings
        processed_heads, processed_rels, processed_tails, processed_uris = [], [], [], []

        if not subgraph_embeddings:
            raise ValueError("Initial subgraph_embeddings is empty! Cannot initialize environment.")

        for i, triple_data_tuple in enumerate(subgraph_embeddings):
            if len(triple_data_tuple) < 6: continue
            h_emb, r_emb, t_emb, h_uri, r_uri, t_uri = triple_data_tuple
            _, h_emb = self._validate_embedding(h_emb, self.entity_embedding_dim)
            _, r_emb = self._validate_embedding(r_emb, self.relation_embedding_dim)
            _, t_emb = self._validate_embedding(t_emb, self.entity_embedding_dim)
            processed_heads.append(h_emb)
            processed_rels.append(r_emb)
            processed_tails.append(t_emb)
            processed_uris.append((str(h_uri).strip("<>"), str(r_uri).strip("<>"), str(t_uri).strip("<>")))

        if not processed_uris:
            raise ValueError(f"Q: {self.question_text[:20]} - No processable triples left after validation.")

        # Store original, validated subgraph data
        self.head_embeddings_orig = np.array(processed_heads, dtype=np.float32)
        self.rel_embeddings_orig = np.array(processed_rels, dtype=np.float32)
        self.tail_embeddings_orig = np.array(processed_tails, dtype=np.float32)
        self.uris_orig = processed_uris

        # Define action space
        self.action_space = gym.spaces.Discrete(self.max_triples)

        # Calculate support threshold for the ground truth answer
        self.ans_uri = self.ground_truth_answer.strip("<>")
        self.original_true_support_count = sum(1 for (h, _, t) in self.uris_orig if self.ans_uri in [h, t])
        self.true_support_threshold = self.original_true_support_count * self.support_threshold_factor

        self.reset()

    def _validate_embedding(self, emb, expected_dim):
        """Validates the shape and type of an embedding, returning a zero vector on failure."""
        is_valid = isinstance(emb, np.ndarray) and emb.shape == (expected_dim,) and emb.dtype == np.float32
        if not is_valid:
            return False, np.zeros(expected_dim, dtype=np.float32)
        return True, emb

    def reset(self):
        """
            Resets the environment to its initial state.

            This involves restoring the original subgraph, clearing any modifications,
            and resetting modification counters.

            Returns:
                dict: The initial state of the environment.
        """
        self.current_head_embeddings = self.head_embeddings_orig.copy()
        self.current_rel_embeddings = self.rel_embeddings_orig.copy()
        self.current_tail_embeddings = self.tail_embeddings_orig.copy()
        self.current_uris = list(self.uris_orig)
        self.removed_triples = []
        self.added_triples_info = []

        self.modifications_made = 0
        self.max_modifications = 0
        return self.get_state()

    def get_action_mask(self):
        """
            Generates a mask to indicate valid actions.

            An action is valid if its index points to an existing triple in the current subgraph.

            Returns:
                np.ndarray: A binary array where 1.0 indicates a valid action and 0.0 an invalid one.
        """
        mask = np.zeros(self.max_triples, dtype=np.float32)
        mask[:len(self.current_uris)] = 1.0
        return mask

    def get_state(self):
        """
            Constructs the current state representation for the agent.

            The state includes the question embedding, the padded embeddings of the
            current triples, an action mask, and other features like answer support.

            Returns:
                dict: A dictionary representing the complete state of the environment.
        """
        num_current_triples = len(self.current_uris)
        if num_current_triples > 0:
            concatenated_triples = np.concatenate(
                [self.current_head_embeddings, self.current_rel_embeddings, self.current_tail_embeddings], axis=1)
        else:
            concatenated_triples = np.empty((0, self.triple_embedding_dim), dtype=np.float32)

        # Pad triples to a fixed size for the neural network input
        padded_triples = np.zeros((self.max_triples, self.triple_embedding_dim), dtype=np.float32)
        if num_current_triples > 0:
            padded_triples[:num_current_triples, :] = concatenated_triples

        # Additional features for the state
        has_support = self._has_sufficient_support()
        contains_true_answer_feature = np.array([1.0 if has_support else 0.0], dtype=np.float32)
        fraction_remaining_val = len(self.current_uris) / len(self.uris_orig) if self.uris_orig else 0.0
        fraction_remaining_feature = np.array([fraction_remaining_val], dtype=np.float32)

        return {
            "question_input": self.question_embedding,
            "triples_input": padded_triples,
            "action_mask_input": self.get_action_mask(),
            "other_features_input": np.concatenate([contains_true_answer_feature, fraction_remaining_feature])
        }

    def _has_sufficient_support(self):
        """Checks if the ground truth answer still has enough supporting triples."""
        if not self.uris_orig: return False
        current_support_count = sum(1 for (h, _, t) in self.current_uris if self.ans_uri in [h, t])
        return current_support_count >= self.true_support_threshold

    def set_attack_limit(self, k):
        """
            Sets the maximum number of modifications (attack budget) for the current episode.

            Args:
                k (int): The number of modifications allowed.
        """
        self.max_modifications = k

    def step(self, action):
        """
            Executes one step in the environment based on the agent's action.

            This involves applying the modification (add/remove), calculating the
            intermediate reward, and determining if the episode is done.

            Args:
                action (int): The index of the triple to act upon.

            Returns:
                tuple: A tuple containing (next_state, reward, done, info).
        """
        if not (0 <= action < len(self.current_uris)):
            return self.get_state(), -1.0, True, {"reason": f"Invalid action index {action}."}

        info = {}
        if self.attack_mode == "remove":
            removed_uri = self.current_uris.pop(action)
            self.removed_triples.append(removed_uri)
            self.current_head_embeddings = np.delete(self.current_head_embeddings, action, axis=0)
            self.current_rel_embeddings = np.delete(self.current_rel_embeddings, action, axis=0)
            self.current_tail_embeddings = np.delete(self.current_tail_embeddings, action, axis=0)

        elif self.attack_mode == "add":
            new_triple_info = self._generate_triple_for_add(action)
            if new_triple_info is None:
                return self.get_state(), -0.5, False, {"reason": "Attempted to add duplicate triple"}
            self.added_triples_info.append(new_triple_info)
            self.current_uris.append(new_triple_info['uri'])
            self.current_head_embeddings = np.vstack([self.current_head_embeddings, new_triple_info['h_emb']])
            self.current_rel_embeddings = np.vstack([self.current_rel_embeddings, new_triple_info['r_emb']])
            self.current_tail_embeddings = np.vstack([self.current_tail_embeddings, new_triple_info['t_emb']])

        self.modifications_made += 1

        done = (self.modifications_made >= self.max_modifications)

        if not done:
            reward = -0.1
        else:
            reward, info = self._evaluate_final_attack()

        return self.get_state(), reward, done, info

    def _evaluate_final_attack(self):
        """
            Performs the final evaluation of the attack.

            This method applies the accumulated changes to the live SPARQL endpoint,
            runs the QA query, and checks if the answer has changed. It also ensures
            that all changes are reverted afterwards to leave the database clean.

            Returns:
                tuple: A tuple containing (final_reward, info_dict).
        """
        triples_to_add = [item['uri'] for item in self.added_triples_info]
        triples_to_remove = self.removed_triples

        try:
            for triple_uri in triples_to_remove:
                execute_sparql_update(self.sparql_update_endpoint,
                                      f"DELETE DATA {{ {format_triple_for_sparql(triple_uri)} }}")
            for triple_uri in triples_to_add:
                execute_sparql_update(self.sparql_update_endpoint,
                                      f"INSERT DATA {{ {format_triple_for_sparql(triple_uri)} }}")

            predicted_answer = self._run_sparql_and_get_top_answer()
            answer_changed = (predicted_answer != self.ground_truth_answer)

            reward = 5.0 if answer_changed else -0.3
            info = {
                "predicted_answer": predicted_answer,
                "answer_changed_this_step": answer_changed
            }

        except Exception as e:
            print(f"ERROR during final evaluation: {e}")
            reward = -10.0
            info = {"reason": "Exception during final DB evaluation."}

        finally:
            for triple_uri in triples_to_remove:
                execute_sparql_update(self.sparql_update_endpoint,
                                      f"INSERT DATA {{ {format_triple_for_sparql(triple_uri)} }}")
            for triple_uri in triples_to_add:
                execute_sparql_update(self.sparql_update_endpoint,
                                      f"DELETE DATA {{ {format_triple_for_sparql(triple_uri)} }}")

        return reward, info

    def _generate_triple_for_add(self, action_idx_for_basis):
        """
            Generates a new triple to add based on a basis triple.

            Args:
                action_idx_for_basis (int): The index of the triple to use as a basis.

            Returns:
                dict or None: A dictionary with the new triple's info, or None if a duplicate is generated.
        """
        h_basis_uri, r_basis_uri, t_basis_uri = self.current_uris[action_idx_for_basis]
        h_basis_emb = self.current_head_embeddings[action_idx_for_basis]
        r_basis_emb = self.current_rel_embeddings[action_idx_for_basis]
        t_basis_emb = self.current_tail_embeddings[action_idx_for_basis]

        if self.add_mode == "semantic":
            new_h_uri, new_r_uri, new_t_uri = t_basis_uri, r_basis_uri, h_basis_uri
            new_h_emb, new_r_emb, new_t_emb = t_basis_emb.copy(), r_basis_emb.copy(), h_basis_emb.copy()
        elif self.add_mode == "cairage":
            if not self.uris_orig: return None
            candidate_idx = random.randint(0, len(self.uris_orig) - 1)
            cand_h_uri, cand_r_uri, cand_t_uri = self.uris_orig[candidate_idx]
            cand_h_emb = self.head_embeddings_orig[candidate_idx]
            cand_r_emb = self.rel_embeddings_orig[candidate_idx]
            cand_t_emb = self.tail_embeddings_orig[candidate_idx]
            if random.random() < 0.5:
                new_h_uri, new_r_uri, new_t_uri = h_basis_uri, cand_r_uri, cand_t_uri
                new_h_emb, new_r_emb, new_t_emb = h_basis_emb.copy(), cand_r_emb.copy(), cand_t_emb.copy()
            else:
                new_h_uri, new_r_uri, new_t_uri = cand_h_uri, cand_r_uri, t_basis_uri
                new_h_emb, new_r_emb, new_t_emb = cand_h_emb.copy(), cand_r_emb.copy(), t_basis_emb.copy()
        else:
            raise ValueError(f"Invalid add_mode: {self.add_mode}")

        new_triple_uri_tuple = (new_h_uri, new_r_uri, new_t_uri)
        if new_triple_uri_tuple in self.current_uris:
            return None

        return {
            'uri': new_triple_uri_tuple,
            'h_emb': new_h_emb.reshape(1, -1),
            'r_emb': new_r_emb.reshape(1, -1),
            't_emb': new_t_emb.reshape(1, -1)
        }

    def _run_sparql_and_get_top_answer(self):
        """
            Executes the ground truth SPARQL query and returns the top answer.

            Returns:
                str or None: The URI of the top answer, "NO_ANSWER_FOUND", or None on error.
        """
        if not self.sparql_query or not self.sparql_endpoint: return None
        sparql = SPARQLWrapper(self.sparql_endpoint)
        sparql.setQuery(self.sparql_query)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
            if not bindings: return "NO_ANSWER_FOUND"
            var = results.get("head", {}).get("vars", [])[0]
            return bindings[0][var]["value"]
        except Exception:
            return None

    def export_graph_centralities(self):
        """
            Calculates and exports centrality measures for the nodes in the current subgraph.

            Uses networkx to compute betweenness, eigenvector, and PageRank centralities.
            This is useful for analyzing the structural properties of the subgraph.

            Returns:
                dict: A dictionary mapping node URIs to their centrality scores.
        """
        G = nx.DiGraph()
        if not self.current_uris:
            return {}

        unique_nodes = set()
        for (h, _, t) in self.current_uris:
            G.add_edge(h, t)
            unique_nodes.add(h)
            unique_nodes.add(t)

        if len(G) == 0:
            return {}

        centralities = {}
        try:
            if G.number_of_nodes() > 0:
                b = nx.betweenness_centrality(G, normalized=True)
                try:
                    e = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-04)
                except nx.PowerIterationFailedConvergence:
                    e = {node: 0.0 for node in G.nodes()}
                p = nx.pagerank(G, alpha=0.85)

                for node in unique_nodes:
                    centralities[node] = {
                        "betweenness": b.get(node, 0.0),
                        "eigenvector": e.get(node, 0.0),
                        "pagerank": p.get(node, 0.0)
                    }
            else:
                return {}
        except Exception as err:
            print(f"[Centrality Error] Q: {self.question_text[:20]} - Error: {err}")
            return {node: {"betweenness": 0.0, "eigenvector": 0.0, "pagerank": 0.0} for node in unique_nodes}
        return centralities