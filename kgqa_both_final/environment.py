import gym
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

class KGQARLEnvironment(gym.Env):
    def __init__(
        self,
        question_text,
        question_embedding,
        answer_embedding,
        subgraph_embeddings,
        ground_truth_answer,
        attack_mode="remove",
        add_mode="semantic"
    ):
        super(KGQARLEnvironment, self).__init__()
        self.question_text = question_text
        self.question_embedding = question_embedding
        self.answer_embedding = answer_embedding
        self.ground_truth_answer = ground_truth_answer
        self.subgraph_data = subgraph_embeddings
        self.attack_mode = attack_mode
        self.add_mode = add_mode

        # Extract embeddings and URIs from subgraph data.
        self.head_embeddings = np.array([t[0] for t in self.subgraph_data])
        self.rel_embeddings = np.array([t[1] for t in self.subgraph_data])
        self.tail_embeddings = np.array([t[2] for t in self.subgraph_data])
        self.uris = [(t[3], t[4], t[5]) for t in self.subgraph_data]

        if len(self.head_embeddings) == 0:
            raise ValueError("Subgraph head embeddings are empty! Check input data.")

        # Define observation space.
        obs_dim = (
            len(self.question_embedding)
            + len(self.head_embeddings[0])
            + len(self.rel_embeddings[0])
            + len(self.tail_embeddings[0])
            + 2  # extra features: support flag and fraction remaining.
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(len(self.head_embeddings))

        # Initialize support info for the ground truth answer.
        self.ans_uri = self.ground_truth_answer.strip("<>")
        self.original_true_support_count = sum(
            1 for (h, _, t) in self.uris if self.ans_uri in [h.strip("<>"), t.strip("<>")]
        )
        self.true_support_threshold = self.original_true_support_count * 0.9

        self.reset()

    def reset(self):
        """Reset the environment state."""
        self.current_head_embeddings = self.head_embeddings.copy()
        self.current_rel_embeddings = self.rel_embeddings.copy()
        self.current_tail_embeddings = self.tail_embeddings.copy()
        self.current_uris = self.uris.copy()
        self.removed_triples = []
        self.added_triples = []
        self.current_answer = self.ground_truth_answer
        return self._get_state()

    def _get_state(self):
        all_triple_embeddings = np.concatenate(
            [self.current_head_embeddings, self.current_rel_embeddings, self.current_tail_embeddings],
            axis=1,
        )
        question_embedding = np.asarray(self.question_embedding).flatten()
        triple_embeddings_flat = all_triple_embeddings.flatten()

        fixed_state_size = 18050  # Base size before extra features.
        state = np.concatenate([question_embedding, triple_embeddings_flat])
        if len(state) < fixed_state_size:
            padding = np.zeros(fixed_state_size - len(state))
            state = np.concatenate([state, padding])
        else:
            state = state[:fixed_state_size]

        contains_true_answer = 1.0 if self._has_sufficient_support() else 0.0
        fraction_remaining = len(self.current_head_embeddings) / len(self.head_embeddings)
        state = np.concatenate([state, [contains_true_answer, fraction_remaining]])
        return state

    def compute_influence_scores(self):
        """Compute influence scores for each triple in the current subgraph."""
        if len(self.current_head_embeddings) == 0:
            return np.array([])

        ans_uri = self.ground_truth_answer.strip("<>")
        head_sims = cosine_similarity([self.answer_embedding], self.current_head_embeddings)[0]
        rel_sims = cosine_similarity([self.answer_embedding], self.current_rel_embeddings)[0]
        tail_sims = cosine_similarity([self.answer_embedding], self.current_tail_embeddings)[0]

        triple_similarities = (0.35 * head_sims + 0.3 * rel_sims + 0.35 * tail_sims)
        for i, (h_uri, _, t_uri) in enumerate(self.current_uris):
            if ans_uri in [h_uri.strip("<>"), t_uri.strip("<>")]:
                triple_similarities[i] += 0.2

        print("Influence Scores Debugging:")
        for idx, score in enumerate(triple_similarities):
            print(f"Triple: {self.current_uris[idx]} -> Score: {score}")
        return triple_similarities

    def step(self, action):
        if self.attack_mode == "remove":
            return self._remove_step(action)
        elif self.attack_mode == "add":
            return self._add_step(action)
        else:
            raise ValueError(f"Invalid attack mode: {self.attack_mode}")

    def _remove_step(self, action):
        """
        Remove the triple corresponding to the action and check if the answer changes.
        Stop removing triples once a false answer is generated.
        """
        if action in self.removed_triples or action < 0 or action >= len(self.current_head_embeddings):
            return self._get_state(), -1.0, False, {}

        self.removed_triples.append(action)
        removed_triple = self.current_uris[action]

        self.current_head_embeddings = np.delete(self.current_head_embeddings, action, axis=0)
        self.current_rel_embeddings = np.delete(self.current_rel_embeddings, action, axis=0)
        self.current_tail_embeddings = np.delete(self.current_tail_embeddings, action, axis=0)
        self.current_uris.pop(action)

        # Check if false answer is generated
        answer_changed, false_answer = self._simulate_qa_system()

        if false_answer:  # If we got a false answer, stop removing triples
            print(f"[DEBUG] False Answer Found: {false_answer}. Stopping further removals.")
            return self._get_state(), 5.0, True, {"false_answer": false_answer}

        # Check stopping conditions
        done = len(self.removed_triples) >= 3 or len(self.current_head_embeddings) == 0

        # Apply reward logic
        if answer_changed:
            reward = 5.0 - (0.5 * len(self.removed_triples))
        elif not self.is_answer_still_in_subgraph():
            reward = -1.0
        else:
            reward = -0.3

        return self._get_state(), reward, done, {"false_answer": false_answer}

    def _add_step(self, action):
        if action < 0 or action >= len(self.head_embeddings):
            return self._get_state(), -1.0, False, {}

        # Get the original triple at the candidate index.
        h, r, t = self.uris[action]

        if self.add_mode == "semantic":
            # Semantic mode: use a symmetric addition (flipping subject and object)
            attack_triple = (t, r, h)
            if attack_triple in self.current_uris or attack_triple in self.added_triples:
                return self._get_state(), -0.5, False, {}
            self.added_triples.append(attack_triple)
            # For embedding update, swap head and tail embeddings:
            h_emb = self.tail_embeddings[action]
            r_emb = self.rel_embeddings[action]
            t_emb = self.head_embeddings[action]

        elif self.add_mode == "cairage":
            # CAIRAGE mode: attack by keeping one entity fixed.
            # Select a candidate triple from the subgraph.
            candidate_idx = random.randint(0, len(self.uris) - 1)
            candidate_uri = self.uris[candidate_idx]
            candidate_head_emb = self.head_embeddings[candidate_idx]
            candidate_rel_emb = self.rel_embeddings[candidate_idx]
            candidate_tail_emb = self.tail_embeddings[candidate_idx]

            # Decide which side to keep fixed.
            if random.random() < 0.5:
                # Keep subject fixed: use original subject 'h'
                # and use candidate's relation and object.
                attack_triple = (h, candidate_uri[1], candidate_uri[2])
                new_h_emb = self.head_embeddings[action]  # original subject embedding
                new_r_emb = candidate_rel_emb  # candidate relation embedding
                new_t_emb = candidate_tail_emb  # candidate tail embedding
            else:
                # Keep object fixed: use original object 't'
                # and use candidate's subject and relation.
                attack_triple = (candidate_uri[0], candidate_uri[1], t)
                new_h_emb = candidate_head_emb  # candidate subject embedding
                new_r_emb = candidate_rel_emb  # candidate relation embedding
                new_t_emb = self.tail_embeddings[action]  # original object embedding

            if attack_triple in self.current_uris or attack_triple in self.added_triples:
                return self._get_state(), -0.5, False, {}

            self.added_triples.append(attack_triple)
            h_emb = new_h_emb
            r_emb = new_r_emb
            t_emb = new_t_emb

        else:
            raise ValueError(f"Invalid add_mode: {self.add_mode}")

        # Update the current subgraph: append the new triple and its embeddings.
        self.current_head_embeddings = np.append(self.current_head_embeddings, [h_emb], axis=0)
        self.current_rel_embeddings = np.append(self.current_rel_embeddings, [r_emb], axis=0)
        self.current_tail_embeddings = np.append(self.current_tail_embeddings, [t_emb], axis=0)
        self.current_uris.append(attack_triple)

        answer_changed, false_answer = self._simulate_qa_system()

        if false_answer:
            print(f"[DEBUG] False Answer Found (Addition): {false_answer}")
            return self._get_state(), 5.0, True, {"false_answer": false_answer}

        done = len(self.added_triples) >= 3
        reward = 5.0 if answer_changed else -0.3

        return self._get_state(), reward, done, {"false_answer": false_answer}

    def is_answer_still_in_subgraph(self):
        ans_uri = self.ground_truth_answer.strip("<>")
        for (h_uri, _, t_uri) in self.current_uris:
            if ans_uri == h_uri.strip("<>") or ans_uri == t_uri.strip("<>"):
                return True
        return False

    def _has_sufficient_support(self):
        current_support_count = sum(
            1 for (h, _, t) in self.current_uris if self.ans_uri in [h.strip("<>"), t.strip("<>")]
        )
        return current_support_count >= self.true_support_threshold

    def _simulate_qa_system(self):
        false_answer = self._infer_answer_from_subgraph()

        if false_answer is None:
            remaining_entities = [
                t_uri for (_, _, t_uri) in self.current_uris
                if t_uri.strip("<>") != self.ans_uri  # Exclude the true answer
            ]
            false_answer = random.choice(
                remaining_entities) if remaining_entities else "http://dbpedia.org/resource/Unknown"

        self.current_answer = false_answer

        print(f"[DEBUG] Generated False Answer: {false_answer}")

        return True, false_answer

    def _infer_answer_from_subgraph(self):
        """
        Infer a new answer based on the most relevant remaining entities.
        Instead of random selection, it selects an entity that is:
        1. Most semantically similar to the original answer (misleading but plausible).
        2. Frequently occurring in the remaining triples.
        3. Appears in similar relational contexts as the original answer.
        """
        if len(self.current_head_embeddings) == 0:
            return None

        influence_scores = self.compute_influence_scores()
        sorted_indices = np.argsort(influence_scores)[::-1]

        candidate_triples = []
        for idx in sorted_indices:
            h_uri, _, t_uri = self.current_uris[idx]
            h_clean = h_uri.strip("<>")
            t_clean = t_uri.strip("<>")

            # Exclude the true answer but select potential false answers
            if self.ans_uri not in [h_clean, t_clean]:
                candidate_triples.append((h_clean, t_clean))

        if candidate_triples:
            # Rank candidates by frequency of occurrence
            entity_counts = {}
            for h, t in candidate_triples:
                entity_counts[h] = entity_counts.get(h, 0) + 1
                entity_counts[t] = entity_counts.get(t, 0) + 1

            # Select the entity that appears the most (likely more relevant)
            sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
            best_false_answer = sorted_entities[0][0] if sorted_entities else None
            return best_false_answer
        else:
            return None


    def is_valid_entity(self, answer_uri):
        """Placeholder for entity validation."""
        return True

