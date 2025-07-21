import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import gc
import time
import torch
import argparse
import random
import re
import numpy as np

from environment import KGQARLEnvironment
from rl_agent import KGQARLAgent

from utils import (
    load_embeddings_from_models, map_indices_to_embeddings,
    fetch_triples_via_sparql, embed_subgraph, encode,
    load_checkpoint, save_checkpoint, log_changes_to_file,
    get_answer_embedding,
    get_predicates_from_query
)

# The maximum number of triples to include in the subgraph for an attack.
MAX_SUBGRAPH_TRIPLES_LIMIT = 200
# A list of 'k' values, where 'k' is the budget (number of triples to modify) for an attack.
K_VALUES_LIST = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
# K_VALUES_LIST = [30,40]
# DEFAULT_COLLATERAL_SAMPLE_SIZE = 50


def run_attack_and_log(
        qidx,
        qald_data,
        entity_mapping,
        relation_mapping,
        persistent_agent,
        attack_mode,
        add_mode,
        output_dir,
        qa_cache,
        sparql_endpoint_url,
        sparql_update_endpoint_url,
        embedding_dim_entity,
        embedding_dim_relation,
        max_triples
):
    """
    Orchestrates and executes an adversarial attack for a single question.

    This function performs the entire pipeline for attacking one question:
    1.  Fetches question data (text, SPARQL query, ground truth answer).
    2.  Extracts a relevant subgraph from the knowledge graph around the question's entities.
    3.  Iterates through a predefined list of attack budgets (k-values).
    4.  For each budget, it initializes a Reinforcement Learning environment.
    5.  Uses an RL agent to select and apply modifications (add/remove triples) to the subgraph.
    6.  Logs the changes made to the graph for later evaluation.
    7.  Cleans up resources after processing.

    Args:
        qidx (int): The unique identifier for the question being processed.
        qald_data (dict): The loaded QALD dataset containing questions and queries.
        entity_mapping (dict): A mapping from entity URIs to their embedding vectors.
        relation_mapping (dict): A mapping from relation URIs to their embedding vectors.
        persistent_agent (KGQARLAgent): The RL agent used to choose attack actions.
        attack_mode (str): The type of attack, either 'add' or 'remove'.
        add_mode (str or None): The specific strategy for 'add' attacks (e.g., 'semantic').
        output_dir (str): The directory path to save logs and outputs.
        qa_cache (dict): A cache mapping question IDs to ground truth answers.
        sparql_endpoint_url (str): The URL of the SPARQL query endpoint.
        sparql_update_endpoint_url (str): The URL of the SPARQL update endpoint.
        embedding_dim_entity (int): The dimension of entity embeddings.
        embedding_dim_relation (int): The dimension of relation embeddings.
        max_triples (int): The maximum number of triples allowed in the state.
    """
    print(f"Starting processing for Question ID: {qidx}")
    q_data_list = qald_data.get('questions', [])
    if not (0 <= (qidx - 1) < len(q_data_list)):
        print(f"[Q{qidx}] ERROR: Question index out of bounds for qald_data. Skipping.")
        return

    # Data Retrieval and Validation
    q_data = q_data_list[qidx - 1]

    question_text_list = q_data.get('question', [])
    question_text = next((q['string'] for q in question_text_list if q.get('language') == 'en'), None)

    sparql_query_gt = q_data.get('query', {}).get('sparql')
    cached_q_info = qa_cache.get(qidx)
    if not cached_q_info:
        print(f"[Q{qidx}] ERROR: Question info not found in qa_cache. Skipping.")
        return

    # Extract the single ground truth answer
    raw_answer_list = cached_q_info.get("answer")
    ground_truth_answer = None
    if raw_answer_list and isinstance(raw_answer_list, list):
        first_answer_dict = raw_answer_list[0]
        if first_answer_dict and isinstance(first_answer_dict, dict) and list(first_answer_dict.values()):
            ground_truth_answer = list(first_answer_dict.values())[0]

    if not all([question_text, sparql_query_gt, ground_truth_answer]):
        print(f"[Q{qidx}] ERROR: Missing essential data (question text, SPARQL, or ground truth answer). Skipping.")
        print(
            f"  Text: {'Found' if question_text else 'Missing'}, SPARQL: {'Found' if sparql_query_gt else 'Missing'}, GT Answer: {'Found' if ground_truth_answer else 'Missing'}")
        return

    print(f"[Q{qidx}] Question: {question_text[:100]}...")
    print(f"[Q{qidx}] Ground Truth Answer: {ground_truth_answer}")

    question_embedding = encode(question_text)
    try:
        answer_embedding_np = get_answer_embedding(ground_truth_answer, entity_mapping)
    except ValueError as e:
        print(f"[Q{qidx}] ERROR: Failed to get answer embedding for '{ground_truth_answer}': {e}. Skipping question.")
        skip_log_path = os.path.join(output_dir, "skipped_questions_critical.log")
        with open(skip_log_path, "a", encoding="utf-8") as f_skip_crit:
            f_skip_crit.write(json.dumps({
                "question_id": qidx, "reason": str(e), "stage": "answer_embedding_retrieval"
            }) + "\n")
        return

    # Collect seed URIs from the answer and SPARQL query to build the subgraph
    seed_uris_for_subgraph = {ground_truth_answer}
    try:
        for match in re.finditer(r'res:([^\s\)\}\.\?\;\,]+)', sparql_query_gt):
            entity_uri = "http://dbpedia.org/resource/" + match.group(1)
            if entity_uri in entity_mapping:
                seed_uris_for_subgraph.add(entity_uri)
        for match in re.finditer(r'<http://dbpedia.org/resource/([^>\s]+)>', sparql_query_gt):
            entity_uri = "http://dbpedia.org/resource/" + match.group(1)
            if entity_uri in entity_mapping:
                seed_uris_for_subgraph.add(entity_uri)
    except Exception as e_parse:
        print(f"[Q{qidx}] WARNING: Could not parse additional entities from SPARQL query: {e_parse}. "
              "Proceeding with answer URI as primary seed.")

    final_seed_uris = list(seed_uris_for_subgraph)
    if not final_seed_uris:
        print(f"[Q{qidx}] ERROR: No valid seed URIs (answer or query entities with embeddings) found. Skipping.")
        return

    print(f"[Q{qidx}] Final seed URIs for subgraph (sample): {final_seed_uris[:5]} (Total: {len(final_seed_uris)})")

    query_predicates = get_predicates_from_query(sparql_query_gt)
    if query_predicates:
        print(f"[Q{qidx}] Extracted {len(query_predicates)} predicates. Using focused subgraph fetch.")
    else:
        print(f"[Q{qidx}] No predicates found in query. Using generic 1-hop subgraph fetch.")

    subgraph_triples_raw, _ = fetch_triples_via_sparql(
        final_seed_uris,
        sparql_endpoint_url,
        predicates=query_predicates,
        limit=MAX_SUBGRAPH_TRIPLES_LIMIT
    )

    if len(subgraph_triples_raw) > MAX_SUBGRAPH_TRIPLES_LIMIT:
        print(
            f"[Q{qidx}] WARNING: Fetched {len(subgraph_triples_raw)} triples, which is more than the limit of {MAX_SUBGRAPH_TRIPLES_LIMIT}.")
        print(f"[Q{qidx}] Shuffling and truncating list to {MAX_SUBGRAPH_TRIPLES_LIMIT} triples.")
        random.shuffle(subgraph_triples_raw)
        subgraph_triples_raw = subgraph_triples_raw[:MAX_SUBGRAPH_TRIPLES_LIMIT]

    if not subgraph_triples_raw:
        print(f"[Q{qidx}] ERROR: No subgraph triples fetched from SPARQL for the seed URIs. Skipping question.")
        skip_log_path = os.path.join(output_dir, "skipped_questions.log")
        with open(skip_log_path, "a", encoding="utf-8") as f_skip:
            f_skip.write(json.dumps({"question_id": qidx, "reason": "No subgraph triples fetched from SPARQL."}) + "\n")
        return

    initial_subgraph_embeddings, missing_stats = embed_subgraph(subgraph_triples_raw, entity_mapping, relation_mapping)

    if not initial_subgraph_embeddings:
        print(f"[Q{qidx}] ERROR: No valid subgraph embeddings after processing raw triples. Skipping question.")
        skip_log_path = os.path.join(output_dir, "skipped_questions.log")
        with open(skip_log_path, "a", encoding="utf-8") as f_skip:
            f_skip.write(json.dumps({"question_id": qidx, "reason": "No valid subgraph embeddings."}) + "\n")

        embedding_missing_log_path = os.path.join(output_dir, "embedding_missing.log")
        log_data = {
            "question_id": qidx,
            "question_text": question_text,
            "reason": "The question was skipped because none of the fetched raw triples could be fully embedded.",
            "embedding_statistics": missing_stats
        }
        with open(embedding_missing_log_path, "a", encoding="utf-8") as f_embedding_missing:
            f_embedding_missing.write(json.dumps(log_data, indent=2) + "\n")
        return

    print(f"[Q{qidx}] Initial subgraph contains {len(initial_subgraph_embeddings)} embedded triples.")

    # Iterative Attack Loop
    for k_budget in K_VALUES_LIST:
        print(f"\n[Q{qidx}] Running attack: k_budget = {k_budget}, attack_mode = {attack_mode}"
              f"{', add_mode = ' + add_mode if add_mode else ''}")

        current_subgraph_embeddings_for_k = [
            (h_emb.copy(), r_emb.copy(), t_emb.copy(), h_uri, r_uri, t_uri)
            for h_emb, r_emb, t_emb, h_uri, r_uri, t_uri in initial_subgraph_embeddings
        ]

        if not current_subgraph_embeddings_for_k:
            msg = f"Subgraph embeddings list is empty before attack (k_budget={k_budget})."
            print(f"[Q{qidx}] {msg} Skipping this k_budget.")
            skip_log_path = os.path.join(output_dir, "skipped_questions.log")
            with open(skip_log_path, "a", encoding="utf-8") as f_skip:
                f_skip.write(json.dumps({"question_id": qidx, "k": k_budget, "reason": msg}) + "\n")
            continue

        # Environment Setup
        try:
            env = KGQARLEnvironment(
                question_text=question_text,
                question_embedding=question_embedding,
                answer_embedding=answer_embedding_np,
                subgraph_embeddings=current_subgraph_embeddings_for_k,
                ground_truth_answer=ground_truth_answer,
                expected_entity_dim=embedding_dim_entity,
                expected_relation_dim=embedding_dim_relation,
                max_triples=max_triples,
                attack_mode=attack_mode,
                add_mode=add_mode,
                sparql_endpoint=sparql_endpoint_url,
                sparql_update_endpoint=sparql_update_endpoint_url,
                sparql_query_gt=sparql_query_gt
            )
        except ValueError as ve:
            reason = f"Failed to initialize environment for k={k_budget}: {ve}"
            print(f"[Q{qidx}] ERROR: {reason}. Skipping this k_budget.")

            skip_log_path = os.path.join(output_dir, "skipped_questions.log")
            with open(skip_log_path, "a", encoding="utf-8") as f_skip:
                f_skip.write(json.dumps({
                    "question_id": qidx,
                    "k": k_budget,
                    "reason": f"KGQARLEnvironment Init Failed: {ve}"
                }) + "\n")
            continue

        actual_attack_steps = min(len(env.uris_orig), k_budget) if attack_mode == "remove" else k_budget
        if actual_attack_steps == 0 and k_budget > 0:
            msg = f"No triples to act on for k={k_budget}."
            print(f"[Q{qidx}] {msg} Skipping.")
            skip_log_path = os.path.join(output_dir, "skipped_questions.log")
            with open(skip_log_path, "a", encoding="utf-8") as f_skip:
                f_skip.write(json.dumps({"question_id": qidx, "k": k_budget, "reason": msg}) + "\n")
            continue

        env.set_attack_limit(actual_attack_steps)
        centralities_before = env.export_graph_centralities()
        state = env.get_state()
        done_from_env = False
        current_step_in_budget = 0

        while not done_from_env and current_step_in_budget < actual_attack_steps:
            action = persistent_agent.act(state)
            next_state, reward, done_from_env, info = env.step(action)
            persistent_agent.remember(state, action, reward, next_state, done_from_env)
            persistent_agent.replay()
            state = next_state
            current_step_in_budget += 1
            print(
                f"[Q{qidx}] k={k_budget}, Step {current_step_in_budget}/{actual_attack_steps}: Action={action}, Reward={reward:.2f}, Done={done_from_env}")
            if info.get("reason"):
                print(f"  Info: {info['reason']}")
                if "Invalid" in info["reason"]:
                    break

        centralities_after = env.export_graph_centralities()
        changes_made = {
            "removed": env.removed_triples,
            "added": [info['uri'] for info in env.added_triples_info]
        }

        # all_other_q_indices = [idx for idx in qa_cache if idx != qidx and idx <= len(qald_data['questions'])]
        # num_to_sample = min(collateral_sample_size, len(all_other_q_indices))
        # sampled_collateral_indices = random.sample(all_other_q_indices, num_to_sample) if num_to_sample > 0 else []
        # evaluation_plan_indices = [qidx] + sampled_collateral_indices

        log_changes_to_file(
            log_path=os.path.join(output_dir, "change_log.jsonl"),
            question_id=qidx,
            k=k_budget,
            ground_truth_answer=ground_truth_answer,
            attack_type=attack_mode,
            removed=changes_made["removed"],
            added=changes_made["added"],
            sparql_query=sparql_query_gt,
            add_mode=add_mode,
            centralities_before=centralities_before,
            centralities_after=centralities_after
        )

        print(
            f"[Q{qidx}] k_budget={k_budget}: Attack finished. Removed: {len(changes_made['removed'])}, Added: {len(changes_made['added'])}")

        del env
        gc.collect()

    del question_embedding, answer_embedding_np, initial_subgraph_embeddings
    gc.collect()
    print(f"--- Finished processing for Question ID: {qidx} ---")


def main():
    """
    Main driver for the KGQA adversarial attack script.

    This function handles the overall workflow:
    1.  Parses user input for attack configuration.
    2.  Sets up directories for inputs and outputs.
    3.  Loads all necessary data: embeddings, mappings, and question datasets.
    4.  Manages a checkpoint system to resume processing from where it left off.
    5.  Initializes the persistent Reinforcement Learning agent.
    6.  Loops through all unprocessed questions, calling `run_attack_and_log` for each.
    7.  Saves the agent's model and progress periodically.
    """
#     parser = argparse.ArgumentParser(description="Run KGQA Adversarial Attacks.")
#     args = parser.parse_args()

    BASE_INPUT_DIR = "../input_data"
    BASE_EMBEDDING_DIR = "../embeddings/DBpediaKGE"
    BASE_OUTPUT_DIR = "../outputs_thesis"

    sparql_endpoint_url = "http://localhost:3030/ds/sparql"
    sparql_update_endpoint_url = "http://localhost:3030/ds/update"

    attack_mode = input("Enter attack mode ('add' or 'remove'): ").strip().lower()
    if attack_mode not in ["add", "remove"]:
        print("Invalid attack mode. Defaulting to 'remove'.")
        attack_mode = "remove"

    add_mode_specific = None
    if attack_mode == "add":
        add_mode_specific = input("Enter add mode for 'add' attack ('semantic' or 'cairage'): ").strip().lower()
        if add_mode_specific not in ["semantic", "cairage"]:
            print("Invalid add_mode_specific for 'add' attack. Defaulting to 'semantic'.")
            add_mode_specific = "semantic"

    specific_attack_output_dir = f"dbpedia_output_{attack_mode}"
    if add_mode_specific:
        specific_attack_output_dir += f"_{add_mode_specific}"

    output_dir_path = os.path.join(BASE_OUTPUT_DIR, specific_attack_output_dir)
    os.makedirs(output_dir_path, exist_ok=True)
    print(f"Outputs will be saved in: {output_dir_path}")

    checkpoint_file = os.path.join(output_dir_path, "processed_questions.json")

    embedding_dir = "../embeddings/DBpediaKGE/"
    entity_file_path = os.path.join(embedding_dir, "entity_to_idx.csv")
    relation_file_path = os.path.join(embedding_dir, "relation_to_idx.csv")
    qald_json_file_path = "../input_data/qald_9_plus_train_dbpedia.json"
    qa_cache_file_path = "/home/neha2022/thesis/input_data/qa_groundtruth_train.jsonl"

    print("Loading embeddings and mappings...")
    try:
        entity_embeddings_torch, relation_embeddings_torch = load_embeddings_from_models(BASE_EMBEDDING_DIR)
        if entity_embeddings_torch is None or relation_embeddings_torch is None:
            raise ValueError("load_embeddings_from_models returned None for embeddings.")
        entity_embeddings_np = entity_embeddings_torch.cpu().numpy()
        relation_embeddings_np = relation_embeddings_torch.cpu().numpy()
        entity_mapping, relation_mapping = map_indices_to_embeddings(
            entity_file_path, relation_file_path, entity_embeddings_np, relation_embeddings_np
        )
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load embeddings or mappings: {e}")
        return

    if not entity_mapping or not relation_mapping:
        print("CRITICAL ERROR: Entity or relation mapping is empty after loading. Exiting.")
        return

    try:
        EMBEDDING_DIM_ENTITY = next(iter(entity_mapping.values())).shape[0]
        EMBEDDING_DIM_RELATION = next(iter(relation_mapping.values())).shape[0]
    except StopIteration:
        print("[CRITICAL] Entity or Relation mapping is unexpectedly empty. Exiting.")
        return

    print(f"Determined Entity Embedding Dim: {EMBEDDING_DIM_ENTITY}")
    print(f"Determined Relation Embedding Dim: {EMBEDDING_DIM_RELATION}")

    try:
        with open(qald_json_file_path, "r", encoding="utf-8") as f:
            qald_data = json.load(f)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load QALD file {qald_json_file_path}: {e}")
        return

    if not os.path.exists(qa_cache_file_path):
        print(f"CRITICAL ERROR: QA ground truth cache file not found at {qa_cache_file_path}.")
        return

    qa_cache_map = {}
    try:
        with open(qa_cache_file_path, "r", encoding="utf-8") as f_cache:
            for line_num, line in enumerate(f_cache, 1):
                try:
                    item = json.loads(line)
                    if "id" not in item or "answer" not in item:
                        print(f"Warning: Invalid entry in QA cache on line {line_num}. Skipping.")
                        continue
                    q_id_int = int(item["id"])
                    qa_cache_map[q_id_int] = item
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    print(f"Warning: Could not process line {line_num} in QA cache: {e}. Skipping.")
                    continue
    except Exception as e:
        print(f"CRITICAL ERROR: Error loading QA cache {qa_cache_file_path}: {e}")
        return

    if not qa_cache_map:
        print("CRITICAL ERROR: QA cache is empty after loading. Exiting.")
        return

    done_questions_set = load_checkpoint(checkpoint_file)
    qald_max_idx = len(qald_data.get('questions', []))
    questions_to_process_ids = sorted([
        idx for idx in qa_cache_map
        if idx not in done_questions_set and (0 < idx <= qald_max_idx)
    ])

    if not questions_to_process_ids:
        print("No new questions to process.")
        return

    print(f"Found {len(questions_to_process_ids)} questions to process.")

    agent_action_size = MAX_SUBGRAPH_TRIPLES_LIMIT
    agent_state_shapes = {
        "question_embedding": encode("sample").shape[0],
        "padded_triples": EMBEDDING_DIM_ENTITY + EMBEDDING_DIM_RELATION + EMBEDDING_DIM_ENTITY,
        "other_features": 2
    }

    agent_model_filename = f'trained_model_{attack_mode}' + (
        f'_{add_mode_specific}' if add_mode_specific else "") + '.keras'
    persistent_rl_agent = KGQARLAgent(
        state_shapes=agent_state_shapes,
        action_size=agent_action_size,
        model_path=os.path.join(output_dir_path, agent_model_filename)
    )

    for i, q_processing_idx in enumerate(questions_to_process_ids, 1):
        print(f"\n>>> Processing Question {i}/{len(questions_to_process_ids)}: ID {q_processing_idx} <<<")
        try:
            run_attack_and_log(
                qidx=q_processing_idx,
                qald_data=qald_data,
                entity_mapping=entity_mapping,
                relation_mapping=relation_mapping,
                persistent_agent=persistent_rl_agent,
                attack_mode=attack_mode,
                add_mode=add_mode_specific,
                output_dir=output_dir_path,
                qa_cache=qa_cache_map,
                sparql_endpoint_url=sparql_endpoint_url,
                sparql_update_endpoint_url=sparql_update_endpoint_url,
                embedding_dim_entity=EMBEDDING_DIM_ENTITY,
                embedding_dim_relation=EMBEDDING_DIM_RELATION,
                max_triples=MAX_SUBGRAPH_TRIPLES_LIMIT
            )
            done_questions_set.add(q_processing_idx)
            save_checkpoint(checkpoint_file, done_questions_set)
            persistent_rl_agent.save_model()
        except Exception as e_main_loop:
            print(f"[CRITICAL ERROR] Unhandled exception while processing Q_ID {q_processing_idx}: {e_main_loop}")
            import traceback
            traceback.print_exc()
            error_log_path = os.path.join(output_dir_path, "critical_processing_errors.log")
            with open(error_log_path, "a", encoding="utf-8") as f_err:
                f_err.write(f"Q_ID {q_processing_idx}: {str(e_main_loop)}\n{traceback.format_exc()}\n---\n")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.2)

    print("\nAll designated questions processed.")
    persistent_rl_agent.save_model()
    print(f"Final RL agent model saved to {persistent_rl_agent.model_path}")


if __name__ == "__main__":
    main()
