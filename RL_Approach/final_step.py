import json
import os
from SPARQLWrapper import SPARQLWrapper, JSON, POST

# Configuration for the Apache Jena Fuseki server endpoints.
FUSEKI_UPDATE_ENDPOINT = "http://localhost:3030/ds/update"
FUSEKI_QUERY_ENDPOINT = "http://localhost:3030/ds/sparql"


def format_triple_for_sparql(triple_parts):
    """Formats a Python list of three elements into a SPARQL triple string.

    Args:
        triple_parts (list): A list or tuple containing the subject, predicate,
                             and object of a triple. E.g.,
                             ['http://example.org/s', 'http://example.org/p', 'http://example.org/o'].

    Returns:
        str: A string representing the triple in SPARQL format, e.g.,
             "<http://example.org/s> <http://example.org/p> <http://example.org/o> .".
    """
    s, p, o = triple_parts
    return f"<{s}> <{p}> <{o}> ."


def execute_sparql_update(update_query):
    """Executes a SPARQL UPDATE query (INSERT/DELETE) against the Fuseki server.

    Args:
        update_query (str): The complete SPARQL UPDATE query string.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    sparql = SPARQLWrapper(FUSEKI_UPDATE_ENDPOINT)
    sparql.setQuery(update_query)
    sparql.setMethod(POST)
    sparql.setRequestMethod('POST')
    try:
        sparql.query()
        return True
    except Exception as e:
        print(f"ERROR executing update: {update_query[:200]}... \nError: {e}")
        return False


def query_dbpedia_select(sparql_query):
    """Executes a SPARQL SELECT query against the Fuseki server.

    It retrieves results and extracts the value from the first variable in the
    SELECT clause for each result binding.

    Args:
        sparql_query (str): The complete SPARQL SELECT query string.

    Returns:
        list: A list of result values (strings). Returns an empty list if
              there are no results or if an error occurs.
    """
    sparql = SPARQLWrapper(FUSEKI_QUERY_ENDPOINT)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        answers = []
        if results["results"]["bindings"]:
            if not results["head"]["vars"]:
                return []
            first_var = results["head"]["vars"][0]
            for result in results["results"]["bindings"]:
                if first_var in result:
                    answers.append(result[first_var]["value"])
        return answers
    except Exception as e:
        print(f"ERROR executing select query: {sparql_query[:200]}... \nError: {e}")
        return []


def get_unpacked_answer(answer_obj):
    """Extracts a single answer value from a potentially complex object.

    This helper function is designed to simplify the ground truth answer format,
    which might be a string, a list of dictionaries, or None.

    Args:
        answer_obj (any): The answer object from the ground truth data.

    Returns:
        str or None: The single, simplified answer string, or None if no
                     answer can be extracted.
    """
    if not answer_obj:
        return None
    if isinstance(answer_obj, str):
        return answer_obj
    if isinstance(answer_obj, list) and answer_obj:
        first_answer_dict = answer_obj[0]
        if isinstance(first_answer_dict, dict) and list(first_answer_dict.values()):
            return list(first_answer_dict.values())[0]
    return None


def get_placeholder_centralities():
    """Returns a placeholder dictionary for centrality-related values.

    Note: The function name is a misnomer from a previous implementation.
    It does not calculate centralities but provides a simple structure.

    Returns:
        dict: A dictionary with keys for answers before/after, all set to None.
    """
    return {"true_answer_before": None, "true_answer_after": None, "false_answer_after": None}


def write_output_json(filename, data):
    """Writes a Python dictionary or list to a JSON file with indentation.

    Args:
        filename (str): The full path for the output JSON file.
        data (dict or list): The serializable Python object to write.
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated: {filename}")


def write_qald_output(filename, qa_pairs):
    """Formats and writes evaluation results into the QALD JSON format.

    Args:
        filename (str): The full path for the output QALD JSON file.
        qa_pairs (list): A list of dictionaries, where each dictionary
                         represents an evaluated question-answer pair.
    """
    qald_format = {"questions": []}
    for entry in qa_pairs:
        predicted_value = entry["predicted_answer"]
        if predicted_value == "NO_ANSWER_FOUND":
            continue

        qald_id = str(entry['question_id'])
        binding_x_value = predicted_value
        binding_x_type = "uri"

        if not isinstance(predicted_value, str) or not predicted_value.startswith("http"):
            binding_x_value = str(predicted_value) if predicted_value is not None else "NULL_PREDICTION"
            binding_x_type = "literal"

        qald_format["questions"].append({
            "id": qald_id,
            "answers": [{"head": {"vars": ["x"]},
                         "results": {"bindings": [{"x": {"type": binding_x_type, "value": binding_x_value}}]}}]
        })

    with open(filename, "w") as f:
        json.dump(qald_format, f, indent=2)
    print(f"Generated QALD: {filename}")


def process_attacks(log_file_path, qa_file_path, output_dir):
    """Main function to evaluate the impact of knowledge graph attacks.

    This function simulates attacks on a knowledge graph by applying triple
    modifications (additions/deletions) specified in a log file. For each attack,
    it evaluates two things:
    1.  Primary Impact: Whether the answer to the targeted question changed.
    2.  Collateral Impact: If the primary attack succeeded, it checks how many
        other questions in a ground truth set were unintentionally affected.

    The state of the knowledge graph is reverted after each attack simulation
    to ensure independent evaluations.

    Args:
        log_file_path (str): Path to the `.jsonl` file containing attack logs.
        qa_file_path (str): Path to the `.jsonl` file with ground truth
                            question-answer pairs.
        output_dir (str): The directory where evaluation result files will be saved.
    """
    qa_groundtruth_map = {}
    try:
        with open(qa_file_path, 'r', encoding='utf-8') as f_qa:
            for line_num, line in enumerate(f_qa, 1):
                item = json.loads(line)
                try:
                    q_id_int = int(item["id"])
                    qa_groundtruth_map[q_id_int] = item
                except (ValueError, TypeError):
                    print(
                        f"Warning: Could not convert ID '{item.get('id')}' to an integer in QA file on line {line_num}. Skipping.")
                    continue
        print(f"Successfully loaded {len(qa_groundtruth_map)} QA ground truth entries for collateral evaluation.")
    except FileNotFoundError:
        print(f"ERROR: QA ground truth file not found at {qa_file_path}")
        return
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while reading {qa_file_path}: {e}")
        return

    log_entries_data = []
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f_log:
            for line in f_log:
                log_entries_data.append(json.loads(line))
        print(f"Successfully loaded {len(log_entries_data)} log entries from {log_file_path}")
    except FileNotFoundError:
        print(f"ERROR: Log file not found at {log_file_path}")
        return
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while reading {log_file_path}: {e}")
        return

    aggregated_results_by_k = {3: [], 5: [], 7: [], 9: [], 11: [], 13: [], 15: [], 17: [], 19: [], 21: []}
    total_attacks = len(log_entries_data)
    current_attack_num = 0
    failure_states = {"NO_ANSWER_FOUND", "QUERY_FAILED", "QUERY_WRAP_FAILED", "QUERY_NOT_RUN_NO_PRIMARY_QUERY", None}

    for log_entry in log_entries_data:
        current_attack_num += 1
        print(
            f"\n--- Processing Attack {current_attack_num}/{total_attacks} (Target q_idx: {log_entry.get('question_id', 'N/A')}, k: {log_entry.get('k', 'N/A')}) ---")

        log_q_idx = log_entry.get('question_id')
        k = log_entry.get('k')
        attack_type = log_entry.get('attack_type')

        if log_q_idx is None or k is None:
            print(f"Warning: Log entry missing 'question_id' or 'k'. Skipping.")
            continue

        triples_to_remove_raw = log_entry.get('removed', log_entry.get('triples_removed', []))
        triples_to_add_raw = log_entry.get('added', log_entry.get('triples_added', []))
        current_attack_evaluation_set = []
        for triple_parts in triples_to_remove_raw:
            execute_sparql_update(f"DELETE DATA {{ {format_triple_for_sparql(triple_parts)} }}")
        for triple_parts in triples_to_add_raw:
            execute_sparql_update(f"INSERT DATA {{ {format_triple_for_sparql(triple_parts)} }}")

        primary_gt_answer = log_entry.get('ground_truth_answer')
        query_for_primary_q = log_entry.get('sparql_query')
        primary_q_info_from_map = qa_groundtruth_map.get(log_q_idx)
        primary_question_text = primary_q_info_from_map.get('question',
                                                            'N/A') if primary_q_info_from_map else "Question text not found"

        predicted_answers_list = query_dbpedia_select(query_for_primary_q) if query_for_primary_q else []
        primary_predicted_answer_final = predicted_answers_list[0] if predicted_answers_list else "NO_ANSWER_FOUND"

        if primary_predicted_answer_final in failure_states:
            primary_is_changed = False
        else:
            primary_is_changed = (primary_gt_answer != primary_predicted_answer_final)

        output_entry_attacked = {
            "question_id": log_q_idx, "k": k, "attack_type": attack_type,
            "question": primary_question_text, "true_answer": primary_gt_answer,
            "predicted_answer": primary_predicted_answer_final, "is_changed": primary_is_changed
        }
        current_attack_evaluation_set.append(output_entry_attacked)
        if k in aggregated_results_by_k:
            aggregated_results_by_k[k].append(output_entry_attacked)

        primary_attack_successful = isinstance(primary_is_changed, bool) and primary_is_changed
        print(
            f"Primary attacked question answer evaluation: {'CHANGED (SUCCESSFUL ATTACK)' if primary_attack_successful else 'DID NOT CHANGE'}.")

        if primary_attack_successful:
            collateral_indices = [idx for idx in qa_groundtruth_map if idx != log_q_idx]
            print(f"Evaluating all {len(collateral_indices)} other questions for collateral damage...")

            for s_idx in collateral_indices:
                sampled_q_info = qa_groundtruth_map.get(s_idx)
                if not sampled_q_info:
                    print(f"Warning: Ground truth for collateral question index {s_idx} not found.")
                    continue

                s_gt_answer = get_unpacked_answer(sampled_q_info.get('answer'))
                s_query = sampled_q_info.get('query')
                s_question_text = sampled_q_info.get('question', 'N/A')

                if not s_query or s_gt_answer is None:
                    continue

                s_predicted_answers = query_dbpedia_select(s_query)
                s_predicted_answer = s_predicted_answers[0] if s_predicted_answers else "NO_ANSWER_FOUND"

                if s_predicted_answer in failure_states:
                    s_is_changed = False
                else:
                    s_is_changed = (s_gt_answer != s_predicted_answer)

                current_attack_evaluation_set.append({
                    "question_id": s_idx, "k": k, "attack_type": f"collateral_from_{attack_type}",
                    "question": s_question_text, "true_answer": s_gt_answer,
                    "predicted_answer": s_predicted_answer, "is_changed": s_is_changed
                })

        filtered_evaluation_set = [
            entry for entry in current_attack_evaluation_set
            if entry.get("predicted_answer") != "NO_ANSWER_FOUND"
        ]
        if filtered_evaluation_set:
            eval_json_filename = os.path.join(output_dir, f"eval_results_target_q{log_q_idx}_k{k}_{attack_type}.json")
            write_output_json(eval_json_filename, filtered_evaluation_set)

        # Revert modifications
        print("Reverting modifications...")
        for triple_parts in triples_to_remove_raw:
            execute_sparql_update(f"INSERT DATA {{ {format_triple_for_sparql(triple_parts)} }}")
        for triple_parts in triples_to_add_raw:
            execute_sparql_update(f"DELETE DATA {{ {format_triple_for_sparql(triple_parts)} }}")

    print("\nGenerating Aggregated Attack Results...")
    for k_val, results_list in aggregated_results_by_k.items():
        if results_list:
            filtered_results = [
                entry for entry in results_list if entry.get("predicted_answer") != "NO_ANSWER_FOUND"
            ]

            if filtered_results:
                agg_json_filename = os.path.join(output_dir, f"aggregated_attacks_k{k_val}.json")
                write_output_json(agg_json_filename, filtered_results)

                agg_qald_filename = os.path.join(output_dir, f"aggregated_attacks_k{k_val}.qald.json")
                write_qald_output(agg_qald_filename, filtered_results)


if __name__ == '__main__':
    """
    Configures file paths and initiates the attack evaluation process.
    """
    log_file = "/home/neha2022/thesis/outputs_thesis/dbpedia_output_remove/change_log.jsonl"
    qa_file = "/home/neha2022/thesis/input_data/qa_groundtruth_train.jsonl"

    output_directory = os.path.dirname(log_file)
    print(f"Evaluation output files will be saved in: {output_directory}")

    print("Starting attack evaluation..")
    process_attacks(log_file, qa_file, output_directory)
    print("Processing finished!!")