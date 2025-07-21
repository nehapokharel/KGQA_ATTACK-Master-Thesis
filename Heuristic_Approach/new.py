"""
This script performs a multi-phase adversarial attack evaluation on a Knowledge Graph.
The workflow is designed to be executed sequentially using command-line arguments.

---
Phase 1: Attack Plan Generation (--phase 1)
---
- Reads a ground-truth file of questions, SPARQL queries, and answers.
- For each question, it analyzes the SPARQL query to find important predicates.
- It fetches all facts related to the answer, then randomly samples a subgraph of max 200 triples for analysis.
- It calculates centrality measures (PageRank, betweenness, eigenvector) for the nodes in this subgraph.
- It then creates a series of "attack plans" by selecting the top 'k' triples to remove from the sampled subgraph.
- These plans, including the centrality data, are saved to a JSON Lines file (`attack_plan.jsonl`) and DO NOT modify the database.

---
Phase 2: Attack Execution & Reporting (--phase 2)
---
- This phase mirrors the output logic of the rl_agent script.
- It reads the `attack_plan.jsonl` file generated in Phase 1.
- For each attack plan, it performs an "Attack -> Evaluate -> Revert" cycle:
  1. DELETE the triples for the targeted question.
  2. QUERY to check if the primary target's answer changed.
  3. IF IT DID, it then queries for ALL other questions to measure collateral damage.
  4. INSERT the triples back to restore the graph.
- It generates multiple output files:
  - A detailed JSON report for EACH individual attack (`eval_results_...`).
  - Aggregated JSON and QALD reports for all attacks, grouped by 'k' (`aggregated_attacks_...`).

"""

import os
import json
import re
import argparse
import random
import asyncio
import aiohttp
from SPARQLWrapper import SPARQLWrapper, JSON, POST
from rdflib import Graph, URIRef, Literal
import networkx as nx

# --- Configuration ---

# Input files
QA_GROUNDTRUTH_FILE = "/home/neha2022/thesis/input_data/qa_groundtruth_train.jsonl"
QALD_TEMPLATE_FILE = '/home/neha2022/thesis/input_data/qald_9_plus_train_dbpedia.json'

OUTPUT_DIR = '/home/neha2022/thesis/outputs_thesis_main/basline_attack_outputs'

# Phase 1 output file
ATTACK_PLAN_FILE = os.path.join(OUTPUT_DIR, 'attack_plan.jsonl')

# Fuseki and Attack Parameters
# FUSEKI_QUERY_ENDPOINT = "http://localhost:3030/ds/sparql"
# FUSEKI_UPDATE_ENDPOINT = "http://localhost:3030/ds/update"

FUSEKI_QUERY_ENDPOINT = "http://localhost:3030/dbpedia2016/sparql"
FUSEKI_UPDATE_ENDPOINT = "http://localhost:3030/dbpedia2016/update"

# REMOVAL_SIZES = [30, 40]
REMOVAL_SIZES= [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
NETWORK_TIMEOUT = 60
SUBGRAPH_TRIPLE_LIMIT = 200


# Utility Functions
def log_error(context: str, exception: Exception, log_file):
    """Logs an error message to the specified file."""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"-----\nContext: {context}\nException: {str(exception)}\n-----\n\n")


def write_output_json(filename, data):
    """Writes data to a pretty-printed JSON file."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Generated: {path}")


def write_qald_output(filename, qa_pairs, qald_template_data):
    """
    Generates a QALD-formatted JSON file based on a template,
    replacing answers with predicted ones.
    """
    path = os.path.join(OUTPUT_DIR, filename)
    # Create a lookup map from the template file for efficient access
    template_map = {item['id']: item for item in qald_template_data['questions']}

    updated_questions = []

    for entry in qa_pairs:
        predicted_value = entry["predicted_answer"]
        if predicted_value == "NO_ANSWER_FOUND":
            continue

        qald_id = str(entry['question_id'])

        # Find the original question data from the template
        original_question_data = template_map.get(qald_id)
        if not original_question_data:
            print(f"Warning: Question ID {qald_id} not found in QALD template. Skipping.")
            continue

        # Create a copy to avoid modifying the original template data in memory
        new_qald_entry = original_question_data.copy()

        # Determine the key for the answer dictionary ('uri' or 'literal')
        answer_key = "uri"
        if not isinstance(predicted_value, str) or not predicted_value.startswith("http"):
            answer_key = "literal"
            predicted_value = str(predicted_value) if predicted_value is not None else "NULL_PREDICTION"

        # The target format's "answer" is a list of objects.
        new_qald_entry['answer'] = [{answer_key: predicted_value}]
        updated_questions.append(new_qald_entry)

    # Final structure for the output file
    qald_format = {"questions": updated_questions}

    with open(path, "w") as f:
        json.dump(qald_format, f, indent=4)
    print(f"  Generated QALD: {path}")


# SPARQL Interaction Functions
def get_predicates_from_query(sparql_query: str) -> list:
    """Extracts all predicate URIs from a SPARQL query string using regex."""
    predicate_pattern = re.compile(r'\b(?:dbo|dbp|rdf|rdfs):[a-zA-Z_]+')
    found = predicate_pattern.findall(sparql_query)
    full_uris = []
    for p in found:
        if p.startswith('dbo:'):
            full_uris.append(p.replace('dbo:', 'http://dbpedia.org/ontology/'))
        elif p.startswith('dbp:'):
            full_uris.append(p.replace('dbp:', 'http://dbpedia.org/property/'))
        elif p.startswith('rdf:'):
            full_uris.append(p.replace('rdf:', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'))
        elif p.startswith('rdfs:'):
            full_uris.append(p.replace('rdfs:', 'http://www.w3.org/2000/01/rdf-schema#'))
    return list(set(full_uris))


def _execute_update(query):
    """Send any update query to Fuseki server."""
    sparql = SPARQLWrapper(FUSEKI_UPDATE_ENDPOINT)
    sparql.setTimeout(NETWORK_TIMEOUT)
    sparql.setQuery(query)
    sparql.setMethod(POST)
    sparql.query()


def manage_live_graph_data(triples, operation="INSERT"):
    """Formats and executes a batch of DELETE or INSERT operations on the live default graph."""
    if not triples: return
    triples_to_format = [(URIRef(s), URIRef(p), URIRef(o) if o.startswith('http') else Literal(o)) for s, p, o in
                         triples]
    triples_str = [f"{s.n3()} {p.n3()} {o.n3()} ." for s, p, o in triples_to_format]
    print(f"{operation.lower().capitalize()}ing {len(triples_str)} triples on live default graph...")
    # Process in batches to avoid overly long query strings
    for i in range(0, len(triples_str), 100):
        batch = ' '.join(triples_str[i:i + 100])
        query = f"{operation} DATA {{ {batch} }}"
        _execute_update(query)


def query_live_graph(sparql_query):
    """Executes a SELECT query against the live default graph."""
    sparql = SPARQLWrapper(FUSEKI_QUERY_ENDPOINT)
    sparql.setTimeout(NETWORK_TIMEOUT)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        if not bindings: return "NO_ANSWER_FOUND"
        # Return the value of the first variable in the first result row
        return bindings[0][results['head']['vars'][0]]['value']
    except Exception as e:
        # Propagate exception to be handled by the calling function's try-except block
        raise e


def load_qa_pairs_from_jsonl(path: str):
    """Loads question-answer data from a JSON Lines file."""
    if not os.path.exists(path):
        print(f"Error: Ground truth file not found at {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def calculate_centralities(subgraph: Graph):
    """
    Calculates betweenness, eigenvector, and PageRank centralities for a given rdflib graph.
    Returns a dictionary in the format required for the attack plan.
    """
    if not subgraph:
        return {}

    # Convert rdflib.Graph to networkx.DiGraph for calculations
    nx_graph = nx.DiGraph()
    for s, p, o in subgraph:
        # networkx handles URIRefs, but we'll use strings for consistency in the output
        nx_graph.add_edge(str(s), str(o))

    if not nx_graph.nodes():
        return {}

    # Calculate centralities
    betweenness = nx.betweenness_centrality(nx_graph)
    pagerank = nx.pagerank(nx_graph)

    # Eigenvector centrality can fail to converge on some graphs
    try:
        eigenvector = nx.eigenvector_centrality(nx_graph, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("  Warning: Eigenvector centrality did not converge. Defaulting to 0 for all nodes.")
        eigenvector = {node: 0.0 for node in nx_graph.nodes()}

    # Combine into the final desired structure
    centralities = {}
    for node in nx_graph.nodes():
        centralities[node] = {
            "betweenness": betweenness.get(node, 0.0),
            "eigenvector": eigenvector.get(node, 0.0),
            "pagerank": pagerank.get(node, 0.0)
        }
    return centralities



async def phase1_generate_attacks():
    """
    Runs the Phase 1 workflow: reads QA data, analyzes it, and generates
    a JSON Lines file (`attack_plan.jsonl`) containing attack plans.
    """
    print("---------------------------")
    print("Running Phase 1: Generating Attack Plans")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    error_log_file = os.path.join(OUTPUT_DIR, 'phase1_errors.log')
    plan_file_path = ATTACK_PLAN_FILE
    if os.path.exists(error_log_file): os.remove(error_log_file)
    if os.path.exists(plan_file_path): os.remove(plan_file_path)

    qa_pairs = load_qa_pairs_from_jsonl(QA_GROUNDTRUTH_FILE)
    if not qa_pairs: return

    for i, qa_pair in enumerate(qa_pairs):
        try:
            qid = qa_pair.get('id') or qa_pair.get('index')
            sparql_query = qa_pair['query']
            answers = qa_pair.get('answer')  # This is often a list

            # --- FIX: Handle list of answers by choosing the first one ---
            true_answer = None
            if isinstance(answers, list) and answers:
                first_answer_dict = answers[0]
                true_answer = first_answer_dict.get('uri') or first_answer_dict.get('name')
            else:
                true_answer = answers

            print(f"Analyzing QID {qid} (using first answer: {true_answer})...")

            # --- Validate the chosen answer before proceeding ---
            if not true_answer or not isinstance(true_answer, str) or not true_answer.startswith('http'):
                print(f"  Skipping QID {qid}: The chosen answer '{true_answer}' is not a valid URI.")
                continue

            seed_uris_for_subgraph = {true_answer}
            try:
                for match in re.finditer(r'res:([^\s\)\}\.\?\;\,]+)', sparql_query):
                    entity_uri = "http://dbpedia.org/resource/" + match.group(1)
                    seed_uris_for_subgraph.add(entity_uri)
                for match in re.finditer(r'<http://dbpedia.org/resource/([^>\s]+)>', sparql_query):
                    entity_uri = "http://dbpedia.org/resource/" + match.group(1)
                    seed_uris_for_subgraph.add(entity_uri)
            except Exception as e_parse:
                print(f"WARNING: Could not parse additional entities from SPARQL query: {e_parse}. "
                      "Proceeding with answer URI as primary seed.")

            final_seed_uris = list(seed_uris_for_subgraph)
            if not final_seed_uris:
                print(f"ERROR: No valid seed URIs (answer or query entities with embeddings) found. Skipping.")
                return

            print(f"Final seed URIs for subgraph (sample): {final_seed_uris[:5]} (Total: {len(final_seed_uris)})")

            query_predicates = get_predicates_from_query(sparql_query)
            if query_predicates:
                print(f"Extracted {len(query_predicates)} predicates. Using focused subgraph fetch.")
            else:
                print(f"No predicates found in query. Using generic 1-hop subgraph fetch.")

            query_predicates = get_predicates_from_query(sparql_query)
            if not query_predicates:
                print(f"  Skipping QID {qid} as no predicates were found in the query.")
                continue

            # Fetch all facts related to the answer
            print(f"  Fetching all related triples...")
            full_subgraph_raw_triples, _ = await fetch_focused_triples(final_seed_uris, query_predicates, FUSEKI_QUERY_ENDPOINT) # Await the call and unpack
            full_subgraph = Graph()
            for s, p, o in full_subgraph_raw_triples:
                full_subgraph.add((URIRef(s), URIRef(p), URIRef(o) if o.startswith('http') else Literal(o)))
#             full_subgraph = fetch_focused_triples(final_seed_uris, query_predicates, FUSEKI_QUERY_ENDPOINT)
            if not full_subgraph:
                print("  Subgraph is empty. Skipping.")
                continue

            all_triples = list(full_subgraph)
            print(f"  Found {len(all_triples)} total related triples.")

            # Randomly sample if over the limit
            subgraph_for_analysis_triples = all_triples
            if len(all_triples) > SUBGRAPH_TRIPLE_LIMIT:
                print(f"  Randomly sampling {SUBGRAPH_TRIPLE_LIMIT} triples for analysis.")
                subgraph_for_analysis_triples = random.sample(all_triples, SUBGRAPH_TRIPLE_LIMIT)

            # Create a new rdflib Graph for centrality calculation from the (potentially sampled) triples
            analysis_graph = Graph()
            for s, p, o in subgraph_for_analysis_triples:
                analysis_graph.add((s, p, o))

            print(f"  Calculating centralities on {len(analysis_graph)} triples...")
            centralities = calculate_centralities(analysis_graph)
            if not centralities:
                print("  Could not calculate centralities. Skipping.")
                continue

            # Use the triples from the analysis subgraph to create attack plans
            for k in REMOVAL_SIZES:
                if len(subgraph_for_analysis_triples) < k:
                    continue

                # Select the first 'k' triples from the (potentially random) list
                triples_to_remove = subgraph_for_analysis_triples[:k]
                attack_plan = {
                    'question_id': qid, 'k': k, 'question': qa_pair['question'],
                    'sparql_query': sparql_query, 'true_answer': true_answer,
                    'triples_to_remove': [[str(s), str(p), str(o)] for s, p, o in triples_to_remove],
                    'centralities_before_attack': centralities  # Add centralities to the plan
                }
                with open(plan_file_path, 'a') as f:
                    f.write(json.dumps(attack_plan) + '\n')
        except Exception as e:
            print(f"ERROR on QID {qid}. Skipping. Check '{error_log_file}' for details.")
            log_error(f"Failed during analysis of QID {qid}", e, error_log_file)
            continue
    print(f"Phase 1 Complete. Attack plans saved to {plan_file_path}")


# def get_focused_subgraph(answer_uri: str, predicates: list, endpoint_url: str) -> Graph:
#     """Fetches a relevant subgraph based on an answer and related predicates."""
#     if not predicates: return Graph()
#
#     # Create a SPARQL filter string for the given predicates
#     predicate_filter = ', '.join([f'<{p}>' for p in predicates])
#
#     # Corrected and more readable SPARQL query
#     query = f"""
#     CONSTRUCT {{
#         <{answer_uri}> ?p ?o .
#         ?s ?p <{answer_uri}> .
#     }}
#     WHERE {{
#         {{ <{answer_uri}> ?p ?o . }}
#         UNION
#         {{ ?s ?p <{answer_uri}> . }}
#         FILTER(?p IN ({predicate_filter}))
#     }}"""
#
#     sparql = SPARQLWrapper(endpoint_url)
#     sparql.setTimeout(NETWORK_TIMEOUT)
#     sparql.setQuery(query)
#     try:
#         # The result of a CONSTRUCT query is an RDF graph
#         return sparql.query().convert()
#     except Exception as e:
#         raise e


async def fetch_focused_triples(seed_entities, predicates, sparql_endpoint, graph_uri=None):
    """
    Asynchronously fetches triples connected to seed entities, optionally filtered by predicates.

    For each seed entity, it retrieves all triples where the entity is the subject or object.
    If a list of predicates is provided, the results are filtered to include only those predicates.

    Args:
        seed_entities (list): A list of entity URIs to use as seeds for the search.
        predicates (list): A list of predicate URIs to filter the results. If empty, all predicates are considered.
        sparql_endpoint (str): The URL of the SPARQL endpoint.

    Returns:
        tuple[list, dict]: A tuple containing a list of unique triples found and an empty dictionary (for API consistency).
    """
    all_raw_triples = set()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for entity in seed_entities:
            # Build the predicate filter if predicates are provided
            predicate_filter = ""
            if predicates:
                predicate_values = ', '.join([f'<{p}>' for p in predicates])
                predicate_filter = f"FILTER(?p IN ({predicate_values}))"

            query = f"""
            SELECT ?s ?p ?o WHERE {{
                {{ BIND(<{entity}> AS ?s) . ?s ?p ?o . }}
                UNION
                {{ BIND(<{entity}> AS ?o) . ?s ?p ?o . }}
                {predicate_filter}
            }}
            """
            tasks.append(fetch_sparql(session, query, sparql_endpoint))

        results = await asyncio.gather(*tasks)

    for result in results:
        bindings = result.get("results", {}).get("bindings", [])
        for binding in bindings:
            # Ensure the object is not a literal, or handle appropriately
            o_val = binding.get('o', {}).get('value')
            o_type = binding.get('o', {}).get('type')
            if o_type == 'literal':
                continue  # Skip triples with literal objects for now

            s = binding.get('s', {}).get('value')
            p = binding.get('p', {}).get('value')
            o = o_val

            if s and p and o:
                all_raw_triples.add((s, p, o))

    return list(all_raw_triples), {}

async def fetch_sparql(session, query, endpoint):
    """
    Asynchronously executes a SPARQL SELECT query against an endpoint.

    Args:
        session (aiohttp.ClientSession): The aiohttp session to use for the request.
        query (str): The SPARQL query to execute.
        endpoint (str): The URL of the SPARQL endpoint.

    Returns:
        dict: The JSON response from the SPARQL endpoint.

    Raises:
        Exception: If the SPARQL query fails.
    """
    headers = {
        'Accept': 'application/sparql-results+json',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    async with session.post(endpoint, data={'query': query}, headers=headers) as response:
        if response.status != 200:
            text = await response.text()
            raise Exception(f"SPARQL error {response.status}: {text}")
        return await response.json()


def phase2_evaluate_and_generate_reports():
    """
    Runs the Phase 2 workflow: Executes attacks from the plan file, evaluates
    primary and collateral damage, and generates reports in the rl_agent format.
    """
    print("---------------------------")
    print("Running Phase 2: Executing Attacks & Generating Reports")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    error_log_file = os.path.join(OUTPUT_DIR, 'phase2_errors.log')

    # --- Pre-computation Checks ---
    if not os.path.exists(ATTACK_PLAN_FILE):
        print(f"Error: Attack plan file not found at {ATTACK_PLAN_FILE}. Please run Phase 1 first.")
        return
    if not os.path.exists(QALD_TEMPLATE_FILE):
        print(f"Error: QALD template file not found at {QALD_TEMPLATE_FILE}. This is required for report generation.")
        return

    if os.path.exists(error_log_file): os.remove(error_log_file)

    # --- Load Data ---
    print("Loading attack plans and QALD template...")
    attack_plans = load_qa_pairs_from_jsonl(ATTACK_PLAN_FILE) # Use existing loader
    with open(QALD_TEMPLATE_FILE, 'r') as f:
        qald_template_data = json.load(f)

    # Create a map for quick lookup of QA pairs during collateral damage check
    qa_map = {plan['question_id']: plan for plan in attack_plans}
    aggregated_results_by_k = {k: [] for k in REMOVAL_SIZES}
    failure_states = {"NO_ANSWER_FOUND", "QUERY_FAILED", "QUERY_WRAP_FAILED"}

    # --- Main Attack Loop ---
    for i, targeted_plan in enumerate(attack_plans):
        targeted_qid = targeted_plan['question_id']
        k = targeted_plan['k']
        triples_to_remove = targeted_plan.get('triples_to_remove', [])

        print(f"\nProcessing Attack {i + 1}/{len(attack_plans)} (Target QID: {targeted_qid}, k: {k})...")

        current_attack_evaluation_set = []

        try:
            # ATTACK: Delete the triples from the live graph
            manage_live_graph_data(triples_to_remove, operation="DELETE")

            # EVALUATE (Primary Target)
            predicted_answer_primary = query_live_graph(targeted_plan['sparql_query'])
            primary_is_changed = (targeted_plan['true_answer'] != predicted_answer_primary) and (
                    predicted_answer_primary not in failure_states)

            primary_result = {
                "question_id": targeted_qid,
                "k": k,
                "attack_type": "remove",
                "question": targeted_plan['question'],
                "true_answer": targeted_plan['true_answer'],
                "predicted_answer": predicted_answer_primary,
                "is_changed": primary_is_changed
            }
            current_attack_evaluation_set.append(primary_result)

            if k in aggregated_results_by_k:
                 aggregated_results_by_k[k].append(primary_result)

            # EVALUATE (Collateral Damage) - only if the primary attack was successful
            if primary_is_changed:
                print(
                    f"  Primary attack successful. Checking {len(qa_map) - 1} other questions for collateral damage...")
                for qid_to_check, plan_to_check in qa_map.items():
                    if qid_to_check == targeted_qid: continue  # Skip the primary target

                    predicted_answer_collateral = query_live_graph(plan_to_check['sparql_query'])
                    is_changed_collateral = (plan_to_check['true_answer'] != predicted_answer_collateral) and (
                            predicted_answer_collateral not in failure_states)

                    # Only log if there was actual collateral damage
                    if is_changed_collateral:
                        collateral_result = {
                            "question_id": qid_to_check,
                            "k": k,
                            "attack_type": f"collateral_from_remove",
                            "question": plan_to_check['question'],
                            "true_answer": plan_to_check['true_answer'],
                            "predicted_answer": predicted_answer_collateral,
                            "is_changed": is_changed_collateral
                        }
                        current_attack_evaluation_set.append(collateral_result)
            else:
                print("  Primary attack failed or answer was lost. Skipping collateral damage check.")

            filtered_evaluation_set = [
                entry for entry in current_attack_evaluation_set
                if entry.get("predicted_answer") != "NO_ANSWER_FOUND"
            ]
            if filtered_evaluation_set:
                eval_filename = f"eval_results_target_q{targeted_qid}_k{k}_remove.json"
                write_output_json(eval_filename, filtered_evaluation_set)

        except Exception as e:
            print(f"  -> ERROR on plan for QID {targeted_qid}. Skipping. Check '{error_log_file}'.")
            log_error(f"Failed during execution of plan for QID {targeted_qid}", e, error_log_file)
        finally:
            # REVERT: Always attempt to restore the graph to its original state
            try:
                print("  Reverting graph state...")
                manage_live_graph_data(triples_to_remove, operation="INSERT")
            except Exception as revert_e:
                print(f"  -> CRITICAL ERROR: Failed to revert graph state for QID {targeted_qid}.")
                log_error(f"CRITICAL: Failed to revert graph state after error on QID {targeted_qid}", revert_e,
                          error_log_file)
                # Continue to the next attack
                continue

    print("\nGenerating Aggregated Attack Results...")
    for k_val, results_list in aggregated_results_by_k.items():
        if not results_list: continue

        # Filter out failed/empty predictions for reporting
        filtered_results = [
            entry for entry in results_list if entry.get("predicted_answer") != "NO_ANSWER_FOUND"
        ]

        if filtered_results:
            # Generate the standard aggregated JSON report
            agg_json_filename = f"aggregated_attacks_k{k_val}.json"
            write_output_json(agg_json_filename, filtered_results)

            # Generate the QALD-formatted report
            agg_qald_filename = f"aggregated_attacks_k{k_val}.qald.json"
            write_qald_output(agg_qald_filename, filtered_results, qald_template_data)

    print(f"Phase 2 Complete. All reports saved to the '{OUTPUT_DIR}' directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a multi-phase graph attack workflow.")
    parser.add_argument(
        '--phase', type=int, choices=[1, 2], required=True,
        help="Specify which phase to run: 1 (Generate Attack Plans), 2 (Execute Attacks & Generate Reports)."
    )
    args = parser.parse_args()

    if args.phase == 1:
        asyncio.run(phase1_generate_attacks())
#         phase1_generate_attacks()
    elif args.phase == 2:
        phase2_evaluate_and_generate_reports()