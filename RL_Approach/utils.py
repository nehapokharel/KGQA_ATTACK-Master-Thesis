import json
import os
import torch
import pandas as pd
import asyncio
import aiohttp
import re
from SPARQLWrapper import SPARQLWrapper, POST
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

# Set the computation device to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a pre-trained sentence transformer model for encoding text
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

def encode(text):
    """
    Encodes a given text into a dense vector embedding using the pre-loaded model.

    Args:
        text (str): The input text to encode.

    Returns:
        np.ndarray: The vector embedding of the text.
    """
    return model.encode(text)

def get_predicates_from_query(sparql_query: str) -> list:
    """
    Extracts all unique predicate URIs from a SPARQL query string.

    This function uses regular expressions to find common predicate prefixes (dbo, dbp, rdf, rdfs)
    and expands them into their full URIs.

    Args:
        sparql_query (str): The SPARQL query string to parse.

    Returns:
        list: A list of unique, full predicate URIs found in the query.
    """
    if not sparql_query: return []
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


def get_answer_embedding(answer_uri, entity_mapping):
    """
    Retrieves the embedding for a single answer URI from the entity mapping.

    Args:
        answer_uri (str): The URI of the answer entity.
        entity_mapping (dict): A dictionary mapping entity URIs to their embeddings.

    Returns:
        np.ndarray: The embedding vector for the given URI.

    Raises:
        ValueError: If the answer URI is not found in the entity mapping.
    """
    if answer_uri not in entity_mapping:
        raise ValueError(f"Answer URI '{answer_uri}' not found in entity mapping.")
    return entity_mapping[answer_uri]


def load_embeddings_from_models(model_dir):
    """
        Loads and concatenates entity and relation embeddings from multiple model files.

        This function assumes embeddings are stored in multiple PyTorch model files
        (e.g., model_0.pt, model_1.pt) and need to be combined.

        Args:
            model_dir (str): The directory containing the model parts.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the concatenated
                                               entity embeddings and relation embeddings.
    """
    entity_embeddings_list = []
    relation_embeddings_list = []
    model_files = sorted(
        [os.path.join(model_dir, f)
         for f in os.listdir(model_dir)
         if f.startswith("model_") and f.endswith(".pt")]
    )

    if not model_files:
        raise FileNotFoundError("No valid model files (model_0.pt, model_1.pt, etc.) found.")

    for model_file in model_files:
        print(f"Loading: {model_file}")
        state_dict = torch.load(model_file)
        entity_key = '_orig_mod.entity_embeddings.weight'
        relation_key = '_orig_mod.relation_embeddings.weight'

        if entity_key not in state_dict or relation_key not in state_dict:
            raise KeyError(f"{model_file} missing keys: {entity_key}, {relation_key}")

        entity_embeddings_list.append(state_dict[entity_key])
        relation_embeddings_list.append(state_dict[relation_key])

    entity_embeddings = torch.cat(entity_embeddings_list, dim=1)
    relation_embeddings = torch.cat(relation_embeddings_list, dim=1)

    return entity_embeddings, relation_embeddings


def map_indices_to_embeddings(entity_file, relation_file, entity_emb_np, relation_emb_np):
    """
    Creates a mapping from entity/relation URIs to their embedding vectors.

    It reads CSV files containing URI-to-index mappings and uses them to build
    dictionaries that link a URI directly to its corresponding embedding.

    Args:
        entity_file (str): Path to the CSV file with entity-to-index mappings.
        relation_file (str): Path to the CSV file with relation-to-index mappings.
        entity_emb_np (np.ndarray): The numpy array of all entity embeddings.
        relation_emb_np (np.ndarray): The numpy array of all relation embeddings.

    Returns:
        tuple[dict, dict]: A tuple containing the entity URI-to-embedding mapping
                           and the relation URI-to-embedding mapping.
    """
    entity_df = pd.read_csv(entity_file)
    relation_df = pd.read_csv(relation_file)

    entity_mapping = {}
    for _, row in entity_df.iterrows():
        uri = row['entity'].strip("<>")
        idx = int(row['index'])
        if 0 <= idx < len(entity_emb_np):
            entity_mapping[uri] = entity_emb_np[idx]
        else:
            print(f"[WARNING] Entity index out of range: {uri} -> {idx}")

    relation_mapping = {}
    for _, row in relation_df.iterrows():
        uri = row['relation'].strip("<>")
        idx = int(row['index'])
        if 0 <= idx < len(relation_emb_np):
            relation_mapping[uri] = relation_emb_np[idx]
        else:
            print(f"[WARNING] Relation index out of range: {uri} -> {idx}")

    return entity_mapping, relation_mapping


def get_top_k_entities(entity_mapping, answer_uri, k=50, chunk_size=1500):
    """
    Finds the top k most similar entities to a given answer URI based on cosine similarity.

    Args:
        entity_mapping (dict): A dictionary mapping entity URIs to their embeddings.
        answer_uri (str): The URI of the entity to find neighbors for.
        k (int): The number of top similar entities to return.
        chunk_size (int): The batch size for processing similarities to manage memory.

    Returns:
        list: A list of the top k most similar entity URIs.
    """
    answer_uri = answer_uri.strip("<>")
    entity_mapping = {k.strip("<>"): v for k, v in entity_mapping.items()}

    if answer_uri not in entity_mapping:
        print(f"[WARNING] Answer URI not found in entity_mapping: {answer_uri}")

        # Log the missing answer URI in an error file
        with open("error_log.txt", "a") as error_file:
            error_file.write(f"Missing Answer URI: {answer_uri}\n")
        return []

    answer_emb = entity_mapping[answer_uri]
    answer_norm = answer_emb / np.linalg.norm(answer_emb)

    all_uris = list(entity_mapping.keys())
    similarities = []

    for start in range(0, len(all_uris), chunk_size):
        end = start + chunk_size
        uris_chunk = all_uris[start:end]
        emb_chunk = np.stack([entity_mapping[u] for u in uris_chunk])
        norms_chunk = np.linalg.norm(emb_chunk, axis=1, keepdims=True)
        emb_chunk_norm = emb_chunk / (norms_chunk + 1e-12)

        chunk_sims = np.dot(emb_chunk_norm, answer_norm)
        for i, uri_val in enumerate(uris_chunk):
            similarities.append((uri_val, chunk_sims[i]))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k_uris = [t[0] for t in similarities[:k]]

    return top_k_uris


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
                continue

            s = binding.get('s', {}).get('value')
            p = binding.get('p', {}).get('value')
            o = o_val

            if s and p and o:
                all_raw_triples.add((s, p, o))

    return list(all_raw_triples), {}


def fetch_triples_via_sparql(top_entities, sparql_endpoint, predicates=None, limit=111, graph_uri=None):
    """
    A wrapper function to fetch triples from a SPARQL endpoint using asyncio.

    This function initializes an asyncio event loop to run the asynchronous
    `fetch_focused_triples` function.

    Args:
        top_entities (list): The list of seed entity URIs.
        sparql_endpoint (str): The URL of the SPARQL endpoint.
        predicates (list, optional): A list of predicates to filter by. Defaults to None.

    Returns:
        tuple[list, dict]: A tuple containing the list of fetched triples and an empty dictionary.
    """
    if predicates is None:
        predicates = []

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop = asyncio.get_event_loop()

    # Use the focused fetching logic. It naturally handles the case of empty predicates.
    triples, _ = loop.run_until_complete(fetch_focused_triples(top_entities, predicates, sparql_endpoint, graph_uri))
    return triples, {}


# def embed_subgraph(subgraph_triples, entity_mapping, relation_mapping):
#     """
#         Converts a list of raw triples into their corresponding embedding representations.
#
#         For each triple, it looks up the embeddings for its head, relation, and tail.
#         It skips triples where any of the components cannot be found in the provided mappings.
#
#         Args:
#             subgraph_triples (list): A list of tuples, where each tuple is a (head, relation, tail) triple.
#             entity_mapping (dict): A dictionary mapping entity URIs to embeddings.
#             relation_mapping (dict): A dictionary mapping relation URIs to embeddings.
#
#         Returns:
#             list: A list of tuples, where each tuple contains the (head_emb, rel_emb, tail_emb, h_uri, r_uri, t_uri).
#      """
#     print(f"[INFO] Embedding {len(subgraph_triples)} triples")
#     triple_embeddings = []
#
#     if not entity_mapping:
#         print("[ERROR] Entity mapping is empty. Cannot create default entity embedding or proceed.")
#         return []
#     if not relation_mapping:
#         print("[ERROR] Relation mapping is empty. Cannot create default relation embedding or proceed.")
#         return []
#
#     default_entity_emb = np.zeros((next(iter(entity_mapping.values())).shape[0],))
#     default_relation_emb = np.zeros((next(iter(relation_mapping.values())).shape[0],))
#
#     missing_head = 0
#     missing_rel = 0
#     missing_tail = 0
#
#     for triple_data in subgraph_triples:
#         if len(triple_data) < 3:
#             print(f"[WARNING] Skipping malformed triple data (less than 3 elements): {triple_data}")
#             continue
#
#         h_uri, r_uri, t_uri = triple_data[0], triple_data[1], triple_data[2]
#
#         h_emb_orig = entity_mapping.get(h_uri.strip("<>"), default_entity_emb)
#         r_emb_orig = relation_mapping.get(r_uri.strip("<>"), default_relation_emb)
#         t_emb_orig = entity_mapping.get(t_uri.strip("<>"), default_entity_emb)
#
#         is_h_default = np.array_equal(h_emb_orig, default_entity_emb)
#         is_r_default = np.array_equal(r_emb_orig, default_relation_emb)
#         is_t_default = np.array_equal(t_emb_orig, default_entity_emb)
#
#         if is_h_default:
#             missing_head += 1
#         if is_r_default:
#             missing_rel += 1
#         if is_t_default:
#             missing_tail += 1
#
#         if is_h_default or is_r_default or is_t_default:
#             continue
#
#         h_emb = np.asarray(h_emb_orig).flatten()
#         r_emb = np.asarray(r_emb_orig).flatten()
#         t_emb = np.asarray(t_emb_orig).flatten()
#
#         if h_emb.shape != r_emb.shape or h_emb.shape != t_emb.shape:
#             print(
#                 f"[ERROR] Embedding shape mismatch after flatten: h={h_emb.shape}, r={r_emb.shape}, t={t_emb.shape} for triple ({h_uri}, {r_uri}, {t_uri})")
#             continue
#
#         triple_embeddings.append((h_emb, r_emb, t_emb, h_uri, r_uri, t_uri))
#
#     print(f"Generated embeddings for {len(triple_embeddings)} triples.")
#     print(f"Total Missing Head URIs encountered (attempted default): {missing_head}")
#     print(f"Total Missing Relation URIs encountered (attempted default): {missing_rel}")
#     print(f"Total Missing Tail URIs encountered (attempted default): {missing_tail}")
#     return triple_embeddings


def embed_subgraph(subgraph_triples, entity_mapping, relation_mapping):
    """
    Converts a list of raw triples into their corresponding embedding representations.
    It skips triples where any of the components cannot be found in the provided mappings
    and returns detailed statistics on what was missing.

    Args:
        subgraph_triples (list): A list of tuples, where each tuple is a (head, relation, tail) triple.
        entity_mapping (dict): A dictionary mapping entity URIs to embeddings.
        relation_mapping (dict): A dictionary mapping relation URIs to embeddings.

    Returns:
        tuple[list, dict]: A tuple containing the list of successfully embedded triples
                           and a dictionary with statistics on missing embeddings.
    """
    print(f"[INFO] Embedding {len(subgraph_triples)} triples")
    triple_embeddings = []

    if not entity_mapping or not relation_mapping:
        print("[ERROR] Entity or relation mapping is empty. Cannot proceed.")
        error_reason = "Entity mapping is empty." if not entity_mapping else "Relation mapping is empty."
        return [], {"error": error_reason, "total_triples_processed": len(subgraph_triples)}

    # Initialize statistics dictionary
    missing_stats = {
        "total_triples_processed": len(subgraph_triples),
        "triples_with_any_missing_part": 0,
        "successfully_embedded_triples": 0,
        "missing_head_uris": [],
        "missing_relation_uris": [],
        "missing_tail_uris": []
    }

    # Use sets for efficient unique URI storage
    missing_h_set, missing_r_set, missing_t_set = set(), set(), set()

    for triple_data in subgraph_triples:
        if len(triple_data) < 3:
            print(f"[WARNING] Skipping malformed triple data: {triple_data}")
            continue

        h_uri, r_uri, t_uri = triple_data[0], triple_data[1], triple_data[2]

        h_emb_orig = entity_mapping.get(h_uri.strip("<>"))
        r_emb_orig = relation_mapping.get(r_uri.strip("<>"))
        t_emb_orig = entity_mapping.get(t_uri.strip("<>"))

        is_h_missing = h_emb_orig is None
        is_r_missing = r_emb_orig is None
        is_t_missing = t_emb_orig is None

        if is_h_missing or is_r_missing or is_t_missing:
            missing_stats["triples_with_any_missing_part"] += 1
            if is_h_missing: missing_h_set.add(h_uri)
            if is_r_missing: missing_r_set.add(r_uri)
            if is_t_missing: missing_t_set.add(t_uri)
            continue  # Skip this triple as it's incomplete

        h_emb = np.asarray(h_emb_orig).flatten()
        r_emb = np.asarray(r_emb_orig).flatten()
        t_emb = np.asarray(t_emb_orig).flatten()

        if h_emb.shape != r_emb.shape or h_emb.shape != t_emb.shape:
            print(f"[ERROR] Embedding shape mismatch for triple ({h_uri}, {r_uri}, {t_uri})")
            continue

        triple_embeddings.append((h_emb, r_emb, t_emb, h_uri, r_uri, t_uri))

    # Finalize statistics
    missing_stats["successfully_embedded_triples"] = len(triple_embeddings)
    missing_stats["missing_head_uris"] = sorted(list(missing_h_set))
    missing_stats["missing_relation_uris"] = sorted(list(missing_r_set))
    missing_stats["missing_tail_uris"] = sorted(list(missing_t_set))

    print(
        f"Generated embeddings for {len(triple_embeddings)}/{len(subgraph_triples)} triples. Triples with missing parts: {missing_stats['triples_with_any_missing_part']}")

    return triple_embeddings, missing_stats


def format_triple_for_sparql(triple_parts):
    """Formats a triple tuple/list into a SPARQL triple string."""
    s, p, o = triple_parts
    return f"<{s}> <{p}> <{o}> ."


def execute_sparql_update(endpoint_url, update_query):
    """
    Executes a SPARQL UPDATE query (INSERT/DELETE) against an endpoint.

    Args:
        endpoint_url (str): The URL of the SPARQL update endpoint.
        update_query (str): The SPARQL UPDATE query string.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(update_query)
    sparql.setMethod(POST)
    sparql.setRequestMethod('POST')
    try:
        sparql.query()
        return True
    except Exception as e:
        print(f"ERROR executing update: {update_query[:250]}... \nError: {e}")
        return False


def load_checkpoint(cp_file):
    return set(json.load(open(cp_file))) if os.path.exists(cp_file) else set()


def save_checkpoint(cp_file, done_set):
    with open(cp_file, "w") as f:
        json.dump(sorted(list(done_set)), f)

def send_sparql_update(endpoint_url, sparql_update):
    headers = {'Content-Type': 'application/sparql-update'}
    response = requests.post(endpoint_url, data=sparql_update, headers=headers)
    if response.status_code != 200:
        print(f"[ERROR] SPARQL update failed: {response.text}")
    return response


def clear_graph(endpoint_url, graph_uri):
    sparql = f"CLEAR GRAPH <{graph_uri}>"
    send_sparql_update(endpoint_url, sparql)


def log_changes_to_file(log_path, question_id, k, ground_truth_answer, attack_type, removed, added, sparql_query, add_mode,
                         centralities_before, centralities_after):
    """
        Logs a dictionary of changes to a JSONL file.

        Each call appends a new JSON object to the file, making it easy to parse later.

        Args:
            log_path (str): The path to the log file.
            **kwargs: A dictionary of key-value pairs to log.
    """
    entry = {
        "question_id": question_id,
        "k": k,
        "ground_truth_answer": ground_truth_answer,
        "attack_type": attack_type,
        "triples_removed": removed,
        "triples_added": added,
        "sparql_query": sparql_query,
        "add_mode": add_mode,
        "centralities_before_attack": centralities_before,
        "centralities_after_attack": centralities_after
    }

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
