import json
import os
import csv
import numpy as np
import torch
import random
import pandas as pd

import asyncio
import aiohttp

from sentence_transformers import SentenceTransformer
from SPARQLWrapper import SPARQLWrapper, JSON

from environment import KGQARLEnvironment
from rl_agent import KGQARLAgent

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

###############################################################################
# LOADING THE KGE MODEL EMBEDDINGS
###############################################################################
def load_embeddings_from_models(model_dir):
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


###############################################################################
# TOP-K NEAREST ENTITIES IN EMBEDDING SPACE
###############################################################################
def get_top_k_entities(entity_mapping, answer_uri, k=15, chunk_size=1500):
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


###############################################################################
# ASYNCHRONOUS SPARQL QUERIES
###############################################################################
async def fetch_sparql(session, query, endpoint):
    """Send an asynchronous SPARQL query."""
    headers = {'Accept': 'application/json'}
    async with session.get(endpoint, params={'query': query}, headers=headers) as response:
        return await response.json()

async def fetch_entity_types(top_entities, sparql_endpoint):
    """Retrieve RDF types (ontology classes) for each entity asynchronously."""
    queries = {
        entity: f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        SELECT ?class WHERE {{
            <{entity}> rdf:type ?class .
            FILTER(STRSTARTS(STR(?class), "http://dbpedia.org/ontology/"))
        }} LIMIT 1
        """ for entity in top_entities
    }

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_sparql(session, q, sparql_endpoint) for q in queries.values()]
        results = await asyncio.gather(*tasks)

    entity_types = {}
    for entity, result in zip(queries.keys(), results):
        if "results" in result and "bindings" in result["results"] and result["results"]["bindings"]:
            entity_types[entity] = result["results"]["bindings"][0]["class"]["value"]

    return entity_types

async def fetch_triples_for_entities(entity_types, sparql_endpoint, limit=10):
    """Fetch triples for entities, filtered by their RDF types."""
    queries = {
        entity: f"""
        SELECT ?related ?p (SAMPLE(?o) AS ?o)
        WHERE {{
            <{entity}> ?p ?related .
            ?related rdf:type <{entity_type}> .
            ?related ?p ?o .
        }}
        GROUP BY ?related ?p
        LIMIT {limit}
        """ for entity, entity_type in entity_types.items()
    }

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_sparql(session, q, sparql_endpoint) for q in queries.values()]
        results = await asyncio.gather(*tasks)

    subgraph_triples = []
    for entity, result in zip(queries.keys(), results):
        if "results" in result and "bindings" in result["results"]:
            for triple in result["results"]["bindings"]:
                subgraph_triples.append((
                    entity,
                    triple["p"]["value"],
                    triple["related"]["value"],
                    triple["o"]["value"]
                ))

    return subgraph_triples

def fetch_triples_via_sparql(top_entities, sparql_endpoint, limit=10):
    """Wrapper to run async SPARQL fetching in a synchronous environment."""
    loop = asyncio.get_event_loop()
    entity_types = loop.run_until_complete(fetch_entity_types(top_entities, sparql_endpoint))
    return loop.run_until_complete(fetch_triples_for_entities(entity_types, sparql_endpoint, limit))


###############################################################################
# EMBED THE EXTRACTED SUBGRAPH
###############################################################################
def embed_subgraph(subgraph_triples, entity_mapping, relation_mapping):
    triple_embeddings = []
    default_entity_emb = np.zeros((next(iter(entity_mapping.values())).shape[0],))
    default_relation_emb = np.zeros((next(iter(relation_mapping.values())).shape[0],))

    missing_head = 0
    missing_rel = 0
    missing_tail = 0

    for (h_uri, r_uri, t_uri, _) in subgraph_triples:
        h_emb = entity_mapping.get(h_uri.strip("<>"), default_entity_emb)
        r_emb = relation_mapping.get(r_uri.strip("<>"), default_relation_emb)
        t_emb = entity_mapping.get(t_uri.strip("<>"), default_entity_emb)

        if np.array_equal(h_emb, default_entity_emb):
            missing_head += 1
        if np.array_equal(r_emb, default_relation_emb):
            missing_rel += 1
        if np.array_equal(t_emb, default_entity_emb):
            missing_tail += 1

        if (np.array_equal(h_emb, default_entity_emb) or
            np.array_equal(r_emb, default_relation_emb) or
            np.array_equal(t_emb, default_entity_emb)):
            continue  # Skip incomplete triples

        h_emb = np.asarray(h_emb).flatten()
        r_emb = np.asarray(r_emb).flatten()
        t_emb = np.asarray(t_emb).flatten()

        triple_embeddings.append((h_emb, r_emb, t_emb, h_uri, r_uri, t_uri))

    print(f"Generated embeddings for {len(triple_embeddings)} triples.")
    print(f"Missing Head Embeddings: {missing_head}")
    print(f"Missing Relation Embeddings: {missing_rel}")
    print(f"Missing Tail Embeddings: {missing_tail}")
    return triple_embeddings


###############################################################################
# RL TRAINING LOGIC
###############################################################################
def train_rl_agent(agent, question, question_embedding, subgraph_embeddings, ground_truth_answer, episodes, output_file,attack_mode):
    if len(subgraph_embeddings) == 0:
        print("[WARNING] No subgraph embeddings to train on.")
        return

    print(f"\n[DEBUG] Training RL on: {question}")
    print(f"[DEBUG] Ground Truth Answer: {ground_truth_answer}")

    agent.load_experience()
    env = KGQARLEnvironment(
        question_text=question,
        question_embedding=question_embedding,
        answer_embedding=np.mean([t[0] for t in subgraph_embeddings], axis=0),
        subgraph_embeddings=subgraph_embeddings,
        ground_truth_answer=ground_truth_answer,
        attack_mode=attack_mode,
        add_mode=add_mode
    )

    final_false_answer = None  # Track the final false answer
    final_step_count = 0   # Track number of triples removed

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        info = {}  # Ensure `info` is initialized before use

        while not done:
            step_count += 1
            similarity_scores = env.compute_influence_scores()
            action = agent.act(state, similarity_scores)

            next_state, reward, done, info = env.step(action)  # `info` is always assigned
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

            if step_count >= 20:
                print("[WARNING] Max steps reached in this episode, forcing termination.")
                done = True

        print(f"Episode {episode + 1} - Total Reward: {total_reward}")

        # Update the final false answer and removed count at the last episode
        if episode == episodes - 1:
            final_false_answer = info.get("false_answer", None)
            final_step_count = step_count

    if final_false_answer:
        with open(output_file, "a") as f:
            f.write(f"--- Final Episode {episodes} ---\n")
            f.write(f"Question: {question}\n")
            f.write(f"True Answer: {ground_truth_answer}\n")
            f.write(f"False Answer: {final_false_answer}\n")
            f.write(f"Steps Taken: {final_step_count}\n")
            if attack_mode == "add":
                added_count = len(env.added_triples)
                f.write(f"Triples Added: {added_count}\n")
                for triple in env.added_triples:
                    f.write(f"  {triple}\n")
            elif attack_mode == "remove":
                removed_count = len(env.removed_triples)
                f.write(f"Triples Removed: {removed_count} \n")
                for idx in env.removed_triples:
                    f.write(f"  {env.uris[idx]}\n")
            f.write("\n")
            f.flush()
    agent.save_experience()


###############################################################################
# MAIN PIPELINE
###############################################################################
def main():
    attack_mode = input("Enter attack mode ('add' for addition, 'remove' for removal): ").strip().lower()
    if attack_mode not in ["add", "remove"]:
        print("Invalid mode, defaulting to 'remove'.")
        attack_mode = "remove"

    global add_mode
    if attack_mode == "add":
        add_mode = input(
            "Enter add mode ('semantic' for inference pattern or 'cairage' for CAIRAGE style): ").strip().lower()
        if add_mode not in ["semantic", "cairage"]:
            print("Invalid add mode, defaulting to 'semantic'.")
            add_mode = "semantic"
    else:
        add_mode = None

    embedding_dir = "../embeddings/DBpediaKGE/"
    entity_file = "../embeddings/DBpediaKGE/entity_to_idx.csv"
    relation_file = "../embeddings/DBpediaKGE/relation_to_idx.csv"
    question_file = "../input_data/qald_9_plus_test_dbpedia.json"
    sparql_endpoint = "https://dbpedia-2016-10.data.dice-research.org/sparql"

    output_dir = f"../output_{attack_mode}"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"../output_{attack_mode}/false_answer_train.txt"
    processed_questions_file = f"../output_{attack_mode}/processed_questions.txt"
    model_path = f"../output_{attack_mode}/trained_model.keras"

    print("Loading KGE model embeddings...")
    entity_embeddings_torch, relation_embeddings_torch = load_embeddings_from_models(embedding_dir)
    entity_embeddings_np = entity_embeddings_torch.cpu().numpy()
    relation_embeddings_np = relation_embeddings_torch.cpu().numpy()

    if os.path.exists(processed_questions_file):
        with open(processed_questions_file, "r") as f:
            processed_questions = set(f.read().splitlines())
    else:
        processed_questions = set()

    print("Mapping indices -> embeddings...")
    entity_mapping, relation_mapping = map_indices_to_embeddings(
        entity_file, relation_file, entity_embeddings_np, relation_embeddings_np
    )

    model = SentenceTransformer('all-MiniLM-L6-v2')
    persistent_agent = KGQARLAgent(state_size=18052, action_size=1000, model_path=model_path)

    with open(question_file, 'r', encoding='utf-8') as f:
        qald_data = json.load(f)

    for qidx, q_data in enumerate(qald_data['questions'], start=1):
        question_entry = next((q for q in q_data['question'] if q.get('language', '').lower() == 'en'), None)
        if not question_entry:
            continue
        question = question_entry['string']
        if question in processed_questions:
            continue
        answers = q_data.get('answers', [])
        if not answers:
            continue
        answer_block = answers[0]
        vars_list = answer_block.get("head", {}).get("vars", [])
        bindings = answer_block.get("results", {}).get("bindings", [])
        if not vars_list or not bindings:
            continue
        answer_var = vars_list[0]
        ground_truth_answers = [b[answer_var]['value'] for b in bindings if answer_var in b]
        if not ground_truth_answers:
            continue
        ground_truth_answer = random.choice(ground_truth_answers)
        if not (ground_truth_answer.startswith("http://") or ground_truth_answer.startswith("https://")):
            continue
        top_k_uris = get_top_k_entities(entity_mapping, ground_truth_answer, k=15, chunk_size=1500)
        if not top_k_uris:
            continue
        try:
            subgraph_triples = fetch_triples_via_sparql(top_entities=top_k_uris, sparql_endpoint=sparql_endpoint,
                                                        limit=1000)
        except Exception as e:
            print(f"[ERROR] SPARQL error: {e}")
            continue
        if not subgraph_triples:
            continue
        subgraph_with_embeddings = embed_subgraph(subgraph_triples, entity_mapping, relation_mapping)
        if not subgraph_with_embeddings:
            continue
        question_embedding = model.encode(question)
        train_rl_agent(
            agent=persistent_agent,
            question=question,
            question_embedding=question_embedding,
            subgraph_embeddings=subgraph_with_embeddings,
            ground_truth_answer=ground_truth_answer,
            episodes=10,
            output_file=output_file,
            attack_mode=attack_mode
        )
        persistent_agent.save_model()
        with open(processed_questions_file, "a") as f:
            f.write(question + "\n")
    print("\nAll questions processed.")


if __name__ == "__main__":
    main()