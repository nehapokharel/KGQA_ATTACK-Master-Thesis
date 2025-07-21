import json
import requests
from SPARQLWrapper import SPARQLWrapper, JSON


def get_sparql_query(question: str, timeout: int = 60) -> str:
    """
    Calls an API to convert a natural language question into a SPARQL query.
    """
    api_url = 'http://kgqa.cs.upb.de:8185/fetch-sparql'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'lang': 'en', 'query': question}

    try:
        response = requests.post(api_url, headers=headers, data=data, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.exceptions.Timeout:
        raise Exception(f"API call timed out after {timeout} seconds.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch SPARQL query: {e}")

# "https://dbpedia-2016-10.data.dice-research.org/sparql/"

def query_dbpedia(sparql_query: str, endpoint_url: str = "http://localhost:3030/ds/sparql", timeout: int = 60):
    """
    Executes a SPARQL query against a specified DBpedia endpoint.
    """
    try:
        sparql = SPARQLWrapper(endpoint_url)
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        sparql.setTimeout(timeout)

        results = sparql.query().convert()
        answers = []
        for result in results["results"]["bindings"]:
            answer = {key: result[key]["value"] for key in result}
            answers.append(answer)
        return answers
    except Exception as e:
        raise Exception(f"Failed to query DBpedia: {e}")


def load_questions_from_json(file_path: str) -> dict:
    """
    Loads the entire JSON data structure from a file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['questions']


def get_english_questions(questions_data: list) -> list:
    """
    Extracts only the English questions and their IDs from the raw data.
    """
    english_questions = []
    for question_data in questions_data:
        for question in question_data['question']:
            if question['language'] == 'en':
                english_questions.append({
                    'id': question_data['id'],
                    'question': question['string']
                })
                break
    return english_questions

if __name__ == "__main__":
    json_file_path = '/home/neha2022/thesis/input_data/qald_9_plus_train_dbpedia.json'
    output_file_path = '/home/neha2022/thesis/input_data/qa_groundtruth_train.jsonl'

    try:
        all_questions_data = load_questions_from_json(json_file_path)
        english_questions_list = get_english_questions(all_questions_data)
        print(f"Successfully loaded {len(english_questions_list)} English questions.")
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        english_questions_list = []
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {e}")
        english_questions_list = []

    if english_questions_list:
        records_saved_count = 0
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for item in english_questions_list:
                question_id = item['id']
                question_text = item['question']

                print(f"Processing Question ID: {question_id}")
                print(f"Question: {question_text}")

                try:
                    # Get the SPARQL query
                    print("Step 1: Generating SPARQL query...")
                    sparql_query = get_sparql_query(question_text)  # Uses the new function with timeout
                    print(f"Generated SPARQL Query:\n{sparql_query}")

                    # Execute the query to get the answer
                    print("Step 2: Executing query against local DBpedia...")
                    answers = query_dbpedia(sparql_query)  # Uses the new function with timeout

                    if answers:
                        result_data = {
                            'id': question_id,
                            'question': question_text,
                            'query': sparql_query,
                            'answer': answers
                        }
                        outfile.write(json.dumps(result_data) + '\n')
                        records_saved_count += 1
                        print(f"SUCCESS: Found {len(answers)} answer(s). Record saved.")
                    else:
                        print("INFO: Query executed successfully, but no answers were found. Record not saved.")

                except Exception as e:
                    # This block will now catch timeouts
                    print(f"[ERROR] Could not process question ID {question_id}. Record not saved. Reason: {e}")

        print("\n" + "=" * 60)
        print("Processing complete.")
        print(f"Total records saved to '{output_file_path}': {records_saved_count}")