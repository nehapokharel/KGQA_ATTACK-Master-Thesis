## Evaluating Robustness on Knowledge Graph Based Question Answering System

This is the Master Thesis which performs adversarial attacks on Knowledge Graph Question Answering (KG-QA) systems using the QALD-9 dataset and DBpedia embeddings. It focuses on manipulating the knowledge graph to evaluate the robustness of KG-QA systems.

### Input Datasets


### 1. DBpedia Entity Embeddings
- **URL**: [DBpedia Embeddings](https://files.dice-research.org/projects/DiceEmbeddings/DBpedia)
- **Description**: Pre-trained embeddings of DBpedia.

### 2. QALD-9 Question-Answer Dataset
- **URL**: [Question Answer](https://github.com/KGQA/QALD_9_plus/blob/main/data/qald_9_plus_test_dbpedia.json)
- **Description**: Contains natural language questions and their SPARQL queries over DBpedia. Used for evaluation before and after adversarial attacks.

### 3. DBpedia RDF Dump (2016-10)
- **URL**: [DBpedia Dump for QALD-9](https://files.dice-research.org/datasets/DBPedia/dbpedia-2016-10.nt.zst)
- **Description**: The full DBpedia knowledge graph in RDF N-Triples format. Required for SPARQL execution and graph modification.



## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nehapokharel/KGQA_ATTACK-Master-Thesis.git
   
