### Prerequisite
- Docker (https://docs.docker.com/engine/install/)
- Prodigal (https://github.com/hyattpd/prodigal)
- HF Access Token (https://huggingface.co/docs/hub/en/security-tokens)

### Clone the repo
```
git clone https://github.com/Rajan-sust/GeneAnnotation.git
cd GeneAnnotation
```

### Python Environment Creation & Pkg Installation
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Pull the Qdrant Vector Database Image & run the db server
```
# Prerequisite: Docker Installation
docker pull qdrant/qdrant

docker run -d -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### Build Vector DB of Protein

###### Features

- Support for multiple protein embedding models:
  - ProtBERT (Rostlab/prot_bert)
  - ESM2 (facebook/esm2)
- Multi-threaded processing of FASTA files
- Integration with Qdrant vector database
- Flexible and modular architecture

###### Command Line Arguments

- `--fasta_path`: Path to input FASTA file (required)
- `--db_name`: Name of the database to create (required)
- `--model_name`: Protein embedding model to use (choices: "prot_bert", "esm2", default: "prot_bert")
- `--batch_size`: Batch size for processing sequences (default: 50)
- `--qdrant_url`: URL for Qdrant server (default: "http://localhost:6333")
- `--num_threads`: Number of worker threads (default: 2)

###### Usage
```
python3 db_build.py --fasta_path ~/Desktop/UP000000212_1234679.fasta --db_name my_esm2_db --model_name esm2 --num_threads 2
```



### Gene Predictions
```
prodigal -i my.genome.fna  -g 11 -a protein.translations.faa
```

### Protein Annotation


### Command Line Arguments

- `--input_faa`: Path to input FASTA file containing protein sequences (required)
- `--db_name`: Name of the Qdrant collection to search against (required)
- `--output_file`: Path to output TSV file for results (required)
- `--threshold`: Similarity threshold for annotations (default: 0.98)
- `--batch_size`: Number of sequences to process in each batch (default: 5)
- `--num_threads`: Number of parallel threads to use (default: 2)
- `--model_name`: Protein embedding model to use ["prot_bert", "esm2"] (default: "esm2")
- `--qdrant_url`: URL for Qdrant server (default: "http://localhost:6333")

###### Example

```
python3 annotate.py --input_faa data/proteins.faa \
                    --db_name my_esm2_db \
                    --output_file results.tsv \
                    --threshold 0.95 \
                    --batch_size 10 \
                    --num_threads 4 \
                    --model_name esm2
```

###### Output Format

The tool generates a TSV file with the following columns:
- `Query_ID`: Identifier of the input sequence
- `Annotation`: Predicted protein annotation
- `Similarity_Score`: Similarity score (0-1) with the matched database entry
- `Status`: Processing status ('success', 'below_threshold', 'embedding_failed', or 'error')

[comment]: <> (## Project Structure)

[comment]: <> (- `main.py`: Entry point for the application)

[comment]: <> (- `config.py`: Configuration and argument parsing)

[comment]: <> (- `embedders.py`: Protein embedding models)

[comment]: <> (- `database.py`: Vector database operations)

[comment]: <> (- `processor.py`: Protein sequence processing workflow)

[comment]: <> (## Adding a New Embedding Model)

[comment]: <> (To add a new embedding model:)

[comment]: <> (1. Implement a new class in `embedders.py` that inherits from `ProteinEmbedder`)

[comment]: <> (2. Add the new model to the `get_embedder` factory function)

[comment]: <> (3. Update the command line choices in `config.py`)
