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
```
python3 annotate.py --input_faa protein.translations.faa \
                    --db_name prot_vec \
                    --num_threads 5 \
                    --threshold 0.98 \
                    --output_file annotation.tsv
```

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



