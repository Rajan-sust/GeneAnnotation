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

### Build Protein Vector DB
```
python db_build.py --fasta ~/Desktop/bacteria.fasta --db_name prot_vec --num_threads 4
```

### Gene Predictions
```
prodigal -i my.genome.fna  -g 11 -a protein.translations.faa
```

### Protein Annotation
```
python3 annotate.py --input_faa protein.translations.faa  --db_name prot_vec --output_file annotation.tsv
```


