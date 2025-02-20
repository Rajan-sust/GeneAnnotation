```
git clone
cd 
```
### Python Env Creation
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```






```
docker pull qdrant/qdrant
```
```
docker run -d -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```
```buildoutcfg
 python db_build.py --fasta ~/Desktop/UP000000212_1234679.fasta --db_name prot_vec
```# GeneAnnotation
