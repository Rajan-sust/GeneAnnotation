#!/bin/bash

# Clone the repo from here (https://github.com/Rajan-sust/GeneAnnotation) and run the benchmarking. 

GeneAnnotation=/Users/rashedulislam/Documents/git_repos/GeneAnnotation
output=/Users/rashedulislam/Documents/git_repos/GeneAnnotation

cd $GeneAnnotation

# Build database and run annotation
# for model in prot_bert prot_t5 esm2_small esm2_large; do
for model in esm2_large; do  
    for f in data/uniprot_sprot_5000.fasta; do 
        echo "Processing file: $f with model: $model"
        filename=$(basename "$f" .fasta)  
        echo $filename

        #Build database with CPU optimization for 24 cores
        python3 db_build.py \
            --fasta_path "$f" \
            --collection "${filename}_${model}_tbl" \
            --model_name "$model" \
            --batch_size 50 \
            --embedding_batch_size 12 \
            --num_workers 20

        # Annotate with the same model
        python3 annotate.py \
            --input_faa "data/uniprot_sprot_5000.fasta" \
            --collection "${filename}_${model}_tbl" \
            --output_file $output/"${filename}_${model}_results.tsv" \
            --model_name "$model"
    done 
done

