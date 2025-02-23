from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import re
from qdrant_client import QdrantClient
from Bio import SeqIO
import argparse
import platform
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description='Annotate proteins using vector database')
    parser.add_argument('--input_faa', type=str, required=True, help='Path to input FAA file to annotate')
    parser.add_argument('--db_name', type=str, required=True, help='Name of the database to search against')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output TSV file')
    parser.add_argument('--threshold', type=float, default=0.97, help='Similarity threshold for annotations')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of sequences to process in each batch')
    parser.add_argument('--num_threads', type=int, default=2,
                        help='Number of threads to use (default: number of CPU cores - 1)')
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available() and platform.system() == 'Darwin':
        return torch.device('mps')
    return torch.device('cpu')


def normalize_l2(x):
    norm = np.linalg.norm(x)
    return x / (norm if norm > 0 else 1)


class ProteinEmbedder:
    def __init__(self):
        self.device = get_device()
        print(f"Using device: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.model = BertModel.from_pretrained("Rostlab/prot_bert")
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_protein_embedding(self, sequence: str) -> np.ndarray:
        try:
            sequence = " ".join(re.sub(r"[UZOB]", "X", sequence))
            encoded_input = self.tokenizer(sequence, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            with torch.no_grad():
                outputs = self.model(**encoded_input)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = embeddings.cpu()

            return normalize_l2(embeddings.numpy()[0])
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None


def process_sequence_batch(args: Tuple[List[str], ProteinEmbedder, QdrantClient, float]) -> List[dict]:
    """Process a batch of sequences and return their annotations."""
    sequences, embedder, qdrant_client, threshold, db_name = args
    results = []

    for seq_record in sequences:
        try:
            seq_id = seq_record.id
            embedding = embedder.get_protein_embedding(str(seq_record.seq))

            if embedding is None:
                results.append({
                    'Query_ID': seq_id,
                    'Annotation': 'hypothetical protein',
                    'Similarity_Score': 0.0
                })
                continue

            search_results = qdrant_client.search(
                collection_name=db_name,
                query_vector=embedding.tolist(),
                limit=1
            )

            if search_results and search_results[0].score >= threshold:
                results.append({
                    'Query_ID': seq_id,
                    'Annotation': search_results[0].payload['protein_info'],
                    'Similarity_Score': float(search_results[0].score)
                })
            else:
                results.append({
                    'Query_ID': seq_id,
                    'Annotation': 'hypothetical protein',
                    'Similarity_Score': 0.0
                })

        except Exception as e:
            print(f"Error processing sequence {seq_id}: {str(e)}")
            results.append({
                'Query_ID': seq_id,
                'Annotation': 'hypothetical protein',
                'Similarity_Score': 0.0
            })

    return results


def annotate_proteins(args):
    # Initialize clients
    qdrant_client = QdrantClient("http://localhost:6333")
    embedder = ProteinEmbedder()

    # Determine number of threads
    num_threads = args.num_threads
    print(f"Using {num_threads} threads")

    # Read all sequences
    sequences = list(SeqIO.parse(args.input_faa, "fasta"))
    total_sequences = len(sequences)
    print(f"Found {total_sequences} sequences to process")

    # Prepare batches
    batch_size = args.batch_size
    sequence_batches = [
        sequences[i:i + batch_size]
        for i in range(0, len(sequences), batch_size)
    ]

    all_results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                process_sequence_batch,
                (batch, embedder, qdrant_client, args.threshold, args.db_name)
            )
            for batch in sequence_batches
        ]

        # Process results as they complete
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                print(f"{len(all_results)} seq completed!")
            except Exception as e:
                print(f"Batch processing failed: {str(e)}")

    # Convert results to DataFrame and save
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_file, sep='\t', index=False)
    print(f"Results saved to {args.output_file}")


if __name__ == '__main__':
    args = parse_args()
    annotate_proteins(args)
