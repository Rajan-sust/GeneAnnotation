from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import re
from qdrant_client import QdrantClient
from Bio import SeqIO
import argparse
import platform


def parse_args():
    parser = argparse.ArgumentParser(description='Annotate proteins using vector database')
    parser.add_argument('--input_faa', type=str, required=True, help='Path to input FAA file to annotate')
    parser.add_argument('--db_name', type=str, required=True, help='Name of the database to search against')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output annotation file')
    parser.add_argument('--threshold', type=float, default=0.98, help='Similarity threshold for annotations')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu'], default='cuda',
                        help='Device to use (cuda/cpu)')
    return parser.parse_args()


def get_device(device_preference='cuda'):
    if device_preference == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available() and platform.system() == 'Darwin':
        return torch.device('mps')
    return torch.device('cpu')


def normalize_l2(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm


class ProteinEmbedder:
    def __init__(self, device=None):
        self.device = device if device is not None else get_device()
        print(f"Using device: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.model = BertModel.from_pretrained("Rostlab/prot_bert")
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_protein_embedding(self, sequence):
        sequence = " ".join(re.sub(r"[UZOB]", "X", sequence))
        encoded_input = self.tokenizer(sequence, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            outputs = self.model(**encoded_input)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings.cpu()

        return normalize_l2(embeddings.numpy()[0])


def annotate_proteins(input_faa, db_name, output_file, threshold=0.98, device=None):
    # Initialize Qdrant client and embedder
    qdrant_client = QdrantClient("http://localhost:6333")
    embedder = ProteinEmbedder(device)

    # Open output file and write header
    with open(output_file, 'w') as out_f:
        out_f.write("Query_ID\tTop_Match_ID\tSimilarity_Score\tAnnotation\n")

        # Process sequences one by one
        for idx, seq_record in enumerate(SeqIO.parse(input_faa, "fasta")):
            print(f"Processing query sequence {idx + 1}: {seq_record.description}")

            # Get embedding for current sequence
            embedding = embedder.get_protein_embedding(str(seq_record.seq))

            # Search database
            search_results = qdrant_client.search(
                collection_name=db_name,
                query_vector=embedding,
                limit=1
            )

            # Write results to output file
            for result in search_results:
                if result.score >= threshold:
                    out_f.write(f"{seq_record.description}\t{result.payload['protein_info']}\t"
                                f"{result.score:.4f}\t{result.payload['protein_info']}\n")
                else:
                    out_f.write(f"{seq_record.description}\tNo match above threshold\t"
                                f"{result.score:.4f}\tNo annotation\n")


if __name__ == '__main__':
    args = parse_args()
    device = get_device(args.device)

    annotate_proteins(
        input_faa=args.input_faa,
        db_name=args.db_name,
        output_file=args.output_file,
        threshold=args.threshold,
        device=device
    )