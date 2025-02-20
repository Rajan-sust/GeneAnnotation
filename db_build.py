from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import re
from qdrant_client import QdrantClient, models
from Bio import SeqIO
import argparse
import platform


def parse_args():
    parser = argparse.ArgumentParser(description='Build protein vector database from FASTA file')
    parser.add_argument('--fasta_path', type=str, required=True, help='Path to input FASTA file')
    parser.add_argument('--db_name', type=str, required=True, help='Name of the database to create')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for processing sequences')
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

        # Load model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.model = BertModel.from_pretrained("Rostlab/prot_bert")
        self.model = self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

    def get_protein_embedding(self, sequence):
        # Add spaces between amino acids
        sequence = " ".join(re.sub(r"[UZOB]", "X", sequence))

        # Encode sequence
        encoded_input = self.tokenizer(sequence, return_tensors='pt')

        # Move encoded_input to the same device as the model
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings.cpu()

        e = embeddings.numpy()[0]
        return normalize_l2(e)


def build_vector_database_of_protein(fasta_file_path, db_name, batch_size=50, device=None):
    # Initialize Qdrant client
    qdrant_client = QdrantClient("http://localhost:6333")

    # Initialize embedder
    embedder = ProteinEmbedder(device)

    # Create collection if it doesn't exist
    if not qdrant_client.collection_exists(db_name):
        qdrant_client.create_collection(
            collection_name=db_name,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.DOT),
        )

    points = []
    for idx, seq_record in enumerate(SeqIO.parse(fasta_file_path, "fasta")):
        print(f"Processing sequence {idx + 1}: {seq_record.description}")

        points.append(
            models.PointStruct(
                id=idx,
                vector=embedder.get_protein_embedding(str(seq_record.seq)),
                payload={'protein_info': seq_record.description}
            )
        )

        # Upload batch when it reaches batch_size
        if len(points) >= batch_size:
            qdrant_client.upload_points(
                collection_name=db_name,
                points=points
            )
            points = []

    # Upload any remaining points
    if points:
        qdrant_client.upload_points(
            collection_name=db_name,
            points=points
        )


if __name__ == '__main__':
    args = parse_args()
    device = get_device(args.device)

    build_vector_database_of_protein(
        fasta_file_path=args.fasta_path,
        db_name=args.db_name,
        batch_size=args.batch_size,
        device=device
    )
