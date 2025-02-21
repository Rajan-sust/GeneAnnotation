from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import re
from qdrant_client import QdrantClient, models
from Bio import SeqIO
import argparse
import platform
import uuid
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm
import sys
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('protein_embedder.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for tracking processing progress and errors."""
    total_sequences: int = 0
    processed_sequences: int = 0
    failed_sequences: int = 0
    empty_sequences: int = 0


class ConfigurationError(Exception):
    """Raised when there's an error in the configuration."""
    pass


class EmbeddingError(Exception):
    """Raised when there's an error during embedding generation."""
    pass


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments with input validation.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Build protein vector database from FASTA file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--fasta_path', type=str, required=True, help='Path to input FASTA file')
    parser.add_argument('--db_name', type=str, required=True, help='Name of the database to create')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for processing sequences')
    parser.add_argument('--qdrant_url', type=str, default="http://localhost:6333", help='URL for Qdrant server')

    args = parser.parse_args()

    # Validate arguments
    if not Path(args.fasta_path).exists():
        raise ConfigurationError(f"FASTA file not found: {args.fasta_path}")
    if args.batch_size < 1:
        raise ConfigurationError("Batch size must be positive")

    return args


def get_device() -> torch.device:
    """
    Determine the appropriate device for computation.

    Returns:
        torch.device: Selected computation device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available() and platform.system() == 'Darwin':
        return torch.device('mps')
    else:
        return torch.device('cpu')


def normalize_l2(x: np.ndarray) -> np.ndarray:
    """
    Normalize vector using L2 normalization.

    Args:
        x: Input vector

    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(x)
    return x / (norm if norm > 0 else 1)


class ProteinEmbedder:
    """Handles protein sequence embedding using BERT model."""

    def __init__(self):
        """
        Initialize the embedder with specified device.

        Args:
            device: Computation device to use
        """
        self.device = get_device()
        logger.info(f"Initializing ProteinEmbedder using device: {self.device}")

        try:
            self.tokenizer = BertTokenizer.from_pretrained(
                "Rostlab/prot_bert",
                do_lower_case=False
            )
            self.model = BertModel.from_pretrained("Rostlab/prot_bert")
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize BERT model: {str(e)}")

    def get_protein_embedding(self, sequence: str) -> List[float]:
        """
        Generate embedding for a protein sequence.

        Args:
            sequence: Protein sequence string

        Returns:
            List of embedding values

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            # Preprocess sequence
            sequence = " ".join(re.sub(r"[UZOB]", "X", sequence))

            # Encode sequence
            encoded_input = self.tokenizer(sequence, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**encoded_input)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = embeddings.cpu()

            # Post-process embedding
            normalized_embedding = normalize_l2(embeddings.numpy()[0])
            embedding_list = normalized_embedding.tolist()

            # Validate embedding
            if not embedding_list:
                raise EmbeddingError("Generated embedding is empty")
            if not all(isinstance(x, float) for x in embedding_list):
                raise EmbeddingError("Non-float values in embedding")
            if any(np.isnan(x) or np.isinf(x) for x in embedding_list):
                raise EmbeddingError("NaN or Inf values in embedding")

            return embedding_list

        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")


def build_vector_database_of_protein(
        fasta_file_path: str,
        db_name: str,
        qdrant_url: str = "http://localhost:6333",
        batch_size: int = 50
) -> ProcessingStats:
    """
    Build vector database from protein sequences in FASTA file.

    Args:
        fasta_file_path: Path to input FASTA file
        db_name: Name of the database to create
        qdrant_url: URL for Qdrant server
        batch_size: Number of sequences to process in each batch


    Returns:
        ProcessingStats: Statistics about the processing run
    """
    stats = ProcessingStats()

    try:
        # Initialize Qdrant client
        qdrant_client = QdrantClient(qdrant_url)

        # Initialize embedder
        embedder = ProteinEmbedder()

        # Create or check collection
        if not qdrant_client.collection_exists(db_name):
            logger.info(f"Creating new collection: {db_name}")
            qdrant_client.create_collection(
                collection_name=db_name,
                vectors_config=models.VectorParams(
                    size=1024,
                    distance=models.Distance.DOT
                )
            )
        else:
            logger.info(f"Collection {db_name} already exists")

        # Count total sequences
        stats.total_sequences = sum(1 for _ in SeqIO.parse(fasta_file_path, "fasta"))

        points = []
        with tqdm(total=stats.total_sequences, desc="Processing sequences") as pbar:
            for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
                seq = str(seq_record.seq).strip('*')

                if not seq:
                    stats.empty_sequences += 1
                    logger.warning(f"Empty sequence found: {seq_record.description}")
                    continue

                try:
                    embedding = embedder.get_protein_embedding(seq)
                    points.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                'protein_info': seq_record.description,
                                'sequence_length': len(seq)
                            }
                        )
                    )
                    stats.processed_sequences += 1

                    # Upload batch when it reaches batch_size
                    if len(points) >= batch_size:
                        qdrant_client.upload_points(
                            collection_name=db_name,
                            points=points
                        )
                        points = []

                except EmbeddingError as e:
                    stats.failed_sequences += 1
                    logger.error(f"Failed to process sequence: {str(e)}")

                pbar.update(1)

        # Upload any remaining points
        if points:
            qdrant_client.upload_points(
                collection_name=db_name,
                points=points
            )

        logger.info(f"Processing complete. "
                    f"Processed: {stats.processed_sequences}, "
                    f"Failed: {stats.failed_sequences}, "
                    f"Empty: {stats.empty_sequences}")

        return stats

    except Exception as e:
        logger.error(f"Database building failed: {str(e)}")
        raise


if __name__ == '__main__':
    try:
        args = parse_args()

        stats = build_vector_database_of_protein(
            fasta_file_path=args.fasta_path,
            db_name=args.db_name,
            qdrant_url=args.qdrant_url,
            batch_size=args.batch_size
        )

        logger.info("Program completed successfully")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        sys.exit(1)
