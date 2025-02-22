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
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
import sys
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from threading import Lock
import multiprocessing

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
    lock: Lock = Lock()  # Thread-safe counter updates

    def increment_processed(self):
        with self.lock:
            self.processed_sequences += 1

    def increment_failed(self):
        with self.lock:
            self.failed_sequences += 1

    def increment_empty(self):
        with self.lock:
            self.empty_sequences += 1


class ConfigurationError(Exception):
    """Raised when there's an error in the configuration."""
    pass


class EmbeddingError(Exception):
    """Raised when there's an error during embedding generation."""
    pass


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with input validation."""
    parser = argparse.ArgumentParser(
        description='Build protein vector database from FASTA file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--fasta_path', type=str, required=True, help='Path to input FASTA file')
    parser.add_argument('--db_name', type=str, required=True, help='Name of the database to create')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for processing sequences')
    parser.add_argument('--qdrant_url', type=str, default="http://localhost:6333", help='URL for Qdrant server')
    parser.add_argument('--num_threads', type=int, default=max(1, multiprocessing.cpu_count() - 1), help='Number of worker threads')

    args = parser.parse_args()

    if not Path(args.fasta_path).exists():
        raise ConfigurationError(f"FASTA file not found: {args.fasta_path}")
    if args.batch_size < 1:
        raise ConfigurationError("Batch size must be positive")
    if args.num_threads < 1:
        raise ConfigurationError("Number of threads must be positive")

    return args


def get_device() -> torch.device:
    """Determine the appropriate device for computation."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available() and platform.system() == 'Darwin':
        return torch.device('mps')
    else:
        return torch.device('cpu')


def normalize_l2(x: np.ndarray) -> np.ndarray:
    """Normalize vector using L2 normalization."""
    norm = np.linalg.norm(x)
    return x / (norm if norm > 0 else 1)


class ProteinEmbedder:
    """Handles protein sequence embedding using BERT model."""

    def __init__(self):
        """Initialize the embedder with specified device."""
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
        """Generate embedding for a protein sequence."""
        try:
            sequence = " ".join(re.sub(r"[UZOB]", "X", sequence))

            encoded_input = self.tokenizer(sequence, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            with torch.no_grad():
                outputs = self.model(**encoded_input)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = embeddings.cpu()

            normalized_embedding = normalize_l2(embeddings.numpy()[0])
            embedding_list = normalized_embedding.tolist()

            if not embedding_list:
                raise EmbeddingError("Generated embedding is empty")
            if not all(isinstance(x, float) for x in embedding_list):
                raise EmbeddingError("Non-float values in embedding")
            if any(np.isnan(x) or np.isinf(x) for x in embedding_list):
                raise EmbeddingError("NaN or Inf values in embedding")

            return embedding_list

        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")


def process_sequence(embedder: ProteinEmbedder, seq_record, stats: ProcessingStats) -> Optional[models.PointStruct]:
    """Process a single sequence and return a PointStruct for database insertion."""
    seq = str(seq_record.seq).strip('*')

    if not seq:
        stats.increment_empty()
        logger.warning(f"Empty sequence found: {seq_record.description}")
        return None

    try:
        embedding = embedder.get_protein_embedding(seq)
        stats.increment_processed()

        return models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                'protein_info': seq_record.description,
                'sequence_length': len(seq)
            }
        )

    except EmbeddingError as e:
        stats.increment_failed()
        logger.error(f"Failed to process sequence: {str(e)}")
        return None


def build_vector_database_of_protein(
        fasta_file_path: str,
        db_name: str,
        qdrant_url: str = "http://localhost:6333",
        batch_size: int = 50,
        num_threads: int = 4
) -> ProcessingStats:
    """Build vector database from protein sequences in FASTA file using multiple threads."""
    stats = ProcessingStats()

    try:
        # Initialize Qdrant client
        qdrant_client = QdrantClient(qdrant_url)

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

        # Create thread-safe queue for batch processing
        points_queue = queue.Queue()

        # Initialize embedders for each thread
        embedders = [ProteinEmbedder() for _ in range(num_threads)]

        def upload_batch():
            points = []
            while not points_queue.empty() and len(points) < batch_size:
                point = points_queue.get()
                if point is not None:
                    points.append(point)

            if points:
                qdrant_client.upload_points(
                    collection_name=db_name,
                    points=points
                )

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []

            # Submit sequences for processing
            for i, seq_record in enumerate(SeqIO.parse(fasta_file_path, "fasta")):
                embedder = embedders[i % num_threads]  # Round-robin assignment of embedders
                future = executor.submit(process_sequence, embedder, seq_record, stats)
                futures.append(future)

                # Process completed futures and upload in batches
                if len(futures) >= batch_size:
                    for completed in as_completed(futures):
                        point = completed.result()
                        if point is not None:
                            points_queue.put(point)

                    upload_batch()
                    futures = []

            # Process remaining futures
            for completed in as_completed(futures):
                point = completed.result()
                if point is not None:
                    points_queue.put(point)

            # Upload final batch
            upload_batch()

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
            batch_size=args.batch_size,
            num_threads=args.num_threads
        )

        logger.info("Program completed successfully")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        sys.exit(1)
