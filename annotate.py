import numpy as np
from qdrant_client import QdrantClient
from Bio import SeqIO
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
import logging
import sys
from tqdm import tqdm
from embedders import get_embedder
import threading
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('annotation.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Annotate proteins using vector database'
    )
    parser.add_argument('--input_faa', type=str, required=True, help='Path to input FAA file to annotate')
    parser.add_argument('--db_name', type=str, required=True, help='Name of the database to search against')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output TSV file')
    parser.add_argument('--threshold', type=float, default=0.98, help='Similarity threshold for annotations')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of sequences to process in each batch')
    parser.add_argument('--num_threads', type=int, default=2, help='Number of threads to use')
    parser.add_argument('--model_name', type=str, default="esm2",
                        choices=["prot_bert", "esm2"],
                        help='Protein embedding model to use')
    parser.add_argument('--qdrant_url', type=str, default="http://localhost:6333", help='URL for Qdrant server')
    return parser.parse_args()


def normalize_l2(x: np.ndarray) -> np.ndarray:
    """Normalize vector using L2 normalization."""
    norm = np.linalg.norm(x)
    return x / (norm if norm > 0 else 1)


class Stats:
    """Simple statistics tracker."""

    def __init__(self):
        self.processed = 0
        self.failed = 0
        self.empty = 0
        self.lock = threading.Lock()

    def increment_processed(self):
        with self.lock:
            self.processed += 1

    def increment_failed(self):
        with self.lock:
            self.failed += 1

    def increment_empty(self):
        with self.lock:
            self.empty += 1


class ProteinProcessor:
    """Handles processing of protein sequences and insertion into database."""

    def __init__(self, embedder, db, batch_size=50, num_threads=2):
        """Initialize the processor with necessary components."""
        self.embedder = embedder
        self.db = db
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.embedder_lock = threading.Lock()
        self.stats = Stats()

        # Add a semaphore to limit concurrent access to the embedder
        # Allow a few concurrent accesses if the embedder supports it
        self.embedder_semaphore = threading.Semaphore(1)

    def get_embedding_with_retry(self, seq, max_retries=3, initial_wait=0.1):
        """Get embedding with retry logic and backoff."""
        for attempt in range(max_retries):
            try:
                # Use semaphore to control concurrent access
                with self.embedder_semaphore:
                    # Additional lock for complete thread safety
                    with self.embedder_lock:
                        embedding = self.embedder.get_embedding(seq)
                        return embedding
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
                time.sleep(wait_time)
        return None

    def process_sequence(self, seq_record) -> Tuple[str, Optional[np.ndarray], str]:
        """Process a single sequence and return its embedding with status."""
        seq_id = seq_record.id
        seq = str(seq_record.seq).strip('*')

        if not seq:
            self.stats.increment_empty()
            logger.warning(f"Empty sequence found: {seq_record.description}")
            return seq_id, None, 'empty_sequence'

        try:
            embedding = self.get_embedding_with_retry(seq)

            if embedding is not None:
                self.stats.increment_processed()
                return seq_id, embedding, 'success'
            return seq_id, None, 'embedding_failed'
        except Exception as e:
            self.stats.increment_failed()
            logger.error(f"Failed to process sequence: {seq_record.description[:30]}... - {str(e)}")
            return seq_id, None, 'error'

    def process_batch(self, batch: List) -> List[Tuple[str, Optional[np.ndarray], str]]:
        """Process a batch of sequences and return their embeddings."""
        results = []
        for seq_record in batch:
            result = self.process_sequence(seq_record)
            results.append(result)
        return results


class ProteinAnnotator:
    """Class for managing protein annotation pipeline."""

    def __init__(self, args):
        """Initialize annotator with command line arguments."""
        self.args = args
        # self.device = get_device()
        self.embedder = get_embedder(self.args.model_name)
        print("found embedder", type(self.embedder))
        self.qdrant_client = self._initialize_qdrant()
        # Initialize ProteinProcessor
        self.processor = ProteinProcessor(
            embedder=self.embedder,
            db=self.qdrant_client,
            batch_size=self.args.batch_size,
            num_threads=self.args.num_threads
        )

    def _initialize_qdrant(self) -> QdrantClient:
        """Initialize and verify Qdrant connection."""
        try:
            client = QdrantClient(self.args.qdrant_url)
            # Verify connection and collection
            collections = client.get_collections()
            if not any(collection.name == self.args.db_name for collection in collections.collections):
                raise ValueError(f"Collection {self.args.db_name} not found in database")
            return client
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant database: {str(e)}")

    def process_sequence_batch(self, batch: List) -> List[Dict]:
        """
        Process a batch of sequences and return their annotations using ProteinProcessor.

        Args:
            batch: List of SeqRecord objects

        Returns:
            List of dictionaries containing annotation results
        """
        results = []

        # Use ProteinProcessor to get embeddings
        processed_sequences = self.processor.process_batch(batch)

        for seq_id, embedding, status in processed_sequences:
            if status != 'success' or embedding is None:
                results.append(self._create_result(
                    seq_id,
                    status=status if status != 'success' else 'embedding_failed'
                ))
                continue

            try:
                search_results = self.qdrant_client.search(
                    collection_name=self.args.db_name,
                    query_vector=embedding,
                    limit=1
                )

                if search_results and search_results[0].score >= self.args.threshold:
                    results.append(self._create_result(
                        seq_id,
                        annotation=search_results[0].payload['protein_info'],
                        score=search_results[0].score,
                        status='success'
                    ))
                else:
                    score = search_results[0].score if search_results else 0.0
                    results.append(self._create_result(
                        seq_id,
                        annotation=search_results[0].payload['protein_info'],
                        score=score,
                        status='below_threshold'
                    ))

            except Exception as e:
                logger.error(f"Error during database search for sequence {seq_id}: {str(e)}")
                results.append(self._create_result(seq_id, status='search_error'))

        return results

    def _create_result(self, seq_id: str, annotation: str = 'hypothetical protein',
                       score: float = 0.0, status: str = 'error') -> Dict:
        """Create a standardized result dictionary."""
        return {
            'Query_ID': seq_id,
            'Annotation': annotation,
            'Similarity_Score': float(score),
            'Status': status
        }

    def run(self):
        """Execute the annotation pipeline."""
        logger.info("Starting annotation pipeline...")

        try:
            # Read sequences
            sequences = list(SeqIO.parse(self.args.input_faa, "fasta"))
            if not sequences:
                raise ValueError("No sequences found in input file")

            total_sequences = len(sequences)
            logger.info(f"Found {total_sequences} sequences to process")

            # Prepare batches
            sequence_batches = [
                sequences[i:i + self.args.batch_size]
                for i in range(0, len(sequences), self.args.batch_size)
            ]

            # Process batches
            all_results = []
            with ThreadPoolExecutor(max_workers=self.args.num_threads) as executor:
                futures = [
                    executor.submit(self.process_sequence_batch, batch)
                    for batch in sequence_batches
                ]

                # Process results with progress bar
                with tqdm(total=len(futures), desc="Processing batches") as pbar:
                    for future in as_completed(futures):
                        try:
                            batch_results = future.result()
                            all_results.extend(batch_results)
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Batch processing failed: {str(e)}")

            # Save results
            df = pd.DataFrame(all_results)
            df.to_csv(self.args.output_file, sep='\t', index=False)
            logger.info(f"Results saved to {self.args.output_file}")

            # Print summary
            self._print_summary(df)

            # Print processor statistics
            logger.info("\nProcessor Statistics:")
            logger.info(f"Sequences processed successfully: {self.processor.stats.processed}")
            logger.info(f"Sequences failed: {self.processor.stats.failed}")
            logger.info(f"Empty sequences: {self.processor.stats.empty}")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics of the annotation results."""
        status_counts = df['Status'].value_counts()
        logger.info("\nAnnotation Summary:")
        logger.info(f"Total sequences processed: {len(df)}")
        logger.info(f"Successful annotations: {status_counts.get('success', 0)}")
        logger.info(f"Below threshold: {status_counts.get('below_threshold', 0)}")
        logger.info(f"Failed embeddings: {status_counts.get('embedding_failed', 0)}")
        logger.info(f"Errors: {status_counts.get('error', 0)}")
        logger.info(f"Search errors: {status_counts.get('search_error', 0)}")
        logger.info(f"Empty sequences: {status_counts.get('empty_sequence', 0)}")


def main():
    """Main entry point for the script."""
    args = parse_args()
    annotator = ProteinAnnotator(args)
    annotator.run()


if __name__ == '__main__':
    main()
