import logging
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from Bio import SeqIO
from qdrant_client import models
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for tracking progress and errors."""
    total_sequences: int = 0
    processed_sequences: int = 0
    failed_sequences: int = 0
    empty_sequences: int = 0
    lock: Lock = field(default_factory=Lock)

    def increment_processed(self):
        with self.lock:
            self.processed_sequences += 1

    def increment_failed(self):
        with self.lock:
            self.failed_sequences += 1

    def increment_empty(self):
        with self.lock:
            self.empty_sequences += 1


class ProteinProcessor:
    """Handles processing of protein sequences and insertion into database."""

    def __init__(self, embedder, db, batch_size=50, num_threads=2):
        """Initialize the processor with necessary components."""
        self.embedder = embedder
        self.db = db
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.stats = ProcessingStats()
        self.embedder_lock = threading.Lock()

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

    def process_sequence(self, seq_record) -> Optional[models.PointStruct]:
        """Process a single sequence and return a PointStruct for database insertion."""
        seq = str(seq_record.seq).strip('*')

        if not seq:
            self.stats.increment_empty()
            logger.warning(f"Empty sequence found: {seq_record.description}")
            return None

        try:
            embedding = self.get_embedding_with_retry(seq)

            if embedding is not None:
                self.stats.increment_processed()
                return models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        'protein_info': seq_record.description,
                        'sequence_length': len(seq)
                    }
                )
            return None

        except Exception as e:
            self.stats.increment_failed()
            logger.error(f"Failed to process sequence: {seq_record.description[:30]}... - {str(e)}")
            return None

    def process_batch(self, seq_records):
        """Process a batch of sequences."""
        results = []
        for seq_record in seq_records:
            result = self.process_sequence(seq_record)
            if result is not None:
                results.append(result)
        return results

    def process_fasta_file(self, fasta_path: str) -> ProcessingStats:
        """Process all sequences in a FASTA file and insert into database."""
        # Count total sequences
        self.stats.total_sequences = sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))
        logger.info(f"Processing {self.stats.total_sequences} sequences from {fasta_path}")

        # Create thread-safe queue for batch processing
        points_queue = queue.Queue()
        upload_lock = threading.Lock()

        def upload_batch():
            """Upload a batch of points to the database."""
            points = []
            try:
                while not points_queue.empty() and len(points) < self.batch_size:
                    point = points_queue.get_nowait()
                    if point is not None:
                        points.append(point)
            except queue.Empty:
                pass

            if points:
                with upload_lock:
                    try:
                        self.db.upload_points(points)
                        logger.info(f"Uploaded batch of {len(points)} points")
                    except Exception as e:
                        logger.error(f"Failed to upload batch: {str(e)}")
                        # Put points back in queue for retry
                        for point in points:
                            points_queue.put(point)

        # Process in batches using thread pool
        batch = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []

            for i, seq_record in enumerate(SeqIO.parse(fasta_path, "fasta"), 1):
                batch.append(seq_record)

                if len(batch) >= self.batch_size:
                    future = executor.submit(self.process_batch, batch.copy())
                    futures.append(future)
                    batch = []

                # if i % 100 == 0:
                #     logger.info(f"Processing sequence {i}/{self.stats.total_sequences}")

            # Process remaining sequences
            if batch:
                future = executor.submit(self.process_batch, batch)
                futures.append(future)

            # Handle results
            for future in as_completed(futures):
                try:
                    results = future.result()
                    for point in results:
                        points_queue.put(point)

                    if points_queue.qsize() >= self.batch_size:
                        upload_batch()
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")

            # Upload remaining points
            while not points_queue.empty():
                upload_batch()

        return self.stats