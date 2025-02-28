import logging
from dataclasses import dataclass
from typing import List
from Bio import SeqIO
from qdrant_client import models

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for tracking progress and errors."""
    total_sequences: int = 0
    processed_sequences: int = 0
    failed_sequences: int = 0
    empty_sequences: int = 0

    def increment_processed(self):
        self.processed_sequences += 1

    def increment_failed(self):
        self.failed_sequences += 1

    def increment_empty(self):
        self.empty_sequences += 1


class ProteinProcessor:
    """Handles processing of protein sequences and insertion into database."""

    def __init__(self, embedder, db, batch_size=50):
        """Initialize the processor with necessary components."""
        self.embedder = embedder
        self.db = db
        self.batch_size = batch_size
        self.points = []
        self.stats = ProcessingStats()
    

    def process_fasta_file(self, fasta_path: str) -> ProcessingStats:
        """Process all sequences in a FASTA file and insert into database."""
        # Count total sequences
        self.stats.total_sequences = sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))
        logger.info(f"Processing {self.stats.total_sequences} sequences from {fasta_path}")
        
        for i, seq_record in enumerate(SeqIO.parse(fasta_path, "fasta"), 1):
            seq = str(seq_record.seq)
            if not seq:
                self.stats.increment_empty()
                continue
            try:
                v = self.embedder.get_embedding(seq)
                self.points.append(
                    models.PointStruct(
                        id=i,
                        vector=v,
                        payload={
                            'protein_info': seq_record.description,
                            'sequence_length': len(seq)
                        }
                    )
                )
                self.stats.increment_processed()

            except Exception as e:
                self.stats.increment_failed()
                logger.error(f"Failed to process sequence: {seq_record.id} ... - {str(e)}")
            
            if len(self.points) >= self.batch_size:
                try:
                    self.db.upload_points(self.points)
                    
                    self.points = []
                    logger.info(f'Successfully Processed {self.stats.processed_sequences}/{self.stats.total_sequences}')
                except Exception as e:
                    logger.error(f"Failed to upload points: {str(e)}")
        
        # Upload remaining sequences in the final batch if any
        if self.points:
            try:
                self.db.upload_points(self.points)
                logger.info(f"Uploaded final batch of {len(self.points)} points")
            except Exception as e:
                logger.error(f"Failed to upload final batch of points: {str(e)}")

        return self.stats
