#!/usr/bin/env python3
"""
Main entry point for the protein vector database builder application.
"""

import argparse
import logging
import sys
from embedders import get_embedder
from database import VectorDatabase
from processor import ProteinProcessor
from config import parse_args, ConfigurationError

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


def main():
    """Main function to run the protein database builder."""
    try:
        # Parse command line arguments
        args = parse_args()

        # Initialize embedder based on model name
        embedder = get_embedder(args.model_name)
        print('Embedding model loaded done')

        # Initialize vector database
        db = VectorDatabase(
            db_name=args.db_name,
            qdrant_url=args.qdrant_url,
            vector_size=embedder.vector_size
        )

        # Initialize processor
        processor = ProteinProcessor(
            embedder=embedder,
            db=db,
            batch_size=args.batch_size
        )
        print('Processor initialized done')

        # Process the FASTA file
        stats = processor.process_fasta_file(args.fasta_path)

        logger.info("Program completed successfully")
        logger.info(f"Total sequences: {stats.total_sequences}, "
                    f"Processed: {stats.processed_sequences}, "
                    f"Failed: {stats.failed_sequences}, "
                    f"Empty: {stats.empty_sequences}")

        return 0

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Program failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
