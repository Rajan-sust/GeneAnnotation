"""
Configuration and argument parsing module for protein database builder.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Raised when there's an error in the configuration."""
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
    parser.add_argument('--num_threads', type=int, default=2, help='Number of worker threads')
    parser.add_argument('--model_name', type=str, default="prot_bert",
                       choices=["prot_bert", "esm2"],
                       help='Protein embedding model to use')

    args = parser.parse_args()

    # Validate arguments
    if not Path(args.fasta_path).exists():
        raise ConfigurationError(f"FASTA file not found: {args.fasta_path}")
    if args.batch_size < 1:
        raise ConfigurationError("Batch size must be positive")
    if args.num_threads < 1:
        raise ConfigurationError("Number of threads must be positive")

    return args
