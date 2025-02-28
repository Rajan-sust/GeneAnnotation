import logging
import sys
import argparse
import pandas as pd
from Bio import SeqIO
from qdrant_client import QdrantClient
from embedders import get_embedder

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
    parser.add_argument('--input_faa', type=str, required=True, 
                       help='Path to input FAA file to annotate')
    parser.add_argument('--db_name', type=str, required=True, 
                       help='Name of the database to search against')
    parser.add_argument('--output_file', type=str, required=True, 
                       help='Path to output TSV file')
    parser.add_argument('--threshold', type=float, default=0.98, 
                       help='Similarity threshold for annotations')
    parser.add_argument('--model_name', type=str, default="esm2",
                       choices=["prot_bert", "esm2"],
                       help='Protein embedding model to use')
    parser.add_argument('--qdrant_url', type=str, default="http://localhost:6333", 
                       help='URL for Qdrant server')
    return parser.parse_args()

class ProteinAnnotator:
    def __init__(self, args):
        self.args = args
        self.embedder = get_embedder(self.args.model_name)
        self.qdrant_client = self._initialize_qdrant()
        self.results = []
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'empty': 0,
            'below_threshold': 0
        }

    def _initialize_qdrant(self):
        try:
            client = QdrantClient(self.args.qdrant_url)
            collections = client.get_collections()
            if not any(collection.name == self.args.db_name for collection in collections.collections):
                raise ValueError(f"Collection {self.args.db_name} not found in database")
            return client
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant database: {str(e)}")

    def process_sequence(self, seq_record):
        """Process a single sequence and find its annotation."""
        seq_id = seq_record.id
        seq = str(seq_record.seq).strip('*')
        self.stats['total'] += 1

        if not seq:
            self.stats['empty'] += 1
            return {
                'Query_ID': seq_id,
                'Annotation': 'N/A',
                'Similarity_Score': 0.0,
                'Status': 'empty_sequence'
            }

        try:
            # Get embedding
            embedding = self.embedder.get_embedding(seq)
            
            # Search in database
            search_results = self.qdrant_client.search(
                collection_name=self.args.db_name,
                query_vector=embedding,
                limit=1
            )

            if search_results and search_results[0].score >= self.args.threshold:
                self.stats['success'] += 1
                return {
                    'Query_ID': seq_id,
                    'Annotation': search_results[0].payload['protein_info'],
                    'Similarity_Score': float(search_results[0].score),
                    'Status': 'success'
                }
            else:
                self.stats['below_threshold'] += 1
                return {
                    'Query_ID': seq_id,
                    'Annotation': search_results[0].payload['protein_info'] if search_results else 'hypothetical protein',
                    'Similarity_Score': float(search_results[0].score) if search_results else 0.0,
                    'Status': 'below_threshold'
                }

        except Exception as e:
            self.stats['failed'] += 1
            logger.error(f"Error processing sequence {seq_id}: {str(e)}")
            return {
                'Query_ID': seq_id,
                'Annotation': 'hypothetical protein',
                'Similarity_Score': 0.0,
                'Status': 'error'
            }

    def run(self):
        """Execute the annotation pipeline."""
        logger.info("Starting annotation pipeline...")
        
        try:
            # Process sequences one by one
            for seq_record in SeqIO.parse(self.args.input_faa, "fasta"):
                result = self.process_sequence(seq_record)
                self.results.append(result)
                
                # Log progress every 100 sequences
                if len(self.results) % 100 == 0:
                    logger.info(f"Processed {len(self.results)} sequences...")

            # Save results
            df = pd.DataFrame(self.results)
            df.to_csv(self.args.output_file, sep='\t', index=False)
            logger.info(f"Results saved to {self.args.output_file}")

            # Print summary
            self._print_summary()

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _print_summary(self):
        """Print summary statistics of the annotation results."""
        logger.info("\nAnnotation Summary:")
        logger.info(f"Total sequences processed: {self.stats['total']}")
        logger.info(f"Successful annotations: {self.stats['success']}")
        logger.info(f"Below threshold: {self.stats['below_threshold']}")
        logger.info(f"Failed sequences: {self.stats['failed']}")
        logger.info(f"Empty sequences: {self.stats['empty']}")

def main():
    args = parse_args()
    annotator = ProteinAnnotator(args)
    annotator.run()

if __name__ == '__main__':
    main()
