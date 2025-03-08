import logging
import sys
import argparse
import pandas as pd
from Bio import SeqIO
from pymilvus import MilvusClient
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
    parser.add_argument('--collection', type=str, required=True, 
                       help='Name of the collection to search against')
    parser.add_argument('--output_file', type=str, required=True, 
                       help='Path to output TSV file')
    parser.add_argument('--threshold', type=float, default=0.98, 
                       help='Similarity threshold for annotations')
    parser.add_argument('--model_name', type=str, default="esm2",
                       choices=["prot_bert", "esm2", "openai"],
                       help='Protein embedding model to use')
    return parser.parse_args()

class ProteinAnnotator:
    def __init__(self, args):
        self.args = args
        self.embedder = get_embedder(self.args.model_name)
        self.milvas_client = self._initialize_milvas()
        self.results = []
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'empty': 0,
            'below_threshold': 0
        }

    def _initialize_milvas(self):
        try:
            client = MilvusClient('./vector.db')
            # 7. Load the collection
            client.load_collection(
                collection_name=self.args.collection
            )
            return client
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant database: {str(e)}")

    def process_sequence(self, seq_record):
        """Process a single sequence and find its annotation."""
        seq_id = seq_record.id
        logger.info(f"Processing sequence {seq_id}...")
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
            res = self.milvas_client.search(
                collection_name=self.args.collection,
                anns_field="vector",
                data=[embedding],
                limit=10,
                search_params={"metric_type": "IP"},
                output_fields=["protein_info"]
            )
            self.stats['success'] += 1
            search_result = [
                    {
                        'Query': seq_record.description,
                        'Annotation': hit['entity']['protein_info'],
                        'Similarity_Score': hit['distance'],
                        'Status': 'success'
                    }
                    for hits in res for hit in hits
            ]
            return search_result
                    

            # if search_result and search_result['distance'] >= self.args.threshold:
            #     self.stats['success'] += 1
            #     return {
            #         'Query_ID': seq_id,
            #         'Annotation': search_result['entity']['protein_info'],
            #         'Similarity_Score': search_result['distance'],
            #         'Status': 'success'
            #     }
            # else:
            #     self.stats['below_threshold'] += 1
            #     return {
            #         'Query_ID': seq_id,
            #         'Annotation':  search_result['entity']['protein_info'] if search_result else 'hypothetical protein',
            #         'Similarity_Score': search_result['distance'] if search_result else 0.0,
            #         'Status': 'below_threshold'
            #     }

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
                rets = self.process_sequence(seq_record)
                # self.results.extend(rets)
                if isinstance(rets, list):
                    self.results.extend(rets)
                else:
                    self.results.append(rets)
                # print(self.results)            
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
