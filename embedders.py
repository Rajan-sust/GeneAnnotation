"""
Module containing classes for generating protein embeddings using different models.
"""

import re
import logging
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from config import get_device

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when there's an error during embedding generation."""
    pass



def normalize_l2(x: np.ndarray) -> np.ndarray:
    """Normalize vector using L2 normalization."""
    norm = np.linalg.norm(x)
    return x / (norm if norm > 0 else 1)


class ProteinEmbedder(ABC):
    """Abstract base class for protein embedders."""

    def __init__(self):
        # Set device explicitly
        self.device = get_device()
        logger.info(f"Initializing {self.__class__.__name__} using device: {self.device}")

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Return the size of the generated embedding vector."""
        pass

    @abstractmethod
    def get_embedding(self, sequence: str) -> List[float]:
        """Generate embedding for a protein sequence."""
        pass

    def validate_embedding(self, embedding: List[float]) -> List[float]:
        """Validate embeddings to ensure they are well-formed."""
        if not embedding:
            raise EmbeddingError("Generated embedding is empty")
        if not all(isinstance(x, float) for x in embedding):
            raise EmbeddingError("Non-float values in embedding")
        if any(np.isnan(x) or np.isinf(x) for x in embedding):
            raise EmbeddingError("NaN or Inf values in embedding")
        return embedding


class ProtBertEmbedder(ProteinEmbedder):
    """Protein embedder using the ProtBERT model."""

    def __init__(self):
        # Ensure device is set before model initialization
        self.device = get_device()
        logger.info(f"Initializing ProtBertEmbedder using device: {self.device}")

        try:
            from transformers import BertModel, BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(
                "Rostlab/prot_bert",
                do_lower_case=False
            )
            self.model = BertModel.from_pretrained("Rostlab/prot_bert")
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to initialize BERT model: {str(e)}")
            raise EmbeddingError(f"Failed to initialize BERT model: {str(e)}")

    @property
    def vector_size(self) -> int:
        return 1024

    def get_embedding(self, sequence: str) -> List[float]:
        try:
            # Preprocess sequence: replace rare amino acids with X and space out
            sequence = " ".join(re.sub(r"[UZOB]", "X", sequence))

            # Tokenize and encode
            encoded_input = self.tokenizer(sequence, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**encoded_input)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = embeddings.cpu()

            # Normalize and convert to list
            normalized_embedding = normalize_l2(embeddings.numpy()[0])
            embedding_list = normalized_embedding.tolist()

            return self.validate_embedding(embedding_list)

        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")


class ESM2Embedder(ProteinEmbedder):
    """Protein embedder using the ESM2 model."""

    def __init__(self, model_name="facebook/esm2_t12_35M_UR50D"):
        # IMPORTANT: Set device explicitly before any other initialization
        self.device = get_device()
        logger.info(f"Initializing ESM2Embedder using device: {self.device}")

        try:
            from transformers import AutoTokenizer, AutoModel
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Get embedding size from model config
            self._vector_size = self.model.config.hidden_size

        except Exception as e:
            logger.error(f"Failed to initialize ESM2 model: {str(e)}")
            raise EmbeddingError(f"Failed to initialize ESM2 model: {str(e)}")

    @property
    def vector_size(self) -> int:
        return self._vector_size

    def _get_chunk_embedding(self, sequence_chunk: str, max_length: int = 1024) -> np.ndarray:
        """Process a single chunk of sequence and return its embedding."""
        # Tokenize and encode with attention mask to handle padding properly
        inputs = self.tokenizer(
            sequence_chunk,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )

        # Move to device
        # print(f"DEBUG: self.device = {self.device}")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get sequence embeddings
        # For ESM2, we can use the last hidden state
        last_hidden_state = outputs.last_hidden_state

        # Mean pooling - take average of all token embeddings
        emb = torch.mean(last_hidden_state, dim=1)

        return emb.cpu().numpy()[0]

    def get_embedding(self, sequence: str, max_length: int = 1024) -> List[float]:
        try:
            # Preprocess sequence - some ESM models require specific formatting
            # Remove any whitespace and non-amino acid characters
            sequence = re.sub(r'[^A-Z]', '', sequence.upper())

            # Replace rare amino acids with X
            sequence = re.sub(r'[UZOB]', 'X', sequence)

            # Calculate how many chunks we need (subtract 2 to account for special tokens [CLS] and [SEP])
            effective_max_length = max_length - 2

            # If sequence is shorter than max_length, process it directly
            if len(sequence) <= effective_max_length:
                # print('hi...')
                embedding = self._get_chunk_embedding(sequence, max_length)
                normalized_embedding = normalize_l2(embedding)
                return self.validate_embedding(normalized_embedding.tolist())

            # For long sequences, split into chunks and process each separately
            embeddings = []

            # Process complete chunks of size effective_max_length
            for i in range(0, len(sequence), effective_max_length):
                chunk = sequence[i:i + effective_max_length]
                if len(chunk) > 0:  # Only process non-empty chunks
                    chunk_embedding = self._get_chunk_embedding(chunk, max_length)
                    embeddings.append(chunk_embedding)

            # Average all chunk embeddings
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                normalized_embedding = normalize_l2(avg_embedding)
                return self.validate_embedding(normalized_embedding.tolist())
            else:
                raise EmbeddingError("No valid chunks were processed")

        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")


def get_embedder(model_name: str) -> ProteinEmbedder:
    """Factory function to get the appropriate embedder."""
    try:
        if model_name.lower() == "prot_bert":
            return ProtBertEmbedder()
        elif model_name.lower() == "esm2":
            return ESM2Embedder()
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    except Exception as e:
        logger.error(f"Error creating embedder '{model_name}': {str(e)}")
        raise
