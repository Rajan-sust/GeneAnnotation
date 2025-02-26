"""
Module for managing the vector database connection and operations.
"""

import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Raised when there's an error with database operations."""
    pass


class VectorDatabase:
    """Class to handle vector database operations."""

    def __init__(self, db_name: str, qdrant_url: str = "http://localhost:6333", vector_size: int = 1024):
        """Initialize the vector database connection."""
        self.db_name = db_name
        self.vector_size = vector_size

        try:
            self.client = QdrantClient(qdrant_url)
            self._initialize_collection()
        except Exception as e:
            raise DatabaseError(f"Failed to initialize database: {str(e)}")

    def _initialize_collection(self):
        """Create or recreate collection."""
        try:
            # Check if collection exists and delete if it does
            if self.client.collection_exists(self.db_name):
                logger.info(f"Deleting existing collection: {self.db_name}")
                self.client.delete_collection(collection_name=self.db_name)

            # Create new collection
            self.client.create_collection(
                collection_name=self.db_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.DOT
                )
            )
            logger.info(f"Created collection: {self.db_name} with vector size {self.vector_size}")
        except Exception as e:
            raise DatabaseError(f"Failed to initialize collection: {str(e)}")

    def upload_points(self, points: List[models.PointStruct]):
        """Upload points to the database."""
        if not points:
            return

        try:
            self.client.upload_points(
                collection_name=self.db_name,
                points=points
            )
            logger.debug(f"Uploaded {len(points)} points to {self.db_name}")
        except Exception as e:
            raise DatabaseError(f"Failed to upload points: {str(e)}")
