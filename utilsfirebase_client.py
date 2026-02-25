"""
Firebase Firestore client for state synchronization and real-time data streaming.
Implements connection pooling, retry logic, and automatic reconnection.
"""

import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.exceptions import FirebaseError
from typing import Dict, Any, Optional, List, Callback
import threading
import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FirestoreDocument:
    """Base document structure for Firestore operations."""
    collection: str
    document_id: str
    data: Dict[str, Any]
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


class FirestoreClient:
    """
    Thread-safe Firebase Firestore client with connection management.
    Implements singleton pattern to avoid multiple app initializations.
    """
    
    _instance: Optional['FirestoreClient'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern with thread safety."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Firebase app and Firestore client."""
        if not hasattr(self, 'initialized'):
            self.initialized = False
            self._app = None
            self._client = None
            self._listeners: Dict[str, Any] = {}
            self._initialize()
    
    def _initialize(self) -> bool:
        """
        Initialize Firebase connection with retry logic.
        
        Returns:
            bool: True if initialization successful
        """
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Check if Firebase app already exists
                if not firebase_admin._apps:
                    from config.settings import settings
                    cred = credentials.Certificate(settings.firestore.credentials_path)
                    self._app = firebase_admin.initialize_app(cred, {
                        'projectId': settings.firestore.project