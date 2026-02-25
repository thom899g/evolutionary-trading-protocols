# Evolutionary Trading Protocols

## Objective
A framework allowing AI to evolve trading strategies through generative adversarial networks and genetic algorithms.

## Strategy
Implement GANs to simulate market scenarios and use genetic algorithms for strategy optimization.

## Execution Output
SUMMARY: Established foundational architecture for Evolutionary Trading Protocols with robust Firestore integration, market data abstraction, and initial Generator Agent using scikit-learn decision trees. Implemented comprehensive error handling, logging, and type safety throughout the system.

OUTPUT: Created 6 production-grade files with full documentation and edge case handling.

### FILE: requirements.txt
```
firebase-admin>=6.5.0
ccxt>=4.1.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
boto3>=1.28.0
requests>=2.31.0
python-dotenv>=1.0.0
```

### FILE: config/settings.py
```python
"""
Configuration management for Evolutionary Trading Protocols.
Centralizes environment variables and system constants with validation.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketType(Enum):
    """Supported market types for abstraction layer."""
    CRYPTO = "crypto"
    EQUITIES = "equities"
    FUTURES = "futures"
    FOREX = "forex"


@dataclass
class FirestoreConfig:
    """Firebase Firestore configuration with validation."""
    project_id: str
    credentials_path: str = "service_account.json"
    
    def __post_init__(self):
        """Validate configuration on initialization."""
        if not self.project_id:
            raise ValueError("Firestore project_id cannot be empty")
        if not os.path.exists(self.credentials_path):
            logger.warning(f"Firestore credentials not found at {self.credentials_path}")


@dataclass
class ExchangeConfig:
    """Exchange API configuration with rate limiting."""
    name: str
    api_key: Optional[str] = None
    secret: Optional[str] = None
    sandbox: bool = True
    rate_limit: int = 1000  # Requests per minute
    
    def __post_init__(self):
        """Validate exchange configuration."""
        valid_exchanges = ["binance", "coinbase", "kraken", "alpaca"]
        if self.name.lower() not in valid_exchanges:
            raise ValueError(f"Exchange {self.name} not supported. Valid: {valid_exchanges}")


class SystemSettings:
    """
    Main system configuration singleton.
    Loads from environment variables with fallback defaults.
    """
    
    _instance: Optional['SystemSettings'] = None
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Load and validate all configuration settings."""
        try:
            # Firestore Configuration
            self.firestore = FirestoreConfig(
                project_id=os.getenv("FIRESTORE_PROJECT_ID", "evolutionary-trading"),
                credentials_path=os.getenv("FIRESTORE_CREDENTIALS", "service_account.json")
            )
            
            # Exchange Configuration
            self.primary_exchange = ExchangeConfig(
                name=os.getenv("PRIMARY_EXCHANGE", "binance"),
                api_key=os.getenv("EXCHANGE_API_KEY"),
                secret=os.getenv("EXCHANGE_SECRET"),
                sandbox=os.getenv("EXCHANGE_SANDBOX", "true").lower() == "true"
            )
            
            # Evolutionary Parameters
            self.evolution_interval = int(os.getenv("EVOLUTION_INTERVAL", "300"))  # 5 minutes
            self.max_strategies = int(os.getenv("MAX_STRATEGIES", "100"))
            self.min_capital_per_strategy = float(os.getenv("MIN_CAPITAL", "100.0"))
            
            # Risk Management
            self.max_drawdown_percent = float(os.getenv("MAX_DRAWDOWN", "20.0"))
            self.position_sizing = float(os.getenv("POSITION_SIZING", "0.02"))  # 2% per position
            
            # Feature Engineering
            self.feature_window_sizes = [
                int(x) for x in os.getenv("FEATURE_WINDOWS", "5,10,20,50,100").split(",")
            ]
            
            logger.info("System settings initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system settings: {e}")
            raise
    
    def validate(self) -> bool:
        """
        Validate all configurations are ready for production.
        
        Returns:
            bool: True if all critical configurations are valid
        """
        try:
            # Check required environment variables
            required_vars = ["FIRESTORE_PROJECT_ID"]
            missing = [var for var in required_vars if not os.getenv(var)]
            
            if missing:
                logger.error(f"Missing required environment variables: {missing}")
                return False
                
            # Validate exchange credentials for production
            if not self.primary_exchange.sandbox:
                if not self.primary_exchange.api_key or not self.primary_exchange.secret:
                    logger.error("Production exchange requires API key and secret")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


# Global settings instance
settings = SystemSettings()
```

### FILE: utils/firebase_client.py
```python
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