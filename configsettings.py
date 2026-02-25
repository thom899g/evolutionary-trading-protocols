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