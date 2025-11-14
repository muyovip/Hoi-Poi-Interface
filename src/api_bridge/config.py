"""
Configuration management for the Game API Bridge.

This module handles environment variables, configuration settings,
and service discovery for the bridge components.
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class CapsuleOSConfig:
    """Configuration for CapsuleOS connection."""
    api_url: str = "http://localhost:8080"
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class VLLMConfig:
    """Configuration for vLLM connection."""
    api_url: str = "http://localhost:8000"
    timeout: int = 300  # 5 minutes for generation
    retry_attempts: int = 2
    retry_delay: float = 2.0


@dataclass
class ServerConfig:
    """Configuration for the API server."""
    host: str = "0.0.0.0"
    port: int = 9000
    reload: bool = False
    log_level: str = "info"
    workers: int = 1


@dataclass
class BridgeConfig:
    """Main configuration for the Game API Bridge."""
    capsulos: CapsuleOSConfig
    vllm: VLLMConfig
    server: ServerConfig
    environment: str = "development"
    debug: bool = False
    max_concurrent_requests: int = 10
    request_timeout: int = 600  # 10 minutes


def load_config() -> BridgeConfig:
    """
    Load configuration from environment variables.

    Returns:
        BridgeConfig instance with settings from environment
    """
    return BridgeConfig(
        capsulos=CapsuleOSConfig(
            api_url=os.getenv("CAPSULOS_API_URL", "http://localhost:8080"),
            timeout=int(os.getenv("CAPSULOS_TIMEOUT", "30")),
            retry_attempts=int(os.getenv("CAPSULOS_RETRY_ATTEMPTS", "3")),
            retry_delay=float(os.getenv("CAPSULOS_RETRY_DELAY", "1.0"))
        ),
        vllm=VLLMConfig(
            api_url=os.getenv("VLLM_API_URL", "http://localhost:8000"),
            timeout=int(os.getenv("VLLM_TIMEOUT", "300")),
            retry_attempts=int(os.getenv("VLLM_RETRY_ATTEMPTS", "2")),
            retry_delay=float(os.getenv("VLLM_RETRY_DELAY", "2.0"))
        ),
        server=ServerConfig(
            host=os.getenv("SERVER_HOST", "0.0.0.0"),
            port=int(os.getenv("SERVER_PORT", "9000")),
            reload=os.getenv("SERVER_RELOAD", "false").lower() == "true",
            log_level=os.getenv("SERVER_LOG_LEVEL", "info"),
            workers=int(os.getenv("SERVER_WORKERS", "1"))
        ),
        environment=os.getenv("ENVIRONMENT", "development"),
        debug=os.getenv("DEBUG", "false").lower() == "true",
        max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "600"))
    )


def create_example_env_file():
    """Create an example .env file for configuration."""
    env_content = """# Game API Bridge Configuration

# CapsuleOS Settings
CAPSULOS_API_URL=http://localhost:8080
CAPSULOS_TIMEOUT=30
CAPSULOS_RETRY_ATTEMPTS=3
CAPSULOS_RETRY_DELAY=1.0

# vLLM Settings
VLLM_API_URL=http://localhost:8000
VLLM_TIMEOUT=300
VLLM_RETRY_ATTEMPTS=2
VLLM_RETRY_DELAY=2.0

# Server Settings
SERVER_HOST=0.0.0.0
SERVER_PORT=9000
SERVER_RELOAD=false
SERVER_LOG_LEVEL=info
SERVER_WORKERS=1

# Environment
ENVIRONMENT=development
DEBUG=false
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=600
"""

    with open(".env.example", "w") as f:
        f.write(env_content)

    print("Created .env.example file with default configuration")


if __name__ == "__main__":
    # Create example environment file
    create_example_env_file()

    # Load and print current configuration
    config = load_config()
    print("\nCurrent Configuration:")
    print(f"Environment: {config.environment}")
    print(f"Debug: {config.debug}")
    print(f"CapsuleOS API: {config.capsulos.api_url}")
    print(f"vLLM API: {config.vllm.api_url}")
    print(f"Server: {config.server.host}:{config.server.port}")