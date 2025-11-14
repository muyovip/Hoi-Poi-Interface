"""
Cross-Repository API Bridge for Multi-LLM Game Generation System

This module provides the bridge between:
- Hoi-Poi-Interface (Python orchestration and vLLM)
- CapsuleOS (Rust game capsule storage)

The bridge handles communication, protocol translation, and synchronization
between the two systems.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

import aiohttp
import requests
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BridgeError(Exception):
    """Base exception for API bridge errors."""
    pass


class ConnectionError(BridgeError):
    """Exception for connection failures."""
    pass


class SerializationError(BridgeError):
    """Exception for data serialization failures."""
    pass


class ValidationError(BridgeError):
    """Exception for data validation failures."""
    pass


class GameStatus(str, Enum):
    """Game generation status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EVOLVED = "evolved"


class LLMRole(str, Enum):
    """LLM role enumeration matching the CapsuleOS definitions."""
    NARRATIVE = "narrative"
    MECHANICS = "mechanics"
    ASSETS = "assets"
    BALANCE = "balance"


# ============================================================================
# Data Models for API Bridge
# ============================================================================

@dataclass
class LLMOutput:
    """LLM output structure matching CapsuleOS format."""
    llm_name: str
    llm_role: LLMRole
    glyph_expression: str  # GΛLYPH λ-expression
    processed_at: int
    confidence: Optional[float] = None


@dataclass
class GameManifest:
    """Game manifest structure for cross-system compatibility."""
    id: str
    title: str
    story: str
    rules: Dict[str, Any]
    code: str
    balance: float
    genre: Optional[str] = None
    theme: Optional[str] = None
    created_at: int = None
    llm_outputs: List[LLMOutput] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = int(datetime.now(timezone.utc).timestamp())
        if self.llm_outputs is None:
            self.llm_outputs = []


class GameGenerationRequest(BaseModel):
    """Request model for game generation."""
    user_id: str
    input_text: str
    context: Optional[Dict[str, Any]] = None
    parent_cid: Optional[str] = None  # For evolution


class GameGenerationResponse(BaseModel):
    """Response model for game generation."""
    request_id: str
    status: GameStatus
    game_cid: Optional[str] = None
    message: Optional[str] = None
    progress: Optional[float] = None
    estimated_time: Optional[int] = None


class GameRetrievalRequest(BaseModel):
    """Request model for game retrieval."""
    game_cid: str
    user_id: Optional[str] = None


class GameRetrievalResponse(BaseModel):
    """Response model for game retrieval."""
    game_cid: str
    manifest: Optional[Dict[str, Any]] = None
    exists: bool
    status: GameStatus


class GameEvolutionRequest(BaseModel):
    """Request model for game evolution."""
    parent_cid: str
    user_id: str
    evolution_instructions: str
    context: Optional[Dict[str, Any]] = None


class BridgeStatusResponse(BaseModel):
    """Response model for bridge status."""
    bridge_version: str
    capsulos_connected: bool
    vllm_connected: bool
    active_requests: int
    total_games_stored: int


# ============================================================================
# Core API Bridge Implementation
# ============================================================================

class CapsuleOSBridge:
    """
    Bridge interface for CapsuleOS game storage system.

    This class handles communication with the CapsuleOS game capsule storage,
    including game storage, retrieval, and evolution operations.
    """

    def __init__(self, capsulos_api_url: str = "http://localhost:8080"):
        """
        Initialize CapsuleOS bridge.

        Args:
            capsulos_api_url: Base URL for CapsuleOS API
        """
        self.capsulos_api_url = capsulos_api_url.rstrip('/')
        self.session = None
        self.connected = False

    async def connect(self) -> bool:
        """Establish connection to CapsuleOS API."""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )

            # Test connection
            async with self.session.get(f"{self.capsulos_api_url}/health") as response:
                if response.status == 200:
                    self.connected = True
                    logger.info("Connected to CapsuleOS API")
                    return True
                else:
                    logger.error(f"CapsuleOS API returned status {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Failed to connect to CapsuleOS API: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Close connection to CapsuleOS API."""
        if self.session:
            await self.session.close()
            self.session = None
        self.connected = False
        logger.info("Disconnected from CapsuleOS API")

    async def store_game(self, manifest: GameManifest, user_id: Optional[str] = None) -> str:
        """
        Store a game manifest in CapsuleOS.

        Args:
            manifest: Game manifest to store
            user_id: Optional user ID for ownership

        Returns:
            Game CID (content identifier)

        Raises:
            ConnectionError: If CapsuleOS is not connected
            SerializationError: If manifest cannot be serialized
        """
        if not self.connected or not self.session:
            raise ConnectionError("Not connected to CapsuleOS API")

        try:
            # Convert to CapsuleOS format
            capsulos_manifest = self._convert_to_capsulos_format(manifest)

            # Send to CapsuleOS
            payload = {
                "manifest": capsulos_manifest,
                "user_id": user_id
            }

            async with self.session.post(
                f"{self.capsulos_api_url}/api/v1/games",
                json=payload
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    game_cid = result.get("game_cid")
                    logger.info(f"Stored game {manifest.id} with CID {game_cid}")
                    return game_cid
                else:
                    error_text = await response.text()
                    raise ConnectionError(f"Failed to store game: {error_text}")

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Network error storing game: {e}")
        except Exception as e:
            raise SerializationError(f"Error serializing game manifest: {e}")

    async def retrieve_game(self, game_cid: str) -> Optional[GameManifest]:
        """
        Retrieve a game manifest from CapsuleOS.

        Args:
            game_cid: Game content identifier

        Returns:
            Game manifest if found, None otherwise

        Raises:
            ConnectionError: If CapsuleOS is not connected
        """
        if not self.connected or not self.session:
            raise ConnectionError("Not connected to CapsuleOS API")

        try:
            async with self.session.get(
                f"{self.capsulos_api_url}/api/v1/games/{game_cid}"
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    capsulos_manifest = result.get("manifest")

                    if capsulos_manifest:
                        manifest = self._convert_from_capsulos_format(capsulos_manifest)
                        logger.info(f"Retrieved game {game_cid}")
                        return manifest
                    else:
                        return None
                elif response.status == 404:
                    logger.warning(f"Game {game_cid} not found")
                    return None
                else:
                    error_text = await response.text()
                    raise ConnectionError(f"Failed to retrieve game: {error_text}")

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Network error retrieving game: {e}")
        except Exception as e:
            raise SerializationError(f"Error deserializing game manifest: {e}")

    async def evolve_game(
        self,
        parent_cid: str,
        new_manifest: GameManifest,
        user_id: Optional[str] = None
    ) -> str:
        """
        Evolve an existing game in CapsuleOS.

        Args:
            parent_cid: Parent game CID
            new_manifest: New game manifest
            user_id: Optional user ID for ownership

        Returns:
            New game CID

        Raises:
            ConnectionError: If CapsuleOS is not connected
            ValidationError: If parent game doesn't exist
        """
        if not self.connected or not self.session:
            raise ConnectionError("Not connected to CapsuleOS API")

        try:
            # Verify parent exists
            parent_game = await self.retrieve_game(parent_cid)
            if not parent_game:
                raise ValidationError(f"Parent game {parent_cid} not found")

            # Convert to CapsuleOS format
            capsulos_manifest = self._convert_to_capsulos_format(new_manifest)

            # Send evolution request
            payload = {
                "parent_cid": parent_cid,
                "manifest": capsulos_manifest,
                "user_id": user_id
            }

            async with self.session.post(
                f"{self.capsulos_api_url}/api/v1/games/evolve",
                json=payload
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    new_cid = result.get("game_cid")
                    logger.info(f"Evolved game {parent_cid} to {new_cid}")
                    return new_cid
                else:
                    error_text = await response.text()
                    raise ConnectionError(f"Failed to evolve game: {error_text}")

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Network error evolving game: {e}")
        except Exception as e:
            raise SerializationError(f"Error processing evolution: {e}")

    async def get_game_lineage(self, game_cid: str) -> List[str]:
        """
        Get the evolution lineage for a game.

        Args:
            game_cid: Game content identifier

        Returns:
            List of CIDs from oldest to newest
        """
        if not self.connected or not self.session:
            raise ConnectionError("Not connected to CapsuleOS API")

        try:
            async with self.session.get(
                f"{self.capsulos_api_url}/api/v1/games/{game_cid}/lineage"
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("lineage", [])
                else:
                    return []

        except aiohttp.ClientError as e:
            logger.error(f"Error getting game lineage: {e}")
            return []

    def _convert_to_capsulos_format(self, manifest: GameManifest) -> Dict[str, Any]:
        """Convert Python GameManifest to CapsuleOS-compatible format."""
        return {
            "id": manifest.id,
            "title": manifest.title,
            "story": manifest.story,
            "rules": manifest.rules,
            "code": manifest.code,
            "balance": manifest.balance,
            "genre": manifest.genre,
            "theme": manifest.theme,
            "created_at": manifest.created_at,
            "llm_outputs": [
                {
                    "llm_name": output.llm_name,
                    "llm_role": output.llm_role.value,
                    "glyph_expression": output.glyph_expression,
                    "processed_at": output.processed_at,
                    "confidence": output.confidence
                }
                for output in manifest.llm_outputs
            ]
        }

    def _convert_from_capsulos_format(self, capsulos_manifest: Dict[str, Any]) -> GameManifest:
        """Convert CapsuleOS format to Python GameManifest."""
        llm_outputs = []
        for output_data in capsulos_manifest.get("llm_outputs", []):
            llm_outputs.append(LLMOutput(
                llm_name=output_data["llm_name"],
                llm_role=LLMRole(output_data["llm_role"]),
                glyph_expression=output_data["glyph_expression"],
                processed_at=output_data["processed_at"],
                confidence=output_data.get("confidence")
            ))

        return GameManifest(
            id=capsulos_manifest["id"],
            title=capsulos_manifest["title"],
            story=capsulos_manifest["story"],
            rules=capsulos_manifest["rules"],
            code=capsulos_manifest["code"],
            balance=capsulos_manifest["balance"],
            genre=capsulos_manifest.get("genre"),
            theme=capsulos_manifest.get("theme"),
            created_at=capsulos_manifest["created_at"],
            llm_outputs=llm_outputs
        )


class VLLMBridge:
    """
    Bridge interface for vLLM inference service.

    This class handles communication with the vLLM API for LLM inference
    and orchestration.
    """

    def __init__(self, vllm_api_url: str = "http://localhost:8000"):
        """
        Initialize vLLM bridge.

        Args:
            vllm_api_url: Base URL for vLLM API
        """
        self.vllm_api_url = vllm_api_url.rstrip('/')
        self.session = None
        self.connected = False

    async def connect(self) -> bool:
        """Establish connection to vLLM API."""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout for generation
            )

            # Test connection
            async with self.session.get(f"{self.vllm_api_url}/health") as response:
                if response.status == 200:
                    self.connected = True
                    logger.info("Connected to vLLM API")
                    return True
                else:
                    logger.error(f"vLLM API returned status {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Failed to connect to vLLM API: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Close connection to vLLM API."""
        if self.session:
            await self.session.close()
            self.session = None
        self.connected = False
        logger.info("Disconnected from vLLM API")

    async def generate_game(self, request: GameGenerationRequest) -> GameManifest:
        """
        Generate a game using the vLLM orchestration service.

        Args:
            request: Game generation request

        Returns:
            Generated game manifest

        Raises:
            ConnectionError: If vLLM is not connected
        """
        if not self.connected or not self.session:
            raise ConnectionError("Not connected to vLLM API")

        try:
            payload = {
                "input_text": request.input_text,
                "user_id": request.user_id,
                "context": request.context or {},
                "parent_cid": request.parent_cid
            }

            async with self.session.post(
                f"{self.vllm_api_url}/v1/game/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._parse_generation_response(result)
                else:
                    error_text = await response.text()
                    raise ConnectionError(f"Failed to generate game: {error_text}")

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Network error generating game: {e}")

    async def get_generation_status(self, request_id: str) -> GameGenerationResponse:
        """
        Get the status of an ongoing game generation.

        Args:
            request_id: Generation request identifier

        Returns:
            Generation status response
        """
        if not self.connected or not self.session:
            raise ConnectionError("Not connected to vLLM API")

        try:
            async with self.session.get(
                f"{self.vllm_api_url}/v1/game/status/{request_id}"
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return GameGenerationResponse(**result)
                else:
                    error_text = await response.text()
                    raise ConnectionError(f"Failed to get generation status: {error_text}")

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Network error getting status: {e}")

    def _parse_generation_response(self, response: Dict[str, Any]) -> GameManifest:
        """Parse vLLM generation response into GameManifest."""
        data = response.get("data", {})

        llm_outputs = []
        for output in data.get("llm_outputs", []):
            llm_outputs.append(LLMOutput(
                llm_name=output["llm_name"],
                llm_role=LLMRole(output["llm_role"]),
                glyph_expression=output["glyph_expression"],
                processed_at=output["processed_at"],
                confidence=output.get("confidence")
            ))

        return GameManifest(
            id=data.get("id", str(uuid.uuid4())),
            title=data["title"],
            story=data["story"],
            rules=data["rules"],
            code=data["code"],
            balance=data["balance"],
            genre=data.get("genre"),
            theme=data.get("theme"),
            created_at=data.get("created_at", int(datetime.now(timezone.utc).timestamp())),
            llm_outputs=llm_outputs
        )


# ============================================================================
# Main API Bridge Orchestrator
# ============================================================================

class GameAPIBridge:
    """
    Main orchestrator for the cross-repository API bridge.

    This class coordinates between the vLLM orchestration service and
    CapsuleOS storage system to provide a unified game generation API.
    """

    def __init__(
        self,
        capsulos_api_url: str = "http://localhost:8080",
        vllm_api_url: str = "http://localhost:8000"
    ):
        """
        Initialize the API bridge.

        Args:
            capsulos_api_url: Base URL for CapsuleOS API
            vllm_api_url: Base URL for vLLM API
        """
        self.capsulos_bridge = CapsuleOSBridge(capsulos_api_url)
        self.vllm_bridge = VLLMBridge(vllm_api_url)
        self.active_requests: Dict[str, GameGenerationRequest] = {}

    async def start(self) -> bool:
        """Start the bridge and connect to both services."""
        logger.info("Starting Game API Bridge...")

        capsulos_connected = await self.capsulos_bridge.connect()
        vllm_connected = await self.vllm_bridge.connect()

        if capsulos_connected and vllm_connected:
            logger.info("Game API Bridge started successfully")
            return True
        else:
            logger.error("Failed to start Game API Bridge")
            return False

    async def stop(self):
        """Stop the bridge and disconnect from services."""
        logger.info("Stopping Game API Bridge...")

        await self.capsulos_bridge.disconnect()
        await self.vllm_bridge.disconnect()

        self.active_requests.clear()
        logger.info("Game API Bridge stopped")

    async def generate_game(self, request: GameGenerationRequest) -> GameGenerationResponse:
        """
        Generate a new game end-to-end.

        This method orchestrates the full game generation process:
        1. Sends request to vLLM for generation
        2. Stores result in CapsuleOS
        3. Returns the game CID

        Args:
            request: Game generation request

        Returns:
            Generation response with game CID
        """
        request_id = str(uuid.uuid4())
        self.active_requests[request_id] = request

        try:
            logger.info(f"Starting game generation for user {request.user_id}")

            # Step 1: Generate game using vLLM
            manifest = await self.vllm_bridge.generate_game(request)

            # Step 2: Store game in CapsuleOS
            game_cid = await self.capsulos_bridge.store_game(manifest, request.user_id)

            logger.info(f"Successfully generated and stored game {game_cid}")

            return GameGenerationResponse(
                request_id=request_id,
                status=GameStatus.COMPLETED,
                game_cid=game_cid,
                message="Game generated successfully"
            )

        except Exception as e:
            logger.error(f"Game generation failed: {e}")
            return GameGenerationResponse(
                request_id=request_id,
                status=GameStatus.FAILED,
                message=f"Generation failed: {str(e)}"
            )
        finally:
            self.active_requests.pop(request_id, None)

    async def retrieve_game(self, request: GameRetrievalRequest) -> GameRetrievalResponse:
        """
        Retrieve a stored game.

        Args:
            request: Game retrieval request

        Returns:
            Game retrieval response
        """
        try:
            manifest = await self.capsulos_bridge.retrieve_game(request.game_cid)

            if manifest:
                return GameRetrievalResponse(
                    game_cid=request.game_cid,
                    manifest=asdict(manifest),
                    exists=True,
                    status=GameStatus.COMPLETED
                )
            else:
                return GameRetrievalResponse(
                    game_cid=request.game_cid,
                    exists=False,
                    status=GameStatus.FAILED
                )

        except Exception as e:
            logger.error(f"Game retrieval failed: {e}")
            return GameRetrievalResponse(
                game_cid=request.game_cid,
                exists=False,
                status=GameStatus.FAILED
            )

    async def evolve_game(self, request: GameEvolutionRequest) -> GameGenerationResponse:
        """
        Evolve an existing game.

        Args:
            request: Game evolution request

        Returns:
            Evolution response with new game CID
        """
        request_id = str(uuid.uuid4())

        try:
            logger.info(f"Starting game evolution from {request.parent_cid}")

            # Step 1: Generate evolved game using vLLM
            generation_request = GameGenerationRequest(
                user_id=request.user_id,
                input_text=request.evolution_instructions,
                context=request.context,
                parent_cid=request.parent_cid
            )

            new_manifest = await self.vllm_bridge.generate_game(generation_request)

            # Step 2: Store evolution in CapsuleOS
            new_cid = await self.capsulos_bridge.evolve_game(
                request.parent_cid,
                new_manifest,
                request.user_id
            )

            logger.info(f"Successfully evolved game to {new_cid}")

            return GameGenerationResponse(
                request_id=request_id,
                status=GameStatus.EVOLVED,
                game_cid=new_cid,
                message="Game evolved successfully"
            )

        except Exception as e:
            logger.error(f"Game evolution failed: {e}")
            return GameGenerationResponse(
                request_id=request_id,
                status=GameStatus.FAILED,
                message=f"Evolution failed: {str(e)}"
            )

    async def get_game_lineage(self, game_cid: str) -> List[str]:
        """
        Get the evolution lineage for a game.

        Args:
            game_cid: Game content identifier

        Returns:
            List of CIDs from oldest to newest
        """
        try:
            return await self.capsulos_bridge.get_game_lineage(game_cid)
        except Exception as e:
            logger.error(f"Failed to get game lineage: {e}")
            return []

    async def get_status(self) -> BridgeStatusResponse:
        """
        Get the current status of the API bridge.

        Returns:
            Bridge status response
        """
        return BridgeStatusResponse(
            bridge_version="0.1.0",
            capsulos_connected=self.capsulos_bridge.connected,
            vllm_connected=self.vllm_bridge.connected,
            active_requests=len(self.active_requests),
            total_games_stored=0  # TODO: Get from CapsuleOS
        )


# ============================================================================
# Convenience Functions
# ============================================================================

async def create_bridge(
    capsulos_api_url: str = "http://localhost:8080",
    vllm_api_url: str = "http://localhost:8000"
) -> GameAPIBridge:
    """
    Create and initialize a Game API Bridge.

    Args:
        capsulos_api_url: Base URL for CapsuleOS API
        vllm_api_url: Base URL for vLLM API

    Returns:
        Initialized GameAPIBridge instance
    """
    bridge = GameAPIBridge(capsulos_api_url, vllm_api_url)
    await bridge.start()
    return bridge


# ============================================================================
# Example Usage and Testing
# ============================================================================

async def example_usage():
    """Example usage of the Game API Bridge."""

    # Create and start the bridge
    bridge = await create_bridge()

    try:
        # Generate a new game
        generation_request = GameGenerationRequest(
            user_id="user123",
            input_text="Create a space exploration game with turn-based combat",
            context={"genre": "sci-fi", "complexity": "medium"}
        )

        generation_response = await bridge.generate_game(generation_request)

        if generation_response.status == GameStatus.COMPLETED:
            game_cid = generation_response.game_cid
            print(f"Generated game with CID: {game_cid}")

            # Retrieve the game
            retrieval_request = GameRetrievalRequest(game_cid=game_cid)
            retrieval_response = await bridge.retrieve_game(retrieval_request)

            if retrieval_response.exists:
                manifest = retrieval_response.manifest
                print(f"Retrieved game: {manifest['title']}")

            # Evolve the game
            evolution_request = GameEvolutionRequest(
                parent_cid=game_cid,
                user_id="user123",
                evolution_instructions="Add multiplayer support",
                context={"multiplayer": True}
            )

            evolution_response = await bridge.evolve_game(evolution_request)

            if evolution_response.status == GameStatus.EVOLVED:
                new_cid = evolution_response.game_cid
                print(f"Evolved game to: {new_cid}")

                # Get lineage
                lineage = await bridge.get_game_lineage(new_cid)
                print(f"Game lineage: {lineage}")

        # Get bridge status
        status = await bridge.get_status()
        print(f"Bridge status: {status}")

    finally:
        # Stop the bridge
        await bridge.stop()


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())