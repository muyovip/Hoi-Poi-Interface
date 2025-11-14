"""
FastAPI Server for Game API Bridge

This module provides a REST API interface for the cross-repository game generation system,
exposing the bridge functionality via HTTP endpoints.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .mod import (
    GameAPIBridge,
    GameGenerationRequest,
    GameGenerationResponse,
    GameRetrievalRequest,
    GameRetrievalResponse,
    GameEvolutionRequest,
    BridgeStatusResponse,
    GameStatus,
    create_bridge,
    BridgeError,
    ConnectionError,
    ValidationError,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global bridge instance
bridge: Optional[GameAPIBridge] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the bridge lifecycle."""
    global bridge

    logger.info("Starting Game API Bridge server...")

    # Initialize the bridge
    try:
        bridge = await create_bridge()
        logger.info("Bridge initialized successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize bridge: {e}")
        raise

    finally:
        # Cleanup
        if bridge:
            await bridge.stop()
            logger.info("Bridge stopped")


# Create FastAPI application
app = FastAPI(
    title="Game Generation API Bridge",
    description="Cross-repository API bridge for multi-LLM game generation",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Dependency Injection
# ============================================================================

async def get_bridge() -> GameAPIBridge:
    """Get the bridge instance."""
    if bridge is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Bridge not initialized"
        )
    return bridge


# ============================================================================
# Health and Status Endpoints
# ============================================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "game-api-bridge"}


@app.get("/status", response_model=BridgeStatusResponse, tags=["Status"])
async def get_bridge_status(br: GameAPIBridge = Depends(get_bridge)):
    """Get detailed bridge status."""
    try:
        return await br.get_status()
    except Exception as e:
        logger.error(f"Error getting bridge status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get bridge status"
        )


# ============================================================================
# Game Generation Endpoints
# ============================================================================

@app.post("/api/v1/games/generate", response_model=GameGenerationResponse, tags=["Games"])
async def generate_game(
    request: GameGenerationRequest,
    background_tasks: BackgroundTasks,
    br: GameAPIBridge = Depends(get_bridge)
):
    """
    Generate a new game using the multi-LLM orchestration system.

    This endpoint initiates game generation, which involves:
    1. Processing the input text through the orchestration engine
    2. Coordinating between 4 specialized LLMs
    3. Merging GΛLYPH expressions
    4. Storing the result in CapsuleOS

    The process typically takes 30-120 seconds depending on complexity.
    """
    try:
        logger.info(f"Received game generation request from user {request.user_id}")

        # Generate the game
        response = await br.generate_game(request)

        if response.status == GameStatus.COMPLETED:
            logger.info(f"Successfully generated game {response.game_cid}")
            return response
        else:
            logger.error(f"Game generation failed: {response.message}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.message or "Game generation failed"
            )

    except ConnectionError as e:
        logger.error(f"Connection error during game generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable"
        )
    except ValidationError as e:
        logger.error(f"Validation error during game generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during game generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.get("/api/v1/games/{game_cid}", response_model=GameRetrievalResponse, tags=["Games"])
async def retrieve_game(
    game_cid: str,
    user_id: Optional[str] = None,
    br: GameAPIBridge = Depends(get_bridge)
):
    """
    Retrieve a stored game by its content identifier (CID).

    Optionally requires user_id to enforce access control.
    """
    try:
        logger.info(f"Retrieving game {game_cid}")

        request = GameRetrievalRequest(game_cid=game_cid, user_id=user_id)
        response = await br.retrieve_game(request)

        if response.exists:
            logger.info(f"Successfully retrieved game {game_cid}")
            return response
        else:
            logger.warning(f"Game {game_cid} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game not found"
            )

    except ConnectionError as e:
        logger.error(f"Connection error during game retrieval: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable"
        )
    except Exception as e:
        logger.error(f"Unexpected error during game retrieval: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/api/v1/games/{game_cid}/evolve", response_model=GameGenerationResponse, tags=["Games"])
async def evolve_game(
    game_cid: str,
    request: GameEvolutionRequest,
    br: GameAPIBridge = Depends(get_bridge)
):
    """
    Evolve an existing game based on instructions.

    This creates a new version of the game while maintaining
    lineage from the original. The evolution process:
    1. Validates the parent game exists
    2. Processes evolution instructions through LLMs
    3. Creates new game with inherited properties
    4. Stores as new version with lineage tracking
    """
    try:
        logger.info(f"Evolving game {game_cid}")

        # Ensure the parent CID matches the URL parameter
        if request.parent_cid != game_cid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Parent CID in request must match URL parameter"
            )

        response = await br.evolve_game(request)

        if response.status == GameStatus.EVOLVED:
            logger.info(f"Successfully evolved game to {response.game_cid}")
            return response
        else:
            logger.error(f"Game evolution failed: {response.message}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.message or "Game evolution failed"
            )

    except ValidationError as e:
        logger.error(f"Validation error during game evolution: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ConnectionError as e:
        logger.error(f"Connection error during game evolution: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable"
        )
    except Exception as e:
        logger.error(f"Unexpected error during game evolution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.get("/api/v1/games/{game_cid}/lineage", tags=["Games"])
async def get_game_lineage(
    game_cid: str,
    br: GameAPIBridge = Depends(get_bridge)
):
    """
    Get the evolution lineage for a game.

    Returns a list of CIDs representing the evolution chain
    from the original game to the current version.
    """
    try:
        logger.info(f"Getting lineage for game {game_cid}")

        lineage = await br.get_game_lineage(game_cid)

        return {
            "game_cid": game_cid,
            "lineage": lineage,
            "generation": len(lineage)
        }

    except Exception as e:
        logger.error(f"Error getting game lineage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get game lineage"
        )


# ============================================================================
# User Game Management Endpoints
# ============================================================================

@app.get("/api/v1/users/{user_id}/games", tags=["Users"])
async def list_user_games(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    br: GameAPIBridge = Depends(get_bridge)
):
    """
    List games owned by a specific user.

    This endpoint would typically query CapsuleOS for games
    belonging to the specified user. For now, it returns
    a placeholder response.
    """
    try:
        logger.info(f"Listing games for user {user_id}")

        # TODO: Implement user game listing in CapsuleOS bridge
        # This would require CapsuleOS to support user-based queries

        return {
            "user_id": user_id,
            "games": [],
            "total": 0,
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"Error listing user games: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list user games"
        )


# ============================================================================
# Administrative Endpoints
# ============================================================================

@app.get("/api/v1/admin/stats", tags=["Admin"])
async def get_admin_stats(br: GameAPIBridge = Depends(get_bridge)):
    """
    Get administrative statistics about the system.

    This endpoint provides metrics for monitoring and
    administration of the game generation system.
    """
    try:
        bridge_status = await br.get_status()

        # TODO: Add more detailed statistics from both services
        stats = {
            "bridge": bridge_status.dict(),
            "system": {
                "uptime": "unknown",  # TODO: Track actual uptime
                "total_requests": 0,  # TODO: Track request statistics
                "successful_generations": 0,
                "failed_generations": 0,
                "average_generation_time": 0.0,
            },
            "storage": {
                "total_games": 0,  # TODO: Get from CapsuleOS
                "active_users": 0,
                "storage_usage": "unknown"
            }
        }

        return stats

    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get admin statistics"
        )


@app.post("/api/v1/admin/bridge/restart", tags=["Admin"])
async def restart_bridge():
    """
    Restart the bridge connection to backend services.

    This administrative endpoint can be used to recover
    from connection issues without restarting the entire server.
    """
    global bridge

    try:
        logger.info("Restarting bridge connections...")

        # Stop existing bridge
        if bridge:
            await bridge.stop()

        # Start new bridge
        bridge = await create_bridge()

        return {"status": "success", "message": "Bridge restarted successfully"}

    except Exception as e:
        logger.error(f"Failed to restart bridge: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restart bridge: {str(e)}"
        )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(BridgeError)
async def bridge_error_handler(request, exc: BridgeError):
    """Handle bridge-specific errors."""
    logger.error(f"Bridge error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Bridge error", "detail": str(exc)}
    )


@app.exception_handler(ConnectionError)
async def connection_error_handler(request, exc: ConnectionError):
    """Handle connection errors."""
    logger.error(f"Connection error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "Service unavailable", "detail": str(exc)}
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Validation error", "detail": str(exc)}
    )


# ============================================================================
# Server Entry Point
# ============================================================================

def main(
    host: str = "0.0.0.0",
    port: int = 9000,
    reload: bool = False,
    log_level: str = "info"
):
    """Run the API bridge server."""
    logger.info(f"Starting Game API Bridge server on {host}:{port}")

    uvicorn.run(
        "src.api_bridge.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Game Generation API Bridge Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])

    args = parser.parse_args()

    main(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )