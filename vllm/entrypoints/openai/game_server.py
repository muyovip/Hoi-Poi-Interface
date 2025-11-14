"""
Game Generation Server Extension

Extends vLLM OpenAI-compatible API with game generation endpoints.
Handles text-to-game generation using multi-LLM orchestration.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.protocol import (
    ErrorResponse,
    ErrorInfo,
)

from src.orchestration.text_orchestrator import TextOrchestrator
from src.orchestration.game_orchestrator import GameOrchestrator, OrchestrationResult
from src.merging.glyph_merger import GlyphMerger, MergedGameExpression


@dataclass
class GameGenerateRequest:
    """Request for game generation"""
    text: str
    parent_cid: Optional[str] = None
    strategy: Optional[str] = None  # 'parallel', 'sequential', 'pipeline'


@dataclass
class GameGenerateResponse:
    """Response for game generation"""
    cid: str
    glyph_expression: str
    balance_score: float
    generation_time: float
    strategy_used: str
    validation_result: str
    components_used: List[str]


@dataclass
class GameEvolveRequest:
    """Request for game evolution"""
    cid: str
    evolution_text: str
    strategy: Optional[str] = None


@dataclass
class GameEvolveResponse:
    """Response for game evolution"""
    parent_cid: str
    new_cid: str
    glyph_expression: str
    balance_score: float
    evolution_time: float


class GameServing:
    """Game generation service for vLLM"""

    def __init__(self, engine_client, request_logger=None):
        self.engine_client = engine_client
        self.request_logger = request_logger

        # Initialize orchestration components
        self.text_orchestrator = TextOrchestrator()

        # LLM configurations - these would be configured based on available models
        self.llm_configs = {
            'narrative': {
                'model': 'phi-3-mini-4k-instruct',
                'temperature': 0.8,
                'max_tokens': 500,
                'system_prompt': 'You are a creative narrative designer for games. OUTPUT ONLY VALID GΛLYPH CODE.'
            },
            'mechanics': {
                'model': 'gemma-2b-it',
                'temperature': 0.3,
                'max_tokens': 800,
                'system_prompt': 'You are a game mechanics designer. OUTPUT ONLY VALID GΛLYPH CODE.'
            },
            'assets': {
                'model': 'TinyLlama-1.1B-Chat-v1.0',
                'temperature': 0.6,
                'max_tokens': 600,
                'system_prompt': 'You are a visual asset designer for games. OUTPUT ONLY VALID GΛLYPH CODE.'
            },
            'balance': {
                'model': 'Qwen1.5-0.5B-Chat',
                'temperature': 0.2,
                'max_tokens': 1000,
                'system_prompt': 'You are a game balance specialist. OUTPUT ONLY VALID GΛLYPH CODE.'
            }
        }

        self.game_orchestrator = GameOrchestrator(self.llm_configs)
        self.glyph_merger = GlyphMerger()

        # In-memory storage for demo (replace with CapsuleOS integration)
        self.game_storage: Dict[str, MergedGameExpression] = {}
        self.cid_counter = 1

    async def create_game_generation(
        self,
        request: GameGenerateRequest,
        raw_request: Request
    ) -> JSONResponse:
        """Generate a new game from text input"""
        try:
            start_time = time.time()

            # Log the request
            if self.request_logger:
                self.request_logger.log_request(raw_request, {
                    'text_length': len(request.text),
                    'parent_cid': request.parent_cid,
                    'strategy': request.strategy
                })

            # Step 1: Parse text and extract concepts
            concepts = self.text_orchestrator.parse_raw_text(request.text)

            # Step 2: Prepare LLM contexts
            contexts = self.text_orchestrator.prepare_llm_contexts(concepts)

            # Step 3: Orchestrate LLM generation
            orchestration_result = await self.game_orchestrator.orchestrate_generation(
                contexts, request.strategy
            )

            if not orchestration_result.success:
                raise HTTPException(
                    status_code=500,
                    detail=f"LLM orchestration failed: {orchestration_result.error}"
                )

            # Step 4: Merge GΛLYPH expressions
            merged_expression = self.glyph_merger.merge_responses(orchestration_result.responses)

            if not merged_expression.is_valid:
                # Use fallback expression
                merged_expression.glyph_expression = self.glyph_merger.create_fallback_expression(
                    f"Merge validation failed: {merged_expression.validation_result.value}"
                )
                merged_expression.is_valid = True
                merged_expression.validation_result = None

            # Step 5: Generate CID and store
            cid = self._generate_cid()
            self.game_storage[cid] = merged_expression

            # Step 6: Store in CapsuleOS (when integrated)
            # TODO: Add CapsuleOS integration

            generation_time = time.time() - start_time

            # Create response
            response = GameGenerateResponse(
                cid=cid,
                glyph_expression=merged_expression.glyph_expression,
                balance_score=merged_expression.balance_score,
                generation_time=generation_time,
                strategy_used=orchestration_result.strategy.value,
                validation_result=merged_expression.validation_result.value if merged_expression.validation_result else "valid",
                components_used=list(merged_expression.components.keys())
            )

            # Log successful generation
            if self.request_logger:
                self.request_logger.log_response({
                    'cid': cid,
                    'success': True,
                    'generation_time': generation_time,
                    'components_used': response.components_used,
                    'balance_score': response.balance_score
                })

            return JSONResponse(content=asdict(response))

        except HTTPException:
            raise
        except Exception as e:
            # Log error
            if self.request_logger:
                self.request_logger.log_response({
                    'success': False,
                    'error': str(e)
                })

            error = ErrorResponse(
                error=ErrorInfo(
                    message=f"Game generation failed: {str(e)}",
                    type="generation_error",
                    code=500
                )
            )
            return JSONResponse(
                content=error.model_dump(),
                status_code=500
            )

    async def get_game(self, cid: str, raw_request: Request) -> JSONResponse:
        """Retrieve a game by CID"""
        try:
            # Check if game exists
            if cid not in self.game_storage:
                raise HTTPException(
                    status_code=404,
                    detail=f"Game with CID {cid} not found"
                )

            game = self.game_storage[cid]

            # Log retrieval
            if self.request_logger:
                self.request_logger.log_request(raw_request, {'cid': cid})

            response = {
                'cid': cid,
                'glyph_expression': game.glyph_expression,
                'balance_score': game.balance_score,
                'is_valid': game.is_valid,
                'validation_result': game.validation_result.value if game.validation_result else "valid",
                'components': {
                    comp_type: {
                        'is_valid': comp.is_valid,
                        'lambda_name': comp.lambda_name,
                        'binding_count': len(comp.bindings)
                    }
                    for comp_type, comp in game.components.items()
                },
                'merge_time': game.merge_time
            }

            return JSONResponse(content=response)

        except HTTPException:
            raise
        except Exception as e:
            error = ErrorResponse(
                error=ErrorInfo(
                    message=f"Failed to retrieve game: {str(e)}",
                    type="retrieval_error",
                    code=500
                )
            )
            return JSONResponse(
                content=error.model_dump(),
                status_code=500
            )

    async def evolve_game(
        self,
        cid: str,
        request: GameEvolveRequest,
        raw_request: Request
    ) -> JSONResponse:
        """Evolve an existing game with new text"""
        try:
            start_time = time.time()

            # Check if parent game exists
            if cid not in self.game_storage:
                raise HTTPException(
                    status_code=404,
                    detail=f"Parent game with CID {cid} not found"
                )

            parent_game = self.game_storage[cid]

            # Log evolution request
            if self.request_logger:
                self.request_logger.log_request(raw_request, {
                    'parent_cid': cid,
                    'evolution_text_length': len(request.evolution_text),
                    'strategy': request.strategy
                })

            # Step 1: Parse evolution text with context from parent
            combined_text = f"Original theme: {parent_game.components.get('narrative', {}).raw_expression if parent_game.components.get('narrative') else ''}\n\nEvolution request: {request.evolution_text}"
            concepts = self.text_orchestrator.parse_raw_text(combined_text)

            # Set intent to evolution
            from src.orchestration.text_orchestrator import IntentType
            concepts.intent = IntentType.EVOLUTION

            # Step 2: Prepare LLM contexts with parent context
            contexts = self.text_orchestrator.prepare_llm_contexts(concepts)

            # Add parent game expressions to context constraints
            for context in contexts:
                if context.llm_type in parent_game.components:
                    parent_component = parent_game.components[context.llm_type]
                    context.constraints['parent_expression'] = parent_component.glyph_expression

            # Step 3: Generate evolved game
            orchestration_result = await self.game_orchestrator.orchestrate_generation(
                contexts, request.strategy
            )

            if not orchestration_result.success:
                raise HTTPException(
                    status_code=500,
                    detail=f"Evolution orchestration failed: {orchestration_result.error}"
                )

            # Step 4: Merge evolved expressions
            merged_expression = self.glyph_merger.merge_responses(orchestration_result.responses)

            # Step 5: Generate new CID and store
            new_cid = self._generate_cid()
            self.game_storage[new_cid] = merged_expression

            evolution_time = time.time() - start_time

            response = GameEvolveResponse(
                parent_cid=cid,
                new_cid=new_cid,
                glyph_expression=merged_expression.glyph_expression,
                balance_score=merged_expression.balance_score,
                evolution_time=evolution_time
            )

            # Log successful evolution
            if self.request_logger:
                self.request_logger.log_response({
                    'parent_cid': cid,
                    'new_cid': new_cid,
                    'success': True,
                    'evolution_time': evolution_time
                })

            return JSONResponse(content=asdict(response))

        except HTTPException:
            raise
        except Exception as e:
            error = ErrorResponse(
                error=Info(
                    message=f"Game evolution failed: {str(e)}",
                    type="evolution_error",
                    code=500
                )
            )
            return JSONResponse(
                content=error.model_dump(),
                status_code=500
            )

    def _generate_cid(self) -> str:
        """Generate a content-addressable ID"""
        # Simple CID generation (replace with actual CapsuleOS CID generation)
        cid = f"⊙{self.cid_counter:09d}"
        self.cid_counter += 1
        return cid

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        total_games = len(self.game_storage)
        valid_games = sum(1 for game in self.game_storage.values() if game.is_valid)

        return {
            'status': 'healthy',
            'total_games_generated': total_games,
            'valid_games': valid_games,
            'llm_configs': list(self.llm_configs.keys()),
            'orchestrator_available': True,
            'merger_available': True,
            'storage_type': 'in_memory_demo'  # Will change to 'capsuleos' when integrated
        }

    def create_error_response(self, message: str, code: int = 500) -> JSONResponse:
        """Create standardized error response"""
        error = ErrorResponse(
            error=ErrorInfo(
                message=message,
                type="game_server_error",
                code=code
            )
        )
        return JSONResponse(
            content=error.model_dump(),
            status_code=code
        )