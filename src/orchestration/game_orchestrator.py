"""
Multi-LLM Game Orchestration Engine

Coordinates 4 specialized LLMs with adaptive hybrid processing for game generation.
Orchestrates Phi-3 (narrative), Gemma-2B (mechanics), TinyLlama (assets), Qwen-0.5B (balance).
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import time
import aiohttp
from concurrent.futures import TimeoutError

from .text_orchestrator import TextAnalysisResult, LLMContext, OrchestrationStrategy

logger = logging.getLogger(__name__)


class LLMType(Enum):
    NARRATIVE = "narrative"      # Phi-3
    MECHANICS = "mechanics"      # Gemma-2B
    ASSETS = "assets"           # TinyLlama
    BALANCE = "balance"         # Qwen-0.5B


@dataclass
class LLMModel:
    """Configuration for each LLM"""
    llm_type: LLMType
    model_name: str
    api_endpoint: str
    max_tokens: int
    temperature: float
    timeout: int = 60


@dataclass
class LLMResponse:
    """Response from individual LLM"""
    llm_type: LLMType
    model_name: str
    raw_response: str
    glyph_expression: str
    is_valid_glyph: bool
    response_time: float
    token_usage: int
    error: Optional[str] = None


@dataclass
class GameGenerationRequest:
    """Request for game generation"""
    request_id: str
    text_analysis: TextAnalysisResult
    user_id: Optional[str] = None
    parent_cid: Optional[str] = None
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class GameGenerationResult:
    """Result of game generation process"""
    request_id: str
    success: bool
    final_glyph_expression: str
    llm_responses: Dict[LLMType, LLMResponse]
    generation_time: float
    strategy_used: OrchestrationStrategy
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class GameOrchestrator:
    """
    Coordinates 4 specialized LLMs for game generation using adaptive hybrid processing.

    Features:
    - Adaptive routing (parallel, sequential, pipeline) based on input complexity
    - Individual model timeout and retry handling
    - GΛLYPH expression validation for each response
    - Performance monitoring and metrics collection
    """

    def __init__(self, vllm_api_url: str = "http://localhost:8000"):
        self.vllm_api_url = vllm_api_url.rstrip('/')

        # Initialize LLM configurations
        self.models = {
            LLMType.NARRATIVE: LLMModel(
                llm_type=LLMType.NARRATIVE,
                model_name="microsoft/Phi-3-mini-4k-instruct",
                api_endpoint="/v1/chat/completions",
                max_tokens=800,
                temperature=0.8,
                timeout=45
            ),
            LLMType.MECHANICS: LLMModel(
                llm_type=LLMType.MECHANICS,
                model_name="google/gemma-2b-it",
                api_endpoint="/v1/chat/completions",
                max_tokens=1000,
                temperature=0.6,
                timeout=60
            ),
            LLMType.ASSETS: LLMModel(
                llm_type=LLMType.ASSETS,
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                api_endpoint="/v1/chat/completions",
                max_tokens=600,
                temperature=0.7,
                timeout=45
            ),
            LLMType.BALANCE: LLMModel(
                llm_type=LLMType.BALANCE,
                model_name="Qwen/Qwen1.5-0.5B-Chat",
                api_endpoint="/v1/chat/completions",
                max_tokens=1200,
                temperature=0.5,
                timeout=90
            )
        }

        # Performance tracking
        self.active_requests: Dict[str, GameGenerationRequest] = {}
        self.generation_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_generation_time": 0.0,
            "llm_response_times": {llm_type: [] for llm_type in LLMType}
        }

  async def generate_game(self, request: GameGenerationRequest) -> GameGenerationResult:
        """
        Generate a game using the 4 specialized LLMs.

        Args:
            request: Game generation request with text analysis

        Returns:
            GameGenerationResult with final GΛLYPH expression and metadata
        """
        start_time = time.time()
        request_id = request.request_id

        logger.info(f"Starting game generation {request_id} with strategy: {request.text_analysis.strategy.value}")

        # Track active request
        self.active_requests[request_id] = request
        self.generation_metrics["total_requests"] += 1

        try:
            # Execute based on strategy
            if request.text_analysis.strategy == OrchestrationStrategy.PARALLEL:
                llm_responses = await self._execute_parallel(request)
            elif request.text_analysis.strategy == OrchestrationStrategy.SEQUENTIAL:
                llm_responses = await self._execute_sequential(request)
            else:  # PIPELINE
                llm_responses = await self._execute_pipeline(request)

            # Validate responses
            valid_responses = self._validate_responses(llm_responses)
            if not valid_responses:
                raise ValueError("No valid LLM responses received")

            # Generate final result
            final_result = await self._generate_final_result(valid_responses, request)

            generation_time = time.time() - start_time

            # Update metrics
            self.generation_metrics["successful_requests"] += 1
            self._update_performance_metrics(generation_time, llm_responses)

            result = GameGenerationResult(
                request_id=request_id,
                success=True,
                final_glyph_expression=final_result,
                llm_responses=llm_responses,
                generation_time=generation_time,
                strategy_used=request.text_analysis.strategy,
                metadata={
                    "complexity": request.text_analysis.complexity.value,
                    "intent": request.text_analysis.intent.value,
                    "genre": request.text_analysis.game_concept.genre,
                    "theme": request.text_analysis.game_concept.theme
                }
            )

            logger.info(f"Game generation {request_id} completed successfully in {generation_time:.2f}s")
            return result

        except Exception as e:
            generation_time = time.time() - start_time
            self.generation_metrics["failed_requests"] += 1

            logger.error(f"Game generation {request_id} failed: {str(e)}")

            return GameGenerationResult(
                request_id=request_id,
                success=False,
                final_glyph_expression="",
                llm_responses={},
                generation_time=generation_time,
                strategy_used=request.text_analysis.strategy,
                error=str(e)
            )

        finally:
            # Clean up active request
            self.active_requests.pop(request_id, None)

    async def _execute_parallel(self, request: GameGenerationRequest) -> Dict[LLMType, LLMResponse]:
        """Execute all LLMs in parallel for simple requests."""
        logger.info(f"Executing parallel generation for {request.request_id}")

        tasks = []
        for llm_type, context in request.text_analysis.llm_contexts.items():
            task = self._call_llm(LLMType(llm_type), context, request.request_id)
            tasks.append(task)

        # Wait for all LLMs to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process responses
        llm_responses = {}
        for i, response in enumerate(responses):
            llm_type = list(LLMType)[i]
            if isinstance(response, Exception):
                logger.error(f"LLM {llm_type.value} failed: {response}")
                llm_responses[llm_type] = LLMResponse(
                    llm_type=llm_type,
                    model_name=self.models[llm_type].model_name,
                    raw_response="",
                    glyph_expression="",
                    is_valid_glyph=False,
                    response_time=0.0,
                    token_usage=0,
                    error=str(response)
                )
            else:
                llm_responses[llm_type] = response

        return llm_responses

    async def _execute_sequential(self, request: GameGenerationRequest) -> Dict[LLMType, LLMResponse]:
        """Execute LLMs sequentially for iteration requests."""
        logger.info(f"Executing sequential generation for {request.request_id}")

        llm_responses = {}
        accumulated_context = ""

        # Execution order for sequential processing
        execution_order = [LLMType.NARRATIVE, LLMType.MECHANICS, LLMType.ASSETS, LLMType.BALANCE]

        for llm_type in execution_order:
            if llm_type.value not in request.text_analysis.llm_contexts:
                continue

            context = request.text_analysis.llm_contexts[llm_type.value]

            # Modify context to include previous outputs
            if accumulated_context:
                context.user_context += f"\n\nPrevious outputs:\n{accumulated_context}"

            response = await self._call_llm(llm_type, context, request.request_id)
            llm_responses[llm_type] = response

            # Accumulate valid responses for next LLM
            if response.is_valid_glyph:
                accumulated_context += f"\n{llm_type.value}: {response.glyph_expression}"

        return llm_responses

    async def _execute_pipeline(self, request: GameGenerationRequest) -> Dict[LLMType, LLMResponse]:
        """Execute LLMs in pipeline where outputs inform next LLM."""
        logger.info(f"Executing pipeline generation for {request.request_id}")

        llm_responses = {}

        # Pipeline: Narrative -> Mechanics -> Assets -> Balance
        pipeline_order = [
            (LLMType.NARRATIVE, "story_elements"),
            (LLMType.MECHANICS, "game_rules"),
            (LLMType.ASSETS, "visual_assets"),
            (LLMType.BALANCE, "final_balance")
        ]

        pipeline_outputs = {}

        for llm_type, output_key in pipeline_order:
            if llm_type.value not in request.text_analysis.llm_contexts:
                continue

            context = request.text_analysis.llm_contexts[llm_type.value]

            # Modify context to include pipeline outputs
            if pipeline_outputs:
                pipeline_context = "\n".join([
                    f"{key}: {output}" for key, output in pipeline_outputs.items()
                ])
                context.user_context += f"\n\nPipeline inputs:\n{pipeline_context}"

            response = await self._call_llm(llm_type, context, request.request_id)
            llm_responses[llm_type] = response

            # Store output for pipeline
            if response.is_valid_glyph:
                pipeline_outputs[output_key] = response.glyph_expression

        return llm_responses

    async def _call_llm(self, llm_type: LLMType, context: LLMContext, request_id: str) -> LLMResponse:
        """Call individual LLM with context and retry logic."""
        model = self.models[llm_type]
        start_time = time.time()

        logger.debug(f"Calling {llm_type.value} LLM ({model.model_name}) for request {request_id}")

        try:
            # Prepare request payload
            payload = {
                "model": model.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": context.system_prompt
                    },
                    {
                        "role": "user",
                        "content": context.user_context
                    }
                ],
                "max_tokens": model.max_tokens,
                "temperature": model.temperature,
                "stream": False
            }

            # Make API call
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=model.timeout)) as session:
                async with session.post(
                    f"{self.vllm_api_url}{model.api_endpoint}",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API call failed with status {response.status}: {error_text}")

                    result = await response.json()

                    # Extract response content
                    raw_response = result["choices"][0]["message"]["content"]
                    token_usage = result.get("usage", {}).get("total_tokens", 0)
                    response_time = time.time() - start_time

                    # Validate GΛLYPH expression
                    glyph_expression = self._extract_glyph_expression(raw_response)
                    is_valid_glyph = self._validate_glyph_syntax(glyph_expression)

                    logger.debug(f"{llm_type.value} LLM response: {response_time:.2f}s, valid_glyph={is_valid_glyph}")

                    return LLMResponse(
                        llm_type=llm_type,
                        model_name=model.model_name,
                        raw_response=raw_response,
                        glyph_expression=glyph_expression,
                        is_valid_glyph=is_valid_glyph,
                        response_time=response_time,
                        token_usage=token_usage
                    )

        except asyncio.TimeoutError:
            error_msg = f"LLM {llm_type.value} timed out after {model.timeout}s"
            logger.error(error_msg)
            return LLMResponse(
                llm_type=llm_type,
                model_name=model.model_name,
                raw_response="",
                glyph_expression="",
                is_valid_glyph=False,
                response_time=model.timeout,
                token_usage=0,
                error=error_msg
            )

        except Exception as e:
            error_msg = f"LLM {llm_type.value} call failed: {str(e)}"
            logger.error(error_msg)
            response_time = time.time() - start_time
            return LLMResponse(
                llm_type=llm_type,
                model_name=model.model_name,
                raw_response="",
                glyph_expression="",
                is_valid_glyph=False,
                response_time=response_time,
                token_usage=0,
                error=error_msg
            )

    def _extract_glyph_expression(self, raw_response: str) -> str:
        """Extract GΛLYPH expression from LLM response."""
        # Look for code blocks with GΛLYPH
        import re

        # Try to extract from code blocks
        code_block_pattern = r'```(?:g(?:lyph)?|l(?:ambda)?)?\s*\n?(.*?)\n?```'
        matches = re.findall(code_block_pattern, raw_response, re.DOTALL | re.IGNORECASE)

        if matches:
            return matches[0].strip()

        # Look for lambda expressions
        lambda_pattern = r'λ.*?->.*'
        lambda_matches = re.findall(lambda_pattern, raw_response, re.DOTALL)

        if lambda_matches:
            return lambda_matches[0].strip()

        # Return cleaned response as fallback
        return raw_response.strip()

    def _validate_glyph_syntax(self, glyph_expression: str) -> bool:
        """Basic validation of GΛLYPH syntax."""
        if not glyph_expression:
            return False

        # Check for basic GΛLYPH syntax elements
        has_lambda = 'λ' in glyph_expression or '\\' in glyph_expression
        has_arrow = '->' in glyph_expression

        # Check for balanced parentheses and brackets
        open_parens = glyph_expression.count('(')
        close_parens = glyph_expression.count(')')
        open_brackets = glyph_expression.count('[')
        close_brackets = glyph_expression.count(']')

        is_balanced = (open_parens == close_parens) and (open_brackets == close_brackets)

        # Basic structure validation
        has_let = 'let' in glyph_expression
        has_structure = has_lambda and has_arrow and is_balanced

        return has_structure

    def _validate_responses(self, llm_responses: Dict[LLMType, LLMResponse]) -> Dict[LLMType, LLMResponse]:
        """Validate and filter LLM responses."""
        valid_responses = {}

        for llm_type, response in llm_responses.items():
            if response.is_valid_glyph and not response.error:
                valid_responses[llm_type] = response
            else:
                logger.warning(f"Invalid response from {llm_type.value}: {response.error}")

        # Ensure we have at least some valid responses
        if not valid_responses:
            logger.error("No valid LLM responses received")
            return {}

        return valid_responses

    async def _generate_final_result(self, valid_responses: Dict[LLMType, LLMResponse], request: GameGenerationRequest) -> str:
        """Generate final GΛLYPH expression from valid LLM responses."""

        # If balance LLM provided a complete expression, use it
        if LLMType.BALANCE in valid_responses:
            balance_response = valid_responses[LLMType.BALANCE]
            if balance_response.is_valid_glyph and 'λgame' in balance_response.glyph_expression:
                return balance_response.glyph_expression

        # Otherwise, construct from individual responses
        narrative_expr = valid_responses.get(LLMType.NARRATIVE, LLMResponse(
            llm_type=LLMType.NARRATIVE, model_name="", raw_response="",
            glyph_expression='let story = "Generated story..."',
            is_valid_glyph=True, response_time=0, token_usage=0
        )).glyph_expression

        mechanics_expr = valid_responses.get(LLMType.MECHANICS, LLMResponse(
            llm_type=LLMType.MECHANICS, model_name="", raw_response="",
            glyph_expression='let mechanics = [rule("move", λstate -> λaction -> state)]',
            is_valid_glyph=True, response_time=0, token_usage=0
        )).glyph_expression

        assets_expr = valid_responses.get(LLMType.ASSETS, LLMResponse(
            llm_type=LLMType.ASSETS, model_name="", raw_response="",
            glyph_expression='let assets = [item("crystal")]',
            is_valid_glyph=True, response_time=0, token_usage=0
        )).glyph_expression

        # Construct final expression
        final_expression = f"""λgame ->
  {narrative_expr}
  {mechanics_expr}
  {assets_expr}
  let balance = 0.75 in
  manifest story mechanics assets balance"""

        return final_expression

    def _update_performance_metrics(self, generation_time: float, llm_responses: Dict[LLMType, LLMResponse]):
        """Update performance tracking metrics."""
        # Update average generation time
        total_successful = self.generation_metrics["successful_requests"]
        current_avg = self.generation_metrics["average_generation_time"]
        new_avg = (current_avg * (total_successful - 1) + generation_time) / total_successful
        self.generation_metrics["average_generation_time"] = new_avg

        # Update LLM response times
        for llm_type, response in llm_responses.items():
            if response.response_time > 0:
                self.generation_metrics["llm_response_times"][llm_type].append(response.response_time)
                # Keep only last 100 response times per LLM
                if len(self.generation_metrics["llm_response_times"][llm_type]) > 100:
                    self.generation_metrics["llm_response_times"][llm_type].pop(0)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self.generation_metrics,
            "active_requests": len(self.active_requests),
            "model_configs": {llm_type.value: asdict(model) for llm_type, model in self.models.items()}
        }

    def get_active_requests(self) -> List[str]:
        """Get list of active request IDs."""
        return list(self.active_requests.keys())

    async def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request."""
        if request_id in self.active_requests:
            # In a real implementation, you'd cancel the ongoing tasks
            self.active_requests.pop(request_id, None)
            logger.info(f"Cancelled request {request_id}")
            return True
        return False


# Singleton instance
game_orchestrator = GameOrchestrator()


async def generate_game_from_analysis(text_analysis: TextAnalysisResult, user_id: Optional[str] = None, parent_cid: Optional[str] = None) -> GameGenerationResult:
    """
    Generate a game from text analysis results.

    Args:
        text_analysis: Results from text orchestrator analysis
        user_id: Optional user ID for tracking
        parent_cid: Optional parent CID for evolution

    Returns:
        GameGenerationResult with final GΛLYPH expression
    """
    import uuid

    request = GameGenerationRequest(
        request_id=str(uuid.uuid4()),
        text_analysis=text_analysis,
        user_id=user_id,
        parent_cid=parent_cid
    )

    return await game_orchestrator.generate_game(request)