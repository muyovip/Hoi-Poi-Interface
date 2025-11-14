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

    async def orchestrate_generation(
        self,
        contexts: List[LLMContext],
        strategy: Optional[str] = None
    ) -> OrchestrationResult:
        """
        Orchestrate game generation using specified strategy

        Args:
            contexts: List of LLM contexts prepared by text orchestrator
            strategy: Override strategy ('parallel', 'sequential', 'pipeline')

        Returns:
            OrchestrationResult: Complete result with all LLM responses
        """
        start_time = time.time()

        # Determine execution strategy
        if strategy:
            exec_strategy = ExecutionStrategy(strategy)
        else:
            exec_strategy = self._determine_strategy(contexts)

        try:
            if exec_strategy == ExecutionStrategy.PARALLEL:
                responses = await self._execute_parallel(contexts)
            elif exec_strategy == ExecutionStrategy.SEQUENTIAL:
                responses = await self._execute_sequential(contexts)
            else:  # PIPELINE
                responses = await self._execute_pipeline(contexts)

            total_time = time.time() - start_time

            return OrchestrationResult(
                strategy=exec_strategy,
                responses=responses,
                total_time=total_time,
                success=True
            )

        except Exception as e:
            total_time = time.time() - start_time
            return OrchestrationResult(
                strategy=exec_strategy,
                responses=[],
                total_time=total_time,
                success=False,
                error=str(e)
            )

    async def _execute_parallel(self, contexts: List[LLMContext]) -> List[LLMResponse]:
        """Execute all LLMs in parallel"""
        tasks = []
        for context in contexts:
            task = asyncio.create_task(
                self._execute_llm_with_retry(context)
            )
            tasks.append(task)

        # Wait for all tasks to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions and ensure all responses are LLMResponse objects
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                # Create error response
                processed_responses.append(LLMResponse(
                    llm_type=contexts[i].llm_type,
                    raw_response="",
                    glyph_expression="",
                    is_valid_glyph=False,
                    validation_error=str(response)
                ))
            else:
                processed_responses.append(response)

        return processed_responses

    async def _execute_sequential(self, contexts: List[LLMContext]) -> List[LLMResponse]:
        """Execute LLMs sequentially (for iteration/intent refinement)"""
        responses = []
        accumulated_context = {}

        for context in contexts:
            # Add accumulated context from previous LLMs
            if accumulated_context:
                context.constraints.update(accumulated_context)

            response = await self._execute_llm_with_retry(context)
            responses.append(response)

            # Accumulate successful responses for next LLM
            if response.is_valid_glyph:
                accumulated_context[f"previous_{context.llm_type}"] = response.glyph_expression

        return responses

    async def _execute_pipeline(self, contexts: List[LLMContext]) -> List[LLMResponse]:
        """Execute LLMs in dependency pipeline (narrative -> mechanics -> assets -> balance)"""
        # Sort contexts by pipeline order
        context_map = {ctx.llm_type: ctx for ctx in contexts}
        ordered_types = ['narrative', 'mechanics', 'assets', 'balance']

        responses = []
        pipeline_data = {}

        for llm_type in ordered_types:
            if llm_type not in context_map:
                continue

            context = context_map[llm_type]

            # Inject pipeline data from previous stages
            if llm_type == 'mechanics' and 'narrative' in pipeline_data:
                context.constraints['narrative_output'] = pipeline_data['narrative']
            elif llm_type == 'assets' and 'narrative' in pipeline_data:
                context.constraints['narrative_output'] = pipeline_data['narrative']
                if 'mechanics' in pipeline_data:
                    context.constraints['mechanics_output'] = pipeline_data['mechanics']
            elif llm_type == 'balance':
                # Balance LLM gets all previous outputs
                for prev_type, prev_output in pipeline_data.items():
                    context.constraints[f'{prev_type}_output'] = prev_output

            response = await self._execute_llm_with_retry(context)
            responses.append(response)

            # Store successful response for pipeline
            if response.is_valid_glyph:
                pipeline_data[llm_type] = response.glyph_expression

        return responses

    async def _execute_llm_with_retry(self, context: LLMContext) -> LLMResponse:
        """Execute LLM with retry logic"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = await self._execute_single_llm(context)

                # Validate GΛLYPH expression
                if self._validate_glyph_expression(response.glyph_expression):
                    return response
                else:
                    # Invalid GΛLYPH - retry with simplified prompt
                    if attempt < self.max_retries - 1:
                        context = self._simplify_context(context, response.glyph_expression)
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue

        # All retries failed
        return LLMResponse(
            llm_type=context.llm_type,
            raw_response="",
            glyph_expression="",
            is_valid_glyph=False,
            validation_error=str(last_error) if last_error else "Max retries exceeded"
        )

    async def _execute_single_llm(self, context: LLMContext) -> LLMResponse:
        """Execute a single LLM call"""
        start_time = time.time()

        # Prepare the final prompt
        prompt = self._prepare_final_prompt(context)

        # Get LLM configuration
        llm_config = self.llm_configs[context.llm_type]

        # Simulate LLM call (replace with actual vLLM API call)
        raw_response = await self._call_llm_api(prompt, llm_config)

        # Extract GΛLYPH expression from response
        glyph_expression = self._extract_glyph_expression(raw_response)

        execution_time = time.time() - start_time

        return LLMResponse(
            llm_type=context.llm_type,
            raw_response=raw_response,
            glyph_expression=glyph_expression,
            is_valid_glyph=False,  # Will be validated in retry logic
            execution_time=execution_time
        )

    def _prepare_final_prompt(self, context: LLMContext) -> str:
        """Prepare the final prompt for the LLM"""
        # Start with base template
        prompt = context.prompt_template

        # Replace placeholders with context data
        replacements = {
            'theme': context.concepts.theme,
            'genre': context.concepts.genre,
            'target_audience': context.concepts.target_audience,
            'complexity': context.concepts.complexity.value,
            'keywords': ', '.join(context.concepts.keywords)
        }

        # Add constraints-specific replacements
        for key, value in context.constraints.items():
            if isinstance(value, list):
                replacements[key] = ', '.join(str(v) for v in value)
            else:
                replacements[key] = str(value)

        # Replace placeholders
        for placeholder, value in replacements.items():
            prompt = prompt.replace(f'{{{placeholder}}}', value)

        # Add pipeline data if present
        for key, value in context.constraints.items():
            if key.endswith('_output') and isinstance(value, str):
                prompt = prompt.replace(f'[{key.upper()}]', value)

        return prompt

    async def _call_llm_api(self, prompt: str, llm_config: Dict[str, Any]) -> str:
        """
        Call the LLM API (placeholder for actual vLLM integration)

        This would be replaced with actual vLLM OpenAI-compatible API call
        """
        # Placeholder implementation - replace with actual API call
        # For now, return a mock GΛLYPH expression based on the LLM type

        await asyncio.sleep(0.1)  # Simulate API latency

        if 'narrative' in prompt.lower():
            return 'λnarrative -> let setting = "mystical world" in let premise = "explore and discover" in let objective = "find the artifact" in narrative_manifest(setting, premise, objective)'
        elif 'mechanics' in prompt.lower():
            return 'λmechanics -> let move = λstate -> λdirection -> update_position(state, direction) in let interact = λstate -> λobject -> check_collision(state, object) in mechanics_manifest([move, interact])'
        elif 'assets' in prompt.lower():
            return 'λassets -> let player = {sprite: "hero", position: (0, 0)} in let world = {tiles: "ground", size: (100, 100)} in assets_manifest(player, world)'
        elif 'balance' in prompt.lower():
            return 'λgame -> let story = λnarrative -> let setting = "mystical world" in let premise = "explore and discover" in let objective = "find the artifact" in narrative_manifest(setting, premise, objective) in let rules = λmechanics -> let move = λstate -> λdirection -> update_position(state, direction) in let interact = λstate -> λobject -> check_collision(state, object) in mechanics_manifest([move, interact]) in let visuals = λassets -> let player = {sprite: "hero", position: (0, 0)} in let world = {tiles: "ground", size: (100, 100)} in assets_manifest(player, world) in let balance = 0.75 in game_manifest(story, rules, visuals, balance)'
        else:
            return 'λmanifest -> simple_game_manifest'

    def _extract_glyph_expression(self, raw_response: str) -> str:
        """Extract GΛLYPH expression from raw response"""
        # Look for lambda expressions in the response
        import re

        # Pattern to match lambda expressions
        lambda_pattern = r'λ\w+\s*->.*'
        matches = re.findall(lambda_pattern, raw_response, re.MULTILINE | re.DOTALL)

        if matches:
            # Return the longest match (most likely to be complete)
            return max(matches, key=len)

        # If no lambda found, return as-is (might be a simple expression)
        return raw_response.strip()

    def _validate_glyph_expression(self, expression: str) -> bool:
        """
        Validate that the expression is valid GΛLYPH code

        This is a simplified validation - in production, use the actual GΛLYPH parser
        """
        if not expression:
            return False

        # Basic structural validation
        if 'λ' not in expression:
            return False

        # Check for balanced parentheses and brackets
        if not self._check_balanced_delimiters(expression):
            return False

        # Check for GΛLYPH keywords
        glyph_keywords = ['let', 'in', 'match', 'if', 'then', 'else', 'λ']
        if not any(keyword in expression for keyword in glyph_keywords):
            return False

        return True

    def _check_balanced_delimiters(self, expression: str) -> bool:
        """Check if parentheses, brackets, and braces are balanced"""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}

        for char in expression:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                if pairs[stack.pop()] != char:
                    return False

        return not stack

    def _simplify_context(self, context: LLMContext, failed_expression: str) -> LLMContext:
        """Create simplified context for retry"""
        simplified = LLMContext(
            llm_type=context.llm_type,
            prompt_template=context.prompt_template,
            concepts=context.concepts,
            constraints=context.constraints.copy()
        )

        # Add simplification instructions
        simplified.constraints['simplify'] = True
        simplified.constraints['previous_attempt'] = failed_expression[:100]  # First 100 chars

        return simplified

    def _determine_strategy(self, contexts: List[LLMContext]) -> ExecutionStrategy:
        """Determine best execution strategy based on contexts"""
        # Check if any context indicates iteration intent
        for context in contexts:
            if context.concepts.intent.value == 'iteration':
                return ExecutionStrategy.SEQUENTIAL

        # Check complexity levels
        complexity_levels = [ctx.concepts.complexity for ctx in contexts]
        if any(level.value == 'complex' for level in complexity_levels):
            return ExecutionStrategy.PIPELINE

        # Default to parallel for simple/moderate requests
        return ExecutionStrategy.PARALLEL

    def get_performance_metrics(self, result: OrchestrationResult) -> Dict[str, Any]:
        """Extract performance metrics from orchestration result"""
        metrics = {
            'strategy': result.strategy.value,
            'total_time': result.total_time,
            'success': result.success,
            'llm_count': len(result.responses),
            'valid_glyph_count': sum(1 for r in result.responses if r.is_valid_glyph),
            'average_llm_time': 0.0,
            'llm_times': {}
        }

        if result.responses:
            total_llm_time = sum(r.execution_time for r in result.responses)
            metrics['average_llm_time'] = total_llm_time / len(result.responses)

            for response in result.responses:
                metrics['llm_times'][response.llm_type] = response.execution_time

        return metrics