"""
Text Orchestrator for Game Generation

Parses raw text input and prepares structured context for LLM game generation.
Extracts game concepts, themes, mechanics, and determines orchestration strategy.
"""

import re
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InputComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class OrchestrationStrategy(Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    PIPELINE = "pipeline"


class UserIntent(Enum):
    CREATION = "creation"
    ITERATION = "iteration"
    EVOLUTION = "evolution"


@dataclass
class GameConcept:
    """Extracted game concept from raw text"""
    theme: str
    genre: str
    core_mechanics: List[str]
    setting: str
    target_audience: str
    complexity_score: float


@dataclass
class LLMContext:
    """Structured context for each LLM"""
    llm_type: str  # narrative, mechanics, assets, balance
    system_prompt: str
    user_context: str
    constraints: List[str]
    output_format: str
    examples: List[str]


@dataclass
class TextAnalysisResult:
    """Result of text parsing and analysis"""
    original_text: str
    game_concept: GameConcept
    complexity: InputComplexity
    intent: UserIntent
    strategy: OrchestrationStrategy
    llm_contexts: Dict[str, LLMContext]


class TextOrchestrator:
    """
    Parses raw text input and prepares structured context for LLM game generation.

    This component transforms raw user input (PDF parsed to string) into structured
    prompts for each of the 4 specialized LLMs, ensuring all prompts enforce
    GΛLYPH code output only.
    """

    def __init__(self):
        self.complexity_keywords = {
            InputComplexity.SIMPLE: [
                "simple", "basic", "easy", "quick", "minimal", "straightforward"
            ],
            InputComplexity.MODERATE: [
                "interesting", "moderate", "some", "multiple", "several", "standard"
            ],
            InputComplexity.COMPLEX: [
                "complex", "detailed", "comprehensive", "advanced", "intricate",
                "multiple systems", "deep", "rich", "extensive"
            ]
        }

        self.genre_patterns = {
            "puzzle": r"\bpuzzle|riddle|brain.*teaser|logic.*game\b",
            "adventure": r"\badventure|exploration|quest|journey\b",
            "rpg": r"\brpg|role.*play|character|level.*up|stats\b",
            "strategy": r"\bstrategy|tactical|resource.*management|planning\b",
            "simulation": r"\bsimulation|sim|manage|build|create\b",
            "action": r"\baction|fight|combat|battle|shoot\b",
            "card": r"\bcard|deck|hand|collectible\b",
            "board": r"\bboard|game.*board|chess|checkers\b"
        }

        self.mechanic_patterns = {
            "inventory": r"\binventory|items|collect|carry\b",
            "combat": r"\bcombat|fight|battle|attack|damage\b",
            "crafting": r"\bcraft|build|create|combine|make\b",
            "trading": r"\btrade|exchange|buy|sell|market\b",
            "exploration": r"\bexplore|discover|map|area\b",
            "social": r"\bsocial|talk|dialogue|relationship\b",
            "puzzle": r"\bpuzzle|solve|logic|pattern\b",
            "resource": r"\bresource|gather|harvest|mine\b"
        }

    async def analyze_text(self, raw_text: str) -> TextAnalysisResult:
        """
        Analyze raw text input and extract game concepts.

        Args:
            raw_text: Raw text input (PDF parsed to string)

        Returns:
            TextAnalysisResult with extracted concepts and LLM contexts
        """
        logger.info(f"Analyzing text input ({len(raw_text)} characters)")

        # Clean and normalize text
        cleaned_text = self._clean_text(raw_text)

        # Extract game concept
        game_concept = await self._extract_game_concept(cleaned_text)

        # Determine complexity and intent
        complexity = self._determine_complexity(cleaned_text)
        intent = self._determine_intent(cleaned_text)

        # Determine orchestration strategy
        strategy = self._determine_strategy(complexity, intent)

        # Generate LLM contexts
        llm_contexts = await self._generate_llm_contexts(
            game_concept, complexity, strategy
        )

        result = TextAnalysisResult(
            original_text=raw_text,
            game_concept=game_concept,
            complexity=complexity,
            intent=intent,
            strategy=strategy,
            llm_contexts=llm_contexts
        )

        logger.info(f"Analysis complete: {complexity.value} complexity, {strategy.value} strategy")
        return result

    def _clean_text(self, text: str) -> str:
        """Clean and normalize raw text input."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-]', ' ', text)

        # Normalize case
        text = text.lower().strip()

        return text

    async def _extract_game_concept(self, text: str) -> GameConcept:
        """Extract game concept from cleaned text."""

        # Detect genre
        genre = self._detect_genre(text)

        # Extract theme and setting
        theme = self._extract_theme(text)
        setting = self._extract_setting(text)

        # Identify core mechanics
        core_mechanics = self._identify_mechanics(text)

        # Determine target audience
        target_audience = self._determine_audience(text)

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(text)

        return GameConcept(
            theme=theme,
            genre=genre,
            core_mechanics=core_mechanics,
            setting=setting,
            target_audience=target_audience,
            complexity_score=complexity_score
        )

    def _detect_genre(self, text: str) -> str:
        """Detect game genre from text."""
        for genre, pattern in self.genre_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return genre
        return "general"

    def _extract_theme(self, text: str) -> str:
        """Extract theme from text."""
        # Look for theme keywords
        theme_patterns = {
            "space": r"\bspace|planet|star|galaxy|cosmic|alien\b",
            "fantasy": r"\bfantasy|magic|wizard|dragon|medieval|kingdom\b",
            "modern": r"\bmodern|city|contemporary|urban|business\b",
            "historical": r"\bhistorical|ancient|rome|egypt|medieval|war\b",
            "nature": r"\bnature|forest|ocean|animals|environment|wildlife\b",
            "abstract": r"\babstract|geometric|minimal|artistic|creative\b"
        }

        for theme, pattern in theme_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return theme

        return "general"

    def _extract_setting(self, text: str) -> str:
        """Extract setting from text."""
        # Extract setting descriptions
        setting_keywords = [
            "space station", "spaceship", "planet", "castle", "dungeon",
            "city", "town", "forest", "island", "school", "office", "home"
        ]

        found_settings = []
        for keyword in setting_keywords:
            if keyword in text:
                found_settings.append(keyword)

        return found_settings[0] if found_settings else "unspecified"

    def _identify_mechanics(self, text: str) -> List[str]:
        """Identify core game mechanics from text."""
        mechanics = []
        for mechanic, pattern in self.mechanic_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                mechanics.append(mechanic)
        return mechanics

    def _determine_audience(self, text: str) -> str:
        """Determine target audience from text."""
        audience_patterns = {
            "kids": r"\bkids|children|young|family|casual|easy\b",
            "teens": r"\bteen|teenager|young.*adult|school\b",
            "adults": r"\badult|mature|complex|strategic|hard\b",
            "all": r"\ball.*ages|family|everyone|general\b"
        }

        for audience, pattern in audience_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return audience

        return "general"

    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate complexity score (0.0 to 1.0)."""
        score = 0.0

        # Base score from text length
        if len(text) > 500:
            score += 0.2
        if len(text) > 1000:
            score += 0.1

        # Score from identified mechanics
        mechanic_count = len(self._identify_mechanics(text))
        score += min(mechanic_count * 0.1, 0.4)

        # Score from complexity keywords
        for complexity, keywords in self.complexity_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    if complexity == InputComplexity.SIMPLE:
                        score += 0.1
                    elif complexity == InputComplexity.MODERATE:
                        score += 0.2
                    else:  # COMPLEX
                        score += 0.3

        return min(score, 1.0)

    def _determine_complexity(self, text: str) -> InputComplexity:
        """Determine input complexity level."""
        complexity_score = self._calculate_complexity_score(text)

        if complexity_score < 0.3:
            return InputComplexity.SIMPLE
        elif complexity_score < 0.7:
            return InputComplexity.MODERATE
        else:
            return InputComplexity.COMPLEX

    def _determine_intent(self, text: str) -> UserIntent:
        """Determine user intent from text."""
        intent_patterns = {
            UserIntent.ITERATION: r"\brefine|improve|fix|change|modify|update\b",
            UserIntent.EVOLUTION: r"\bevolve|expand|add|extend|continue|sequel\b",
            UserIntent.CREATION: r"\bcreate|make|build|design|generate|new\b"
        }

        for intent, pattern in intent_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return intent

        return UserIntent.CREATION

    def _determine_strategy(self, complexity: InputComplexity, intent: UserIntent) -> OrchestrationStrategy:
        """Determine orchestration strategy based on complexity and intent."""
        if complexity == InputComplexity.SIMPLE:
            return OrchestrationStrategy.PARALLEL
        elif intent == UserIntent.ITERATION:
            return OrchestrationStrategy.SEQUENTIAL
        else:
            return OrchestrationStrategy.PIPELINE

    async def _generate_llm_contexts(self, concept: GameConcept, complexity: InputComplexity, strategy: OrchestrationStrategy) -> Dict[str, LLMContext]:
        """Generate structured context for each LLM."""

        contexts = {}

        # Narrative LLM Context (Phi-3)
        contexts["narrative"] = LLMContext(
            llm_type="narrative",
            system_prompt=self._get_narrative_system_prompt(),
            user_context=self._get_narrative_context(concept),
            constraints=["OUTPUT ONLY VALID GΛLYPH CODE", "Use λlet expressions", "Focus on story elements"],
            output_format="GΛLYPH lambda expression",
            examples=[self._get_narrative_example(concept.genre)]
        )

        # Mechanics LLM Context (Gemma-2B)
        contexts["mechanics"] = LLMContext(
            llm_type="mechanics",
            system_prompt=self._get_mechanics_system_prompt(),
            user_context=self._get_mechanics_context(concept),
            constraints=["OUTPUT ONLY VALID GΛLYPH CODE", "Use functional patterns", "Define game rules"],
            output_format="GΛLYPH lambda expression",
            examples=[self._get_mechanics_example(concept.core_mechanics)]
        )

        # Assets LLM Context (TinyLlama)
        contexts["assets"] = LLMContext(
            llm_type="assets",
            system_prompt=self._get_assets_system_prompt(),
            user_context=self._get_assets_context(concept),
            constraints=["OUTPUT ONLY VALID GΛLYPH CODE", "Describe visual elements", "Define item properties"],
            output_format="GΛLYPH lambda expression",
            examples=[self._get_assets_example(concept.theme)]
        )

        # Balance LLM Context (Qwen-0.5B)
        contexts["balance"] = LLMContext(
            llm_type="balance",
            system_prompt=self._get_balance_system_prompt(),
            user_context=self._get_balance_context(concept, complexity),
            constraints=["OUTPUT ONLY VALID GΛLYPH CODE", "Ensure game balance", "Create single λgame expression"],
            output_format="Single merged GΛLYPH λgame expression",
            examples=[self._get_balance_example()]
        )

        return contexts

    def _get_narrative_system_prompt(self) -> str:
        return """You are a narrative designer for functional games. OUTPUT ONLY VALID GΛLYPH CODE.

Create narrative elements using GΛLYPH lambda calculus expressions. Use λlet expressions to define story components.

Your output must be valid GΛLYPH that can be parsed by glyph_parser."""

    def _get_mechanics_system_prompt(self) -> str:
        return """You are a game mechanics designer. OUTPUT ONLY VALID GΛLYPH CODE.

Define game rules and mechanics using GΛLYPH functional programming patterns. Use lambda expressions and immutable data structures.

Your output must be valid GΛLYPH that can be parsed by glyph_parser."""

    def _get_assets_system_prompt(self) -> str:
        return """You are an asset and visual designer. OUTPUT ONLY VALID GΛLYPH CODE.

Define visual elements, items, and assets using GΛLYPH expressions. Use functional patterns to describe properties.

Your output must be valid GΛLYPH that can be parsed by glyph_parser."""

    def _get_balance_system_prompt(self) -> str:
        return """You are a game balance designer. OUTPUT ONLY VALID GΛLYPH CODE.

Merge all game components into a single λgame expression. Ensure proper balance and coherence.

Your output must be a single valid GΛLYPH λgame expression that can be parsed by glyph_parser."""

    def _get_narrative_context(self, concept: GameConcept) -> str:
        return f"""Create narrative elements for a {concept.genre} game with {concept.theme} theme.
Setting: {concept.setting}
Core mechanics: {', '.join(concept.core_mechanics)}
Target audience: {concept.target_audience}"""

    def _get_mechanics_context(self, concept: GameConcept) -> str:
        return f"""Design game mechanics for a {concept.genre} game.
Core mechanics to implement: {', '.join(concept.core_mechanics)}
Theme: {concept.theme}
Setting: {concept.setting}"""

    def _get_assets_context(self, concept: GameConcept) -> str:
        return f"""Design visual assets for a {concept.theme} themed {concept.genre} game.
Setting: {concept.setting}
Core mechanics: {', '.join(concept.core_mechanics)}"""

    def _get_balance_context(self, concept: GameConcept, complexity: InputComplexity) -> str:
        return f"""Balance and merge game components for a {concept.genre} game.
Complexity level: {complexity.value}
Theme: {concept.theme}
Core mechanics: {', '.join(concept.core_mechanics)}
Create a single cohesive λgame expression."""

    def _get_narrative_example(self, genre: str) -> str:
        return """λlet story = "In a world of endless wonder..." in
λlet protagonist = hero("Ada") in
λlet conflict = quest("find the lost artifact") in
story protagonist conflict"""

    def _get_mechanics_example(self, mechanics: List[str]) -> str:
        return """λlet rules = [
  rule("move", λstate -> λaction -> transition(state, action)),
  rule("collect", λstate -> λitem -> update_inventory(state, item))
] in
λlet win_condition = λstate -> check_victory(state) in
rules win_condition"""

    def _get_assets_example(self, theme: str) -> str:
        return """λlet items = [
  item("crystal", prop("color", "blue"), prop("power", 5)),
  item("key", prop("uses", 1), prop("required", true))
] in
λlet environments = [env("forest"), env("cave")] in
items environments"""

    def _get_balance_example(self) -> str:
        return """λgame ->
  let story = "Adventure awaits..." in
  let mechanics = [move_rule, collect_rule] in
  let assets = [crystal_item, key_item] in
  let balance = 0.75 in
  manifest story mechanics assets balance"""


# Singleton instance
text_orchestrator = TextOrchestrator()


async def analyze_user_input(raw_text: str) -> TextAnalysisResult:
    """
    Analyze user input and prepare LLM contexts.

    Args:
        raw_text: Raw text input from user (PDF parsed to string)

    Returns:
        TextAnalysisResult with structured contexts for all LLMs
    """
    return await text_orchestrator.analyze_text(raw_text)