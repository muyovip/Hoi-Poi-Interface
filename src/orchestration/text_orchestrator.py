"""
Raw Text to GΛLYPH Orchestrator

Parses raw chat text and coordinates LLM GΛLYPH generation.
Extracts game concepts, themes, and mechanics for multi-LLM processing.
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ComplexityLevel(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class IntentType(Enum):
    GENERATION = "generation"
    ITERATION = "iteration"
    EVOLUTION = "evolution"


@dataclass
class GameConcepts:
    """Extracted concepts from raw text"""
    theme: str
    genre: str
    mechanics: List[str]
    assets: List[str]
    target_audience: str
    complexity: ComplexityLevel
    intent: IntentType
    keywords: List[str]
    context: Dict[str, any]


@dataclass
class LLMContext:
    """Context prepared for each LLM"""
    llm_type: str
    prompt_template: str
    concepts: GameConcepts
    constraints: Dict[str, any]
    examples: Optional[List[str]] = None


class TextOrchestrator:
    """Orchestrates parsing of raw text and LLM context preparation"""

    def __init__(self):
        self.genre_patterns = {
            'puzzle': ['puzzle', 'solve', 'brain', 'logic', 'match'],
            'rpg': ['role', 'character', 'level', 'quest', 'story'],
            'strategy': ['strategy', 'manage', 'build', 'resource', 'war'],
            'action': ['action', 'fight', 'battle', 'shoot', 'run'],
            'simulation': ['simulate', 'life', 'world', 'city', 'business'],
            'adventure': ['adventure', 'explore', 'journey', 'discover', 'map'],
            'sports': ['sport', 'team', 'play', 'match', 'score'],
            'racing': ['race', 'drive', 'speed', 'car', 'track']
        }

        self.mechanic_patterns = {
            'inventory': ['inventory', 'items', 'collect', 'carry'],
            'crafting': ['craft', 'combine', 'make', 'build'],
            'combat': ['fight', 'battle', 'attack', 'defend'],
            'trading': ['trade', 'buy', 'sell', 'exchange'],
            'dialogue': ['talk', 'dialogue', 'conversation', 'choice'],
            'exploration': ['explore', 'discover', 'map', 'area'],
            'progression': ['level', 'progress', 'upgrade', 'skill'],
            'multiplayer': ['multiplayer', 'social', 'team', 'versus'],
            'puzzle': ['puzzle', 'solve', 'logic', 'pattern'],
            'resource': ['resource', 'gather', 'manage', 'time']
        }

        # LLM-specific prompt templates
        self.llm_prompts = {
            'narrative': self._create_narrative_prompt(),
            'mechanics': self._create_mechanics_prompt(),
            'assets': self._create_assets_prompt(),
            'balance': self._create_balance_prompt()
        }

    def parse_raw_text(self, text: str) -> GameConcepts:
        """
        Parse raw text and extract game concepts

        Args:
            text: Raw text input (e.g., PDF parsed to string)

        Returns:
            GameConcepts: Extracted concepts and metadata
        """
        # Clean and normalize text
        cleaned_text = self._clean_text(text)

        # Extract basic information
        theme = self._extract_theme(cleaned_text)
        genre = self._detect_genre(cleaned_text)
        mechanics = self._extract_mechanics(cleaned_text)
        assets = self._extract_assets(cleaned_text)
        target_audience = self._detect_audience(cleaned_text)

        # Analyze complexity and intent
        complexity = self._analyze_complexity(cleaned_text, mechanics)
        intent = self._detect_intent(cleaned_text)

        # Extract keywords
        keywords = self._extract_keywords(cleaned_text)

        # Build context
        context = {
            'word_count': len(cleaned_text.split()),
            'sentence_count': len(re.findall(r'[.!?]+', cleaned_text)),
            'has_numbers': bool(re.search(r'\d', cleaned_text)),
            'has_specifics': bool(re.search(r'\b(specific|exactly|number|count)\b', cleaned_text, re.I))
        }

        return GameConcepts(
            theme=theme,
            genre=genre,
            mechanics=mechanics,
            assets=assets,
            target_audience=target_audience,
            complexity=complexity,
            intent=intent,
            keywords=keywords,
            context=context
        )

    def determine_strategy(self, concepts: GameConcepts) -> str:
        """
        Determine orchestration strategy based on input analysis

        Returns: 'parallel', 'sequential', or 'pipeline'
        """
        # Simple requests → parallel processing
        if concepts.complexity == ComplexityLevel.SIMPLE:
            return "parallel"

        # Iteration requests → sequential
        if concepts.intent == IntentType.ITERATION:
            return "sequential"

        # Complex requests → pipeline
        if concepts.complexity == ComplexityLevel.COMPLEX:
            return "pipeline"

        # Default to parallel for moderate complexity
        return "parallel"

    def prepare_llm_contexts(self, concepts: GameConcepts) -> List[LLMContext]:
        """
        Prepare LLM-specific contexts and prompts

        Returns list of contexts in execution order
        """
        strategy = self.determine_strategy(concepts)

        # Base context for all LLMs
        base_context = {
            'theme': concepts.theme,
            'genre': concepts.genre,
            'target_audience': concepts.target_audience,
            'complexity': concepts.complexity.value,
            'keywords': concepts.keywords
        }

        # Narrative LLM context
        narrative_context = LLMContext(
            llm_type="narrative",
            prompt_template=self.llm_prompts['narrative'],
            concepts=concepts,
            constraints={
                'max_length': 500 if concepts.complexity == ComplexityLevel.SIMPLE else 1000,
                'style': 'engaging and immersive',
                'elements': ['setting', 'premise', 'objective']
            },
            examples=self._get_narrative_examples(concepts.genre)
        )

        # Mechanics LLM context
        mechanics_context = LLMContext(
            llm_type="mechanics",
            prompt_template=self.llm_prompts['mechanics'],
            concepts=concepts,
            constraints={
                'max_rules': 3 if concepts.complexity == ComplexityLevel.SIMPLE else 7,
                'complexity_level': concepts.complexity.value,
                'required_mechanics': concepts.mechanics
            },
            examples=self._get_mechanics_examples(concepts.mechanics)
        )

        # Assets LLM context
        assets_context = LLMContext(
            llm_type="assets",
            prompt_template=self.llm_prompts['assets'],
            concepts=concepts,
            constraints={
                'visual_style': 'consistent with theme',
                'asset_count': 5 if concepts.complexity == ComplexityLevel.SIMPLE else 15,
                'types': concepts.assets
            },
            examples=self._get_assets_examples(concepts.genre)
        )

        # Balance LLM context (receives all other outputs)
        balance_context = LLMContext(
            llm_type="balance",
            prompt_template=self.llm_prompts['balance'],
            concepts=concepts,
            constraints={
                'balance_range': (0.1, 0.9),
                'cohesion_weight': 0.4,
                'fun_weight': 0.6
            }
        )

        # Order contexts based on strategy
        if strategy == "pipeline":
            return [narrative_context, mechanics_context, assets_context, balance_context]
        elif strategy == "sequential":
            return [narrative_context, mechanics_context, assets_context, balance_context]
        else:  # parallel
            return [narrative_context, mechanics_context, assets_context, balance_context]

    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text.strip()

    def _extract_theme(self, text: str) -> str:
        """Extract the main theme from text"""
        # Simple keyword extraction for theme
        theme_keywords = {
            'space': ['space', 'planet', 'galaxy', 'star', 'rocket', 'universe'],
            'fantasy': ['magic', 'dragon', 'wizard', 'kingdom', 'sword', 'spell'],
            'modern': ['city', 'car', 'phone', 'computer', 'modern', 'technology'],
            'historical': ['ancient', 'history', 'medieval', 'rome', 'egypt', 'castle'],
            'nature': ['forest', 'animal', 'plant', 'ocean', 'mountain', 'wild'],
            'abstract': ['abstract', 'concept', 'idea', 'philosophy', 'mind', 'thought']
        }

        text_lower = text.lower()
        theme_scores = {}

        for theme, keywords in theme_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                theme_scores[theme] = score

        if theme_scores:
            return max(theme_scores, key=theme_scores.get)
        return "general"

    def _detect_genre(self, text: str) -> str:
        """Detect game genre from text"""
        text_lower = text.lower()
        genre_scores = {}

        for genre, patterns in self.genre_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                genre_scores[genre] = score

        if genre_scores:
            return max(genre_scores, key=genre_scores.get)
        return "general"

    def _extract_mechanics(self, text: str) -> List[str]:
        """Extract game mechanics from text"""
        text_lower = text.lower()
        found_mechanics = []

        for mechanic, patterns in self.mechanic_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                found_mechanics.append(mechanic)

        return found_mechanics

    def _extract_assets(self, text: str) -> List[str]:
        """Extract asset types from text"""
        asset_patterns = {
            'characters': ['character', 'player', 'hero', 'person', 'avatar'],
            'environment': ['world', 'map', 'level', 'environment', 'background'],
            'ui': ['interface', 'menu', 'button', 'display', 'ui'],
            'audio': ['sound', 'music', 'audio', 'effect', 'noise'],
            'items': ['item', 'object', 'tool', 'equipment', 'gear'],
            'effects': ['effect', 'particle', 'animation', 'visual', 'effect']
        }

        text_lower = text.lower()
        found_assets = []

        for asset, patterns in asset_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                found_assets.append(asset)

        return found_assets

    def _detect_audience(self, text: str) -> str:
        """Detect target audience"""
        audience_patterns = {
            'kids': ['kid', 'child', 'young', 'simple', 'easy', 'fun'],
            'teens': ['teen', 'challenge', 'competitive', 'social'],
            'adults': ['adult', 'complex', 'strategic', 'deep', 'mature'],
            'casual': ['casual', 'relax', 'simple', 'quick', 'easy'],
            'hardcore': ['hardcore', 'difficult', 'challenge', 'complex', 'skill']
        }

        text_lower = text.lower()
        audience_scores = {}

        for audience, patterns in audience_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                audience_scores[audience] = score

        if audience_scores:
            return max(audience_scores, key=audience_scores.get)
        return "general"

    def _analyze_complexity(self, text: str, mechanics: List[str]) -> ComplexityLevel:
        """Analyze complexity of the request"""
        complexity_score = 0

        # Word count contribution
        word_count = len(text.split())
        if word_count > 50:
            complexity_score += 1
        if word_count > 100:
            complexity_score += 1

        # Mechanics count contribution
        if len(mechanics) > 3:
            complexity_score += 1
        if len(mechanics) > 5:
            complexity_score += 1

        # Specific requirements contribution
        if re.search(r'\b(specific|exactly|requirement|must|should)\b', text, re.I):
            complexity_score += 1

        # Multi-step or conditional logic
        if re.search(r'\b(if|then|else|when|after|before)\b', text, re.I):
            complexity_score += 1

        if complexity_score <= 2:
            return ComplexityLevel.SIMPLE
        elif complexity_score <= 4:
            return ComplexityLevel.MODERATE
        else:
            return ComplexityLevel.COMPLEX

    def _detect_intent(self, text: str) -> IntentType:
        """Detect user intent from text"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['create', 'generate', 'make', 'build']):
            return IntentType.GENERATION
        elif any(word in text_lower for word in ['update', 'modify', 'change', 'improve', 'add']):
            return IntentType.ITERATION
        elif any(word in text_lower for word in ['evolve', 'expand', 'grow', 'develop']):
            return IntentType.EVOLUTION
        else:
            return IntentType.GENERATION

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction - remove common words and keep important ones
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}

        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in common_words and len(word) > 2]

        # Return top keywords by frequency
        from collections import Counter
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(10)]

    def _create_narrative_prompt(self) -> str:
        """Create prompt template for narrative LLM"""
        return """
OUTPUT ONLY VALID GΛLYPH CODE

Generate a GΛLYPH lambda expression for the game narrative/theme.

Theme: {theme}
Genre: {genre}
Target Audience: {target_audience}
Complexity: {complexity}

Create a narrative expression with:
- Setting description
- Core premise
- Player objective

Format:
λnarrative -> let setting = "..." in let premise = "..." in let objective = "..." in narrative_manifest(setting, premise, objective)

Constraints:
- Max length: {max_length}
- Style: {style}
- Must include: {elements}

OUTPUT ONLY VALID GΛLYPH CODE
"""

    def _create_mechanics_prompt(self) -> str:
        """Create prompt template for mechanics LLM"""
        return """
OUTPUT ONLY VALID GΛLYPH CODE

Generate GΛLYPH lambda expressions for game mechanics.

Theme: {theme}
Required Mechanics: {required_mechanics}
Complexity Level: {complexity}

Create {max_rules} game rule expressions:
- Each rule as a pure function
- No mutable state
- Deterministic outcomes

Format example:
λmechanics -> let move_rule = λstate -> λaction -> ... in let win_rule = λstate -> ... in mechanics_manifest([move_rule, win_rule])

OUTPUT ONLY VALID GΛLYPH CODE
"""

    def _create_assets_prompt(self) -> str:
        """Create prompt template for assets LLM"""
        return """
OUTPUT ONLY VALID GΛLYPH CODE

Generate GΛLYPH lambda expressions for visual assets.

Theme: {theme}
Visual Style: consistent with {theme}
Asset Types: {types}
Asset Count: {asset_count}

Create asset descriptors as immutable data structures:
- Visual properties
- Display rules
- Interaction handlers

Format:
λassets -> let characters = [...] in let environment = [...] in assets_manifest(characters, environment)

OUTPUT ONLY VALID GΛLYPH CODE
"""

    def _create_balance_prompt(self) -> str:
        """Create prompt template for balance LLM"""
        return """
OUTPUT ONLY VALID GΛLYPH CODE

Merge all GΛLYPH expressions into a single game manifest.

Inputs:
- Narrative: [NARRATIVE_OUTPUT]
- Mechanics: [MECHANICS_OUTPUT]
- Assets: [ASSETS_OUTPUT]

Create unified λgame expression:
- Balance all components
- Ensure coherence
- Output final balance score (0.0-1.0)

Format:
λgame -> let story = [merged_narrative] in let rules = [merged_mechanics] in let visuals = [merged_assets] in let balance = 0.75 in game_manifest(story, rules, visuals, balance)

CRITICAL: OUTPUT ONLY VALID GΛLYPH CODE
"""

    def _get_narrative_examples(self, genre: str) -> List[str]:
        """Get example narrative expressions for genre"""
        examples = {
            'puzzle': ['λnarrative -> let setting = "mystical temple" in let premise = "solve ancient puzzles" in let objective = "unlock the treasure" in narrative_manifest(setting, premise, objective)'],
            'rpg': ['λnarrative -> let setting = "fantasy kingdom" in let premise = "save the realm" in let objective = "defeat the dark lord" in narrative_manifest(setting, premise, objective)']
        }
        return examples.get(genre, [])

    def _get_mechanics_examples(self, mechanics: List[str]) -> List[str]:
        """Get example mechanics expressions"""
        base_example = 'λmechanics -> let move = λstate -> λdirection -> update_position(state, direction) in let check_win = λstate -> is_goal_reached(state) in mechanics_manifest([move, check_win])'
        return [base_example]

    def _get_assets_examples(self, genre: str) -> List[str]:
        """Get example assets expressions"""
        base_example = 'λassets -> let player = {sprite: "hero", size: (32, 32)} in let world = {tiles: "grassland", size: (100, 100)} in assets_manifest([player, world])'
        return [base_example]