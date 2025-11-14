"""
GΛLYPH Expression Merger

Merges 4 LLM GΛLYPH outputs into a single λgame expression.
Validates all inputs and ensures final expression coherence.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..orchestration.game_orchestrator import LLMResponse


class MergeError(Exception):
    """Error during GΛLYPH expression merging"""
    pass


class ValidationResult(Enum):
    VALID = "valid"
    INVALID_SYNTAX = "invalid_syntax"
    INVALID_STRUCTURE = "invalid_structure"
    MISSING_COMPONENTS = "missing_components"


@dataclass
class ParsedGlyphComponent:
    """Parsed GΛLYPH expression component"""
    component_type: str  # 'narrative', 'mechanics', 'assets', 'balance'
    raw_expression: str
    lambda_name: str
    bindings: List[str]
    body: str
    is_valid: bool
    validation_error: Optional[str] = None


@dataclass
class MergedGameExpression:
    """Final merged game expression"""
    glyph_expression: str
    components: Dict[str, ParsedGlyphComponent]
    balance_score: float
    is_valid: bool
    validation_result: ValidationResult
    merge_time: float = 0.0


class GlyphMerger:
    """Merges multiple GΛLYPH expressions into unified game manifest"""

    def __init__(self):
        self.required_components = ['narrative', 'mechanics', 'assets', 'balance']
        self.balance_range = (0.0, 1.0)

    def merge_responses(self, responses: List[LLMResponse]) -> MergedGameExpression:
        """
        Merge LLM responses into single game expression

        Args:
            responses: List of LLM responses to merge

        Returns:
            MergedGameExpression: Final merged expression with validation
        """
        import time
        start_time = time.time()

        try:
            # Parse and validate all components
            components = self._parse_and_validate_components(responses)

            # Check for required components
            missing = self._check_required_components(components)
            if missing:
                raise MergeError(f"Missing required components: {missing}")

            # Create merged expression
            merged_expr = self._create_merged_expression(components)

            # Extract and validate balance score
            balance_score = self._extract_balance_score(merged_expr, components.get('balance'))

            # Final validation
            validation_result = self._validate_final_expression(merged_expr)

            merge_time = time.time() - start_time

            return MergedGameExpression(
                glyph_expression=merged_expr,
                components=components,
                balance_score=balance_score,
                is_valid=validation_result == ValidationResult.VALID,
                validation_result=validation_result,
                merge_time=merge_time
            )

        except Exception as e:
            merge_time = time.time() - start_time
            # Return expression with error information
            error_expr = f'λgame -> let error = "{str(e)}" in error_manifest(error)'
            return MergedGameExpression(
                glyph_expression=error_expr,
                components={},
                balance_score=0.0,
                is_valid=False,
                validation_result=ValidationResult.INVALID_STRUCTURE,
                merge_time=merge_time
            )

    def _parse_and_validate_components(self, responses: List[LLMResponse]) -> Dict[str, ParsedGlyphComponent]:
        """Parse and validate individual GΛLYPH components"""
        components = {}

        for response in responses:
            if not response.is_valid_glyph:
                continue

            try:
                component = self._parse_glyph_expression(response.llm_type, response.glyph_expression)
                components[response.llm_type] = component
            except Exception as e:
                # Create invalid component with error
                components[response.llm_type] = ParsedGlyphComponent(
                    component_type=response.llm_type,
                    raw_expression=response.glyph_expression,
                    lambda_name="",
                    bindings=[],
                    body="",
                    is_valid=False,
                    validation_error=str(e)
                )

        return components

    def _parse_glyph_expression(self, component_type: str, expression: str) -> ParsedGlyphComponent:
        """Parse individual GΛLYPH expression"""
        # Basic lambda expression parsing
        lambda_match = re.match(r'λ(\w+)\s*->\s*(.+)', expression.strip(), re.DOTALL)
        if not lambda_match:
            raise MergeError(f"Invalid lambda expression: {expression[:50]}...")

        lambda_name = lambda_match.group(1)
        body = lambda_match.group(2).strip()

        # Extract let bindings
        bindings = self._extract_let_bindings(body)

        return ParsedGlyphComponent(
            component_type=component_type,
            raw_expression=expression,
            lambda_name=lambda_name,
            bindings=bindings,
            body=body,
            is_valid=True
        )

    def _extract_let_bindings(self, body: str) -> List[str]:
        """Extract variable names from let bindings"""
        # Find all let bindings
        let_pattern = r'let\s+(\w+)\s*='
        bindings = re.findall(let_pattern, body)
        return bindings

    def _check_required_components(self, components: Dict[str, ParsedGlyphComponent]) -> List[str]:
        """Check for missing required components"""
        missing = []
        for required in self.required_components:
            if required not in components or not components[required].is_valid:
                missing.append(required)
        return missing

    def _create_merged_expression(self, components: Dict[str, ParsedGlyphComponent]) -> str:
        """Create merged λgame expression from components"""
        narrative = components.get('narrative')
        mechanics = components.get('mechanics')
        assets = components.get('assets')
        balance = components.get('balance')

        # If we have a balance component that's already merged, use it
        if balance and 'game_manifest' in balance.body:
            return balance.raw_expression

        # Otherwise, construct merged expression
        merged_parts = []

        # Add narrative component
        if narrative and narrative.is_valid:
            narrative_expr = self._format_component_for_merge(narrative, 'story')
            merged_parts.append(f"let {narrative_expr}")

        # Add mechanics component
        if mechanics and mechanics.is_valid:
            mechanics_expr = self._format_component_for_merge(mechanics, 'rules')
            merged_parts.append(f"let {mechanics_expr}")

        # Add assets component
        if assets and assets.is_valid:
            assets_expr = self._format_component_for_merge(assets, 'visuals')
            merged_parts.append(f"let {assets_expr}")

        # Add balance (either from balance component or default)
        if balance and balance.is_valid:
            balance_value = self._extract_balance_number(balance.body)
        else:
            balance_value = 0.5  # Default balance

        merged_parts.append(f"let balance = {balance_value}")

        # Create final expression
        if merged_parts:
            body = '\n        in '.join(part for part in merged_parts)
            final_expr = f"λgame -> {body}\n        in game_manifest(story, rules, visuals, balance)"
        else:
            # Fallback expression
            final_expr = 'λgame -> let story = "simple game" in let rules = [] in let visuals = [] in let balance = 0.5 in game_manifest(story, rules, visuals, balance)'

        return final_expr

    def _format_component_for_merge(self, component: ParsedGlyphComponent, new_name: str) -> str:
        """Format component expression for merging with new variable name"""
        # Extract the main body from let bindings
        body = component.body

        # Replace the lambda name with new_name in the body
        # For example: narrative_manifest(...) -> story_manifest(...)
        if component.lambda_name == new_name:
            return body

        # Replace lambda references
        body = body.replace(f"{component.lambda_name}_manifest", f"{new_name}_manifest")

        return f"{new_name} = {body}"

    def _extract_balance_score(self, merged_expr: str, balance_component: Optional[ParsedGlyphComponent]) -> float:
        """Extract balance score from expression"""
        try:
            # First try to extract from balance component
            if balance_component and balance_component.is_valid:
                balance_value = self._extract_balance_number(balance_component.body)
                if self.balance_range[0] <= balance_value <= self.balance_range[1]:
                    return balance_value

            # Extract from merged expression
            balance_match = re.search(r'let\s+balance\s*=\s*([0-9]*\.?[0-9]+)', merged_expr)
            if balance_match:
                balance_value = float(balance_match.group(1))
                if self.balance_range[0] <= balance_value <= self.balance_range[1]:
                    return balance_value

            # Look for balance in game_manifest call
            manifest_match = re.search(r'game_manifest\([^,)]*,\s*[^,)]*,\s*[^,)]*,\s*([0-9]*\.?[0-9]+)\s*\)', merged_expr)
            if manifest_match:
                balance_value = float(manifest_match.group(1))
                if self.balance_range[0] <= balance_value <= self.balance_range[1]:
                    return balance_value

        except (ValueError, AttributeError):
            pass

        # Default balance
        return 0.5

    def _extract_balance_number(self, body: str) -> float:
        """Extract numeric balance value from component body"""
        # Look for balance assignment
        balance_match = re.search(r'let\s+balance\s*=\s*([0-9]*\.?[0-9]+)', body)
        if balance_match:
            return float(balance_match.group(1))

        # Look for numbers in the body
        number_matches = re.findall(r'[0-9]*\.?[0-9]+', body)
        if number_matches:
            return float(number_matches[-1])  # Use last number as balance

        return 0.5  # Default

    def _validate_final_expression(self, expression: str) -> ValidationResult:
        """Validate the final merged expression"""
        try:
            # Basic structure validation
            if not expression.startswith('λgame ->'):
                return ValidationResult.INVALID_STRUCTURE

            # Check for required function call
            if 'game_manifest(' not in expression:
                return ValidationResult.INVALID_STRUCTURE

            # Check for required parameters
            if not all(param in expression for param in ['story', 'rules', 'visuals', 'balance']):
                return ValidationResult.MISSING_COMPONENTS

            # Check syntax (balanced delimiters)
            if not self._check_balanced_delimiters(expression):
                return ValidationResult.INVALID_SYNTAX

            return ValidationResult.VALID

        except Exception:
            return ValidationResult.INVALID_SYNTAX

    def _check_balanced_delimiters(self, expression: str) -> bool:
        """Check if all delimiters are balanced"""
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

    def create_fallback_expression(self, error_message: str) -> str:
        """Create a fallback expression when merging fails"""
        # Escape any quotes in error message
        safe_error = error_message.replace('"', '\\"')
        return f'λgame -> let error = "Generation failed: {safe_error}" in let story = "basic game" in let rules = [λstate -> λaction -> state] in let visuals = [] in let balance = 0.3 in game_manifest(story, rules, visuals, balance)'

    def optimize_expression(self, expression: str) -> str:
        """Optimize the merged expression for better performance"""
        # Remove redundant whitespace
        expression = re.sub(r'\s+', ' ', expression)

        # Remove comments (if any)
        expression = re.sub(r'#.*', '', expression)

        # Ensure proper spacing around operators
        expression = re.sub(r'([=(){}[\],])', r' \1 ', expression)
        expression = re.sub(r'\s+', ' ', expression)

        return expression.strip()

    def get_merge_statistics(self, result: MergedGameExpression) -> Dict[str, Any]:
        """Get statistics about the merge process"""
        stats = {
            'success': result.is_valid,
            'validation_result': result.validation_result.value,
            'merge_time': result.merge_time,
            'balance_score': result.balance_score,
            'component_count': len(result.components),
            'valid_components': sum(1 for c in result.components.values() if c.is_valid),
            'expression_length': len(result.glyph_expression),
            'components': {}
        }

        for comp_type, component in result.components.items():
            stats['components'][comp_type] = {
                'is_valid': component.is_valid,
                'lambda_name': component.lambda_name,
                'binding_count': len(component.bindings),
                'expression_length': len(component.raw_expression)
            }

        return stats