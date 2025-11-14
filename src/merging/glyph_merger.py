"""
GΛLYPH Expression Merger

Merges 4 LLM GΛLYPH outputs into single λgame expression.
Validates syntax, ensures coherence, and creates final balanced expression.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import ast

logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    """Strategy for merging GΛLYPH expressions"""
    SIMPLE = "simple"           # Direct combination
    INTELLIGENT = "intelligent"  # Semantic analysis and merging
    BALANCED = "balanced"       # Focus on game balance and coherence


@dataclass
class GlyphExpression:
    """Parsed GΛLYPH expression with metadata"""
    llm_type: str
    raw_expression: str
    parsed_ast: Optional[Any]
    variables: List[str]
    functions: List[str]
    dependencies: List[str]
    is_valid: bool
    confidence: float = 1.0


@dataclass
class MergeResult:
    """Result of GΛLYPH expression merging"""
    success: bool
    final_expression: str
    merge_strategy: MergeStrategy
    component_expressions: Dict[str, str]
    validation_errors: List[str]
    merge_score: float
    processing_time: float
    metadata: Dict[str, Any]


class GlyphMerger:
    """
    Merges 4 LLM GΛLYPH outputs into single coherent λgame expression.

    Features:
    - Expression parsing and validation
    - Dependency analysis and resolution
    - Variable scoping and conflict resolution
    - Semantic coherence checking
    - Balance scoring and optimization
    """

    def __init__(self):
        self.glyph_keywords = {
            'lambda_symbols': ['λ', '\\'],
            'binding_keywords': ['let', 'in'],
            'control_keywords': ['if', 'then', 'else', 'match'],
            'data_keywords': ['tuple', 'list', 'record'],
            'function_keywords': ['fn', 'func', 'function']
        }

        self.game_components = {
            'narrative': ['story', 'setting', 'premise', 'objective', 'narrative'],
            'mechanics': ['rules', 'mechanics', 'actions', 'transitions', 'logic'],
            'assets': ['assets', 'visuals', 'items', 'environment', 'graphics'],
            'balance': ['balance', 'scoring', 'difficulty', 'progression']
        }

    async def merge_expressions(
        self,
        llm_expressions: Dict[str, str],
        strategy: Optional[MergeStrategy] = None
    ) -> MergeResult:
        """
        Merge 4 LLM GΛLYPH expressions into single λgame expression.

        Args:
            llm_expressions: Dictionary mapping LLM types to their GΛLYPH expressions
            strategy: Optional merge strategy override

        Returns:
            MergeResult with final expression and metadata
        """
        import time
        start_time = time.time()

        logger.info(f"Starting GΛLYPH expression merge with {len(llm_expressions)} expressions")

        try:
            # Parse and validate all expressions
            parsed_expressions = await self._parse_expressions(llm_expressions)

            # Determine merge strategy
            if strategy is None:
                strategy = self._determine_merge_strategy(parsed_expressions)

            # Perform merging based on strategy
            if strategy == MergeStrategy.SIMPLE:
                final_expression = await self._simple_merge(parsed_expressions)
            elif strategy == MergeStrategy.INTELLIGENT:
                final_expression = await self._intelligent_merge(parsed_expressions)
            else:  # BALANCED
                final_expression = await self._balanced_merge(parsed_expressions)

            # Validate final expression
            validation_errors = self._validate_final_expression(final_expression)

            # Calculate merge score
            merge_score = self._calculate_merge_score(parsed_expressions, final_expression)

            processing_time = time.time() - start_time

            result = MergeResult(
                success=len(validation_errors) == 0,
                final_expression=final_expression,
                merge_strategy=strategy,
                component_expressions=llm_expressions,
                validation_errors=validation_errors,
                merge_score=merge_score,
                processing_time=processing_time,
                metadata={
                    'parsed_count': len(parsed_expressions),
                    'valid_count': sum(1 for exp in parsed_expressions if exp.is_valid),
                    'strategy_used': strategy.value
                }
            )

            logger.info(f"GΛLYPH merge completed in {processing_time:.2f}s, score: {merge_score:.2f}")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"GΛLYPH merge failed: {str(e)}")

            return MergeResult(
                success=False,
                final_expression="",
                merge_strategy=strategy or MergeStrategy.SIMPLE,
                component_expressions=llm_expressions,
                validation_errors=[f"Merge failed: {str(e)}"],
                merge_score=0.0,
                processing_time=processing_time,
                metadata={'error': str(e)}
            )

    async def _parse_expressions(self, llm_expressions: Dict[str, str]) -> List[GlyphExpression]:
        """Parse and validate individual GΛLYPH expressions."""
        parsed = []

        for llm_type, expression in llm_expressions.items():
            try:
                # Clean and extract the core expression
                cleaned_expr = self._clean_expression(expression)

                # Parse AST (simplified for demonstration)
                ast_result = self._parse_glyph_ast(cleaned_expr)

                # Extract variables and functions
                variables = self._extract_variables(cleaned_expr)
                functions = self._extract_functions(cleaned_expr)
                dependencies = self._extract_dependencies(cleaned_expr)

                # Validate syntax
                is_valid = self._validate_syntax(cleaned_expr)
                confidence = self._calculate_confidence(cleaned_expr, is_valid)

                glyph_expr = GlyphExpression(
                    llm_type=llm_type,
                    raw_expression=cleaned_expr,
                    parsed_ast=ast_result,
                    variables=variables,
                    functions=functions,
                    dependencies=dependencies,
                    is_valid=is_valid,
                    confidence=confidence
                )

                parsed.append(glyph_expr)
                logger.debug(f"Parsed {llm_type} expression: valid={is_valid}, confidence={confidence:.2f}")

            except Exception as e:
                logger.error(f"Failed to parse {llm_type} expression: {str(e)}")
                # Create invalid expression object
                glyph_expr = GlyphExpression(
                    llm_type=llm_type,
                    raw_expression=expression,
                    parsed_ast=None,
                    variables=[],
                    functions=[],
                    dependencies=[],
                    is_valid=False,
                    confidence=0.0
                )
                parsed.append(glyph_expr)

        return parsed

    def _clean_expression(self, expression: str) -> str:
        """Clean and normalize GΛLYPH expression."""
        # Remove extra whitespace
        expression = re.sub(r'\s+', ' ', expression.strip())

        # Extract code blocks if present
        code_block_pattern = r'```(?:g(?:lyph)?|l(?:ambda)?)?\s*\n?(.*?)\n?```'
        matches = re.findall(code_block_pattern, expression, re.DOTALL | re.IGNORECASE)
        if matches:
            expression = matches[0].strip()

        # Ensure proper lambda notation
        expression = expression.replace('\\', 'λ')

        return expression

    def _parse_glyph_ast(self, expression: str) -> Optional[Any]:
        """Parse GΛLYPH expression into AST (simplified implementation)."""
        try:
            # This is a simplified AST parser
            # In production, use the actual GΛLYPH parser

            # Basic structure detection
            has_lambda = 'λ' in expression
            has_bindings = 'let' in expression and 'in' in expression

            # Create simple AST representation
            ast_structure = {
                'type': 'lambda_expression' if has_lambda else 'expression',
                'has_bindings': has_bindings,
                'complexity': self._estimate_complexity(expression)
            }

            return ast_structure

        except Exception as e:
            logger.warning(f"AST parsing failed: {str(e)}")
            return None

    def _extract_variables(self, expression: str) -> List[str]:
        """Extract variable names from GΛLYPH expression."""
        # Find let-bound variables
        let_pattern = r'let\s+(\w+)\s*='
        let_vars = re.findall(let_pattern, expression)

        # Find lambda parameters
        lambda_pattern = r'λ\s*(\w+)\s*->'
        lambda_vars = re.findall(lambda_pattern, expression)

        # Find function arguments
        func_pattern = r'(\w+)\s*\('
        func_args = re.findall(func_pattern, expression)

        # Combine and deduplicate
        all_vars = list(set(let_vars + lambda_vars + func_args))

        # Filter out common keywords
        keywords = {'let', 'in', 'if', 'then', 'else', 'match', 'with', 'end', 'true', 'false'}
        variables = [var for var in all_vars if var.lower() not in keywords]

        return variables

    def _extract_functions(self, expression: str) -> List[str]:
        """Extract function definitions from GΛLYPH expression."""
        # Find lambda functions
        lambda_pattern = r'λ\s*(\w+)\s*->\s*([^;]+)'
        lambda_funcs = re.findall(lambda_pattern, expression, re.DOTALL)

        # Find let-bound functions
        let_func_pattern = r'let\s+(\w+)\s*=\s*λ[^;]+'
        let_funcs = re.findall(let_func_pattern, expression)

        # Combine function names
        all_funcs = [name for name, _ in lambda_funcs] + let_funcs

        return list(set(all_funcs))

    def _extract_dependencies(self, expression: str) -> List[str]:
        """Extract dependencies (variables used but not defined)."""
        # Find all variable usages
        usage_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        all_usages = re.findall(usage_pattern, expression)

        # Find variable definitions
        def_pattern = r'(?:let|λ)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        definitions = re.findall(def_pattern, expression)

        # Dependencies = usages - definitions
        dependencies = [var for var in all_usages if var not in definitions]

        # Filter out common keywords and literals
        keywords = {'let', 'in', 'if', 'then', 'else', 'match', 'with', 'end', 'true', 'false',
                   'manifest', 'story', 'mechanics', 'assets', 'balance'}
        literals = {'\d+', '"[^"]*"', "'[^']*'"}  # Numbers and strings

        filtered_deps = []
        for dep in dependencies:
            if dep.lower() not in keywords:
                # Check if it's a literal
                is_literal = any(re.fullmatch(pattern, dep) for pattern in literals)
                if not is_literal:
                    filtered_deps.append(dep)

        return list(set(filtered_deps))

    def _validate_syntax(self, expression: str) -> bool:
        """Validate GΛLYPH expression syntax."""
        if not expression:
            return False

        # Check for balanced delimiters
        if not self._check_balanced_delimiters(expression):
            return False

        # Check for lambda expressions
        if 'λ' not in expression and 'let' not in expression:
            return False

        # Check for proper let-in structure
        let_count = expression.count('let')
        in_count = expression.count('in')
        if let_count > 0 and let_count != in_count:
            return False

        # Basic structure validation
        return True

    def _calculate_confidence(self, expression: str, is_valid: bool) -> float:
        """Calculate confidence score for expression."""
        if not is_valid:
            return 0.0

        confidence = 0.5  # Base score for valid syntax

        # Bonus for complete lambda expressions
        if 'λ' in expression and '->' in expression:
            confidence += 0.2

        # Bonus for proper let-in bindings
        if 'let' in expression and 'in' in expression:
            confidence += 0.2

        # Bonus for game-specific keywords
        game_keywords = ['story', 'mechanics', 'assets', 'balance', 'manifest']
        for keyword in game_keywords:
            if keyword in expression:
                confidence += 0.05

        return min(confidence, 1.0)

    def _estimate_complexity(self, expression: str) -> int:
        """Estimate complexity of GΛLYPH expression."""
        complexity = 1

        # Count nested expressions
        complexity += expression.count('λ')
        complexity += expression.count('let')
        complexity += len(re.findall(r'\[.*?\]', expression))  # Lists
        complexity += len(re.findall(r'\(.*?\)', expression))  # Groups

        return complexity

    def _determine_merge_strategy(self, parsed_expressions: List[GlyphExpression]) -> MergeStrategy:
        """Determine best merge strategy based on expressions."""
        valid_count = sum(1 for exp in parsed_expressions if exp.is_valid)
        total_count = len(parsed_expressions)

        # If most expressions are valid, use intelligent merging
        if valid_count / total_count > 0.75:
            return MergeStrategy.INTELLIGENT

        # If we have balance expression, use balanced strategy
        has_balance = any(exp.llm_type == 'balance' and exp.is_valid for exp in parsed_expressions)
        if has_balance:
            return MergeStrategy.BALANCED

        # Default to simple merging
        return MergeStrategy.SIMPLE

    async def _simple_merge(self, parsed_expressions: List[GlyphExpression]) -> str:
        """Simple direct merging of expressions."""
        logger.debug("Using simple merge strategy")

        # Order expressions: narrative -> mechanics -> assets -> balance
        order = ['narrative', 'mechanics', 'assets', 'balance']
        ordered_expressions = {}

        for exp in parsed_expressions:
            if exp.is_valid:
                ordered_expressions[exp.llm_type] = exp.raw_expression

        # If balance expression exists and is complete, use it
        if 'balance' in ordered_expressions and 'λgame' in ordered_expressions['balance']:
            return ordered_expressions['balance']

        # Otherwise construct basic lambda game
        components = []
        for llm_type in order:
            if llm_type in ordered_expressions:
                expr = ordered_expressions[llm_type]
                # Wrap in let-binding if needed
                if not expr.startswith('let'):
                    component_name = self.game_components.get(llm_type, [llm_type])[0]
                    expr = f"let {component_name} = {expr}"
                components.append(expr)

        if not components:
            # Fallback expression
            return 'λgame -> let story = "Basic game" in let mechanics = [] in let assets = [] in let balance = 0.5 in manifest story mechanics assets balance'

        # Combine into final expression
        body = '\n  '.join(components)
        return f"""λgame ->
  {body}
  let balance = 0.75 in
  manifest story mechanics assets balance"""

    async def _intelligent_merge(self, parsed_expressions: List[GlyphExpression]) -> str:
        """Intelligent merging with semantic analysis."""
        logger.debug("Using intelligent merge strategy")

        # Analyze expression dependencies and conflicts
        dependency_graph = self._build_dependency_graph(parsed_expressions)
        conflicts = self._detect_variable_conflicts(parsed_expressions)

        # Resolve conflicts by renaming variables
        resolved_expressions = self._resolve_conflicts(parsed_expressions, conflicts)

        # Order expressions based on dependencies
        ordered_exprs = self._topological_sort(resolved_expressions, dependency_graph)

        # Merge with proper scoping
        merged_expr = self._merge_with_scoping(ordered_exprs)

        return merged_expr

    async def _balanced_merge(self, parsed_expressions: List[GlyphExpression]) -> str:
        """Balanced merging focused on game coherence."""
        logger.debug("Using balanced merge strategy")

        # Find the balance expression if available
        balance_expr = None
        for exp in parsed_expressions:
            if exp.llm_type == 'balance' and exp.is_valid:
                balance_expr = exp
                break

        if balance_expr and 'λgame' in balance_expr.raw_expression:
            # Use balance expression as base and enhance it
            return self._enhance_balance_expression(balance_expr, parsed_expressions)
        else:
            # Create balanced expression from components
            return await self._create_balanced_expression(parsed_expressions)

    def _check_balanced_delimiters(self, expression: str) -> bool:
        """Check if all delimiters are balanced."""
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

    def _build_dependency_graph(self, parsed_expressions: List[GlyphExpression]) -> Dict[str, List[str]]:
        """Build dependency graph between expressions."""
        graph = {}

        for exp in parsed_expressions:
            if not exp.is_valid:
                continue

            # Find what this expression provides
            provides = set(exp.variables + exp.functions)

            # Find what this expression needs
            needs = set(exp.dependencies)

            # Build dependencies
            for other_exp in parsed_expressions:
                if other_exp.llm_type == exp.llm_type or not other_exp.is_valid:
                    continue

                other_provides = set(other_exp.variables + other_exp.functions)

                # If other expression provides what this one needs, add dependency
                if needs & other_provides:
                    if exp.llm_type not in graph:
                        graph[exp.llm_type] = []
                    if other_exp.llm_type not in graph[exp.llm_type]:
                        graph[exp.llm_type].append(other_exp.llm_type)

        return graph

    def _detect_variable_conflicts(self, parsed_expressions: List[GlyphExpression]) -> Dict[str, List[str]]:
        """Detect variable name conflicts between expressions."""
        variable_map = {}
        conflicts = {}

        for exp in parsed_expressions:
            if not exp.is_valid:
                continue

            for var in exp.variables:
                if var in variable_map:
                    if var not in conflicts:
                        conflicts[var] = []
                    conflicts[var].extend([variable_map[var], exp.llm_type])
                else:
                    variable_map[var] = exp.llm_type

        return conflicts

    def _resolve_conflicts(self, parsed_expressions: List[GlyphExpression], conflicts: Dict[str, List[str]]) -> List[GlyphExpression]:
        """Resolve variable name conflicts by renaming."""
        resolved = []

        for exp in parsed_expressions:
            if not exp.is_valid:
                resolved.append(exp)
                continue

            new_expr = exp.raw_expression
            llm_type = exp.llm_type

            # Rename conflicting variables
            for var, conflict_types in conflicts.items():
                if llm_type in conflict_types:
                    # Rename variable for this LLM type
                    new_name = f"{var}_{llm_type}"
                    new_expr = re.sub(r'\b' + re.escape(var) + r'\b', new_name, new_expr)

            # Create resolved expression
            resolved_exp = GlyphExpression(
                llm_type=exp.llm_type,
                raw_expression=new_expr,
                parsed_ast=exp.parsed_ast,
                variables=self._extract_variables(new_expr),
                functions=exp.functions,
                dependencies=exp.dependencies,
                is_valid=self._validate_syntax(new_expr),
                confidence=exp.confidence
            )
            resolved.append(resolved_exp)

        return resolved

    def _topological_sort(self, expressions: List[GlyphExpression], dependency_graph: Dict[str, List[str]]) -> List[GlyphExpression]:
        """Sort expressions based on dependencies."""
        # Simple topological sort
        expr_map = {exp.llm_type: exp for exp in expressions if exp.is_valid}
        visited = set()
        result = []

        def visit(llm_type: str):
            if llm_type in visited or llm_type not in expr_map:
                return

            visited.add(llm_type)

            # Visit dependencies first
            for dep in dependency_graph.get(llm_type, []):
                visit(dep)

            result.append(expr_map[llm_type])

        # Visit all expressions
        for exp in expressions:
            if exp.is_valid:
                visit(exp.llm_type)

        return result

    def _merge_with_scoping(self, ordered_expressions: List[GlyphExpression]) -> str:
        """Merge expressions with proper variable scoping."""
        if not ordered_expressions:
            return 'λgame -> manifest "Empty game" [] [] 0.0'

        # Start with base expression
        base_expr = ordered_expressions[0].raw_expression

        # Merge other expressions with proper scoping
        for exp in ordered_expressions[1:]:
            base_expr = f"{base_expr}\n  {exp.raw_expression}"

        # Wrap in lambda game if not already
        if not base_expr.startswith('λgame'):
            base_expr = f"""λgame ->
  {base_expr}
  let balance = 0.75 in
  manifest story mechanics assets balance"""

        return base_expr

    def _enhance_balance_expression(self, balance_expr: GlyphExpression, all_expressions: List[GlyphExpression]) -> str:
        """Enhance balance expression with components from other LLMs."""
        enhanced = balance_expr.raw_expression

        # Look for missing components and add them
        for exp in all_expressions:
            if exp.llm_type == 'balance' or not exp.is_valid:
                continue

            # Check if this component is missing from balance expression
            component_name = self.game_components.get(exp.llm_type, [exp.llm_type])[0]
            if component_name not in enhanced:
                # Add the component
                enhanced = enhanced.replace(
                    'let balance =',
                    f"let {component_name} = {exp.raw_expression}\n  let balance ="
                )

        return enhanced

    async def _create_balanced_expression(self, parsed_expressions: List[GlyphExpression]) -> str:
        """Create balanced expression from components."""
        # Collect valid expressions by type
        components = {}
        for exp in parsed_expressions:
            if exp.is_valid:
                components[exp.llm_type] = exp

        # Create balanced expression with proper weight
        narrative = components.get('narrative')
        mechanics = components.get('mechanics')
        assets = components.get('assets')

        if not any([narrative, mechanics, assets]):
            return 'λgame -> manifest "Fallback game" [] [] 0.5'

        # Build balanced expression
        expr_parts = []
        if narrative:
            expr_parts.append(f"let story = {narrative.raw_expression}")
        if mechanics:
            expr_parts.append(f"let rules = {mechanics.raw_expression}")
        if assets:
            expr_parts.append(f"let visuals = {assets.raw_expression}")

        # Add balance calculation
        balance_expr = self._calculate_balance_score(components)
        expr_parts.append(f"let balance = {balance_expr}")

        body = '\n  '.join(expr_parts)

        return f"""λgame ->
  {body}
  manifest story rules visuals balance"""

    def _calculate_balance_score(self, components: Dict[str, GlyphExpression]) -> str:
        """Calculate game balance score based on components."""
        # Simple balance calculation based on component complexity
        total_complexity = sum(
            self._estimate_complexity(exp.raw_expression)
            for exp in components.values()
        )

        # Normalize to 0.0-1.0 range
        balance = min(total_complexity / 20.0, 1.0)

        return f"{balance:.2f}"

    def _validate_final_expression(self, expression: str) -> List[str]:
        """Validate the final merged expression."""
        errors = []

        if not expression:
            errors.append("Empty expression")
            return errors

        # Syntax validation
        if not self._validate_syntax(expression):
            errors.append("Invalid GΛLYPH syntax")

        # Structure validation
        if 'λgame' not in expression:
            errors.append("Missing λgame wrapper")

        if 'manifest' not in expression:
            errors.append("Missing manifest call")

        # Component validation
        required_components = ['story', 'mechanics', 'assets', 'balance']
        for component in required_components:
            if component not in expression:
                errors.append(f"Missing {component} component")

        return errors

    def _calculate_merge_score(self, parsed_expressions: List[GlyphExpression], final_expression: str) -> float:
        """Calculate quality score for the merge."""
        if not final_expression:
            return 0.0

        score = 0.0

        # Base score for having any valid expressions
        valid_count = sum(1 for exp in parsed_expressions if exp.is_valid)
        if valid_count > 0:
            score += 0.3

        # Bonus for component completeness
        components = ['story', 'mechanics', 'assets', 'balance']
        found_components = sum(1 for comp in components if comp in final_expression)
        score += (found_components / len(components)) * 0.4

        # Bonus for syntax validity
        if self._validate_syntax(final_expression):
            score += 0.2

        # Bonus for proper structure
        if 'λgame' in final_expression and 'manifest' in final_expression:
            score += 0.1

        return min(score, 1.0)


# Singleton instance
glyph_merger = GlyphMerger()


async def merge_glyph_expressions(
    llm_expressions: Dict[str, str],
    strategy: Optional[MergeStrategy] = None
) -> MergeResult:
    """
    Merge 4 LLM GΛLYPH expressions into single λgame expression.

    Args:
        llm_expressions: Dictionary mapping LLM types to their GΛLYPH expressions
        strategy: Optional merge strategy override

    Returns:
        MergeResult with final expression and metadata
    """
    return await glyph_merger.merge_expressions(llm_expressions, strategy)