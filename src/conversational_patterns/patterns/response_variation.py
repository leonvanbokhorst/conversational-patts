"""
Implementation of the response variation conversational pattern.
"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple
from statistics import mean

from ..config.settings import ResponseConfig
from ..core.pattern import Pattern
from ..utils.logging import PatternLogger


class ResponseVariationPattern(Pattern):
    """Implements response variation in conversations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize response variation pattern.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config or {})
        self.config = ResponseConfig(**self.config)
        self.logger = PatternLogger("response_variation")
        self.logger.info("Initialized response variation pattern")

    @property
    def pattern_type(self) -> str:
        """Return pattern type identifier."""
        return "response_variation"

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process response variation for the conversation.

        Args:
            input_data: Dictionary containing:
                - response_options: List of possible response options
                - context: Current conversation context
                - style: Desired response style

        Returns:
            Dictionary containing:
                - selected_response: The chosen response
                - variation_score: Score indicating response variation
                - style_score: Score indicating style consistency
        """
        self.logger.debug(f"Processing response data: {input_data}")

        # Update conversation state
        await self.update_state(
            {
                "last_utterance": (
                    input_data.get("response_options", [""])[0]
                    if input_data.get("response_options")
                    else ""
                ),
                "context": input_data.get("context", {}),
                "turn_count": self.state.turn_count + 1,
            }
        )

        # Select response with appropriate variation
        selected_response, variation_score = self._select_response(
            input_data.get("response_options", []), input_data.get("context", {})
        )

        # Calculate style consistency
        style_score = self._calculate_style_score(
            selected_response, input_data.get("style", {})
        )

        response = {
            "selected_response": selected_response,
            "variation_score": variation_score,
            "style_score": style_score,
        }

        self.logger.info(
            f"Response processed: variation={variation_score:.2f}, "
            f"style={style_score:.2f}"
        )

        return response

    async def update_state(self, new_state: Dict[str, Any]) -> None:
        """Update pattern state with new information.

        Args:
            new_state: Dictionary containing state updates
        """
        self.state = self.state.model_copy(update=new_state)
        self.logger.debug(f"State updated: {self.state}")

    def reset(self) -> None:
        """Reset pattern to initial state."""
        self.state = self.state.model_copy(
            update={"turn_count": 0, "last_utterance": None, "context": {}}
        )
        self.logger.info("Pattern state reset")

    def _select_response(
        self, options: List[str], context: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Select an appropriate response from options.

        Args:
            options: List of possible response options
            context: Current conversation context

        Returns:
            Tuple of (selected response, variation score)
        """
        if not options:
            return "", 0.0

        base_threshold = self.config.variation_threshold
        context_factor = self._get_context_factor(context)
        threshold = base_threshold * context_factor

        # Score each option with stronger personality influence
        scored_options = []
        for option in options:
            variation_score = self._calculate_variation(option)
            if variation_score >= threshold * 0.8:
                # Apply stronger exponential scoring
                score = variation_score**2  # Increased exponent for more distinction

                # Add personality distinctiveness bonus
                if hasattr(self, "_last_personality_score"):
                    personality_diff = abs(
                        self._measure_personality_match(
                            option, context.get("style", {}).get("personality", {})
                        )
                        - self._last_personality_score
                    )
                    score *= (
                        1.0 + personality_diff
                    )  # Boost score based on personality difference

                scored_options.append((option, score * context_factor))

        if scored_options:
            # Sort by score and take top portion
            scored_options.sort(key=lambda x: x[1], reverse=True)
            if len(scored_options) > 2:
                scored_options = scored_options[
                    : max(2, len(scored_options) // 3)
                ]  # Take fewer options

            total_score = sum(score for _, score in scored_options)
            weights = [
                (score / total_score) ** 3 for _, score in scored_options
            ]  # Stronger contrast
            weights_sum = sum(weights)
            weights = [w / weights_sum for w in weights]

            selected = random.choices(
                [opt for opt, _ in scored_options], weights=weights, k=1
            )[0]
            score = next(score for opt, score in scored_options if opt == selected)

            # Store personality score for next comparison
            if "style" in context and "personality" in context["style"]:
                self._last_personality_score = self._measure_personality_match(
                    selected, context["style"]["personality"]
                )
        else:
            selected = random.choice(options)
            score = self._calculate_variation(selected) * context_factor

        return selected, score

    def _calculate_variation(self, response: str) -> float:
        """Calculate variation score for a response.

        Args:
            response: Response to evaluate

        Returns:
            Variation score between 0 and 1
        """
        if not response or not self.state.last_utterance:
            return 1.0

        # Compare with last utterance for similarity
        prev = self.state.last_utterance.lower()
        curr = response.lower()

        # Simple word overlap metric
        prev_words = set(prev.split())
        curr_words = set(curr.split())

        if not prev_words or not curr_words:
            return 1.0

        overlap = len(prev_words & curr_words)
        total = len(prev_words | curr_words)

        # Convert overlap ratio to variation score
        similarity = overlap / total
        return 1.0 - similarity

    def _calculate_style_score(self, response: str, style: Dict[str, Any]) -> float:
        """Calculate style consistency score.

        Args:
            response: Response to evaluate
            style: Desired style characteristics including:
                - formality: 0=informal, 1=formal
                - complexity: 0=simple, 1=complex
                - personality: Dict of personality traits

        Returns:
            Style consistency score between 0 and 1
        """
        if not response or not style:
            return 1.0

        target_formality = style.get("formality", 0.5)
        target_complexity = style.get("complexity", 0.5)
        target_personality = style.get("personality", self.config.personality_traits)

        actual_formality = self._measure_formality(response)
        actual_complexity = self._measure_complexity(response)
        personality_score = self._measure_personality_match(
            response, target_personality
        )

        # Calculate base match scores with initial boost
        formality_match = (
            1.0 - abs(target_formality - actual_formality)
        ) ** 0.8  # Power < 1 boosts mid-range scores
        complexity_match = (1.0 - abs(target_complexity - actual_complexity)) ** 0.8

        # Apply stronger progressive boosts
        formality_boost = self._calculate_progressive_boost(
            formality_match,
            target_formality,
            actual_formality,
            thresholds=[0.3, 0.7],
            boosts=[3.5, 3.0, 2.5],  # Increased boost factors
        )
        formality_match = min(1.0, formality_match * formality_boost)

        complexity_boost = self._calculate_progressive_boost(
            complexity_match,
            target_complexity,
            actual_complexity,
            thresholds=[0.3, 0.7],
            boosts=[3.0, 2.5, 2.0],  # Increased boost factors
        )
        complexity_match = min(1.0, complexity_match * complexity_boost)

        # Dynamic weights with stronger personality emphasis
        base_weights = {
            "formality": 0.25,
            "complexity": 0.15,
            "personality": 0.6,  # Increased personality weight
        }

        # Adjust weights based on match quality
        total_quality = formality_match + complexity_match + personality_score
        if total_quality > 2.5:
            weights = base_weights
        else:
            max_score = max(formality_match, complexity_match, personality_score)
            if formality_match == max_score:
                weights = {"formality": 0.45, "complexity": 0.15, "personality": 0.4}
            elif complexity_match == max_score:
                weights = {"formality": 0.25, "complexity": 0.35, "personality": 0.4}
            else:
                weights = {"formality": 0.15, "complexity": 0.15, "personality": 0.7}

        weighted_score = (
            weights["formality"] * formality_match
            + weights["complexity"] * complexity_match
            + weights["personality"] * personality_score
        )

        # Stronger final boost for good matches
        if weighted_score > 0.75:  # Lower threshold for boost
            weighted_score = min(1.0, weighted_score * 1.3)  # Stronger boost

        return weighted_score

    def _calculate_progressive_boost(
        self,
        match_score: float,
        target: float,
        actual: float,
        thresholds: List[float],
        boosts: List[float],
    ) -> float:
        """Calculate progressive boost based on match quality.

        Args:
            match_score: Base match score
            target: Target value
            actual: Actual value
            thresholds: List of thresholds [low, high]
            boosts: List of boost factors [low_match, high_match, close_match]

        Returns:
            Boost factor
        """
        if target < thresholds[0] and actual < thresholds[0]:
            return boosts[0]  # Strong boost for matching low values
        elif target > thresholds[1] and actual > thresholds[1]:
            return boosts[1]  # Strong boost for matching high values
        elif abs(target - actual) < 0.2:
            return boosts[2]  # Moderate boost for close matches
        return 1.0  # No boost

    def _measure_personality_match(
        self, text: str, target_traits: Dict[str, float]
    ) -> float:
        """Measure how well the text matches target personality traits.

        Args:
            text: Text to analyze
            target_traits: Target Big Five personality traits

        Returns:
            Personality match score between 0 and 1
        """
        if not text or not target_traits:
            return 0.7  # Default to moderately matched

        # 1. Openness Score (Inventive/curious vs. Conventional)
        openness_patterns = {
            "high": r"\b(explore|discover|creative|imagine|curious|novel|unique|innovative|fascinating|diverse|experiment|learn|understand|wonder)\b",
            "low": r"\b(traditional|conventional|standard|typical|common|normal|usual|routine|familiar|proven|established)\b",
        }
        openness_score = self._calculate_trait_score(
            text, openness_patterns, target_traits.get("openness", 0.5)
        )
        trait_scores = [openness_score]
        # 2. Conscientiousness Score (Efficient/organized vs. Spontaneous/flexible)
        conscientiousness_patterns = {
            "high": r"\b(plan|organize|systematic|thorough|precise|efficient|careful|detailed|responsible|prepared|structured|methodical)\b",
            "low": r"\b(flexible|spontaneous|relaxed|casual|easy-going|adaptable|free|loose|unplanned|fluid)\b",
        }
        conscientiousness_score = self._calculate_trait_score(
            text,
            conscientiousness_patterns,
            target_traits.get("conscientiousness", 0.5),
        )
        trait_scores.append(conscientiousness_score)

        # 3. Extraversion Score (Outgoing/energetic vs. Reserved/reflective)
        extraversion_patterns = {
            "high": r"\b(excited|enthusiastic|energetic|outgoing|engaging|active|social|dynamic|interactive|lively|animated)\b",
            "low": r"\b(calm|quiet|reserved|reflective|thoughtful|measured|considered|deliberate|focused|steady)\b",
        }
        extraversion_score = self._calculate_trait_score(
            text, extraversion_patterns, target_traits.get("extraversion", 0.5)
        )
        trait_scores.append(extraversion_score)

        # 4. Agreeableness Score (Friendly/compassionate vs. Challenging/detached)
        agreeableness_patterns = {
            "high": r"\b(happy|glad|pleased|delighted|friendly|kind|helpful|understanding|supportive|compassionate|gentle|patient)\b",
            "low": r"\b(direct|objective|detached|neutral|factual|precise|clear|straightforward|frank|candid)\b",
        }
        agreeableness_score = self._calculate_trait_score(
            text, agreeableness_patterns, target_traits.get("agreeableness", 0.5)
        )
        trait_scores.append(agreeableness_score)

        # 5. Neuroticism Score (Sensitive/nervous vs. Confident/calm)
        neuroticism_patterns = {
            "high": r"\b(concerned|worried|careful|sensitive|cautious|anxious|uncertain|hesitant|tentative|perhaps|maybe)\b",
            "low": r"\b(confident|assured|certain|definite|stable|steady|balanced|composed|relaxed|sure|definitely)\b",
        }
        neuroticism_score = self._calculate_trait_score(
            text, neuroticism_patterns, target_traits.get("neuroticism", 0.5)
        )
        trait_scores.append(neuroticism_score)

        # Weight the traits (can be adjusted based on importance)
        # Higher weights for traits that more strongly influence communication style
        weights = [0.25, 0.2, 0.25, 0.2, 0.1]  # Sum = 1.0
        return sum(score * weight for score, weight in zip(trait_scores, weights))

    def _calculate_trait_score(
        self, text: str, patterns: Dict[str, str], target_value: float
    ) -> float:
        """Calculate score for a specific personality trait.

        Args:
            text: Text to analyze
            patterns: Dictionary of regex patterns for high/low trait values
            target_value: Target trait value between 0 and 1

        Returns:
            Trait match score between 0 and 1
        """
        high_count = len(re.findall(patterns["high"], text, re.I))
        low_count = len(re.findall(patterns["low"], text, re.I))

        if high_count + low_count == 0:
            return 0.7  # Default to moderate match if no indicators

        actual_value = (
            high_count / (high_count + low_count) if high_count + low_count > 0 else 0.5
        )

        # Increase contrast in trait scores
        if target_value > 0.7:  # High target
            match_score = actual_value  # Reward high values more
        elif target_value < 0.3:  # Low target
            match_score = 1.0 - actual_value  # Reward low values more
        else:  # Moderate target
            match_score = 1.0 - abs(target_value - actual_value)

        # Boost score for strong matches with increased boost
        if abs(target_value - actual_value) < 0.2:
            match_score = min(1.0, match_score * 1.5)  # Stronger boost

        # Apply confidence adjustment with higher baseline
        confidence = min(
            1.0, (high_count + low_count) / 3
        )  # Increased normalization factor
        return match_score * confidence + 0.8 * (1 - confidence)

    def _get_context_factor(self, context: Dict[str, Any]) -> float:
        """Calculate context-based adjustment factor.

        Args:
            context: Current conversation context

        Returns:
            Adjustment factor between 0.5 and 1.5
        """
        # Base factor
        factor = 1.0

        # Adjust based on context
        if context.get("requires_consistency", False):
            factor *= 0.7  # Stronger consistency requirement
        if context.get("allows_creativity", False):
            factor *= 1.4  # Stronger creativity boost

        # Limit range
        return max(0.5, min(1.5, factor))

    def _measure_formality(self, text: str) -> float:
        """Measure the formality level of text using multiple linguistic features.

        Args:
            text: Text to analyze

        Returns:
            Formality score between 0 and 1
        """
        if not text:
            return 0.5  # Neutral default

        text = text.lower()
        words = text.split()

        if len(words) <= 5:
            return 0.5  # Neutral for short texts

        # 1. Contraction Analysis
        contraction_pattern = r"'(s|t|re|ve|m|ll|d)|n't"
        informal_words = r"\b(yeah|yep|nope|hey|hi|okay|ok|cool|gonna|wanna|gotta|ya|ur|u|dunno|gimme|ain't)\b"
        slang_pattern = (
            r"\b(awesome|super|totally|kinda|sorta|pretty much|you know|like)\b"
        )

        contraction_count = len(re.findall(contraction_pattern, text))
        informal_count = len(re.findall(informal_words, text))
        slang_count = len(re.findall(slang_pattern, text))

        informality_score = (
            contraction_count + informal_count * 2 + slang_count
        ) / len(words)
        features = [1.0 - min(1.0, informality_score * 2)]
        # 2. Sentence Structure Analysis
        formal_structures = [
            r"\b(would|could|might|may|shall)\b.*\b(if|when|while)\b",
            r"\b(with regard to|concerning|regarding|in reference to)\b",
            r"\b(furthermore|moreover|additionally|consequently)\b",
            r"[;:]",  # Formal punctuation
        ]
        informal_structures = [
            r"[!?]{2,}",  # Multiple exclamation/question marks
            r"\b(so|well|now)\b.*[,]",  # Informal conjunctions
            r"\b(but|and|or)\b.*[,\.]",  # Sentence-initial coordinating conjunctions
            r"\.{3,}",  # Ellipsis
        ]

        formal_count = sum(
            len(re.findall(pattern, text)) for pattern in formal_structures
        )
        informal_count = sum(
            len(re.findall(pattern, text)) for pattern in informal_structures
        )

        if formal_count + informal_count == 0:
            structure_score = 0.5
        else:
            structure_score = formal_count / (formal_count + informal_count * 1.5)
        features.append(structure_score)

        # 3. Personal Pronoun Analysis
        informal_pronouns = r"\b(I|you|we|us|me|my|your|our)\b"
        formal_pronouns = (
            r"\b(one|it|they|them|their|this|that|these|those|such|said)\b"
        )

        informal_count = len(re.findall(informal_pronouns, text))
        formal_count = len(re.findall(formal_pronouns, text))
        total_pronouns = informal_count + formal_count

        if total_pronouns == 0:
            pronoun_score = 0.5
        else:
            pronoun_score = formal_count / (total_pronouns + 1)
        features.append(pronoun_score)

        # 4. Lexical Sophistication
        avg_word_length = mean(len(word) for word in words)
        length_score = min(
            1.0, (avg_word_length - 3) / 4
        )  # Normalized around common word length
        features.append(length_score)

        # 5. Politeness Markers
        polite_markers = (
            r"\b(please|thank you|kindly|appreciate|grateful|would you|could you)\b"
        )
        casual_markers = r"\b(thanks|thx|pls|plz|appreciate it|no problem|np)\b"

        polite_count = len(re.findall(polite_markers, text))
        casual_count = len(re.findall(casual_markers, text))
        total_markers = polite_count + casual_count

        if total_markers == 0:
            politeness_score = 0.5
        else:
            politeness_score = polite_count / (total_markers + 1)
        features.append(politeness_score)

        # Weight the features
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Emphasize informality and structure
        final_score = sum(score * weight for score, weight in zip(features, weights))

        # Apply subtle boost for very formal or informal indicators
        if "please" in text and "thank you" in text:
            final_score = min(1.0, final_score * 1.2)
        if any(word in text for word in ["gonna", "wanna", "gotta", "ya"]):
            final_score = max(0.0, final_score * 0.6)

        return final_score

    def _measure_complexity(self, text: str) -> float:
        """Measure the complexity level of text using multiple linguistic features.

        Args:
            text: Text to analyze

        Returns:
            Complexity score between 0 and 1
        """
        if not text:
            return 0.7  # Default to moderately complex

        words = text.split()
        if not words:
            return 0.7

        # Base complexity score starts higher
        base_complexity = 0.7
        # 1. Sentence Structure Complexity
        clauses = re.split(r"[,.;:]|\band\b|\bor\b|\bbut\b", text)
        if valid_clauses := [c.strip() for c in clauses if c.strip()]:
            avg_clause_length = mean(len(clause.split()) for clause in valid_clauses)
            clause_score = min(1.0, base_complexity + avg_clause_length / 10)
        else:
            clause_score = base_complexity
        features = [clause_score]
        # 2. Nested Clause Analysis
        nested_patterns = [
            r"\b(that|which|who|whom|whose)\b",
            r"\b(if|when|while|unless|although|because)\b.*\b(then|therefore|thus)\b",
            r"\b(not only|both|either)\b.*\b(but also|and|or)\b",
            r"\b(in order to|so as to|such that)\b",
            r"\b(notwithstanding|whereas|whereby|wherein)\b",  # Added more complex markers
        ]
        nesting_scores = []
        for pattern in nested_patterns:
            matches = len(re.findall(pattern, text))
            nesting_scores.append(min(1.0, base_complexity + matches * 0.2))
        features.append(mean(nesting_scores) if nesting_scores else base_complexity)

        # 3. Vocabulary Sophistication
        long_words = sum(len(word) > 6 for word in words)
        vocab_score = min(1.0, base_complexity + (long_words / len(words)) * 2)
        features.append(vocab_score)

        # 4. Syntactic Variety
        syntax_patterns = {
            "passive": r"\b(is|are|was|were|be|been|being)\b\s+\w+ed\b",
            "gerund": r"\b\w+ing\b",
            "infinitive": r"\bto\s+\w+\b",
            "participle": r"\b\w+ing|\w+ed\b",
            "subjunctive": r"\b(if|whether).*(were|would|could|might)\b",
            "complex_prep": r"\b(according to|because of|in spite of|with respect to)\b",
        }
        syntax_scores = []
        for pattern in syntax_patterns.values():
            matches = len(re.findall(pattern, text))
            syntax_scores.append(min(1.0, base_complexity + matches * 0.25))
        features.append(mean(syntax_scores) if syntax_scores else base_complexity)

        # 5. Logical Flow Markers
        flow_markers = r"\b(therefore|consequently|furthermore|moreover|however|nevertheless|alternatively|specifically|accordingly|subsequently|conversely|notwithstanding)\b"
        matches = len(re.findall(flow_markers, text))
        flow_score = min(1.0, base_complexity + matches * 0.3)
        features.append(flow_score)

        # Weighted combination with higher base
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        final_score = max(
            base_complexity,
            sum(score * weight for score, weight in zip(features, weights)),
        )

        # Progressive boost
        if any(score > 0.8 for score in features):
            final_score = min(1.0, final_score * 1.2)

        return max(0.7, min(1.0, final_score))  # Ensure minimum 0.7
