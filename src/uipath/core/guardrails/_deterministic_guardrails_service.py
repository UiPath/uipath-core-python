from typing import Any

from pydantic import BaseModel

from ..tracing.decorators import traced
from ._evaluators import (
    evaluate_boolean_rule,
    evaluate_number_rule,
    evaluate_universal_rule,
    evaluate_word_rule,
)
from .guardrails import (
    AllFieldsSelector,
    ApplyTo,
    BooleanRule,
    DeterministicGuardrail,
    FieldSource,
    GuardrailValidationResult,
    NumberRule,
    SpecificFieldsSelector,
    UniversalRule,
    WordRule,
)


class DeterministicGuardrailsService(BaseModel):
    @traced("evaluate_pre_deterministic_guardrail", run_type="uipath")
    def evaluate_pre_deterministic_guardrail(
        self,
        input_data: dict[str, Any],
        guardrail: DeterministicGuardrail,
    ) -> GuardrailValidationResult:
        """Evaluate deterministic guardrail rules against input data (pre-execution)."""
        # Check if at least one rule requires output data
        has_output_rule = self._has_output_dependent_rule(guardrail)

        # If guardrail has no output rules and no universal rules, skip evaluation and pass
        if has_output_rule:
            return GuardrailValidationResult(
                validation_passed=True,
                reason="All deterministic guardrail rules passed",
            )
        return self._evaluate_deterministic_guardrail(
            input_data=input_data,
            output_data={},
            guardrail=guardrail,
        )

    @traced("evaluate_post_deterministic_guardrails", run_type="uipath")
    def evaluate_post_deterministic_guardrail(
        self,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        guardrail: DeterministicGuardrail,
    ) -> GuardrailValidationResult:
        """Evaluate deterministic guardrail rules against input and output data."""
        # Check if at least one rule requires output data
        has_output_rule = self._has_output_dependent_rule(guardrail)

        # If guardrail has no output rules and no universal rules, skip evaluation and pass
        if not has_output_rule:
            return GuardrailValidationResult(
                validation_passed=True,
                reason="All deterministic guardrail rules passed",
            )

        return self._evaluate_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

    @staticmethod
    def _has_output_dependent_rule(guardrail: DeterministicGuardrail) -> bool:
        """Check if at least one rule EXCLUSIVELY requires output data.

        Returns:
            True if at least one rule exclusively depends on output data, False otherwise.
        """
        for rule in guardrail.rules:
            # UniversalRule: only return True if it applies to OUTPUT or INPUT_AND_OUTPUT
            if isinstance(rule, UniversalRule):
                if rule.apply_to in (ApplyTo.OUTPUT, ApplyTo.INPUT_AND_OUTPUT):
                    return True
            # Rules with field_selector
            elif isinstance(rule, (WordRule, NumberRule, BooleanRule)):
                field_selector = rule.field_selector
                # AllFieldsSelector applies to both input and output, not exclusively output
                # SpecificFieldsSelector: only return True if at least one field has OUTPUT source
                if isinstance(field_selector, SpecificFieldsSelector):
                    if field_selector.fields and any(
                        field.source == FieldSource.OUTPUT
                        for field in field_selector.fields
                    ):
                        return True
                elif isinstance(field_selector, AllFieldsSelector):
                    if field_selector.source == FieldSource.OUTPUT:
                        return True

        return False

    @staticmethod
    def _evaluate_deterministic_guardrail(
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        guardrail: DeterministicGuardrail,
    ) -> GuardrailValidationResult:
        """Evaluate deterministic guardrail rules against input and output data."""
        for rule in guardrail.rules:
            if isinstance(rule, WordRule):
                passed, reason = evaluate_word_rule(rule, input_data, output_data)
            elif isinstance(rule, NumberRule):
                passed, reason = evaluate_number_rule(rule, input_data, output_data)
            elif isinstance(rule, BooleanRule):
                passed, reason = evaluate_boolean_rule(rule, input_data, output_data)
            elif isinstance(rule, UniversalRule):
                passed, reason = evaluate_universal_rule(rule, output_data)
            else:
                return GuardrailValidationResult(
                    validation_passed=False,
                    reason=f"Unknown rule type: {type(rule)}",
                )

            if not passed:
                return GuardrailValidationResult(
                    validation_passed=False, reason=reason or "Rule validation failed"
                )

        return GuardrailValidationResult(
            validation_passed=True, reason="All deterministic guardrail rules passed"
        )
