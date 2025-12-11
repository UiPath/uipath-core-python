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
    BooleanRule,
    DeterministicGuardrail,
    GuardrailValidationResult,
    NumberRule,
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
        return self._evaluate_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

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
