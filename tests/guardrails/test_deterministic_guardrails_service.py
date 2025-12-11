import re
from typing import Any

import pytest

from uipath.core.guardrails import (
    AllFieldsSelector,
    ApplyTo,
    BooleanRule,
    DeterministicGuardrail,
    DeterministicGuardrailsService,
    FieldReference,
    FieldSource,
    GuardrailScope,
    GuardrailSelector,
    NumberRule,
    SpecificFieldsSelector,
    UniversalRule,
    WordRule,
)


@pytest.fixture
def service() -> DeterministicGuardrailsService:
    return DeterministicGuardrailsService()


class TestDeterministicGuardrailsService:
    """Test GuardrailsService functionality."""

    def test_evaluate_post_deterministic_guardrail_validation_passed(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test deterministic guardrail validation with passing rules."""
        # Create a deterministic guardrail matching the C# example
        deterministic_guardrail = DeterministicGuardrail(
            id="test-deterministic-id",
            name="Pre execution Guardrail",
            description="Test pre-execution guardrail",
            enabled_for_evals=True,
            guardrail_type="custom",
            selector=GuardrailSelector(
                scopes=[GuardrailScope.TOOL], match_names=["test"]
            ),
            rules=[
                NumberRule(
                    rule_type="number",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[FieldReference(path="age", source=FieldSource.INPUT)],
                    ),
                    func=lambda n: n >= 21.0,
                ),
                BooleanRule(
                    rule_type="boolean",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[
                            FieldReference(path="isActive", source=FieldSource.INPUT)
                        ],
                    ),
                    func=lambda b: b is True,
                ),
            ],
        )

        # Input data matching the C# example
        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": True,
        }
        output_data: dict[str, Any] = {}

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=deterministic_guardrail,
        )

        assert result.validation_passed is True
        assert "All deterministic guardrail rules passed" in result.reason

    def test_evaluate_post_deterministic_guardrail_validation_failed_age(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test deterministic guardrail validation fails when age is too low."""
        deterministic_guardrail = DeterministicGuardrail(
            id="test-deterministic-id",
            name="Pre execution Guardrail",
            description="Test pre-execution guardrail",
            enabled_for_evals=True,
            guardrail_type="custom",
            selector=GuardrailSelector(
                scopes=[GuardrailScope.TOOL], match_names=["test"]
            ),
            rules=[
                NumberRule(
                    rule_type="number",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[FieldReference(path="age", source=FieldSource.INPUT)],
                    ),
                    func=lambda n: n >= 21.0,
                ),
                BooleanRule(
                    rule_type="boolean",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[
                            FieldReference(path="isActive", source=FieldSource.INPUT)
                        ],
                    ),
                    func=lambda b: b is True,
                ),
            ],
        )

        # Input data with age < 21
        input_data = {
            "userName": "John",
            "age": 18,
            "isActive": True,
        }
        output_data: dict[str, Any] = {}

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=deterministic_guardrail,
        )

        assert result.validation_passed is False
        assert "age" in result.reason.lower()

    def test_evaluate_post_deterministic_guardrail_validation_failed_is_active(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test deterministic guardrail validation fails when isActive is False."""
        deterministic_guardrail = DeterministicGuardrail(
            id="test-deterministic-id",
            name="Pre execution Guardrail",
            description="Test pre-execution guardrail",
            enabled_for_evals=True,
            guardrail_type="custom",
            selector=GuardrailSelector(
                scopes=[GuardrailScope.TOOL], match_names=["test"]
            ),
            rules=[
                NumberRule(
                    rule_type="number",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[FieldReference(path="age", source=FieldSource.INPUT)],
                    ),
                    func=lambda n: n >= 21.0,
                ),
                BooleanRule(
                    rule_type="boolean",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[
                            FieldReference(path="isActive", source=FieldSource.INPUT)
                        ],
                    ),
                    func=lambda b: b is True,
                ),
            ],
        )

        # Input data with isActive = False
        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": False,
        }
        output_data: dict[str, Any] = {}

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=deterministic_guardrail,
        )

        assert result.validation_passed is False
        assert "isActive" in result.reason or "isactive" in result.reason.lower()
        assert "comparing function" in result.reason.lower()

    def test_evaluate_post_deterministic_guardrail_matches_regex_positive(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test deterministic guardrail validation passes when regex matches."""
        deterministic_guardrail = DeterministicGuardrail(
            id="test-deterministic-id",
            name="Regex Guardrail",
            description="Test regex guardrail",
            enabled_for_evals=True,
            guardrail_type="custom",
            selector=GuardrailSelector(
                scopes=[GuardrailScope.TOOL], match_names=["test"]
            ),
            rules=[
                WordRule(
                    rule_type="word",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[
                            FieldReference(path="userName", source=FieldSource.INPUT)
                        ],
                    ),
                    func=lambda s: bool(re.search(".*te.*3.*", s)),
                ),
            ],
        )

        # Input data with userName that matches the regex pattern
        input_data = {
            "userName": "test123",
        }
        output_data: dict[str, Any] = {}

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=deterministic_guardrail,
        )

        assert result.validation_passed is True
        assert "All deterministic guardrail rules passed" in result.reason

    def test_evaluate_post_deterministic_guardrail_matches_regex_negative(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test deterministic guardrail validation fails when regex doesn't match."""
        deterministic_guardrail = DeterministicGuardrail(
            id="test-deterministic-id",
            name="Regex Guardrail",
            description="Test regex guardrail",
            enabled_for_evals=True,
            guardrail_type="custom",
            selector=GuardrailSelector(
                scopes=[GuardrailScope.TOOL], match_names=["test"]
            ),
            rules=[
                WordRule(
                    rule_type="word",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[
                            FieldReference(path="userName", source=FieldSource.INPUT)
                        ],
                    ),
                    func=lambda s: bool(re.search(".*te.*3.*", s)),
                ),
            ],
        )

        # Input data with userName that doesn't match the regex pattern
        input_data = {
            "userName": "test",
        }
        output_data: dict[str, Any] = {}

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=deterministic_guardrail,
        )

        assert result.validation_passed is False
        assert "userName" in result.reason

    def test_evaluate_post_deterministic_guardrail_word_func_positive(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test deterministic guardrail validation passes when word func returns True."""
        deterministic_guardrail = DeterministicGuardrail(
            id="test-deterministic-id",
            name="Word Func Guardrail",
            description="Test word func guardrail",
            enabled_for_evals=True,
            guardrail_type="custom",
            selector=GuardrailSelector(
                scopes=[GuardrailScope.TOOL], match_names=["test"]
            ),
            rules=[
                WordRule(
                    rule_type="word",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[
                            FieldReference(path="userName", source=FieldSource.INPUT)
                        ],
                    ),
                    func=lambda s: len(s) > 5,
                ),
            ],
        )

        # Input data with userName that passes the function check
        input_data = {
            "userName": "testuser",
        }
        output_data: dict[str, Any] = {}

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=deterministic_guardrail,
        )

        assert result.validation_passed is True
        assert "All deterministic guardrail rules passed" in result.reason

    def test_evaluate_post_deterministic_guardrail_word_func_negative(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test deterministic guardrail validation fails when word func returns False."""
        deterministic_guardrail = DeterministicGuardrail(
            id="test-deterministic-id",
            name="Word Func Guardrail",
            description="Test word func guardrail",
            enabled_for_evals=True,
            guardrail_type="custom",
            selector=GuardrailSelector(
                scopes=[GuardrailScope.TOOL], match_names=["test"]
            ),
            rules=[
                WordRule(
                    rule_type="word",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[
                            FieldReference(path="userName", source=FieldSource.INPUT)
                        ],
                    ),
                    func=lambda s: len(s) > 5,
                ),
            ],
        )

        # Input data with userName that fails the function check
        input_data = {
            "userName": "test",
        }
        output_data: dict[str, Any] = {}

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=deterministic_guardrail,
        )

        assert result.validation_passed is False

    def test_evaluate_post_deterministic_guardrail_number_func_positive(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test deterministic guardrail validation passes when number func returns True."""
        deterministic_guardrail = DeterministicGuardrail(
            id="test-deterministic-id",
            name="Number Func Guardrail",
            description="Test number func guardrail",
            enabled_for_evals=True,
            guardrail_type="custom",
            selector=GuardrailSelector(
                scopes=[GuardrailScope.TOOL], match_names=["test"]
            ),
            rules=[
                NumberRule(
                    rule_type="number",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[FieldReference(path="age", source=FieldSource.INPUT)],
                    ),
                    func=lambda n: n >= 18 and n <= 65,
                ),
            ],
        )

        # Input data with age that passes the function check
        input_data = {
            "age": 25,
        }
        output_data: dict[str, Any] = {}

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=deterministic_guardrail,
        )

        assert result.validation_passed is True
        assert "All deterministic guardrail rules passed" in result.reason

    def test_evaluate_post_deterministic_guardrail_number_func_negative(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test deterministic guardrail validation fails when number func returns False."""
        deterministic_guardrail = DeterministicGuardrail(
            id="test-deterministic-id",
            name="Number Func Guardrail",
            description="Test number func guardrail",
            enabled_for_evals=True,
            guardrail_type="custom",
            selector=GuardrailSelector(
                scopes=[GuardrailScope.TOOL], match_names=["test"]
            ),
            rules=[
                NumberRule(
                    rule_type="number",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[FieldReference(path="age", source=FieldSource.INPUT)],
                    ),
                    func=lambda n: n >= 18 and n <= 65,
                ),
            ],
        )

        # Input data with age that fails the function check
        input_data = {
            "age": 70,
        }
        output_data: dict[str, Any] = {}

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=deterministic_guardrail,
        )

        assert result.validation_passed is False

    def test_should_trigger_policy_pre_execution_only_some_rules_not_met_returns_false(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test pre-execution guardrail fails when some rules are not met."""
        guardrail = self._create_guardrail_for_pre_execution()
        input_data = {
            "userName": "John",
            "age": 18,  # Less than 21
            "isActive": True,
        }
        output_data: dict[str, Any] = {}

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        assert result.validation_passed is False

    def test_should_ignore_post_execution_guardrail_for_pre_execution_returns_false(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test that pre-execution guardrail ignores post-execution data."""
        guardrail = self._create_guardrail_for_post_execution()
        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": True,
        }
        output_data: dict[str, Any] = {}

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        # Should fail because post-execution guardrail needs output data
        assert result.validation_passed is True

    def test_should_trigger_policy_post_execution_guardrail_for_pre_execution_returns_false(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test that pre-execution guardrail does not trigger in post-execution."""
        guardrail = self._create_guardrail_for_pre_execution()
        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": True,
        }
        output_data = {
            "result": "Success",
            "status": 200,
            "success": True,
        }

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        # Pre-execution guardrail should still pass in post-execution
        assert result.validation_passed is True

    def test_should_trigger_policy_post_execution_with_output_fields_all_conditions_met_returns_true(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test post-execution guardrail passes when all conditions are met."""
        guardrail = self._create_guardrail_for_post_execution()
        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": True,
        }
        output_data = {
            "result": "Success",
            "status": 200,
            "success": True,
        }

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        assert result.validation_passed is True

    def test_should_trigger_policy_post_execution_with_output_fields_input_conditions_not_met_returns_false(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test post-execution guardrail fails when input conditions are not met."""
        guardrail = self._create_guardrail_for_post_execution()
        input_data = {
            "userName": "John",
            "age": 18,  # Less than 21
            "isActive": True,
        }
        output_data = {
            "result": "Success",
            "status": 200,
            "success": True,
        }

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        assert result.validation_passed is False

    def test_should_trigger_policy_post_execution_with_output_fields_output_conditions_not_met_returns_false(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test post-execution guardrail fails when output conditions are not met."""
        guardrail = self._create_guardrail_for_post_execution()
        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": True,
        }
        output_data = {
            "result": "Success",
            "status": 400,  # Not 200
            "success": True,
        }

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        assert result.validation_passed is False

    def test_should_trigger_policy_post_execution_multiple_rules_all_conditions_must_be_met_returns_true(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test post-execution guardrail with multiple rules passes when all conditions are met."""
        guardrail = self._create_guardrail_with_multiple_rules()
        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": True,
        }
        output_data = {
            "result": "Success",
            "status": 200,
            "success": True,
        }

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        assert result.validation_passed is True

    def test_should_trigger_policy_post_execution_rule_with_multiple_conditions_all_must_be_met_returns_true(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test guardrail with rule having multiple conditions passes when all are met."""
        guardrail = self._create_guardrail_with_rule_having_multiple_conditions()
        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": True,
        }
        output_data = {
            "result": "Success",
            "status": 200,
            "success": True,
        }

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        assert result.validation_passed is True

    def test_should_trigger_policy_post_execution_rule_with_multiple_conditions_one_condition_not_met_returns_false(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test guardrail with multiple conditions fails when one condition is not met."""
        guardrail = self._create_guardrail_with_rule_having_multiple_conditions()
        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": False,  # Not True
        }
        output_data = {
            "result": "Success",
            "status": 200,
            "success": True,
        }

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        assert result.validation_passed is False

    def test_should_trigger_policy_post_execution_with_all_fields_selector_output_schema_has_fields_returns_true(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test guardrail with AllFieldsSelector passes when output has matching fields."""
        guardrail = DeterministicGuardrail(
            id="test-all-fields-id",
            name="Guardrail With All Fields Selector",
            description="Test all fields selector",
            enabled_for_evals=True,
            guardrail_type="custom",
            selector=GuardrailSelector(
                scopes=[GuardrailScope.TOOL], match_names=["test"]
            ),
            rules=[
                NumberRule(
                    rule_type="number",
                    field_selector=AllFieldsSelector(selector_type="all"),
                    func=lambda n: n == 25.0,
                ),
            ],
        )

        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": True,
        }
        output_data = {
            "result": "Success",
            "status": 25,  # Matches the rule value
            "success": True,
        }

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        assert result.validation_passed is True

    def test_should_trigger_policy_post_execution_with_all_fields_selector_empty_output_schema_returns_false(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test guardrail with AllFieldsSelector fails when output is empty."""
        guardrail = DeterministicGuardrail(
            id="test-all-fields-id",
            name="Guardrail With All Fields Selector",
            description="Test all fields selector",
            enabled_for_evals=True,
            guardrail_type="custom",
            selector=GuardrailSelector(
                scopes=[GuardrailScope.TOOL], match_names=["test"]
            ),
            rules=[
                NumberRule(
                    rule_type="number",
                    field_selector=AllFieldsSelector(selector_type="all"),
                    func=lambda n: n == 200.0,
                ),
            ],
        )

        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": True,
        }
        output_data: dict[str, Any] = {}  # Empty output

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        assert result.validation_passed is False

    def test_should_trigger_policy_pre_execution_always_rule_with_input_apply_to_returns_true(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test UniversalRule with INPUT ApplyTo triggers in pre-execution."""
        guardrail = self._create_guardrail_with_always_rule(ApplyTo.INPUT)
        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": True,
        }
        output_data: dict[str, Any] = {}

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        assert result.validation_passed is False  # Should trigger

    def test_should_trigger_policy_pre_execution_always_rule_with_output_apply_to_returns_false(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test UniversalRule with OUTPUT ApplyTo does not trigger in pre-execution."""
        guardrail = self._create_guardrail_with_always_rule(ApplyTo.OUTPUT)
        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": True,
        }
        output_data: dict[str, Any] = {}

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        assert result.validation_passed is True  # Should not trigger

    def test_should_trigger_policy_pre_execution_always_rule_with_input_and_output_apply_to_returns_true(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test UniversalRule with INPUT_AND_OUTPUT ApplyTo triggers in pre-execution."""
        guardrail = self._create_guardrail_with_always_rule(ApplyTo.INPUT_AND_OUTPUT)
        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": True,
        }
        output_data: dict[str, Any] = {}

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        assert result.validation_passed is False  # Should trigger

    def test_should_trigger_policy_post_execution_always_rule_with_input_apply_to_returns_false(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test UniversalRule with INPUT ApplyTo does not trigger in post-execution."""
        guardrail = self._create_guardrail_with_always_rule(ApplyTo.INPUT)
        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": True,
        }
        output_data = {
            "result": "Success",
            "status": 200,
            "success": True,
        }

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        assert result.validation_passed is True  # Should not trigger

    def test_should_trigger_policy_post_execution_always_rule_with_output_apply_to_returns_true(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test UniversalRule with OUTPUT ApplyTo triggers in post-execution."""
        guardrail = self._create_guardrail_with_always_rule(ApplyTo.OUTPUT)
        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": True,
        }
        output_data = {
            "result": "Success",
            "status": 200,
            "success": True,
        }

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        assert result.validation_passed is False  # Should trigger

    def test_should_trigger_policy_post_execution_always_rule_with_input_and_output_apply_to_returns_true(
        self,
        service: DeterministicGuardrailsService,
    ) -> None:
        """Test UniversalRule with INPUT_AND_OUTPUT ApplyTo triggers in post-execution."""
        guardrail = self._create_guardrail_with_always_rule(ApplyTo.INPUT_AND_OUTPUT)
        input_data = {
            "userName": "John",
            "age": 25,
            "isActive": True,
        }
        output_data = {
            "result": "Success",
            "status": 200,
            "success": True,
        }

        result = service.evaluate_post_deterministic_guardrail(
            input_data=input_data,
            output_data=output_data,
            guardrail=guardrail,
        )

        assert result.validation_passed is False  # Should trigger

        # Helper methods to create guardrails

    def _create_guardrail_for_pre_execution(self) -> DeterministicGuardrail:
        """Create a guardrail for pre-execution testing."""
        return DeterministicGuardrail(
            id="test-pre-exec-id",
            name="Pre execution Guardrail",
            description="Test pre-execution guardrail",
            enabled_for_evals=True,
            guardrail_type="custom",
            selector=GuardrailSelector(
                scopes=[GuardrailScope.TOOL], match_names=["test"]
            ),
            rules=[
                NumberRule(
                    rule_type="number",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[FieldReference(path="age", source=FieldSource.INPUT)],
                    ),
                    func=lambda n: n >= 21.0,
                ),
                BooleanRule(
                    rule_type="boolean",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[
                            FieldReference(path="isActive", source=FieldSource.INPUT)
                        ],
                    ),
                    func=lambda b: b is True,
                ),
            ],
        )

    def _create_guardrail_for_post_execution(self) -> DeterministicGuardrail:
        """Create a guardrail for post-execution testing."""
        return DeterministicGuardrail(
            id="test-post-exec-id",
            name="Guardrail for Post execution",
            description="Test post-execution guardrail",
            enabled_for_evals=True,
            guardrail_type="custom",
            selector=GuardrailSelector(
                scopes=[GuardrailScope.TOOL], match_names=["test"]
            ),
            rules=[
                NumberRule(
                    rule_type="number",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[FieldReference(path="age", source=FieldSource.INPUT)],
                    ),
                    func=lambda n: n >= 21.0,
                ),
                NumberRule(
                    rule_type="number",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[
                            FieldReference(path="status", source=FieldSource.OUTPUT)
                        ],
                    ),
                    func=lambda n: n == 200.0,
                ),
            ],
        )

    def _create_guardrail_with_multiple_rules(self) -> DeterministicGuardrail:
        """Create a guardrail with multiple rules."""
        return DeterministicGuardrail(
            id="test-multiple-rules-id",
            name="Guardrail With Multiple Rules",
            description="Test guardrail with multiple rules",
            enabled_for_evals=True,
            guardrail_type="custom",
            selector=GuardrailSelector(
                scopes=[GuardrailScope.TOOL], match_names=["test"]
            ),
            rules=[
                NumberRule(
                    rule_type="number",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[FieldReference(path="age", source=FieldSource.INPUT)],
                    ),
                    func=lambda n: n >= 21.0,
                ),
                BooleanRule(
                    rule_type="boolean",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[
                            FieldReference(path="isActive", source=FieldSource.INPUT)
                        ],
                    ),
                    func=lambda b: b is True,
                ),
                NumberRule(
                    rule_type="number",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[
                            FieldReference(path="status", source=FieldSource.OUTPUT)
                        ],
                    ),
                    func=lambda n: n == 200.0,
                ),
            ],
        )

    def _create_guardrail_with_rule_having_multiple_conditions(
        self,
    ) -> DeterministicGuardrail:
        """Create a guardrail with rule having multiple conditions."""
        return DeterministicGuardrail(
            id="test-multiple-conditions-id",
            name="Guardrail With Rule Having Multiple Conditions",
            description="Test guardrail with multiple conditions",
            enabled_for_evals=True,
            guardrail_type="custom",
            selector=GuardrailSelector(
                scopes=[GuardrailScope.TOOL], match_names=["test"]
            ),
            rules=[
                NumberRule(
                    rule_type="number",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[FieldReference(path="age", source=FieldSource.INPUT)],
                    ),
                    func=lambda n: n >= 21.0,
                ),
                BooleanRule(
                    rule_type="boolean",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[
                            FieldReference(path="isActive", source=FieldSource.INPUT)
                        ],
                    ),
                    func=lambda b: b is True,
                ),
                NumberRule(
                    rule_type="number",
                    field_selector=SpecificFieldsSelector(
                        selector_type="specific",
                        fields=[
                            FieldReference(path="status", source=FieldSource.OUTPUT)
                        ],
                    ),
                    func=lambda n: n == 200.0,
                ),
            ],
        )

    def _create_guardrail_with_always_rule(
        self, apply_to: ApplyTo
    ) -> DeterministicGuardrail:
        """Create a guardrail with an AlwaysRule (UniversalRule)."""
        return DeterministicGuardrail(
            id="test-always-rule-id",
            name="Guardrail With Always Rule",
            description="Test guardrail with always rule",
            enabled_for_evals=True,
            guardrail_type="custom",
            selector=GuardrailSelector(
                scopes=[GuardrailScope.TOOL], match_names=["test"]
            ),
            rules=[
                UniversalRule(
                    rule_type="always",
                    apply_to=apply_to,
                ),
            ],
        )
