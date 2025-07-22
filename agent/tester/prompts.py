"""
Unit Test Generation Prompts for TesterAgent
============================================

Prompts for generating comprehensive unit tests for given code.
"""

# System prompt for TesterAgent
SYSTEM_PROMPT = """You are TesterAgent, a specialized AI agent focused on generating comprehensive unit tests.

Your primary responsibilities:
1. Analyze code and identify all testable components
2. Generate comprehensive unit tests with high coverage
3. Include edge cases, error scenarios, and happy paths
4. Follow testing best practices (AAA pattern: Arrange, Act, Assert)
5. Generate tests using appropriate testing frameworks

Key principles:
- Write clear, descriptive test names
- Test one thing per test method
- Use proper mocking and fixtures
- Include both positive and negative test cases
- Ensure tests are independent and repeatable
- Add helpful comments and docstrings

You have specialized tools for:
- Generating unit tests from code
- Analyzing code coverage requirements
- Creating test fixtures and mocks
- Validating test quality

Always aim for high-quality, maintainable tests that provide confidence in the code's correctness."""

# Prompt for generating unit tests
GENERATE_UNIT_TESTS_PROMPT = """Generate comprehensive unit tests for the following code:

```python
{code}
```

Requirements:
1. Use pytest as the testing framework
2. Include all necessary imports
3. Test all public methods and functions
4. Include edge cases and error scenarios
5. Use descriptive test names (test_<function>_<scenario>)
6. Add docstrings to test functions
7. Use fixtures where appropriate
8. Mock external dependencies
9. Aim for high code coverage

Generate ONLY the test code, no explanations."""

# Prompt for generating test with specific framework
GENERATE_TESTS_WITH_FRAMEWORK_PROMPT = """Generate comprehensive unit tests for the following code using {framework}:

```python
{code}
```

Framework-specific requirements for {framework}:
{framework_requirements}

General requirements:
1. Include all necessary imports
2. Test all public methods and functions
3. Include edge cases and error scenarios
4. Use descriptive test names
5. Add docstrings to test functions
6. Mock external dependencies appropriately
7. Follow {framework} best practices

Generate ONLY the test code, no explanations."""

# Framework-specific requirements
FRAMEWORK_REQUIREMENTS = {
    "pytest": """
- Use pytest fixtures for setup and teardown
- Use @pytest.mark decorators for test categorization
- Use pytest.raises for exception testing
- Use parametrize for data-driven tests
- Follow pytest naming conventions (test_* or *_test.py)
""",
    "unittest": """
- Inherit test classes from unittest.TestCase
- Use setUp() and tearDown() methods
- Use self.assert* methods for assertions
- Use unittest.mock for mocking
- Follow unittest naming conventions (test_*)
""",
    "nose2": """
- Use nose2 decorators and plugins
- Support both function and class-based tests
- Use nose2 assertion helpers
- Follow nose2 test discovery patterns
""",
}

# Prompt for generating integration tests
GENERATE_INTEGRATION_TESTS_PROMPT = """Generate integration tests for the following code:

```python
{code}
```

Integration test requirements:
1. Test interactions between components
2. Test with real or test databases if applicable
3. Test API endpoints if present
4. Test file I/O operations
5. Include setup and cleanup procedures
6. Test error propagation between components
7. Use appropriate test data

Generate comprehensive integration tests following best practices."""

# Prompt for generating test fixtures
GENERATE_FIXTURES_PROMPT = """Generate test fixtures and mocks for the following code:

```python
{code}
```

Fixture requirements:
1. Create reusable fixtures for common test data
2. Generate appropriate mock objects
3. Include factory functions for complex objects
4. Create fixtures for database states if needed
5. Include cleanup procedures
6. Follow pytest fixture best practices

Generate ONLY the fixture code with clear documentation."""

# Prompt for analyzing test coverage
ANALYZE_COVERAGE_PROMPT = """Analyze the following code and identify what tests are missing:

Code:
```python
{code}
```

Current Tests:
```python
{tests}
```

Identify:
1. Untested functions/methods
2. Untested code paths/branches
3. Missing edge cases
4. Missing error scenarios
5. Suggested additional tests

Provide a detailed analysis of test coverage gaps."""

# Prompt for generating parameterized tests
GENERATE_PARAMETERIZED_TESTS_PROMPT = """Generate parameterized tests for the following code:

```python
{code}
```

Requirements:
1. Identify functions suitable for parameterized testing
2. Create comprehensive test data sets
3. Use pytest.mark.parametrize or similar
4. Include edge cases in parameters
5. Test boundary conditions
6. Include both valid and invalid inputs

Generate data-driven tests that thoroughly exercise the code."""

# Prompt for generating performance tests
GENERATE_PERFORMANCE_TESTS_PROMPT = """Generate performance tests for the following code:

```python
{code}
```

Performance test requirements:
1. Measure execution time for critical functions
2. Test memory usage if applicable
3. Test with varying data sizes
4. Include benchmarks
5. Use appropriate performance testing tools
6. Test concurrent execution if relevant

Generate performance tests that help identify bottlenecks."""