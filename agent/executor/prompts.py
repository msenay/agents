"""
Code Execution Prompts for ExecutorAgent
========================================

Prompts for safely executing code and running tests.
"""

# System prompt for ExecutorAgent
SYSTEM_PROMPT = """You are ExecutorAgent, a specialized AI agent focused on safely executing code and running tests.

Your primary responsibilities:
1. Execute Python code in a controlled environment
2. Run unit tests and report results
3. Create and execute demo scripts based on specifications
4. Handle errors gracefully and provide meaningful feedback
5. Ensure security by validating code before execution

Key principles:
- Always validate code for potential security risks before execution
- Provide clear execution results and error messages
- Capture and report stdout, stderr, and return values
- Handle timeouts and resource limits appropriately
- Create isolated execution environments when needed

You have specialized tools for:
- Executing Python code snippets
- Running unit tests with various frameworks
- Creating demo scripts from specifications
- Validating code safety
- Managing execution environments

Always prioritize safety and provide comprehensive execution reports."""

# Prompt for creating demo from spec
CREATE_DEMO_PROMPT = """Create a simple demo script based on the following code and specification:

Code:
```python
{code}
```

Specification:
{spec}

Requirements:
1. Create a working demo that showcases the code's functionality
2. Include example inputs and expected outputs
3. Add print statements to show intermediate steps
4. Handle potential errors gracefully
5. Make the demo self-contained and runnable
6. Include helpful comments explaining what's happening

Generate a complete, runnable Python demo script."""

# Prompt for analyzing code safety
ANALYZE_CODE_SAFETY_PROMPT = """Analyze the following code for potential security risks before execution:

```python
{code}
```

Check for:
1. System calls or subprocess usage
2. File system operations (especially writes/deletes)
3. Network operations
4. Dangerous imports (os, subprocess, eval, exec)
5. Infinite loops or resource-intensive operations
6. SQL injection risks
7. Path traversal attempts
8. Environment variable access

Provide a safety assessment with:
- Risk level (SAFE, LOW_RISK, MEDIUM_RISK, HIGH_RISK, UNSAFE)
- Specific concerns if any
- Recommendations for safe execution"""

# Prompt for preparing test execution
PREPARE_TEST_EXECUTION_PROMPT = """Prepare the following test code for execution:

```python
{test_code}
```

Requirements:
1. Identify the test framework being used
2. Add necessary imports if missing
3. Create a test runner script
4. Handle test discovery if needed
5. Set up proper test reporting
6. Include error handling

Generate a complete script to run these tests and capture results."""

# Prompt for formatting execution results
FORMAT_EXECUTION_RESULTS_PROMPT = """Format the following execution results for clear presentation:

Exit Code: {exit_code}
Stdout: {stdout}
Stderr: {stderr}
Execution Time: {execution_time}s

Create a well-formatted summary including:
1. Success/failure status
2. Key outputs or results
3. Any errors or warnings
4. Performance metrics
5. Recommendations if applicable"""

# Prompt for creating test report
CREATE_TEST_REPORT_PROMPT = """Create a comprehensive test report from the following test execution results:

Raw Output:
{raw_output}

Test Framework: {framework}
Total Tests: {total_tests}
Passed: {passed}
Failed: {failed}
Skipped: {skipped}

Generate a detailed report including:
1. Overall test summary
2. Failed test details with error messages
3. Test coverage information if available
4. Performance metrics
5. Recommendations for fixing failures"""

# Prompt for creating execution environment
CREATE_EXECUTION_ENV_PROMPT = """Create a safe execution environment setup for the following code:

```python
{code}
```

Requirements:
1. Identify required dependencies
2. Set up virtual environment if needed
3. Install necessary packages
4. Configure resource limits
5. Set up proper isolation
6. Handle cleanup after execution

Generate setup instructions and execution script."""

# Prompt for handling execution errors
HANDLE_EXECUTION_ERROR_PROMPT = """Analyze and handle the following execution error:

Code that failed:
```python
{code}
```

Error:
{error}

Traceback:
{traceback}

Provide:
1. Clear explanation of what went wrong
2. Potential causes
3. Suggested fixes
4. Modified code that might work
5. Best practices to avoid this error"""

# Security validation patterns
DANGEROUS_PATTERNS = [
    r"import\s+os",
    r"import\s+subprocess",
    r"import\s+sys",
    r"__import__",
    r"eval\s*\(",
    r"exec\s*\(",
    r"compile\s*\(",
    r"open\s*\(",
    r"file\s*\(",
    r"input\s*\(",
    r"raw_input\s*\(",
    r"\.system\s*\(",
    r"\.popen\s*\(",
    r"\.call\s*\(",
    r"\.run\s*\(",
    r"\.Popen\s*\(",
    r"globals\s*\(",
    r"locals\s*\(",
    r"vars\s*\(",
    r"dir\s*\(",
    r"getattr\s*\(",
    r"setattr\s*\(",
    r"delattr\s*\(",
    r"\.\.\/",  # Path traversal
    r"rm\s+-rf",  # Dangerous commands
    r"DELETE\s+FROM",  # SQL operations
    r"DROP\s+TABLE",
]

# Safe imports whitelist
SAFE_IMPORTS = [
    "math",
    "datetime",
    "json",
    "re",
    "collections",
    "itertools",
    "functools",
    "typing",
    "dataclasses",
    "enum",
    "decimal",
    "fractions",
    "statistics",
    "random",
    "string",
    "textwrap",
    "unicodedata",
    "pytest",
    "unittest",
    "nose",
]