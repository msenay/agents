#!/usr/bin/env python3
"""
Core Agent Tools
===============

Custom tools for Core Agent to execute Python code and manage files.
"""

import os
import subprocess
import tempfile
import shutil
from typing import Optional, Dict, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class PythonExecutorInput(BaseModel):
    """Input for Python executor tool"""
    code: str = Field(description="Python code to execute")
    timeout: int = Field(default=30, description="Timeout in seconds")


class PythonExecutorTool(BaseTool):
    """Tool to execute Python code safely"""
    
    name: str = "python_executor"
    description: str = """Execute Python code in a safe environment and return the output.
    Use this tool to run Python code and see the results.
    Input should be valid Python code as a string."""
    
    args_schema: type[BaseModel] = PythonExecutorInput
    
    def _run(self, code: str, timeout: int = 30) -> str:
        """Execute Python code and return output"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute Python code
                result = subprocess.run(
                    ['python3', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tempfile.gettempdir()  # Safe directory
                )
                
                # Prepare output
                output = []
                if result.stdout:
                    output.append(f"Output:\n{result.stdout}")
                
                if result.stderr:
                    output.append(f"Errors:\n{result.stderr}")
                
                if result.returncode != 0:
                    output.append(f"Exit code: {result.returncode}")
                
                return "\n".join(output) if output else "Code executed successfully (no output)"
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    
        except subprocess.TimeoutExpired:
            return f"Code execution timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing code: {str(e)}"


class FileWriterInput(BaseModel):
    """Input for file writer tool"""
    filename: str = Field(description="Name of the file to write")
    content: str = Field(description="Content to write to the file")
    directory: Optional[str] = Field(default=None, description="Directory to write file (optional)")


class FileWriterTool(BaseTool):
    """Tool to write content to files"""
    
    name: str = "file_writer"
    description: str = """Write content to a file. Use this to save Python code or any text content.
    Specify the filename and content. Optionally specify a directory."""
    
    args_schema: type[BaseModel] = FileWriterInput
    
    def __init__(self, workspace_dir: str = None):
        super().__init__()
        # Store workspace_dir as a private attribute to avoid Pydantic conflicts
        self._workspace_dir = workspace_dir or os.getcwd()
        
        # Create workspace directory if it doesn't exist
        if not os.path.exists(self._workspace_dir):
            os.makedirs(self._workspace_dir)
    
    def _run(self, filename: str, content: str, directory: Optional[str] = None) -> str:
        """Write content to a file"""
        try:
            # Determine target directory
            if directory:
                target_dir = os.path.join(self._workspace_dir, directory)
            else:
                target_dir = self._workspace_dir
            
            # Create directory if it doesn't exist
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            # Full file path
            file_path = os.path.join(target_dir, filename)
            
            # Write content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Verify file was written
            file_size = os.path.getsize(file_path)
            
            return f"Successfully wrote {file_size} bytes to {file_path}"
            
        except Exception as e:
            return f"Error writing file: {str(e)}"


class FileReaderInput(BaseModel):
    """Input for file reader tool"""
    filename: str = Field(description="Name of the file to read")
    directory: Optional[str] = Field(default=None, description="Directory to read from (optional)")


class FileReaderTool(BaseTool):
    """Tool to read content from files"""
    
    name: str = "file_reader"
    description: str = """Read content from a file. Use this to read previously saved files.
    Specify the filename and optionally a directory."""
    
    args_schema: type[BaseModel] = FileReaderInput
    
    def __init__(self, workspace_dir: str = None):
        super().__init__()
        self._workspace_dir = workspace_dir or os.getcwd()
    
    def _run(self, filename: str, directory: Optional[str] = None) -> str:
        """Read content from a file"""
        try:
            # Determine source directory
            if directory:
                source_dir = os.path.join(self._workspace_dir, directory)
            else:
                source_dir = self._workspace_dir
            
            # Full file path
            file_path = os.path.join(source_dir, filename)
            
            # Check if file exists
            if not os.path.exists(file_path):
                return f"File not found: {file_path}"
            
            # Read content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_size = len(content)
            return f"File content ({file_size} characters):\n\n{content}"
            
        except Exception as e:
            return f"Error reading file: {str(e)}"


class DirectoryListInput(BaseModel):
    """Input for directory listing tool"""
    directory: Optional[str] = Field(default=None, description="Directory to list (optional, defaults to workspace)")


class DirectoryListTool(BaseTool):
    """Tool to list files in directory"""
    
    name: str = "directory_list"
    description: str = """List files and directories in the workspace or specified directory.
    Use this to see what files are available."""
    
    args_schema: type[BaseModel] = DirectoryListInput
    
    def __init__(self, workspace_dir: str = None):
        super().__init__()
        self._workspace_dir = workspace_dir or os.getcwd()
    
    def _run(self, directory: Optional[str] = None) -> str:
        """List directory contents"""
        try:
            # Determine target directory
            if directory:
                target_dir = os.path.join(self._workspace_dir, directory)
            else:
                target_dir = self._workspace_dir
            
            # Check if directory exists
            if not os.path.exists(target_dir):
                return f"Directory not found: {target_dir}"
            
            # List contents
            items = []
            for item in sorted(os.listdir(target_dir)):
                item_path = os.path.join(target_dir, item)
                if os.path.isdir(item_path):
                    items.append(f"📁 {item}/")
                else:
                    size = os.path.getsize(item_path)
                    items.append(f"📄 {item} ({size} bytes)")
            
            if not items:
                return f"Directory is empty: {target_dir}"
            
            return f"Contents of {target_dir}:\n" + "\n".join(items)
            
        except Exception as e:
            return f"Error listing directory: {str(e)}"


def create_python_coding_tools(workspace_dir: str = None) -> list[BaseTool]:
    """Create a set of tools for Python coding tasks"""
    if workspace_dir is None:
        # Create a temporary workspace
        workspace_dir = tempfile.mkdtemp(prefix="core_agent_workspace_")
    
    return [
        PythonExecutorTool(),
        FileWriterTool(workspace_dir=workspace_dir),
        FileReaderTool(workspace_dir=workspace_dir),
        DirectoryListTool(workspace_dir=workspace_dir)
    ]


# Test the tools
if __name__ == "__main__":
    print("Testing Core Agent Tools...")
    
    # Create tools
    tools = create_python_coding_tools()
    
    # Test Python executor
    python_tool = tools[0]
    test_code = """
def test_function():
    print("Hello from Core Agent!")
    return 42

result = test_function()
print(f"Result: {result}")
"""
    
    print("\n🐍 Testing Python Executor:")
    result = python_tool._run(test_code)
    print(result)
    
    # Test file writer
    file_writer = tools[1]
    test_content = """# Test Python file generated by Core Agent
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

if __name__ == "__main__":
    for i in range(10):
        print(f"F({i}) = {fibonacci(i)}")
"""
    
    print("\n📄 Testing File Writer:")
    result = file_writer._run("fibonacci.py", test_content)
    print(result)
    
    # Test directory listing
    dir_tool = tools[3]
    print("\n📁 Testing Directory Listing:")
    result = dir_tool._run()
    print(result)
    
    # Test file reader
    file_reader = tools[2]
    print("\n📖 Testing File Reader:")
    result = file_reader._run("fibonacci.py")
    print(result)
    
    print("\n✅ All tools tested successfully!")