from typing import Optional, List, Dict, Any
import logging
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoreAgentState(BaseModel):
    """State definition for the core agent"""

    messages: List[BaseMessage] = Field(default_factory=list)
    current_task: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    memory: Dict[str, Any] = Field(default_factory=dict)
    tool_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation_results: Dict[str, Any] = Field(default_factory=dict)
    human_feedback: str = ""  # Default empty string for test compatibility
    supervisor_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    next_agent: str = ""  # For multi-agent coordination
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Added for backward compatibility

    # Backward compatibility properties
    @property
    def tool_results(self) -> List[Dict[str, Any]]:
        """Backward compatibility alias for tool_outputs"""
        return self.tool_outputs

    @tool_results.setter
    def tool_results(self, value: List[Dict[str, Any]]):
        """Backward compatibility setter for tool_outputs"""
        self.tool_outputs = value

    class Config:
        arbitrary_types_allowed = True