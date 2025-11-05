
from langchain_classic.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List
from langchain_classic.schema import LLMResult

class AgentCallbackHandler(BaseCallbackHandler):
    """Callback handler for agent actions and observations."""

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str],  **kwargs: Any) -> Any:
        """Run when LLM starts running."""
        print(f"***LLM started with prompts: {prompts}***")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print(f"***LLM RESPONSE: {response.generations[0][0].text}***")
        