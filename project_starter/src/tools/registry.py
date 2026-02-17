import inspect
from typing import Any, Callable, Dict

from pydantic import BaseModel, create_model

class Tool:
    """A callable tool with schema."""
    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description
        self.model = self._create_pydantic_model(func)

    def _create_pydantic_model(self, func: Callable) -> type[BaseModel]:
        """Create a Pydantic model from function signature."""
        sig = inspect.signature(func)
        fields = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue

            annotation = param.annotation
            if annotation == inspect.Parameter.empty:
                annotation = str

            default = param.default
            if default == inspect.Parameter.empty:
                fields[name] = (annotation, ...)
            else:
                fields[name] = (annotation, default)

        return create_model(f"{self.name}Schema", **fields)

    def to_openai_schema(self) -> dict:
        """Convert tool to OpenAI function schema format using Pydantic."""
        schema = self.model.model_json_schema()

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                    "additionalProperties": False,
                },
                "strict": True
            },
        }

    def execute(self, **kwargs) -> Any:
        # Validate arguments using the model
        validated_args = self.model(**kwargs)
        return self.func(**validated_args.model_dump())

class ToolRegistry:
    """Registry for managing available tools."""
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, list[str]] = {}

    def register(self, name: str, description: str, category: str = "general"):
        """
        Decorator to register a function as a tool.
        
        This method uses the Decorator Pattern to dynamically adding functionality 
        (registration) to functions without modifying their structure.
        """
        def decorator(func: Callable):
            # Create a Tool instance from the function
            tool = Tool(name=name, func=func, description=description)
            
            # Register it in self._tools dictionary
            self._tools[name] = tool
            
            # Add to category in self._categories
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(name)
            
            # Return the original function (so it can still be called normally)
            return func
        return decorator

    def get_tool(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get_tools_by_category(self, category: str) -> list[Tool]:
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def execute_tool(self, name: str) -> Callable:
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found in registry")
        return tool.execute

# Global registry instance
registry = ToolRegistry()