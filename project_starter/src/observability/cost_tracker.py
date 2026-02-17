import logging
from dataclasses import dataclass, field
# from litellm import completion_cost

logger = logging.getLogger(__name__)

@dataclass
class StepCost:
    step_number: int
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    is_tool_call: bool = False

@dataclass
class QueryCost:
    query: str
    steps: list[StepCost] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def add_step(self, step: StepCost):
        self.steps.append(step)
        self.total_cost_usd += step.cost_usd
        self.total_input_tokens += step.input_tokens
        self.total_output_tokens += step.output_tokens

class CostTracker:
    """
    Tracks costs across agent executions.
    """
    def __init__(self):
        self.queries: list[QueryCost] = []
        self._current_query: QueryCost | None = None

    def start_query(self, query: str):
        self._current_query = QueryCost(query=query)

    def log_completion(self, step_number: int, response, is_tool_call: bool = False):
        """
        Log a completion response's cost.
        """
        # Check if _current_query exists
        if not self._current_query:
            logger.warning("No active query to log completion to")
            return
        
        # Extract usage stats from response
        usage = getattr(response, 'usage', None)
        if not usage:
            logger.warning("Response has no usage data")
            return
        
        input_tokens = getattr(usage, 'prompt_tokens', 0)
        output_tokens = getattr(usage, 'completion_tokens', 0)
        model = getattr(response, 'model', 'gpt-4o-mini')
        
        # Calculate cost (fallback pricing)
        # GPT-4o-mini: $0.15/1M input, $0.60/1M output
        input_cost = (input_tokens / 1_000_000) * 0.15
        output_cost = (output_tokens / 1_000_000) * 0.60
        total_cost = input_cost + output_cost
        
        # Create StepCost and add to query
        step_cost = StepCost(
            step_number=step_number,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=total_cost,
            is_tool_call=is_tool_call
        )
        
        self._current_query.add_step(step_cost)
        
        logger.info(f"Step {step_number}: {input_tokens} in, {output_tokens} out, ${total_cost:.6f}")

    def end_query(self):
        if self._current_query:
            self.queries.append(self._current_query)
            self._current_query = None

    def print_cost_breakdown(self):
        """Print detailed cost breakdown"""
        if not self.queries:
            print("No queries tracked yet.")
            return
        
        print("\n" + "="*60)
        print("COST BREAKDOWN")
        print("="*60)
        
        total_cost = 0.0
        total_input = 0
        total_output = 0
        
        for i, query in enumerate(self.queries, 1):
            print(f"\nQuery {i}: {query.query[:50]}...")
            print(f"  Total Cost: ${query.total_cost_usd:.6f}")
            print(f"  Tokens: {query.total_input_tokens} in, {query.total_output_tokens} out")
            print(f"  Steps: {len(query.steps)}")
            
            for step in query.steps:
                step_type = "Tool Call" if step.is_tool_call else "Response"
                print(f"    Step {step.step_number} ({step_type}): ${step.cost_usd:.6f} - {step.input_tokens}/{step.output_tokens} tokens")
            
            total_cost += query.total_cost_usd
            total_input += query.total_input_tokens
            total_output += query.total_output_tokens
        
        print("\n" + "="*60)
        print(f"TOTAL: ${total_cost:.6f}")
        print(f"TOKENS: {total_input} input, {total_output} output")
        print("="*60 + "\n")

# Global cost tracker instance
cost_tracker = CostTracker()