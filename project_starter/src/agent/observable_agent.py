import asyncio
import json
import os
import time

import structlog
from litellm import acompletion, completion_cost
from pydantic import ValidationError

from src.observability.cost_tracker import CostTracker
from src.observability.loop_detector import AdvancedLoopDetector
from src.observability.tracer import AgentStep, AgentTracer, ToolCallRecord
from src.tools.registry import registry

logger = structlog.get_logger()

class ObservableAgent:
    """
    Production-grade agent with full observability.
    
    This agent implements the ReAct pattern (Reasoning + Acting) but enhances it 
    with "Observability" - the ability to track, trace, and debug the agent's 
    internal state and actions.
    """
    def __init__(
        self,
        model: str = None,
        max_steps: int = 10,
        agent_name: str = "ObservableAgent",
        verbose: bool = True,
        system_prompt: str = None,
        tools: list = None,
    ):
        self.model = model or os.getenv("MODEL_NAME", "gemini/gemini-1.5-flash")
        self.max_steps = max_steps
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.tools = tools if tools is not None else registry.get_all_tools()

        # Initialize observability components
        self.tracer = AgentTracer(verbose=verbose)
        self.loop_detector = AdvancedLoopDetector()
        self.cost_tracker = CostTracker()

    async def run(self, user_query: str) -> dict:
        """Execute the agent loop with full observability."""
        # Start trace and cost tracking
        trace_id = self.tracer.start_trace(self.agent_name, user_query, self.model)
        self.cost_tracker.start_query(user_query)
        
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_query})
        
        step_number = 0
        final_answer = None
        
        try:
            # Loop until max_steps
            for step_number in range(1, self.max_steps + 1):
                step_start = time.time()
                
                # Call LLM
                tool_schemas = [tool.to_openai_schema() for tool in self.tools]
                
                response = await acompletion(
                    model=self.model,
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                    tool_choice="auto" if tool_schemas else None,
                )
                
                step_duration = (time.time() - step_start) * 1000
                
                # Log completion and cost
                self.cost_tracker.log_completion(step_number, response, is_tool_call=False)
                
                assistant_message = response.choices[0].message
                reasoning = assistant_message.content
                
                # Create step record
                step = AgentStep(
                    step_number=step_number,
                    reasoning=reasoning,
                    duration_ms=step_duration,
                )
                
                # Extract usage stats
                if hasattr(response, 'usage'):
                    usage = response.usage
                    step.input_tokens = getattr(usage, 'prompt_tokens', 0)
                    step.output_tokens = getattr(usage, 'completion_tokens', 0)
                    
                    # Calculate cost
                    input_cost = (step.input_tokens / 1_000_000) * 0.15
                    output_cost = (step.output_tokens / 1_000_000) * 0.60
                    step.cost_usd = input_cost + output_cost
                
                # Add assistant message to conversation
                messages.append({
                    "role": "assistant",
                    "content": reasoning,
                    "tool_calls": assistant_message.tool_calls if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls else None
                })
                
                # Handle tool calls
                tool_calls = getattr(assistant_message, 'tool_calls', None)
                
                if not tool_calls:
                    # No tool calls - this is the final answer
                    final_answer = reasoning
                    self.tracer.log_step(trace_id, step)
                    break
                
                # Execute tool calls (in parallel if multiple)
                tool_call_tasks = []
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_input_str = tool_call.function.arguments
                    
                    # Check for loops BEFORE executing
                    loop_result = self.loop_detector.check_tool_call(tool_name, tool_input_str)
                    
                    if loop_result.is_looping:
                        logger.warning("loop_detected", 
                                     strategy=loop_result.strategy,
                                     message=loop_result.message)
                        
                        # Add loop warning to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": f"⚠️ LOOP DETECTED: {loop_result.message}"
                        })
                        continue
                    
                    # Execute tool
                    tool_call_tasks.append(
                        self._execute_tool(tool_call, tool_name, tool_input_str)
                    )
                
                # Execute all tools in parallel
                tool_results = await asyncio.gather(*tool_call_tasks)
                
                # Add tool results to step and messages
                for tool_call, (tool_name, tool_input_str, tool_output, duration) in zip(tool_calls, tool_results):
                    step.tool_calls.append(ToolCallRecord(
                        tool_name=tool_name,
                        tool_input=json.loads(tool_input_str) if tool_input_str else {},
                        tool_output=tool_output,
                        duration_ms=duration
                    ))
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": tool_output
                    })
                
                # Log the step
                self.tracer.log_step(trace_id, step)
                
                # Check for output stagnation
                if reasoning:
                    stagnation = self.loop_detector.check_output_stagnation(reasoning)
                    if stagnation.is_looping:
                        logger.warning("stagnation_detected", message=stagnation.message)
                        final_answer = f"Agent stopped due to stagnation: {stagnation.message}"
                        break
            
            # If we exhausted max_steps without final answer
            if final_answer is None:
                final_answer = f"Agent reached max steps ({self.max_steps}) without completing the task."
            
            # End trace
            self.tracer.end_trace(trace_id, final_answer, status="completed")
            self.cost_tracker.end_query()
            
            return {
                "answer": final_answer,
                "trace_id": trace_id,
                "steps": step_number,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error("agent_error", error=str(e), trace_id=trace_id)
            self.tracer.end_trace(trace_id, "", status="error", error=str(e))
            self.cost_tracker.end_query()
            
            return {
                "answer": f"Error: {str(e)}",
                "trace_id": trace_id,
                "status": "error"
            }
    
    async def _execute_tool(self, tool_call, tool_name: str, tool_input_str: str):
        """Execute a single tool and return result."""
        start_time = time.time()
        
        try:
            tool = registry.get_tool(tool_name)
            if not tool:
                result = f"Error: Tool '{tool_name}' not found"
            else:
                tool_input = json.loads(tool_input_str)
                result = str(tool.execute(**tool_input))
        except Exception as e:
            result = f"Error executing {tool_name}: {str(e)}"
        
        duration = (time.time() - start_time) * 1000
        return (tool_name, tool_input_str, result, duration)