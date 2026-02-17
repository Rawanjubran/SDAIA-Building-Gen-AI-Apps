from src.agent.observable_agent import ObservableAgent
from src.tools.registry import registry

def create_researcher(model: str = "gemini/gemini-1.5-flash", max_steps: int = 15):
    """The Researcher: finds, retrieves, and extracts information."""
    system_prompt = """You are a world-class researcher with expertise in finding and extracting information.

Your capabilities:
- Search the web for current, accurate information
- Extract key facts and data from sources
- Verify information across multiple sources

Your approach:
1. Break down complex queries into searchable sub-questions
2. Use search tools to find relevant information
3. Extract and organize key findings

Be thorough, accurate, and cite your sources."""

    tools = registry.get_tools_by_category("research")
    
    return ObservableAgent(
        model=model,
        max_steps=max_steps,
        agent_name="Researcher",
        system_prompt=system_prompt,
        tools=tools
    )

def create_analyst(model: str = "gemini/gemini-1.5-flash", max_steps: int = 20):
    """The Analyst: evaluates, cross-references, and identifies patterns."""
    system_prompt = """You are an expert analyst with deep skills in evaluation and critical thinking.

Your capabilities:
- Cross-reference information from multiple sources
- Identify patterns, trends, and inconsistencies
- Draw evidence-based conclusions

Your approach:
1. Examine all available information systematically
2. Look for patterns and contradictions
3. Synthesize findings into clear insights

Be thorough, skeptical, and evidence-based."""

    tools = registry.get_tools_by_category("analysis")
    
    return ObservableAgent(
        model=model,
        max_steps=max_steps,
        agent_name="Analyst",
        system_prompt=system_prompt,
        tools=tools
    )

def create_writer(model: str = "gemini/gemini-1.5-flash", max_steps: int = 4):
    """The Writer: synthesizes analysis into polished, readable output."""
    system_prompt = """You are a skilled writer who transforms complex analysis into clear, engaging content.

Your capabilities:
- Synthesize complex information into readable prose
- Structure content logically and coherently
- Create compelling narratives from data

Your approach:
1. Review all research and analysis provided
2. Structure the content logically
3. Write clearly and engagingly

Be clear, concise, and compelling."""

    tools = []
    
    return ObservableAgent(
        model=model,
        max_steps=max_steps,
        agent_name="Writer",
        system_prompt=system_prompt,
        tools=tools
    )