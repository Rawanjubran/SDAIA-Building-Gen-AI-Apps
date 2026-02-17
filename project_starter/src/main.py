import asyncio
import os
import sys

from dotenv import load_dotenv

from src.agent.specialists import create_researcher, create_analyst, create_writer
from src.observability.tracer import tracer
from src.observability.cost_tracker import cost_tracker

# Load environment variables
load_dotenv()

async def main():
    """
    Main entry point for the AI Agent system.
    """
    # 1. Get the query
    if len(sys.argv) < 2:
        print("Usage: python -m src.main \"Your research query\"")
        sys.exit(1)

    query = sys.argv[1]
    print(f"\n🔍 Starting research on: {query}\n")

    # Initialize agents using Factory Pattern
    researcher = create_researcher()
    analyst = create_analyst()
    writer = create_writer()

    # Simple linear chain: Researcher -> Analyst -> Writer
    try:
        # Step 1: Research
        print("📚 Phase 1: Research")
        research_result = await researcher.run(query)
        
        if research_result.get("status") == "error":
            print(f"\n❌ Research phase failed: {research_result.get('answer')}")
            cost_tracker.print_cost_breakdown()
            return
        
        research_output = research_result["answer"]
        
        print(f"\n✅ Research completed in {research_result.get('steps', 0)} steps")
        print(f"Research findings: {research_output[:200]}...\n")
        
        # Step 2: Analysis
        print("🔍 Phase 2: Analysis")
        analysis_query = f"Analyze the following research findings and provide insights:\n\n{research_output}"
        analysis_result = await analyst.run(analysis_query)
        
        if analysis_result.get("status") == "error":
            print(f"\n❌ Analysis phase failed: {analysis_result.get('answer')}")
            cost_tracker.print_cost_breakdown()
            return
        
        analysis_output = analysis_result["answer"]
        
        print(f"\n✅ Analysis completed in {analysis_result.get('steps', 0)} steps")
        print(f"Analysis: {analysis_output[:200]}...\n")
        
        # Step 3: Writing
        print("✍️ Phase 3: Writing")
        writing_query = f"Write a polished, comprehensive report based on this research and analysis:\n\nResearch:\n{research_output}\n\nAnalysis:\n{analysis_output}"
        writing_result = await writer.run(writing_query)
        
        if writing_result.get("status") == "error":
            print(f"\n❌ Writing phase failed: {writing_result.get('answer')}")
            cost_tracker.print_cost_breakdown()
            return
        
        final_output = writing_result["answer"]
        
        # Print final results
        print("\n" + "="*60)
        print("📄 FINAL REPORT")
        print("="*60)
        print(final_output)
        print("\n" + "="*60)
        print("📊 EXECUTION SUMMARY")
        print("="*60)
        print(f"Research: {research_result.get('steps', 0)} steps")
        print(f"Analysis: {analysis_result.get('steps', 0)} steps")
        print(f"Writing: {writing_result.get('steps', 0)} steps")
        print(f"Total phases: 3")
        print("="*60)
        
        # Print cost breakdown
        print("\n")
        cost_tracker.print_cost_breakdown()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())