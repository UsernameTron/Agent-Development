"""
Basic integration example for Local O1 components
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import core components
from agents.agents import Agent, create_executor, create_ceo
from memory.vector_memory import vector_memory
from orchestration.advanced_orchestrator import run_advanced_pipeline

def main():
    """Example usage of Local O1 components"""
    print("🚀 Local O1 Integration Example")
    
    # Simple agent usage
    ceo = create_ceo()
    plan = ceo.run("Break down the task of building a web application")
    print(f"CEO Plan: {plan}")
    
    # Vector memory usage
    vector_memory.add("test task", "test output")
    cached = vector_memory.retrieve("test task")
    print(f"Cached result: {cached}")
    
    # Advanced orchestration
    result = run_advanced_pipeline("Design a simple API")
    print(f"Orchestration result: {result}")

if __name__ == "__main__":
    main()
