"""
Simple test of Local O1 components without problematic imports
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

def test_basic_imports():
    """Test basic imports without running complex operations"""
    try:
        print("🚀 Testing Local O1 Basic Imports...")
        
        # Test config import
        from config.config import CEO_MODEL, FAST_MODEL
        print(f"✅ Config loaded - CEO Model: {CEO_MODEL}, Fast Model: {FAST_MODEL}")
        
        # Test basic agent classes
        from agents.agents import Agent, create_ceo
        print("✅ Agent classes imported successfully")
        
        # Test creating a simple agent
        ceo = create_ceo()
        print(f"✅ CEO agent created: {ceo.name}")
        
        # Test vector memory class (but don't initialize)
        from memory.vector_memory import VectorMemory
        print("✅ VectorMemory class imported successfully")
        
        print("\n🎉 All basic imports successful!")
        print("The Local O1 components are properly installed and accessible.")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
        
    return True

def test_simple_agent():
    """Test a simple agent operation that doesn't require Ollama"""
    try:
        print("\n🤖 Testing simple agent functionality...")
        from agents.agents import Agent
        
        # Create a simple agent
        agent = Agent("TestAgent", "phi3.5")
        print(f"✅ Created agent: {agent.name}")
        
        # Test basic agent properties
        print(f"✅ Agent model: {agent.model}")
        print(f"✅ Agent initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("LOCAL O1 SIMPLE INTEGRATION TEST")
    print("=" * 50)
    
    success = True
    success &= test_basic_imports()
    success &= test_simple_agent()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("Local O1 components are ready to use.")
        print("\nNext steps:")
        print("1. Ensure Ollama is installed and running")
        print("2. Pull the required models (phi3.5)")  
        print("3. Run full integration tests")
    else:
        print("❌ Some tests failed. Check the errors above.")
    print("=" * 50)
