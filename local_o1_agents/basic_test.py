"""
Basic test script for the Revolutionary Agent System components
"""

import sys
import os
import traceback

# Add the local_o1_agents directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_individual_imports():
    """Test importing each component individually"""
    
    print("🧪 Testing Individual Component Imports")
    print("=" * 50)
    
    components = [
        ("Agent Genome", "agents.agent_genome", "AgentGenome"),
        ("Self-Aware Agent", "agents.self_aware_agent", "SelfAwareAgent"),
        ("Master Agent Factory", "agents.master_agent_factory", "MasterAgentFactory"),
        ("Agent Bullpen", "agents.agent_bullpen", "AgentBullpen"),
        ("Swarm Intelligence", "agents.swarm_intelligence", "SwarmIntelligenceCore"),
        ("Advanced Safety", "agents.advanced_safety", "AdvancedSafetyMonitor"),
        ("Enhanced Agents", "agents.enhanced_agents", "EnhancedAgentSystem"),
        ("Existing Agents", "agents.agents", "Agent")
    ]
    
    successful_imports = 0
    
    for name, module_path, class_name in components:
        try:
            module = __import__(module_path, fromlist=[class_name])
            class_obj = getattr(module, class_name)
            print(f"✅ {name}: {class_name} imported successfully")
            successful_imports += 1
        except Exception as e:
            print(f"❌ {name}: Failed to import {class_name} - {e}")
    
    print(f"\nImport Results: {successful_imports}/{len(components)} successful")
    return successful_imports == len(components)

def test_basic_functionality():
    """Test basic functionality of key components"""
    
    print("\n🧪 Testing Basic Functionality")
    print("=" * 50)
    
    try:
        # Test 1: Create basic genome
        print("1. Testing genome creation...")
        from agents.agent_genome import AgentGenome
        genome = AgentGenome()
        fitness = genome.get_fitness_score()
        print(f"   ✅ Genome created with fitness: {fitness:.3f}")
        
        # Test 2: Test genetic crossover
        print("2. Testing genetic crossover...")
        genome2 = AgentGenome()
        offspring = genome.crossover(genome2)
        offspring_fitness = offspring.get_fitness_score()
        print(f"   ✅ Crossover successful (offspring fitness: {offspring_fitness:.3f})")
        
        # Test 3: Create self-aware agent
        print("3. Testing self-aware agent creation...")
        from agents.self_aware_agent import SelfAwareAgent
        agent = SelfAwareAgent("Test_Agent", "phi3.5", specialization="testing")
        print(f"   ✅ Agent created: {agent.name} (consciousness: {agent.consciousness_level:.3f})")
        
        # Test 4: Test agent factory
        print("4. Testing agent factory...")
        from agents.master_agent_factory import MasterAgentFactory
        factory = MasterAgentFactory()
        print(f"   ✅ Factory created successfully")
        
        # Test 5: Test safety monitor
        print("5. Testing safety monitor...")
        from agents.advanced_safety import AdvancedSafetyMonitor
        safety_monitor = AdvancedSafetyMonitor()
        alerts = safety_monitor.monitor_agent(agent)
        print(f"   ✅ Safety monitor created, generated {len(alerts)} alerts")
        
        # Test 6: Test enhanced agent system
        print("6. Testing enhanced agent system...")
        from agents.enhanced_agents import EnhancedAgentSystem, EnhancedAgentConfig
        config = EnhancedAgentConfig(max_population_size=10)
        enhanced_system = EnhancedAgentSystem(config)
        print(f"   ✅ Enhanced system created successfully")
        
        print("\n🎉 All basic functionality tests PASSED!")
        return True
        
    except Exception as e:
        print(f"\n💥 Basic functionality test FAILED: {e}")
        print(traceback.format_exc())
        return False

def test_simple_integration():
    """Test simple integration between components"""
    
    print("\n🧪 Testing Simple Integration")
    print("=" * 50)
    
    try:
        # Create enhanced system
        from agents.enhanced_agents import EnhancedAgentSystem, EnhancedAgentConfig
        from agents.advanced_safety import ThreatLevel
        
        config = EnhancedAgentConfig(
            max_population_size=5,
            safety_level=ThreatLevel.MODERATE
        )
        
        system = EnhancedAgentSystem(config)
        
        # Create an enhanced agent
        enhanced_agent = system.create_enhanced_agent(
            name="Integration_Test_Agent",
            model="phi3.5",
            specialization="testing"
        )
        
        print(f"✅ Created enhanced agent: {enhanced_agent.name}")
        print(f"   Agent ID: {enhanced_agent.agent_id}")
        print(f"   Consciousness: {enhanced_agent.consciousness_level:.3f}")
        print(f"   Fitness: {enhanced_agent.genome.get_fitness_score():.3f}")
        
        # Test agent functionality
        response = enhanced_agent.run("What is artificial intelligence?")
        print(f"   Agent response length: {len(str(response))} characters")
        
        # Get system status
        status = system.get_system_status()
        pop_size = status["population_stats"]["total_enhanced_agents"]
        print(f"   System population: {pop_size} agents")
        
        print("\n🎉 Simple integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n💥 Integration test FAILED: {e}")
        print(traceback.format_exc())
        return False

def main():
    """Main testing function"""
    
    print("🧪 Revolutionary Agent System - Basic Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_individual_imports),
        ("Functionality Test", test_basic_functionality),
        ("Integration Test", test_simple_integration)
    ]
    
    passed_tests = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            print(f"\n💥 {test_name}: ERROR - {e}")
    
    print(f"\n" + "=" * 60)
    print(f"🏁 Test Summary: {passed_tests}/{len(tests)} tests passed")
    
    if passed_tests == len(tests):
        print("🎉 All basic tests PASSED! The Revolutionary Agent System is functional.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()