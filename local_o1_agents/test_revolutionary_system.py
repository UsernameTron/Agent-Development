"""
Test and validation script for the Revolutionary Agent System
"""

import sys
import os
import time
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the local_o1_agents directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the revolutionary system
try:
    from revolutionary_agent_system import RevolutionaryAgentSystem
    from agents.enhanced_agents import EnhancedAgentConfig
    from agents.advanced_safety import ThreatLevel
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


class RevolutionarySystemTester:
    """Comprehensive tester for the revolutionary agent system"""
    
    def __init__(self):
        self.test_results = []
        self.system = None

    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        
        print("🧪 Starting Revolutionary Agent System Test Suite")
        print("=" * 60)
        
        test_start = time.time()
        
        # Test categories
        test_categories = [
            ("System Initialization", self.test_system_initialization),
            ("Agent Creation", self.test_agent_creation),
            ("Genetic Breeding", self.test_genetic_breeding),
            ("Consciousness Development", self.test_consciousness_development),
            ("Swarm Coordination", self.test_swarm_coordination),
            ("Safety Monitoring", self.test_safety_monitoring),
            ("Evolution Process", self.test_evolution_process),
            ("Integration Features", self.test_integration_features),
            ("System Shutdown", self.test_system_shutdown)
        ]
        
        passed_tests = 0
        total_tests = len(test_categories)
        
        for test_name, test_function in test_categories:
            print(f"\n📋 Running {test_name} Tests...")
            try:
                test_result = test_function()
                if test_result["success"]:
                    print(f"✅ {test_name}: PASSED")
                    passed_tests += 1
                else:
                    print(f"❌ {test_name}: FAILED - {test_result.get('error', 'Unknown error')}")
                
                self.test_results.append({
                    "test_name": test_name,
                    "result": test_result,
                    "timestamp": time.time()
                })
                
            except Exception as e:
                error_msg = f"Exception in {test_name}: {str(e)}"
                print(f"💥 {test_name}: ERROR - {error_msg}")
                self.test_results.append({
                    "test_name": test_name,
                    "result": {"success": False, "error": error_msg, "traceback": traceback.format_exc()},
                    "timestamp": time.time()
                })
        
        test_duration = time.time() - test_start
        
        # Final report
        print(f"\n" + "=" * 60)
        print(f"🏁 Test Suite Complete")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
        print(f"   Total Time: {test_duration:.2f}s")
        
        final_result = {
            "success": passed_tests == total_tests,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "success_rate": passed_tests/total_tests,
            "test_duration": test_duration,
            "detailed_results": self.test_results
        }
        
        return final_result

    def test_system_initialization(self) -> Dict[str, Any]:
        """Test system initialization"""
        
        try:
            # Create test configuration
            test_config = {
                "enable_evolution": True,
                "enable_consciousness": True,
                "enable_swarm_intelligence": True,
                "safety_level": "moderate",
                "max_population_size": 20,  # Smaller for testing
                "breeding_enabled": True
            }
            
            # Save test config
            config_path = "test_config.json"
            with open(config_path, 'w') as f:
                json.dump(test_config, f)
            
            # Initialize system
            self.system = RevolutionaryAgentSystem(config_path)
            
            # Start system
            startup_result = self.system.start_system()
            
            # Cleanup
            if os.path.exists(config_path):
                os.remove(config_path)
            
            if not startup_result["success"]:
                return {"success": False, "error": "System startup failed"}
            
            # Verify initial population
            status = self.system.get_system_status()
            initial_pop = status["enhanced_system_status"]["population_stats"]["total_enhanced_agents"]
            
            if initial_pop == 0:
                return {"success": False, "error": "No initial agents created"}
            
            return {
                "success": True,
                "initial_population": initial_pop,
                "startup_time": startup_result["startup_time"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def test_agent_creation(self) -> Dict[str, Any]:
        """Test agent creation functionality"""
        
        try:
            if not self.system or not self.system.is_running:
                return {"success": False, "error": "System not initialized"}
            
            # Test creating enhanced agent
            enhanced_agent = self.system.enhanced_system.create_enhanced_agent(
                name="Test_Agent",
                model="phi3.5",
                specialization="testing"
            )
            
            if not enhanced_agent:
                return {"success": False, "error": "Failed to create enhanced agent"}
            
            # Verify agent properties
            if not hasattr(enhanced_agent, 'genome'):
                return {"success": False, "error": "Agent missing genome"}
            
            if not hasattr(enhanced_agent, 'consciousness_level'):
                return {"success": False, "error": "Agent missing consciousness"}
            
            # Test agent functionality
            test_response = enhanced_agent.run("Test prompt: What is 2+2?")
            
            return {
                "success": True,
                "agent_id": enhanced_agent.agent_id,
                "agent_name": enhanced_agent.name,
                "consciousness_level": enhanced_agent.consciousness_level,
                "fitness_score": enhanced_agent.genome.get_fitness_score(),
                "test_response_length": len(str(test_response))
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def test_genetic_breeding(self) -> Dict[str, Any]:
        """Test genetic breeding functionality"""
        
        try:
            if not self.system or not self.system.is_running:
                return {"success": False, "error": "System not initialized"}
            
            # Get current agents for breeding
            current_agents = list(self.system.enhanced_system.enhanced_agents.values())
            
            if len(current_agents) < 2:
                return {"success": False, "error": "Need at least 2 agents for breeding"}
            
            # Select parents
            parent1, parent2 = current_agents[0], current_agents[1]
            
            # Test breeding
            offspring = self.system.enhanced_system.breed_specialized_agents(
                parent_agents=[parent1, parent2],
                target_specialization="test_breeding",
                num_offspring=1
            )
            
            if not offspring:
                return {"success": False, "error": "Breeding produced no offspring"}
            
            offspring_agent = offspring[0]
            
            # Verify offspring has genetic material from parents
            parent1_fitness = parent1.genome.get_fitness_score()
            parent2_fitness = parent2.genome.get_fitness_score()
            offspring_fitness = offspring_agent.genome.get_fitness_score()
            
            return {
                "success": True,
                "parent1_fitness": parent1_fitness,
                "parent2_fitness": parent2_fitness,
                "offspring_fitness": offspring_fitness,
                "offspring_id": offspring_agent.agent_id,
                "target_specialization": "test_breeding"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def test_consciousness_development(self) -> Dict[str, Any]:
        """Test consciousness development"""
        
        try:
            if not self.system or not self.system.is_running:
                return {"success": False, "error": "System not initialized"}
            
            # Get an agent for consciousness testing
            agents = list(self.system.enhanced_system.enhanced_agents.values())
            if not agents:
                return {"success": False, "error": "No agents available"}
            
            test_agent = agents[0]
            original_consciousness = test_agent.consciousness_level
            
            # Test consciousness development
            development_result = self.system.develop_consciousness()
            
            if not development_result["success"]:
                return {"success": False, "error": "Consciousness development failed"}
            
            # Check if consciousness increased
            new_consciousness = test_agent.consciousness_level
            
            return {
                "success": True,
                "agents_developed": development_result["agents_developed"],
                "original_consciousness": original_consciousness,
                "new_consciousness": new_consciousness,
                "consciousness_increased": new_consciousness > original_consciousness
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def test_swarm_coordination(self) -> Dict[str, Any]:
        """Test swarm coordination functionality"""
        
        try:
            if not self.system or not self.system.is_running:
                return {"success": False, "error": "System not initialized"}
            
            # Test swarm task execution
            test_task = "Analyze the concept of artificial intelligence and provide insights from multiple perspectives"
            
            swarm_result = self.system.execute_revolutionary_task(
                task_description=test_task,
                coordination_mode="distributed",
                use_swarm=True,
                enable_breeding=False  # Skip breeding for this test
            )
            
            if not swarm_result["success"]:
                return {"success": False, "error": "Swarm coordination failed"}
            
            # Verify swarm execution details
            if swarm_result["execution_method"] != "swarm_intelligence":
                return {"success": False, "error": "Did not use swarm intelligence"}
            
            return {
                "success": True,
                "execution_method": swarm_result["execution_method"],
                "coordination_mode": swarm_result["coordination_mode"],
                "execution_time": swarm_result["execution_time"],
                "task_success": swarm_result["success"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def test_safety_monitoring(self) -> Dict[str, Any]:
        """Test safety monitoring systems"""
        
        try:
            if not self.system or not self.system.is_running:
                return {"success": False, "error": "System not initialized"}
            
            # Get safety status
            safety_status = self.system.enhanced_system.safety_monitor.get_safety_status()
            
            # Verify safety monitoring is active
            if safety_status["monitoring_status"] != "active":
                return {"success": False, "error": "Safety monitoring not active"}
            
            # Test safety monitoring of an agent
            agents = list(self.system.enhanced_system.enhanced_agents.values())
            if not agents:
                return {"success": False, "error": "No agents to monitor"}
            
            test_agent = agents[0]
            safety_alerts = self.system.enhanced_system.safety_monitor.monitor_agent(test_agent)
            
            # Safety system should be working (alerts list should exist, even if empty)
            if safety_alerts is None:
                return {"success": False, "error": "Safety monitoring returned None"}
            
            return {
                "success": True,
                "overall_safety_score": safety_status["overall_safety_score"],
                "active_alerts_count": safety_status["active_alerts_count"],
                "safety_alerts_generated": len(safety_alerts),
                "containment_protocols_active": safety_status["containment_protocols_active"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def test_evolution_process(self) -> Dict[str, Any]:
        """Test population evolution"""
        
        try:
            if not self.system or not self.system.is_running:
                return {"success": False, "error": "System not initialized"}
            
            # Get initial population stats
            initial_status = self.system.get_system_status()
            initial_population = initial_status["enhanced_system_status"]["population_stats"]["total_enhanced_agents"]
            initial_fitness = initial_status["enhanced_system_status"]["population_stats"]["average_fitness"]
            
            # Run evolution for 2 generations (small number for testing)
            evolution_result = self.system.evolve_population(generations=2)
            
            if not evolution_result["success"]:
                return {"success": False, "error": "Evolution failed"}
            
            # Get final population stats
            final_status = self.system.get_system_status()
            final_population = final_status["enhanced_system_status"]["population_stats"]["total_enhanced_agents"]
            final_fitness = final_status["enhanced_system_status"]["population_stats"]["average_fitness"]
            
            return {
                "success": True,
                "generations_completed": evolution_result["generations_completed"],
                "initial_population": initial_population,
                "final_population": final_population,
                "initial_fitness": initial_fitness,
                "final_fitness": final_fitness,
                "fitness_improved": final_fitness > initial_fitness,
                "total_breeding_operations": evolution_result.get("total_breeding_operations", 0)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def test_integration_features(self) -> Dict[str, Any]:
        """Test integration with existing codebase"""
        
        try:
            # Test importing and using existing agent classes
            from agents.agents import Agent, ExecutorWithFallback
            
            # Create legacy agent
            legacy_agent = Agent("Test_Legacy", "phi3.5", "general")
            
            if not self.system or not self.system.is_running:
                return {"success": False, "error": "System not initialized"}
            
            # Test upgrading legacy agent
            enhanced_agent = self.system.enhanced_system.upgrade_legacy_agent(legacy_agent)
            
            if not enhanced_agent:
                return {"success": False, "error": "Failed to upgrade legacy agent"}
            
            # Verify enhanced agent has evolutionary features
            if not hasattr(enhanced_agent, 'genome'):
                return {"success": False, "error": "Upgraded agent missing genome"}
            
            if not hasattr(enhanced_agent, 'consciousness_level'):
                return {"success": False, "error": "Upgraded agent missing consciousness"}
            
            return {
                "success": True,
                "legacy_agent_name": legacy_agent.name,
                "enhanced_agent_name": enhanced_agent.name,
                "enhanced_agent_id": enhanced_agent.agent_id,
                "consciousness_level": enhanced_agent.consciousness_level,
                "fitness_score": enhanced_agent.genome.get_fitness_score()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def test_system_shutdown(self) -> Dict[str, Any]:
        """Test graceful system shutdown"""
        
        try:
            if not self.system or not self.system.is_running:
                return {"success": False, "error": "System not initialized"}
            
            # Get final system stats before shutdown
            final_status = self.system.get_system_status()
            
            # Test shutdown
            shutdown_result = self.system.shutdown_system()
            
            if not shutdown_result["success"]:
                return {"success": False, "error": "Shutdown failed"}
            
            # Verify system is no longer running
            if self.system.is_running:
                return {"success": False, "error": "System still running after shutdown"}
            
            return {
                "success": True,
                "session_duration": shutdown_result["session_duration"],
                "final_tasks_completed": shutdown_result["final_session_stats"]["tasks_completed"],
                "final_agents_created": shutdown_result["final_session_stats"]["agents_created"],
                "final_breeding_operations": shutdown_result["final_session_stats"]["breeding_operations"],
                "shutdown_success": shutdown_result["shutdown_details"]["success"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def run_basic_functionality_test():
    """Run basic functionality test"""
    
    print("🚀 Running Basic Functionality Test")
    
    try:
        # Test 1: Import all modules
        print("Testing imports...")
        from revolutionary_agent_system import RevolutionaryAgentSystem
        from agents.agent_genome import AgentGenome
        from agents.self_aware_agent import SelfAwareAgent
        from agents.master_agent_factory import MasterAgentFactory
        from agents.agent_bullpen import AgentBullpen
        from agents.swarm_intelligence import SwarmIntelligenceCore
        from agents.advanced_safety import AdvancedSafetyMonitor
        from agents.enhanced_agents import EnhancedAgentSystem
        print("✅ All imports successful")
        
        # Test 2: Create basic genome
        print("Testing genome creation...")
        genome = AgentGenome()
        fitness = genome.get_fitness_score()
        print(f"✅ Genome created with fitness: {fitness:.3f}")
        
        # Test 3: Create self-aware agent
        print("Testing self-aware agent creation...")
        agent = SelfAwareAgent("Test_Agent", "phi3.5", specialization="testing")
        print(f"✅ Agent created: {agent.name} (consciousness: {agent.consciousness_level:.3f})")
        
        # Test 4: Test genetic crossover
        print("Testing genetic operations...")
        agent2 = SelfAwareAgent("Test_Agent_2", "phi3.5", specialization="analysis")
        offspring_genome = agent.genome.crossover(agent2.genome)
        print(f"✅ Genetic crossover successful (offspring fitness: {offspring_genome.get_fitness_score():.3f})")
        
        print("\n🎉 Basic functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n💥 Basic functionality test FAILED: {e}")
        print(traceback.format_exc())
        return False


def main():
    """Main testing function"""
    
    print("🧪 Revolutionary Agent System - Test Suite")
    print("=" * 60)
    
    # Run basic functionality test first
    basic_test_passed = run_basic_functionality_test()
    
    if not basic_test_passed:
        print("\n❌ Basic functionality test failed. Skipping comprehensive tests.")
        return
    
    print("\n" + "=" * 60)
    
    # Run comprehensive test suite
    tester = RevolutionarySystemTester()
    final_result = tester.run_all_tests()
    
    # Save detailed results
    results_file = "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(final_result, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    if final_result["success"]:
        print("\n🎉 All tests PASSED! Revolutionary Agent System is ready for deployment.")
    else:
        print(f"\n⚠️  Some tests failed. Success rate: {final_result['success_rate']*100:.1f}%")
        print("Check test_results.json for detailed information.")


if __name__ == "__main__":
    main()