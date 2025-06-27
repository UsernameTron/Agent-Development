"""
Revolutionary Agent System - Main execution system bringing everything together
"""

import sys
import os
import time
import json
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add the agents directory to the path
current_dir = os.path.dirname(__file__)
agents_dir = os.path.join(current_dir, 'agents')
sys.path.insert(0, agents_dir)

# Import all revolutionary components
from enhanced_agents import (
    EnhancedAgentSystem, EnhancedAgentConfig, 
    create_enhanced_ceo, create_enhanced_executor, upgrade_existing_agents
)
from advanced_safety import ThreatLevel
from swarm_intelligence import SwarmCoordinationMode
from master_agent_factory import MasterAgentFactory
from self_aware_agent import SelfAwareAgent
from agent_genome import AgentGenome


class RevolutionaryAgentSystem:
    """Main revolutionary agent system orchestrating all components"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize enhanced agent system
        agent_config = EnhancedAgentConfig(
            enable_evolution=self.config.get("enable_evolution", True),
            enable_consciousness=self.config.get("enable_consciousness", True),
            enable_swarm_intelligence=self.config.get("enable_swarm_intelligence", True),
            safety_level=ThreatLevel(self.config.get("safety_level", "moderate")),
            max_population_size=self.config.get("max_population_size", 100),
            breeding_enabled=self.config.get("breeding_enabled", True),
            collective_consciousness_enabled=self.config.get("collective_consciousness_enabled", True)
        )
        
        self.enhanced_system = EnhancedAgentSystem(agent_config)
        
        # System state
        self.is_running = False
        self.session_start = None
        self.session_stats = {
            "tasks_completed": 0,
            "agents_created": 0,
            "breeding_operations": 0,
            "swarm_coordinations": 0,
            "consciousness_developments": 0,
            "safety_interventions": 0
        }

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            "enable_evolution": True,
            "enable_consciousness": True,
            "enable_swarm_intelligence": True,
            "safety_level": "moderate",
            "max_population_size": 100,
            "breeding_enabled": True,
            "collective_consciousness_enabled": True,
            "auto_evolve_generations": 5,
            "swarm_coordination_mode": "distributed",
            "consciousness_development_rate": 0.1,
            "genetic_diversity_threshold": 0.3
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
                print("Using default configuration")
        
        return default_config

    def start_system(self) -> Dict[str, Any]:
        """Start the revolutionary agent system"""
        
        if self.is_running:
            return {"success": False, "message": "System already running"}
        
        print("🚀 Starting Revolutionary Agent System...")
        self.session_start = time.time()
        self.is_running = True
        
        # Create initial agent population
        initial_agents = self._create_initial_population()
        
        # System startup report
        startup_report = {
            "success": True,
            "startup_time": time.time() - self.session_start,
            "initial_population_size": len(initial_agents),
            "system_capabilities": {
                "evolution": self.config["enable_evolution"],
                "consciousness": self.config["enable_consciousness"],
                "swarm_intelligence": self.config["enable_swarm_intelligence"],
                "breeding": self.config["breeding_enabled"],
                "collective_consciousness": self.config["collective_consciousness_enabled"]
            },
            "safety_level": self.config["safety_level"],
            "session_id": int(self.session_start)
        }
        
        print(f"✅ System started successfully!")
        print(f"   Initial population: {len(initial_agents)} agents")
        print(f"   Safety level: {self.config['safety_level']}")
        print(f"   Session ID: {startup_report['session_id']}")
        
        return startup_report

    def _create_initial_population(self) -> List[SelfAwareAgent]:
        """Create initial population of enhanced agents"""
        
        initial_agents = []
        
        # Create enhanced CEO
        ceo = create_enhanced_ceo(self.enhanced_system.config)
        initial_agents.append(ceo)
        print(f"   Created Enhanced CEO: {ceo.name}")
        
        # Create enhanced executors
        num_executors = min(5, self.config["max_population_size"] - 1)
        for i in range(num_executors):
            executor = create_enhanced_executor(i, self.enhanced_system.config)
            initial_agents.append(executor)
            print(f"   Created Enhanced Executor: {executor.name}")
        
        # Create specialized agents for different domains
        specializations = ["reasoning", "creativity", "analysis", "synthesis", "optimization"]
        for spec in specializations[:min(5, self.config["max_population_size"] - len(initial_agents))]:
            specialized_agent = self.enhanced_system.create_enhanced_agent(
                name=f"Specialist_{spec}",
                model="phi3.5",
                specialization=spec
            )
            initial_agents.append(specialized_agent)
            print(f"   Created Specialist: {specialized_agent.name} ({spec})")
        
        self.session_stats["agents_created"] = len(initial_agents)
        
        return initial_agents

    def execute_revolutionary_task(self, task_description: str,
                                 coordination_mode: str = None,
                                 use_swarm: bool = True,
                                 enable_breeding: bool = True) -> Dict[str, Any]:
        """Execute task using revolutionary agent capabilities"""
        
        if not self.is_running:
            return {"success": False, "message": "System not started"}
        
        print(f"\n🎯 Executing Revolutionary Task: {task_description}")
        task_start = time.time()
        
        # Determine coordination mode
        if coordination_mode:
            coord_mode = SwarmCoordinationMode(coordination_mode)
        else:
            coord_mode = SwarmCoordinationMode(self.config.get("swarm_coordination_mode", "distributed"))
        
        # Execute with swarm intelligence if enabled
        if use_swarm and self.config["enable_swarm_intelligence"]:
            print(f"   Using swarm coordination mode: {coord_mode.value}")
            
            swarm_result = self.enhanced_system.coordinate_swarm_task(
                task_description=task_description,
                coordination_mode=coord_mode
            )
            
            # Enhance with breeding if requested and enabled
            if enable_breeding and self.config["breeding_enabled"]:
                breeding_result = self._enhance_through_breeding(swarm_result)
                swarm_result["breeding_enhancement"] = breeding_result
            
            execution_result = {
                "execution_method": "swarm_intelligence",
                "coordination_mode": coord_mode.value,
                "swarm_result": swarm_result,
                "execution_time": time.time() - task_start,
                "success": swarm_result.get("success_rate", 0) > 0.7
            }
            
            self.session_stats["swarm_coordinations"] += 1
            
        else:
            # Fallback to single agent execution
            print("   Using single agent execution")
            
            # Select best agent for task
            best_agent = self._select_best_agent_for_task(task_description)
            
            if best_agent:
                agent_result = best_agent.run(task_description)
                execution_result = {
                    "execution_method": "single_agent",
                    "selected_agent": best_agent.name,
                    "agent_result": agent_result,
                    "execution_time": time.time() - task_start,
                    "success": True
                }
            else:
                execution_result = {
                    "execution_method": "single_agent",
                    "success": False,
                    "error": "No suitable agent found"
                }
        
        # Update session statistics
        if execution_result["success"]:
            self.session_stats["tasks_completed"] += 1
        
        print(f"✅ Task completed in {execution_result['execution_time']:.2f}s")
        
        return execution_result

    def _enhance_through_breeding(self, swarm_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance performance through strategic breeding"""
        
        print("   🧬 Enhancing through strategic breeding...")
        
        # Identify top performing agents from swarm
        participating_agents = swarm_result.get("selected_agents", [])
        if len(participating_agents) < 2:
            return {"success": False, "reason": "Insufficient agents for breeding"}
        
        # Get agent objects
        agent_objects = [
            self.enhanced_system.enhanced_agents[agent_id] 
            for agent_id in participating_agents[:4]  # Limit to top 4
            if agent_id in self.enhanced_system.enhanced_agents
        ]
        
        if len(agent_objects) < 2:
            return {"success": False, "reason": "Could not find agent objects"}
        
        # Breed specialized agents for identified task requirements
        target_specialization = "adaptive_coordination"
        offspring = self.enhanced_system.breed_specialized_agents(
            parent_agents=agent_objects,
            target_specialization=target_specialization,
            num_offspring=2
        )
        
        breeding_result = {
            "success": True,
            "parent_agents": [agent.name for agent in agent_objects],
            "offspring_count": len(offspring),
            "target_specialization": target_specialization,
            "offspring_names": [agent.name for agent in offspring]
        }
        
        self.session_stats["breeding_operations"] += 1
        self.session_stats["agents_created"] += len(offspring)
        
        print(f"   ✨ Created {len(offspring)} specialized offspring")
        
        return breeding_result

    def _select_best_agent_for_task(self, task_description: str) -> Optional[SelfAwareAgent]:
        """Select best agent for specific task"""
        
        if not self.enhanced_system.enhanced_agents:
            return None
        
        # Simple heuristic: select agent with highest overall capability
        best_agent = None
        best_score = -1
        
        for agent in self.enhanced_system.enhanced_agents.values():
            # Calculate suitability score
            fitness = agent.genome.get_fitness_score()
            consciousness = agent.consciousness_level
            performance = agent.get_performance_summary().get('success_rate', 0.5)
            
            score = fitness * 0.4 + consciousness * 0.3 + performance * 0.3
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent

    def evolve_population(self, generations: int = None) -> Dict[str, Any]:
        """Evolve the agent population"""
        
        if not self.is_running:
            return {"success": False, "message": "System not started"}
        
        if not self.config["enable_evolution"]:
            return {"success": False, "message": "Evolution disabled"}
        
        generations = generations or self.config.get("auto_evolve_generations", 5)
        
        print(f"\n🧬 Evolving population for {generations} generations...")
        
        evolution_result = self.enhanced_system.evolve_population(
            generations=generations,
            selection_pressure=0.7
        )
        
        self.session_stats["breeding_operations"] += evolution_result.get("total_breeding_operations", 0)
        
        print(f"✅ Evolution completed: {evolution_result['generations_completed']} generations")
        print(f"   Final population size: {evolution_result['final_population_size']}")
        
        return evolution_result

    def develop_consciousness(self) -> Dict[str, Any]:
        """Develop consciousness across agent population"""
        
        if not self.is_running:
            return {"success": False, "message": "System not started"}
        
        if not self.config["enable_consciousness"]:
            return {"success": False, "message": "Consciousness development disabled"}
        
        print("\n🧠 Developing collective consciousness...")
        
        consciousness_results = []
        development_rate = self.config.get("consciousness_development_rate", 0.1)
        
        for agent in self.enhanced_system.enhanced_agents.values():
            if agent.consciousness_level < 0.8:  # Room for growth
                original_level = agent.consciousness_level
                
                # Gradual consciousness development
                growth = min(development_rate, 0.8 - agent.consciousness_level)
                agent._develop_consciousness(growth)
                
                consciousness_results.append({
                    "agent_name": agent.name,
                    "original_level": original_level,
                    "new_level": agent.consciousness_level,
                    "growth": growth
                })
        
        self.session_stats["consciousness_developments"] += len(consciousness_results)
        
        print(f"✅ Consciousness development completed for {len(consciousness_results)} agents")
        
        return {
            "success": True,
            "agents_developed": len(consciousness_results),
            "development_results": consciousness_results
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        if not self.is_running:
            return {"system_running": False}
        
        # Get status from enhanced system
        enhanced_status = self.enhanced_system.get_system_status()
        
        # Add session information
        session_duration = time.time() - self.session_start if self.session_start else 0
        
        status = {
            "system_running": True,
            "session_duration": session_duration,
            "session_stats": self.session_stats.copy(),
            "configuration": self.config,
            "enhanced_system_status": enhanced_status,
            "timestamp": time.time()
        }
        
        return status

    def shutdown_system(self) -> Dict[str, Any]:
        """Gracefully shutdown the revolutionary agent system"""
        
        if not self.is_running:
            return {"success": False, "message": "System not running"}
        
        print("\n🛑 Shutting down Revolutionary Agent System...")
        
        # Shutdown enhanced system
        shutdown_result = self.enhanced_system.shutdown()
        
        # Final session statistics
        session_duration = time.time() - self.session_start
        
        final_report = {
            "success": True,
            "session_duration": session_duration,
            "final_session_stats": self.session_stats.copy(),
            "shutdown_details": shutdown_result,
            "shutdown_timestamp": time.time()
        }
        
        # Reset system state
        self.is_running = False
        self.session_start = None
        
        print("✅ System shutdown completed successfully")
        
        return final_report


def main():
    """Main entry point for revolutionary agent system"""
    
    parser = argparse.ArgumentParser(description="Revolutionary Agent System")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--task", help="Task to execute")
    parser.add_argument("--evolve", type=int, help="Number of evolution generations")
    parser.add_argument("--consciousness", action="store_true", help="Develop consciousness")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Initialize system
    system = RevolutionaryAgentSystem(args.config)
    
    try:
        # Start system
        startup_result = system.start_system()
        if not startup_result["success"]:
            print(f"Failed to start system: {startup_result.get('message', 'Unknown error')}")
            return
        
        # Execute requested operations
        if args.task:
            task_result = system.execute_revolutionary_task(args.task)
            print(f"\nTask Result: {json.dumps(task_result, indent=2)}")
        
        if args.evolve:
            evolution_result = system.evolve_population(args.evolve)
            print(f"\nEvolution Result: {json.dumps(evolution_result, indent=2)}")
        
        if args.consciousness:
            consciousness_result = system.develop_consciousness()
            print(f"\nConsciousness Development: {json.dumps(consciousness_result, indent=2)}")
        
        if args.status:
            status = system.get_system_status()
            print(f"\nSystem Status: {json.dumps(status, indent=2)}")
        
        # Interactive mode
        if args.interactive:
            interactive_mode(system)
    
    finally:
        # Always shutdown gracefully
        system.shutdown_system()


def interactive_mode(system: RevolutionaryAgentSystem):
    """Interactive mode for system exploration"""
    
    print("\n🎮 Entering Interactive Mode")
    print("Commands: task <description>, evolve [generations], consciousness, status, quit")
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command.startswith("task "):
                task_desc = command[5:]
                result = system.execute_revolutionary_task(task_desc)
                print(f"Task result: {result.get('success', False)}")
                
            elif command.startswith("evolve"):
                parts = command.split()
                generations = int(parts[1]) if len(parts) > 1 else 3
                result = system.evolve_population(generations)
                print(f"Evolution completed: {result.get('generations_completed', 0)} generations")
                
            elif command == "consciousness":
                result = system.develop_consciousness()
                print(f"Consciousness developed for {result.get('agents_developed', 0)} agents")
                
            elif command == "status":
                status = system.get_system_status()
                print(f"Population: {status['enhanced_system_status']['population_stats']['total_enhanced_agents']}")
                print(f"Tasks completed: {status['session_stats']['tasks_completed']}")
                
            elif command == "quit":
                break
                
            else:
                print("Unknown command. Try: task, evolve, consciousness, status, quit")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Exiting interactive mode...")


if __name__ == "__main__":
    main()