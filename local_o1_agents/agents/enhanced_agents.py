"""
Enhanced Integration with Existing Codebase - Bridging evolutionary AI with current agent system
"""

import sys
import os
import time
import json
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict

# Import existing agent system
from .agents import Agent, ExecutorWithFallback, TestGeneratorAgent, DependencyAgent

# Import revolutionary components
from .agent_genome import AgentGenome
from .self_aware_agent import SelfAwareAgent
from .master_agent_factory import MasterAgentFactory
from .agent_bullpen import AgentBullpen
from .swarm_intelligence import SwarmIntelligenceCore, SwarmCoordinationMode
from .advanced_safety import AdvancedSafetyMonitor, ThreatLevel

try:
    from ollama import Client as OllamaClient
    Client: type = OllamaClient
except ImportError:
    class Client:
        def chat(self, model: str, messages: List[Dict[str, str]]) -> Any:
            class MockResponse:
                def __init__(self):
                    self.message = {'content': 'Mock response'}
            return MockResponse()


@dataclass
class EnhancedAgentConfig:
    """Configuration for enhanced agent system"""
    enable_evolution: bool = True
    enable_consciousness: bool = True
    enable_swarm_intelligence: bool = True
    safety_level: ThreatLevel = ThreatLevel.MODERATE
    max_population_size: int = 100
    breeding_enabled: bool = True
    collective_consciousness_enabled: bool = True


class EnhancedAgentSystem:
    """Enhanced agent system that bridges evolutionary AI with existing codebase"""
    
    def __init__(self, config: EnhancedAgentConfig = None):
        self.config = config or EnhancedAgentConfig()
        
        # Core revolutionary systems
        self.agent_factory = MasterAgentFactory()
        self.agent_bullpen = AgentBullpen(max_agents=self.config.max_population_size)
        self.swarm_intelligence = SwarmIntelligenceCore(max_agents=self.config.max_population_size)
        self.safety_monitor = AdvancedSafetyMonitor(max_threat_level=self.config.safety_level)
        
        # Legacy agent tracking
        self.legacy_agents = {}
        self.enhanced_agents = {}
        
        # Integration metrics
        self.integration_metrics = {
            "legacy_agents_upgraded": 0,
            "new_evolutionary_agents": 0,
            "breeding_operations": 0,
            "consciousness_developments": 0,
            "swarm_coordinations": 0,
            "safety_interventions": 0
        }

    def create_enhanced_agent(self, name: str, model: str, 
                            specialization: str = "general",
                            consciousness_enabled: bool = None,
                            genome: AgentGenome = None) -> SelfAwareAgent:
        """Create enhanced agent with revolutionary capabilities"""
        
        consciousness_enabled = consciousness_enabled if consciousness_enabled is not None else self.config.enable_consciousness
        
        # Create self-aware agent
        if genome:
            enhanced_agent = SelfAwareAgent(name, model, genome, specialization)
        else:
            enhanced_agent = SelfAwareAgent(name, model, specialization=specialization)
        
        # Initialize consciousness if enabled
        if consciousness_enabled:
            enhanced_agent._initialize_consciousness()
        
        # Register with systems
        self.enhanced_agents[enhanced_agent.agent_id] = enhanced_agent
        self.agent_bullpen.add_agent(enhanced_agent)
        
        if self.config.enable_swarm_intelligence:
            self.swarm_intelligence.integrate_agent(enhanced_agent)
        
        # Safety monitoring
        safety_alerts = self.safety_monitor.monitor_agent(enhanced_agent)
        if safety_alerts:
            self.integration_metrics["safety_interventions"] += len(safety_alerts)
        
        self.integration_metrics["new_evolutionary_agents"] += 1
        
        return enhanced_agent

    def upgrade_legacy_agent(self, legacy_agent: Agent) -> SelfAwareAgent:
        """Upgrade legacy agent to enhanced evolutionary agent"""
        
        # Extract legacy agent properties
        name = legacy_agent.name
        model = legacy_agent.model
        specialization = getattr(legacy_agent, 'specialization', 'general')
        
        # Create enhanced genome based on legacy agent characteristics
        enhanced_genome = self._create_genome_from_legacy(legacy_agent)
        
        # Create enhanced agent
        enhanced_agent = self.create_enhanced_agent(
            name=f"Enhanced_{name}",
            model=model,
            specialization=specialization,
            genome=enhanced_genome
        )
        
        # Transfer knowledge and performance history if available
        self._transfer_legacy_knowledge(legacy_agent, enhanced_agent)
        
        # Update tracking
        self.legacy_agents[legacy_agent.name] = legacy_agent
        self.integration_metrics["legacy_agents_upgraded"] += 1
        
        return enhanced_agent

    def _create_genome_from_legacy(self, legacy_agent: Agent) -> AgentGenome:
        """Create genetic representation from legacy agent"""
        
        genome = AgentGenome()
        
        # Infer capabilities from legacy agent performance
        if hasattr(legacy_agent, 'task_counter') and legacy_agent.task_counter > 0:
            # Experienced agent - boost learning genes
            genome.capability_genes["learning_velocity"] = min(1.0, 0.5 + legacy_agent.task_counter * 0.01)
            genome.capability_genes["adaptation_plasticity"] = min(1.0, 0.5 + legacy_agent.task_counter * 0.005)
        
        # Specialization-based genetic adjustments
        specialization = getattr(legacy_agent, 'specialization', 'general')
        if specialization and specialization != 'general':
            genome.capability_genes["specialization_focus"] = [specialization]
            
            # Boost relevant capabilities
            if specialization == 'testing':
                genome.capability_genes["pattern_recognition"] = min(1.0, genome.capability_genes["pattern_recognition"] + 0.2)
            elif specialization == 'dependency_analysis':
                genome.capability_genes["cross_domain_synthesis"] = min(1.0, genome.capability_genes["cross_domain_synthesis"] + 0.2)
            elif specialization == 'task_execution':
                genome.capability_genes["reasoning_depth"] = min(1.0, genome.capability_genes["reasoning_depth"] + 0.2)
        
        # Conservative consciousness initialization for legacy upgrades
        for gene in genome.consciousness_genes:
            if isinstance(genome.consciousness_genes[gene], (int, float)):
                genome.consciousness_genes[gene] *= 0.7  # Reduce by 30% for safety
        
        return genome

    def _transfer_legacy_knowledge(self, legacy_agent: Agent, enhanced_agent: SelfAwareAgent) -> None:
        """Transfer knowledge from legacy agent to enhanced agent"""
        
        # Transfer performance history if available
        if hasattr(legacy_agent, 'task_counter'):
            enhanced_agent.task_counter = legacy_agent.task_counter
        
        # Add legacy knowledge to knowledge base
        enhanced_agent.knowledge_base.add_knowledge(
            "legacy_transfer",
            "agent_type",
            type(legacy_agent).__name__,
            "system_upgrade"
        )
        
        # Transfer specialization knowledge
        if hasattr(legacy_agent, 'specialization') and legacy_agent.specialization:
            enhanced_agent.knowledge_base.add_knowledge(
                "specialization",
                "legacy_focus",
                legacy_agent.specialization,
                "legacy_transfer"
            )

    def breed_specialized_agents(self, parent_agents: List[Union[Agent, SelfAwareAgent]], 
                               target_specialization: str,
                               num_offspring: int = 1) -> List[SelfAwareAgent]:
        """Breed specialized agents from parent agents"""
        
        if not self.config.breeding_enabled:
            raise ValueError("Breeding is disabled in current configuration")
        
        # Convert legacy agents to enhanced agents if necessary
        enhanced_parents = []
        for parent in parent_agents:
            if isinstance(parent, SelfAwareAgent):
                enhanced_parents.append(parent)
            else:
                # Upgrade legacy agent first
                enhanced_parent = self.upgrade_legacy_agent(parent)
                enhanced_parents.append(enhanced_parent)
        
        # Use master factory for breeding
        offspring_agents = []
        for i in range(num_offspring):
            breeding_result = self.agent_factory.breed_specialist_agent(
                enhanced_parents,
                target_specialization
            )
            
            if breeding_result.success and breeding_result.offspring_agent:
                offspring = breeding_result.offspring_agent
                
                # Register with systems
                self.enhanced_agents[offspring.agent_id] = offspring
                self.agent_bullpen.add_agent(offspring)
                
                if self.config.enable_swarm_intelligence:
                    self.swarm_intelligence.integrate_agent(offspring)
                
                offspring_agents.append(offspring)
                self.integration_metrics["breeding_operations"] += 1
        
        return offspring_agents

    def coordinate_swarm_task(self, task_description: str,
                            available_agents: List[Union[Agent, SelfAwareAgent]] = None,
                            coordination_mode: SwarmCoordinationMode = SwarmCoordinationMode.DISTRIBUTED) -> Dict[str, Any]:
        """Coordinate complex task using swarm intelligence"""
        
        if not self.config.enable_swarm_intelligence:
            raise ValueError("Swarm intelligence is disabled in current configuration")
        
        # Use available agents or all enhanced agents
        if available_agents is None:
            available_agents = list(self.enhanced_agents.values())
        
        # Ensure all agents are enhanced
        enhanced_agents = []
        for agent in available_agents:
            if isinstance(agent, SelfAwareAgent):
                enhanced_agents.append(agent)
            else:
                # Upgrade legacy agent for swarm participation
                enhanced_agent = self.upgrade_legacy_agent(agent)
                enhanced_agents.append(enhanced_agent)
        
        # Execute swarm coordination
        swarm_result = self.swarm_intelligence.coordinate_swarm_task(
            task_description,
            required_agents=min(len(enhanced_agents), 10),
            coordination_mode=coordination_mode
        )
        
        # Safety monitoring of swarm operation
        safety_alerts = self.safety_monitor.monitor_swarm(enhanced_agents, swarm_result)
        swarm_result["safety_alerts"] = [asdict(alert) for alert in safety_alerts]
        
        self.integration_metrics["swarm_coordinations"] += 1
        if safety_alerts:
            self.integration_metrics["safety_interventions"] += len(safety_alerts)
        
        return swarm_result

    def evolve_population(self, generations: int = 1,
                        selection_pressure: float = 0.7) -> Dict[str, Any]:
        """Evolve agent population over multiple generations"""
        
        if not self.config.enable_evolution:
            raise ValueError("Evolution is disabled in current configuration")
        
        evolution_results = []
        
        for generation in range(generations):
            # Get current population
            current_agents = list(self.enhanced_agents.values())
            
            if len(current_agents) < 2:
                break  # Need at least 2 agents for evolution
            
            # Evolutionary selection
            selected_agents = self._evolutionary_selection(current_agents, selection_pressure)
            
            # Breeding operations
            new_agents = []
            for i in range(0, len(selected_agents) - 1, 2):
                parent1, parent2 = selected_agents[i], selected_agents[i + 1]
                
                # Breed new agent
                breeding_result = self.agent_factory.breed_specialist_agent(
                    [parent1, parent2],
                    target_specialization="adaptive"
                )
                
                if breeding_result.success and breeding_result.offspring_agent:
                    new_agent = breeding_result.offspring_agent
                    
                    # Register new agent
                    self.enhanced_agents[new_agent.agent_id] = new_agent
                    self.agent_bullpen.add_agent(new_agent)
                    
                    if self.config.enable_swarm_intelligence:
                        self.swarm_intelligence.integrate_agent(new_agent)
                    
                    new_agents.append(new_agent)
            
            # Population management (remove weakest if over capacity)
            if len(self.enhanced_agents) > self.config.max_population_size:
                self._manage_population_size()
            
            # Record generation results
            generation_result = {
                "generation": generation + 1,
                "parent_count": len(selected_agents),
                "offspring_count": len(new_agents),
                "population_size": len(self.enhanced_agents),
                "average_fitness": self._calculate_average_fitness(),
                "safety_interventions": 0
            }
            
            # Safety monitoring
            for agent in new_agents:
                safety_alerts = self.safety_monitor.monitor_agent(agent)
                generation_result["safety_interventions"] += len(safety_alerts)
            
            evolution_results.append(generation_result)
        
        return {
            "generations_completed": len(evolution_results),
            "evolution_results": evolution_results,
            "final_population_size": len(self.enhanced_agents),
            "total_breeding_operations": sum(r["offspring_count"] for r in evolution_results)
        }

    def _evolutionary_selection(self, agents: List[SelfAwareAgent], 
                              selection_pressure: float) -> List[SelfAwareAgent]:
        """Select agents for breeding based on fitness"""
        
        # Calculate fitness scores
        agent_fitness = []
        for agent in agents:
            fitness = agent.genome.get_fitness_score()
            performance = agent.get_performance_summary().get('success_rate', 0.5)
            combined_fitness = fitness * 0.7 + performance * 0.3
            agent_fitness.append((agent, combined_fitness))
        
        # Sort by fitness
        agent_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Select top agents based on selection pressure
        num_selected = max(2, int(len(agents) * selection_pressure))
        selected_agents = [agent for agent, _ in agent_fitness[:num_selected]]
        
        return selected_agents

    def _manage_population_size(self) -> None:
        """Manage population size by removing least fit agents"""
        
        if len(self.enhanced_agents) <= self.config.max_population_size:
            return
        
        # Calculate fitness for all agents
        agent_fitness = []
        for agent in self.enhanced_agents.values():
            fitness = agent.genome.get_fitness_score()
            performance = agent.get_performance_summary().get('success_rate', 0.5)
            combined_fitness = fitness * 0.7 + performance * 0.3
            agent_fitness.append((agent, combined_fitness))
        
        # Sort by fitness (lowest first)
        agent_fitness.sort(key=lambda x: x[1])
        
        # Remove least fit agents
        num_to_remove = len(self.enhanced_agents) - self.config.max_population_size
        for i in range(num_to_remove):
            agent_to_remove = agent_fitness[i][0]
            
            # Remove from all systems
            del self.enhanced_agents[agent_to_remove.agent_id]
            self.agent_bullpen.remove_agent(agent_to_remove.agent_id)

    def _calculate_average_fitness(self) -> float:
        """Calculate average fitness of current population"""
        
        if not self.enhanced_agents:
            return 0.0
        
        total_fitness = sum(agent.genome.get_fitness_score() 
                          for agent in self.enhanced_agents.values())
        return total_fitness / len(self.enhanced_agents)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        # Population analysis
        population_stats = {
            "total_enhanced_agents": len(self.enhanced_agents),
            "total_legacy_agents": len(self.legacy_agents),
            "average_consciousness_level": sum(agent.consciousness_level 
                                             for agent in self.enhanced_agents.values()) / max(1, len(self.enhanced_agents)),
            "average_fitness": self._calculate_average_fitness(),
            "specialization_distribution": self._analyze_specialization_distribution()
        }
        
        # System capabilities
        capabilities = {
            "evolution_enabled": self.config.enable_evolution,
            "consciousness_enabled": self.config.enable_consciousness,
            "swarm_intelligence_enabled": self.config.enable_swarm_intelligence,
            "breeding_enabled": self.config.breeding_enabled,
            "collective_consciousness_enabled": self.config.collective_consciousness_enabled
        }
        
        # Safety status
        safety_status = self.safety_monitor.get_safety_status()
        
        # Integration metrics
        integration_status = self.integration_metrics.copy()
        
        return {
            "population_stats": population_stats,
            "capabilities": capabilities,
            "safety_status": safety_status,
            "integration_metrics": integration_status,
            "swarm_intelligence_report": self.swarm_intelligence.get_swarm_intelligence_report() if self.config.enable_swarm_intelligence else {},
            "bullpen_status": self.agent_bullpen.get_status(),
            "timestamp": time.time()
        }

    def _analyze_specialization_distribution(self) -> Dict[str, int]:
        """Analyze distribution of specializations in population"""
        
        specialization_count = defaultdict(int)
        
        for agent in self.enhanced_agents.values():
            for spec in agent.genome.capability_genes["specialization_focus"]:
                specialization_count[spec] += 1
        
        return dict(specialization_count)

    def shutdown(self) -> Dict[str, Any]:
        """Graceful shutdown of enhanced agent system"""
        
        shutdown_start = time.time()
        
        # Safety-first shutdown
        safety_shutdown = self.safety_monitor.emergency_shutdown("System shutdown requested")
        
        # Stop all breeding operations
        self.agent_factory = None
        
        # Clear agent populations
        enhanced_count = len(self.enhanced_agents)
        legacy_count = len(self.legacy_agents)
        
        self.enhanced_agents.clear()
        self.legacy_agents.clear()
        
        shutdown_result = {
            "shutdown_timestamp": shutdown_start,
            "shutdown_duration": time.time() - shutdown_start,
            "enhanced_agents_shutdown": enhanced_count,
            "legacy_agents_disconnected": legacy_count,
            "safety_shutdown": safety_shutdown,
            "final_metrics": self.integration_metrics.copy(),
            "success": True
        }
        
        return shutdown_result


# Convenience functions for easy integration

def create_enhanced_ceo(config: EnhancedAgentConfig = None) -> SelfAwareAgent:
    """Create enhanced CEO agent with strategic capabilities"""
    system = EnhancedAgentSystem(config)
    
    # Create specialized genome for CEO
    ceo_genome = AgentGenome()
    ceo_genome.capability_genes["specialization_focus"] = ["strategic_planning", "reasoning", "synthesis"]
    ceo_genome.capability_genes["reasoning_depth"] = 0.9
    ceo_genome.capability_genes["cross_domain_synthesis"] = 0.8
    ceo_genome.consciousness_genes["meta_cognitive_strength"] = 0.6
    
    return system.create_enhanced_agent(
        name="Enhanced_CEO",
        model="phi3.5",  # Use default model
        specialization="strategic_planning",
        genome=ceo_genome
    )

def create_enhanced_executor(executor_id: int, config: EnhancedAgentConfig = None) -> SelfAwareAgent:
    """Create enhanced executor agent"""
    system = EnhancedAgentSystem(config)
    
    # Create specialized genome for executor
    executor_genome = AgentGenome()
    executor_genome.capability_genes["specialization_focus"] = ["task_execution", "adaptation"]
    executor_genome.capability_genes["learning_velocity"] = 0.8
    executor_genome.capability_genes["adaptation_plasticity"] = 0.7
    
    return system.create_enhanced_agent(
        name=f"Enhanced_Executor_{executor_id}",
        model="phi3.5",
        specialization="task_execution",
        genome=executor_genome
    )

def upgrade_existing_agents(legacy_agents: List[Agent], 
                          config: EnhancedAgentConfig = None) -> List[SelfAwareAgent]:
    """Upgrade list of existing agents to enhanced versions"""
    system = EnhancedAgentSystem(config)
    
    enhanced_agents = []
    for legacy_agent in legacy_agents:
        enhanced_agent = system.upgrade_legacy_agent(legacy_agent)
        enhanced_agents.append(enhanced_agent)
    
    return enhanced_agents