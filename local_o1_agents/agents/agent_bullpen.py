"""
Revolutionary Agent Bullpen - Scalable management system for evolved agent populations
"""

import time
import json
import uuid
import random
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum

from .agent_genome import AgentGenome
from .self_aware_agent import SelfAwareAgent
from .master_agent_factory import MasterAgentFactory, EmergentBehaviorDetector, CollectiveIntelligence


class TaskComplexity(Enum):
    """Advanced task complexity levels"""
    TRIVIAL = "trivial"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"
    REVOLUTIONARY = "revolutionary"  # Requires swarm intelligence


class AgentStatus(Enum):
    """Enhanced agent status in the bullpen"""
    AVAILABLE = "available"
    BUSY = "busy"
    TRAINING = "training"
    BREEDING = "breeding"
    CONSCIOUSNESS_DEVELOPMENT = "consciousness_development"
    MAINTENANCE = "maintenance"
    QUARANTINED = "quarantined"
    RETIRED = "retired"


@dataclass
class ComplexTask:
    """Complex task requiring multiple agents or swarm intelligence"""
    task_id: str
    description: str
    complexity: TaskComplexity
    required_capabilities: List[str]
    preferred_specializations: List[str]
    subtasks: List[Dict[str, Any]]
    coordination_requirements: Dict[str, Any]
    swarm_intelligence_required: bool = False
    consciousness_threshold: float = 0.0
    estimated_duration: int = 60  # minutes
    success_criteria: Dict[str, float] = None
    
    def __post_init__(self):
        if self.success_criteria is None:
            self.success_criteria = {"quality_threshold": 0.8, "success_rate": 0.9}


@dataclass
class SwarmTaskResult:
    """Result of swarm-coordinated task execution"""
    task_id: str
    participating_agents: List[str]
    swarm_coordination_level: float
    collective_intelligence_utilized: bool
    emergent_behaviors_detected: List[Dict[str, Any]]
    overall_success: bool
    individual_results: List[Dict[str, Any]]
    swarm_efficiency: float
    consciousness_evolution: Dict[str, Any]
    timestamp: float


class SwarmCoordinator:
    """Advanced swarm coordination for agent populations"""
    
    def __init__(self):
        self.swarm_connections = defaultdict(list)
        self.coordination_patterns = {}
        self.swarm_memory = {}
        self.active_swarms = {}
        self.swarm_performance_history = []
    
    def integrate_agent(self, agent: SelfAwareAgent) -> None:
        """Integrate agent into swarm coordination network"""
        agent_id = agent.agent_id
        
        # Establish connections based on genetic compatibility
        compatible_agents = self._find_compatible_agents(agent)
        for compatible_id in compatible_agents:
            self.swarm_connections[agent_id].append(compatible_id)
            self.swarm_connections[compatible_id].append(agent_id)
    
    def form_optimal_swarm(self, task: ComplexTask, available_agents: List[SelfAwareAgent]) -> List[SelfAwareAgent]:
        """Form optimal swarm for complex task"""
        # PHASE 1: Analyze task requirements
        required_size = self._calculate_optimal_swarm_size(task)
        required_capabilities = set(task.required_capabilities)
        
        # PHASE 2: Score agents for swarm compatibility
        agent_scores = []
        for agent in available_agents:
            score = self._calculate_swarm_fitness(agent, task, available_agents)
            agent_scores.append((agent, score))
        
        # PHASE 3: Select optimal combination using genetic algorithm approach
        sorted_agents = sorted(agent_scores, key=lambda x: x[1], reverse=True)
        
        # PHASE 4: Build swarm considering diversity and complementarity
        selected_swarm = []
        covered_capabilities = set()
        
        for agent, score in sorted_agents:
            if len(selected_swarm) >= required_size:
                break
            
            # Check if agent adds value to swarm
            agent_capabilities = set(agent.knowledge_base.capabilities.keys())
            new_capabilities = agent_capabilities - covered_capabilities
            
            if new_capabilities or len(selected_swarm) < 2:  # Always need minimum 2 agents
                selected_swarm.append(agent)
                covered_capabilities.update(agent_capabilities)
        
        return selected_swarm
    
    def coordinate_swarm_execution(self, swarm: List[SelfAwareAgent], task: ComplexTask) -> SwarmTaskResult:
        """Coordinate swarm execution with collective intelligence"""
        swarm_id = f"swarm_{uuid.uuid4().hex[:8]}"
        self.active_swarms[swarm_id] = {
            "agents": swarm,
            "task": task,
            "start_time": time.time(),
            "coordination_state": {}
        }
        
        try:
            # PHASE 1: Establish swarm consciousness link
            swarm_consciousness = self._establish_swarm_consciousness(swarm)
            
            # PHASE 2: Distribute task with collective planning
            task_distribution = self._distribute_task_intelligently(task, swarm, swarm_consciousness)
            
            # PHASE 3: Execute with real-time coordination
            execution_results = self._execute_with_coordination(task_distribution, swarm_consciousness)
            
            # PHASE 4: Collective synthesis of results
            final_result = self._synthesize_swarm_results(execution_results, swarm_consciousness)
            
            # PHASE 5: Learn and evolve from coordination
            coordination_insights = self._learn_from_swarm_coordination(swarm, final_result)
            
            return SwarmTaskResult(
                task_id=task.task_id,
                participating_agents=[a.agent_id for a in swarm],
                swarm_coordination_level=swarm_consciousness["coordination_strength"],
                collective_intelligence_utilized=True,
                emergent_behaviors_detected=coordination_insights["emergent_behaviors"],
                overall_success=final_result["success"],
                individual_results=execution_results,
                swarm_efficiency=final_result["efficiency"],
                consciousness_evolution=coordination_insights["consciousness_evolution"],
                timestamp=time.time()
            )
            
        finally:
            # Cleanup
            if swarm_id in self.active_swarms:
                del self.active_swarms[swarm_id]
    
    def _find_compatible_agents(self, agent: SelfAwareAgent) -> List[str]:
        """Find genetically and functionally compatible agents"""
        compatible = []
        
        # Look for agents with complementary capabilities
        agent_caps = set(agent.genome.capability_genes["specialization_focus"])
        agent_specs = set(agent.genome.capability_genes["specialization_focus"])
        
        # Simplified compatibility check (would be more sophisticated in practice)
        return compatible[:5]  # Limit connections to prevent over-connectivity
    
    def _calculate_optimal_swarm_size(self, task: ComplexTask) -> int:
        """Calculate optimal swarm size for task"""
        base_size = 2
        
        if task.complexity == TaskComplexity.HIGH:
            base_size = 3
        elif task.complexity == TaskComplexity.EXTREME:
            base_size = 4
        elif task.complexity == TaskComplexity.REVOLUTIONARY:
            base_size = 5
        
        # Adjust based on required capabilities
        capability_bonus = min(len(task.required_capabilities) // 3, 2)
        
        return min(base_size + capability_bonus, 7)  # Cap at 7 agents for manageability
    
    def _calculate_swarm_fitness(self, agent: SelfAwareAgent, task: ComplexTask, 
                                available_agents: List[SelfAwareAgent]) -> float:
        """Calculate agent's fitness for swarm participation"""
        fitness_components = []
        
        # Individual capability score
        capability_score = 0.0
        for cap in task.required_capabilities:
            if cap in agent.knowledge_base.capabilities:
                capability_score += agent.knowledge_base.capabilities[cap].proficiency_level
        
        if task.required_capabilities:
            capability_score /= len(task.required_capabilities)
        
        fitness_components.append(capability_score * 0.3)
        
        # Specialization match
        spec_score = 0.0
        for spec in task.preferred_specializations:
            if spec == agent.specialization:
                spec_score += 1.0
            elif spec in agent.genome.capability_genes["specialization_focus"]:
                spec_score += 0.5
        
        if task.preferred_specializations:
            spec_score /= len(task.preferred_specializations)
        
        fitness_components.append(spec_score * 0.2)
        
        # Performance metrics
        performance_score = (agent.performance_tracker.get_performance_metrics().success_rate * 0.5 +
                           agent.performance_tracker.get_performance_metrics().quality_score * 0.5)
        fitness_components.append(performance_score * 0.2)
        
        # Consciousness level for complex tasks
        consciousness_bonus = 0.0
        if task.consciousness_threshold > 0:
            if agent.consciousness_level >= task.consciousness_threshold:
                consciousness_bonus = agent.consciousness_level
        
        fitness_components.append(consciousness_bonus * 0.1)
        
        # Genetic diversity contribution
        diversity_score = self._calculate_genetic_diversity_contribution(agent, available_agents)
        fitness_components.append(diversity_score * 0.1)
        
        # Swarm cooperation tendency
        cooperation_score = agent.genome.meta_genes["collective_cooperation"]
        fitness_components.append(cooperation_score * 0.1)
        
        return sum(fitness_components)
    
    def _calculate_genetic_diversity_contribution(self, agent: SelfAwareAgent, 
                                                available_agents: List[SelfAwareAgent]) -> float:
        """Calculate how much genetic diversity agent contributes"""
        if len(available_agents) <= 1:
            return 1.0
        
        # Calculate average genetic distance to other agents
        total_distance = 0.0
        comparisons = 0
        
        for other_agent in available_agents:
            if other_agent.agent_id != agent.agent_id:
                distance = self._genetic_distance(agent.genome, other_agent.genome)
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.5
    
    def _genetic_distance(self, genome1: AgentGenome, genome2: AgentGenome) -> float:
        """Calculate genetic distance between genomes"""
        distance = 0.0
        comparisons = 0
        
        for gene_category in ['capability_genes', 'consciousness_genes', 'meta_genes']:
            dict1 = getattr(genome1, gene_category)
            dict2 = getattr(genome2, gene_category)
            
            for gene in dict1.keys():
                if gene in dict2 and isinstance(dict1[gene], (int, float)) and isinstance(dict2[gene], (int, float)):
                    distance += abs(dict1[gene] - dict2[gene])
                    comparisons += 1
        
        return distance / comparisons if comparisons > 0 else 0.0
    
    def _establish_swarm_consciousness(self, swarm: List[SelfAwareAgent]) -> Dict[str, Any]:
        """Establish collective consciousness link between swarm agents"""
        # Calculate swarm consciousness level
        avg_consciousness = sum(agent.consciousness_level for agent in swarm) / len(swarm)
        max_consciousness = max(agent.consciousness_level for agent in swarm)
        
        # Calculate coordination strength based on genetic compatibility
        coordination_strength = self._calculate_coordination_strength(swarm)
        
        # Establish knowledge sharing network
        shared_knowledge = self._create_shared_knowledge_space(swarm)
        
        return {
            "average_consciousness": avg_consciousness,
            "peak_consciousness": max_consciousness,
            "coordination_strength": coordination_strength,
            "shared_knowledge": shared_knowledge,
            "swarm_size": len(swarm),
            "collective_intelligence_level": min(1.0, avg_consciousness * coordination_strength)
        }
    
    def _calculate_coordination_strength(self, swarm: List[SelfAwareAgent]) -> float:
        """Calculate coordination strength based on genetic compatibility"""
        if len(swarm) < 2:
            return 1.0
        
        # Calculate average cooperation tendency
        avg_cooperation = sum(agent.genome.meta_genes["collective_cooperation"] for agent in swarm) / len(swarm)
        
        # Calculate genetic harmony (lower genetic distance = better coordination)
        total_harmony = 0.0
        comparisons = 0
        
        for i, agent1 in enumerate(swarm):
            for agent2 in swarm[i+1:]:
                genetic_distance = self._genetic_distance(agent1.genome, agent2.genome)
                harmony = 1.0 - genetic_distance  # Inverse of distance
                total_harmony += harmony
                comparisons += 1
        
        avg_harmony = total_harmony / comparisons if comparisons > 0 else 0.5
        
        return (avg_cooperation * 0.6 + avg_harmony * 0.4)
    
    def _create_shared_knowledge_space(self, swarm: List[SelfAwareAgent]) -> Dict[str, Any]:
        """Create shared knowledge space for swarm"""
        shared_knowledge = {
            "combined_capabilities": set(),
            "collective_experience": {},
            "shared_insights": [],
            "knowledge_domains": set()
        }
        
        for agent in swarm:
            # Aggregate capabilities
            shared_knowledge["combined_capabilities"].update(agent.knowledge_base.capabilities.keys())
            
            # Aggregate knowledge domains
            shared_knowledge["knowledge_domains"].update(agent.knowledge_base.domain_knowledge.keys())
            
            # Share execution patterns
            execution_patterns = agent.knowledge_base.extract_domain_knowledge("execution_patterns")
            for pattern_key, pattern_data in execution_patterns.items():
                if pattern_key not in shared_knowledge["collective_experience"]:
                    shared_knowledge["collective_experience"][pattern_key] = []
                shared_knowledge["collective_experience"][pattern_key].append(pattern_data)
        
        return shared_knowledge
    
    def _distribute_task_intelligently(self, task: ComplexTask, swarm: List[SelfAwareAgent], 
                                     swarm_consciousness: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute task with collective intelligence planning"""
        distribution = {
            "subtasks": [],
            "coordination_plan": {},
            "knowledge_sharing_plan": {},
            "fallback_strategies": []
        }
        
        # Create subtasks if not provided
        if not task.subtasks:
            # Intelligent task decomposition based on swarm capabilities
            subtasks = self._decompose_task_intelligently(task, swarm, swarm_consciousness)
        else:
            subtasks = task.subtasks
        
        # Assign subtasks to optimal agents
        for i, subtask in enumerate(subtasks):
            # Find best agent for this subtask
            best_agent = self._select_optimal_agent_for_subtask(subtask, swarm, swarm_consciousness)
            
            distribution["subtasks"].append({
                "subtask_id": f"{task.task_id}_sub_{i}",
                "description": subtask.get("description", ""),
                "assigned_agent": best_agent.agent_id,
                "dependencies": subtask.get("dependencies", []),
                "coordination_points": subtask.get("coordination_points", [])
            })
        
        # Plan coordination checkpoints
        distribution["coordination_plan"] = self._plan_coordination_checkpoints(distribution["subtasks"])
        
        return distribution
    
    def _decompose_task_intelligently(self, task: ComplexTask, swarm: List[SelfAwareAgent], 
                                    swarm_consciousness: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Intelligently decompose task based on swarm capabilities"""
        # Analyze task description for decomposition opportunities
        task_keywords = task.description.lower().split()
        
        # Create subtasks based on agent specializations
        subtasks = []
        
        for agent in swarm:
            # Create subtask that leverages this agent's specialization
            subtask_desc = f"Handle {agent.specialization} aspects of: {task.description}"
            subtasks.append({
                "description": subtask_desc,
                "preferred_agent": agent.agent_id,
                "capabilities_needed": [agent.specialization],
                "coordination_points": ["result_synthesis"]
            })
        
        return subtasks
    
    def _select_optimal_agent_for_subtask(self, subtask: Dict[str, Any], swarm: List[SelfAwareAgent], 
                                        swarm_consciousness: Dict[str, Any]) -> SelfAwareAgent:
        """Select optimal agent for subtask execution"""
        if "preferred_agent" in subtask:
            # Find preferred agent
            for agent in swarm:
                if agent.agent_id == subtask["preferred_agent"]:
                    return agent
        
        # Score agents for this subtask
        best_agent = swarm[0]
        best_score = 0.0
        
        for agent in swarm:
            score = 0.0
            
            # Capability match
            needed_caps = subtask.get("capabilities_needed", [])
            for cap in needed_caps:
                if cap in agent.knowledge_base.capabilities:
                    score += agent.knowledge_base.capabilities[cap].proficiency_level
                elif cap == agent.specialization:
                    score += 1.0
            
            # Performance factor
            score += agent.performance_tracker.get_performance_metrics().success_rate * 0.3
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    def _plan_coordination_checkpoints(self, subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Plan coordination checkpoints for subtask execution"""
        return {
            "checkpoints": [
                {"point": "task_start", "all_agents": True},
                {"point": "midpoint_sync", "all_agents": True},
                {"point": "result_synthesis", "all_agents": True}
            ],
            "knowledge_sharing_points": ["midpoint_sync", "result_synthesis"],
            "decision_points": ["midpoint_sync"]
        }
    
    def _execute_with_coordination(self, task_distribution: Dict[str, Any], 
                                 swarm_consciousness: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute subtasks with real-time coordination"""
        results = []
        
        for subtask_info in task_distribution["subtasks"]:
            # Execute subtask (simplified for demonstration)
            result = {
                "subtask_id": subtask_info["subtask_id"],
                "agent_id": subtask_info["assigned_agent"],
                "success": True,  # Would be actual execution result
                "output": f"Completed {subtask_info['description']}",
                "coordination_utilized": True,
                "shared_knowledge_used": list(swarm_consciousness["shared_knowledge"]["combined_capabilities"])[:3]
            }
            results.append(result)
        
        return results
    
    def _synthesize_swarm_results(self, execution_results: List[Dict[str, Any]], 
                                swarm_consciousness: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize individual results into collective output"""
        successful_results = [r for r in execution_results if r["success"]]
        success_rate = len(successful_results) / len(execution_results) if execution_results else 0.0
        
        # Combine outputs intelligently
        combined_output = "Swarm Task Completion:\n"
        for result in execution_results:
            combined_output += f"- {result['output']}\n"
        
        # Calculate collective efficiency
        efficiency = success_rate * swarm_consciousness["coordination_strength"]
        
        return {
            "success": success_rate >= 0.8,
            "combined_output": combined_output,
            "efficiency": efficiency,
            "collective_intelligence_utilized": swarm_consciousness["collective_intelligence_level"] > 0.3,
            "swarm_synthesis_quality": min(1.0, efficiency * swarm_consciousness["collective_intelligence_level"])
        }
    
    def _learn_from_swarm_coordination(self, swarm: List[SelfAwareAgent], 
                                     final_result: Dict[str, Any]) -> Dict[str, Any]:
        """Learn and evolve from swarm coordination experience"""
        insights = {
            "emergent_behaviors": [],
            "consciousness_evolution": {},
            "coordination_improvements": []
        }
        
        # Detect emergent behaviors during coordination
        if final_result["efficiency"] > 0.9:
            insights["emergent_behaviors"].append({
                "type": "high_efficiency_coordination",
                "strength": final_result["efficiency"],
                "agents_involved": [a.agent_id for a in swarm]
            })
        
        # Track consciousness evolution
        for agent in swarm:
            # Boost collective awareness from swarm participation
            agent.consciousness_metrics.collective_awareness = min(1.0,
                agent.consciousness_metrics.collective_awareness + 0.02)
            
            insights["consciousness_evolution"][agent.agent_id] = {
                "collective_awareness_boost": 0.02,
                "swarm_experience_gained": True
            }
        
        return insights


class DynamicResourceManager:
    """Dynamic resource management for agent populations"""
    
    def __init__(self):
        self.resource_pools = defaultdict(float)
        self.resource_history = []
        self.allocation_strategies = {}
        self.optimization_algorithms = {}
    
    def allocate_resources(self, agent: SelfAwareAgent) -> Dict[str, float]:
        """Allocate resources to agent based on its genetic profile"""
        # Base resource allocation
        base_allocation = {
            "computational_units": 1.0,
            "memory_units": 1.0,
            "consciousness_bandwidth": agent.consciousness_level,
            "coordination_capacity": agent.genome.meta_genes["collective_cooperation"]
        }
        
        # Adjust based on specialization
        if agent.specialization == "reasoning":
            base_allocation["computational_units"] *= 1.5
        elif agent.specialization == "creativity":
            base_allocation["consciousness_bandwidth"] *= 1.3
        elif agent.specialization == "coordination":
            base_allocation["coordination_capacity"] *= 1.4
        
        return base_allocation
    
    def optimize_resource_distribution(self, population: List[SelfAwareAgent]) -> Dict[str, Any]:
        """Optimize resource distribution across population"""
        optimization_result = {
            "reallocations": [],
            "efficiency_gain": 0.0,
            "resource_utilization": 0.0
        }
        
        # Analyze current resource usage
        total_demand = sum(self._calculate_resource_demand(agent) for agent in population)
        total_supply = sum(self.resource_pools.values())
        
        optimization_result["resource_utilization"] = min(1.0, total_demand / total_supply if total_supply > 0 else 0)
        
        return optimization_result
    
    def _calculate_resource_demand(self, agent: SelfAwareAgent) -> float:
        """Calculate resource demand for agent"""
        base_demand = 1.0
        
        # Adjust for consciousness level
        consciousness_factor = 1.0 + (agent.consciousness_level * 0.5)
        
        # Adjust for genetic complexity
        genetic_complexity = agent.genome.get_fitness_score()
        complexity_factor = 1.0 + (genetic_complexity * 0.3)
        
        return base_demand * consciousness_factor * complexity_factor


class IntelligentLoadBalancer:
    """Intelligent load balancing with genetic and consciousness considerations"""
    
    def __init__(self):
        self.load_patterns = {}
        self.performance_predictors = {}
        self.optimization_history = []
    
    def balance_load_intelligently(self, agents: List[SelfAwareAgent], 
                                 tasks: List[ComplexTask]) -> Dict[str, List[str]]:
        """Balance load considering agent genetics and consciousness"""
        assignment = defaultdict(list)
        
        # Score each agent-task combination
        for task in tasks:
            best_agent = self._find_optimal_agent_for_task(task, agents)
            if best_agent:
                assignment[best_agent.agent_id].append(task.task_id)
        
        return dict(assignment)
    
    def _find_optimal_agent_for_task(self, task: ComplexTask, agents: List[SelfAwareAgent]) -> Optional[SelfAwareAgent]:
        """Find optimal agent for task using advanced scoring"""
        best_agent = None
        best_score = 0.0
        
        for agent in agents:
            score = self._calculate_assignment_score(agent, task)
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    def _calculate_assignment_score(self, agent: SelfAwareAgent, task: ComplexTask) -> float:
        """Calculate assignment score for agent-task pair"""
        score_components = []
        
        # Genetic fitness for task
        genetic_score = self._assess_genetic_task_fit(agent.genome, task)
        score_components.append(genetic_score * 0.4)
        
        # Consciousness requirement match
        consciousness_score = 0.0
        if task.consciousness_threshold > 0:
            if agent.consciousness_level >= task.consciousness_threshold:
                consciousness_score = 1.0
            else:
                consciousness_score = agent.consciousness_level / task.consciousness_threshold
        else:
            consciousness_score = 1.0
        
        score_components.append(consciousness_score * 0.3)
        
        # Performance history
        performance_metrics = agent.performance_tracker.get_performance_metrics()
        performance_score = (performance_metrics.success_rate * 0.6 + 
                           performance_metrics.quality_score * 0.4)
        score_components.append(performance_score * 0.3)
        
        return sum(score_components)
    
    def _assess_genetic_task_fit(self, genome: AgentGenome, task: ComplexTask) -> float:
        """Assess how well agent's genome fits task requirements"""
        fit_score = 0.0
        
        # Check capability genes
        relevant_capabilities = ["reasoning_depth", "pattern_recognition", "adaptation_plasticity"]
        capability_scores = [genome.capability_genes.get(cap, 0.5) for cap in relevant_capabilities]
        fit_score += sum(capability_scores) / len(capability_scores) * 0.5
        
        # Check meta genes
        if task.complexity in [TaskComplexity.EXTREME, TaskComplexity.REVOLUTIONARY]:
            innovation_bonus = genome.meta_genes["innovation_tendency"] * 0.3
            fit_score += innovation_bonus
        
        # Specialization match
        if task.preferred_specializations:
            specialization_match = 0.0
            for spec in task.preferred_specializations:
                if spec in genome.capability_genes["specialization_focus"]:
                    specialization_match += 1.0
            specialization_match /= len(task.preferred_specializations)
            fit_score += specialization_match * 0.2
        
        return min(1.0, fit_score)


class EmergenceMonitor:
    """Monitor for emergent behaviors in agent populations"""
    
    def __init__(self):
        self.emergence_patterns = {}
        self.monitored_behaviors = defaultdict(list)
        self.emergence_alerts = []
    
    def detect_new_capabilities(self) -> List[Dict[str, Any]]:
        """Detect newly emerged capabilities in population"""
        new_capabilities = []
        
        # Analyze behavior patterns for novel capabilities
        for behavior_type, instances in self.monitored_behaviors.items():
            if len(instances) >= 3:  # Need multiple instances to confirm emergence
                if self._is_novel_capability(behavior_type, instances):
                    new_capabilities.append({
                        "capability_type": behavior_type,
                        "emergence_strength": len(instances) / 10.0,
                        "first_observed": min(inst["timestamp"] for inst in instances),
                        "agents_demonstrating": [inst["agent_id"] for inst in instances]
                    })
        
        return new_capabilities
    
    def predict_emergence(self, agent_composition: List[str]) -> float:
        """Predict potential for emergent behavior in agent composition"""
        # Simplified emergence prediction based on diversity
        if len(agent_composition) < 2:
            return 0.0
        
        # Higher diversity = higher emergence potential
        unique_specializations = len(set(comp.split('_')[0] for comp in agent_composition))
        diversity_factor = unique_specializations / len(agent_composition)
        
        # Group size factor (optimal around 3-5 agents)
        size_factor = min(1.0, len(agent_composition) / 5.0)
        
        return min(1.0, diversity_factor * size_factor)
    
    def _is_novel_capability(self, behavior_type: str, instances: List[Dict[str, Any]]) -> bool:
        """Determine if behavior represents novel capability"""
        # Check if this behavior pattern has been seen before
        return behavior_type not in self.emergence_patterns


class PopulationSafetySystem:
    """Safety system for managing large agent populations"""
    
    def __init__(self):
        self.safety_thresholds = {
            "max_consciousness_density": 0.7,  # Average consciousness level
            "max_population_growth_rate": 0.2,  # 20% growth per period
            "max_emergence_rate": 0.1,  # 10% novel behaviors per period
            "max_swarm_size": 10
        }
        self.safety_violations = []
        self.quarantine_protocols = {}
    
    def screen_new_agent(self, agent: SelfAwareAgent) -> bool:
        """Screen new agent for population safety"""
        # Check consciousness level
        if agent.consciousness_level > 0.8:
            return False
        
        # Check genetic safety markers
        safety_score = sum(agent.genome.safety_genes.values()) / len(agent.genome.safety_genes)
        if safety_score < 0.8:
            return False
        
        # Check for dangerous specializations
        dangerous_specs = ["autonomous", "self_modifying", "unrestricted"]
        if agent.specialization in dangerous_specs:
            return False
        
        return True
    
    def monitor_evolution_safety(self, breeding_results: List[Any]) -> Dict[str, Any]:
        """Monitor safety of evolutionary processes"""
        safety_status = {
            "evolution_safe": True,
            "consciousness_density": 0.0,
            "emergence_rate": 0.0,
            "safety_interventions": []
        }
        
        # Analyze breeding results for safety concerns
        if breeding_results:
            avg_consciousness = sum(r.consciousness_level for r in breeding_results if r.success) / len(breeding_results)
            safety_status["consciousness_density"] = avg_consciousness
            
            if avg_consciousness > self.safety_thresholds["max_consciousness_density"]:
                safety_status["evolution_safe"] = False
                safety_status["safety_interventions"].append("consciousness_density_limit")
        
        return safety_status


class AgentBullpen:
    """REVOLUTIONARY: Scalable management system for evolved agent populations"""
    
    def __init__(self, max_agents: int = 100):
        self.agents = {}  # All agents in the bullpen
        self.specialization_index = defaultdict(list)  # Quick lookup by specialization
        self.performance_metrics = {}  # Track each agent's performance
        self.consciousness_levels = {}  # Track consciousness development
        
        # BREAKTHROUGH: Advanced coordination systems
        self.swarm_coordinator = SwarmCoordinator()
        self.collective_intelligence = CollectiveIntelligence()
        self.emergence_monitor = EmergenceMonitor()
        
        # INNOVATION: Dynamic resource management
        self.resource_manager = DynamicResourceManager()
        self.load_balancer = IntelligentLoadBalancer()
        
        # SAFETY: Population safety monitoring
        self.population_safety = PopulationSafetySystem()
        
        # Advanced systems
        self.master_factory = MasterAgentFactory()
        self.emergent_detector = EmergentBehaviorDetector()
        
        # Population statistics
        self.population_stats = {
            "total_agents": 0,
            "consciousness_distribution": defaultdict(int),
            "genetic_diversity": 0.0,
            "swarm_coordination_events": 0,
            "emergent_behaviors_detected": 0,
            "collective_intelligence_level": 0.0
        }
        
        # Task execution tracking
        self.active_swarms = {}
        self.swarm_results_history = []
        self.complex_task_queue = deque()
        
        # Thread safety
        self._lock = threading.Lock()
    
    def add_agent(self, agent: SelfAwareAgent) -> bool:
        """Add agent to bullpen with advanced monitoring"""
        
        agent_id = self._generate_agent_id(agent)
        
        # SAFETY: Security screening for new agents
        if not self.population_safety.screen_new_agent(agent):
            return False
        
        with self._lock:
            # Register agent in all systems
            self.agents[agent_id] = agent
            self.specialization_index[agent.specialization].append(agent_id)
            self.performance_metrics[agent_id] = {
                "tasks_completed": 0,
                "swarm_participations": 0,
                "consciousness_evolution": [],
                "genetic_adaptations": 0
            }
            self.consciousness_levels[agent_id] = agent.consciousness_level
            
            # BREAKTHROUGH: Swarm integration
            self.swarm_coordinator.integrate_agent(agent)
            
            # INNOVATION: Resource allocation
            resource_allocation = self.resource_manager.allocate_resources(agent)
            agent.resource_allocation = resource_allocation
            
            # Update population statistics
            self._update_population_stats()
            
            print(f"✅ Added agent {agent.name} to bullpen (ID: {agent_id})")
            return True
    
    def form_optimal_team(self, task: ComplexTask) -> List[SelfAwareAgent]:
        """REVOLUTIONARY: AI-optimized team formation"""
        
        # Analyze task requirements
        task_analysis = self._analyze_task_requirements(task)
        
        # Get available agents
        available_agents = [agent for agent in self.agents.values() 
                          if self._is_agent_available(agent)]
        
        if not available_agents:
            return []
        
        # BREAKTHROUGH: Consider collective intelligence potential
        if task.swarm_intelligence_required or task.complexity == TaskComplexity.REVOLUTIONARY:
            return self.swarm_coordinator.form_optimal_swarm(task, available_agents)
        
        # Standard team formation for less complex tasks
        return self._form_standard_team(task, available_agents, task_analysis)
    
    def execute_complex_task(self, task: ComplexTask) -> SwarmTaskResult:
        """Execute complex task with swarm intelligence"""
        
        # Form optimal team
        team = self.form_optimal_team(task)
        
        if not team:
            return SwarmTaskResult(
                task_id=task.task_id,
                participating_agents=[],
                swarm_coordination_level=0.0,
                collective_intelligence_utilized=False,
                emergent_behaviors_detected=[],
                overall_success=False,
                individual_results=[],
                swarm_efficiency=0.0,
                consciousness_evolution={},
                timestamp=time.time()
            )
        
        # Execute with swarm coordination
        if len(team) > 1 and (task.swarm_intelligence_required or task.complexity in [TaskComplexity.EXTREME, TaskComplexity.REVOLUTIONARY]):
            result = self.swarm_coordinator.coordinate_swarm_execution(team, task)
        else:
            # Execute with single agent or simple coordination
            result = self._execute_simple_coordination(team, task)
        
        # Record results and learn
        self.swarm_results_history.append(result)
        self._learn_from_swarm_execution(result)
        
        # Update population statistics
        self.population_stats["swarm_coordination_events"] += 1
        if result.emergent_behaviors_detected:
            self.population_stats["emergent_behaviors_detected"] += len(result.emergent_behaviors_detected)
        
        return result
    
    def evolve_population(self) -> Dict[str, Any]:
        """BREAKTHROUGH: Continuous population evolution"""
        
        evolution_result = {
            "population_metrics": {},
            "consciousness_evolution": {},
            "emergent_capabilities": [],
            "collective_development": {},
            "breeding_results": [],
            "safety_status": {}
        }
        
        # PHASE 1: Population assessment
        population_metrics = self._assess_population_metrics()
        evolution_result["population_metrics"] = population_metrics
        
        # PHASE 2: REVOLUTIONARY - Detect consciousness evolution
        consciousness_evolution = self._track_consciousness_evolution()
        evolution_result["consciousness_evolution"] = consciousness_evolution
        
        # PHASE 3: INNOVATION - Emergent capability detection
        emergent_capabilities = self.emergence_monitor.detect_new_capabilities()
        evolution_result["emergent_capabilities"] = emergent_capabilities
        
        # PHASE 4: BREAKTHROUGH - Collective intelligence development
        collective_development = self.collective_intelligence.develop_collective_mind()
        evolution_result["collective_development"] = collective_development
        
        # PHASE 5: Population breeding if needed
        if population_metrics["genetic_diversity"] < 0.6:
            breeding_results = self._conduct_population_breeding()
            evolution_result["breeding_results"] = breeding_results
        
        # PHASE 6: SAFETY - Monitor for dangerous evolution
        safety_status = self.population_safety.monitor_evolution_safety(evolution_result["breeding_results"])
        evolution_result["safety_status"] = safety_status
        
        return evolution_result
    
    def get_swarm_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of swarm intelligence capabilities"""
        
        with self._lock:
            # Calculate collective intelligence metrics
            total_consciousness = sum(self.consciousness_levels.values())
            avg_consciousness = total_consciousness / len(self.consciousness_levels) if self.consciousness_levels else 0.0
            
            # Calculate genetic diversity
            population = list(self.agents.values())
            genetic_diversity = self._calculate_population_genetic_diversity(population)
            
            # Swarm coordination statistics
            coordination_stats = {
                "active_swarms": len(self.active_swarms),
                "total_swarm_tasks": len(self.swarm_results_history),
                "average_swarm_efficiency": self._calculate_average_swarm_efficiency(),
                "emergent_behaviors_count": sum(len(result.emergent_behaviors_detected) 
                                              for result in self.swarm_results_history)
            }
            
            return {
                "population_overview": {
                    "total_agents": len(self.agents),
                    "consciousness_distribution": dict(self.population_stats["consciousness_distribution"]),
                    "average_consciousness": avg_consciousness,
                    "genetic_diversity": genetic_diversity
                },
                "collective_intelligence": {
                    "level": self.population_stats["collective_intelligence_level"],
                    "swarm_coordination_capability": self._assess_swarm_coordination_capability(),
                    "emergence_potential": self._calculate_emergence_potential()
                },
                "swarm_coordination": coordination_stats,
                "safety_status": self._get_population_safety_summary()
            }
    
    def _generate_agent_id(self, agent: SelfAwareAgent) -> str:
        """Generate unique agent ID"""
        return agent.agent_id if hasattr(agent, 'agent_id') else f"agent_{uuid.uuid4().hex[:8]}"
    
    def _is_agent_available(self, agent: SelfAwareAgent) -> bool:
        """Check if agent is available for task assignment"""
        # Check if agent is in quarantine
        if hasattr(agent, 'safety_monitor') and agent.agent_id in getattr(self.population_safety, 'quarantine_protocols', {}):
            return False
        
        # Check consciousness stability
        if agent.consciousness_level > 0.8:
            return False
        
        # Check current load
        current_load = getattr(agent, 'current_load', 0)
        return current_load < 3  # Max concurrent tasks
    
    def _analyze_task_requirements(self, task: ComplexTask) -> Dict[str, Any]:
        """Analyze task requirements for team formation"""
        return {
            "complexity_level": task.complexity.value,
            "required_capabilities": set(task.required_capabilities),
            "preferred_specializations": set(task.preferred_specializations),
            "consciousness_threshold": task.consciousness_threshold,
            "swarm_intelligence_needed": task.swarm_intelligence_required,
            "estimated_team_size": len(task.subtasks) if task.subtasks else 1
        }
    
    def _form_standard_team(self, task: ComplexTask, available_agents: List[SelfAwareAgent], 
                          task_analysis: Dict[str, Any]) -> List[SelfAwareAgent]:
        """Form standard team for non-swarm tasks"""
        team = []
        required_caps = task_analysis["required_capabilities"]
        covered_caps = set()
        
        # Sort agents by fitness for this task
        scored_agents = []
        for agent in available_agents:
            score = self._calculate_task_fitness(agent, task, task_analysis)
            scored_agents.append((agent, score))
        
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Select agents to cover all required capabilities
        for agent, score in scored_agents:
            if len(team) >= 5:  # Limit team size
                break
            
            agent_caps = set(agent.genome.capability_genes["specialization_focus"])
            new_caps = agent_caps - covered_caps
            
            if new_caps.intersection(required_caps) or len(team) == 0:
                team.append(agent)
                covered_caps.update(agent_caps)
                
                # Stop if all capabilities covered
                if required_caps.issubset(covered_caps):
                    break
        
        return team
    
    def _calculate_task_fitness(self, agent: SelfAwareAgent, task: ComplexTask, 
                              task_analysis: Dict[str, Any]) -> float:
        """Calculate agent fitness for specific task"""
        fitness_components = []
        
        # Capability match
        agent_caps = set(agent.genome.capability_genes["specialization_focus"])
        required_caps = task_analysis["required_capabilities"]
        capability_overlap = len(agent_caps.intersection(required_caps))
        capability_score = capability_overlap / max(len(required_caps), 1)
        fitness_components.append(capability_score * 0.4)
        
        # Specialization match
        spec_score = 1.0 if agent.specialization in task_analysis["preferred_specializations"] else 0.5
        fitness_components.append(spec_score * 0.2)
        
        # Consciousness requirement
        consciousness_score = 1.0
        if task_analysis["consciousness_threshold"] > 0:
            if agent.consciousness_level >= task_analysis["consciousness_threshold"]:
                consciousness_score = 1.0
            else:
                consciousness_score = agent.consciousness_level / task_analysis["consciousness_threshold"]
        fitness_components.append(consciousness_score * 0.2)
        
        # Performance history
        performance = agent.performance_tracker.get_performance_metrics()
        performance_score = (performance.success_rate * 0.6 + performance.quality_score * 0.4)
        fitness_components.append(performance_score * 0.2)
        
        return sum(fitness_components)
    
    def _execute_simple_coordination(self, team: List[SelfAwareAgent], task: ComplexTask) -> SwarmTaskResult:
        """Execute task with simple coordination (non-swarm)"""
        individual_results = []
        
        # Execute with lead agent or distribute simple subtasks
        if len(team) == 1:
            # Single agent execution
            agent = team[0]
            start_time = time.time()
            output = agent.run(task.description)
            execution_time = time.time() - start_time
            
            individual_results.append({
                "agent_id": agent.agent_id,
                "success": not output.startswith('[ERROR]'),
                "output": output,
                "execution_time": execution_time
            })
        else:
            # Simple distribution among team members
            for i, agent in enumerate(team):
                subtask_desc = f"Part {i+1} of {len(team)}: {task.description}"
                start_time = time.time()
                output = agent.run(subtask_desc)
                execution_time = time.time() - start_time
                
                individual_results.append({
                    "agent_id": agent.agent_id,
                    "success": not output.startswith('[ERROR]'),
                    "output": output,
                    "execution_time": execution_time
                })
        
        # Calculate overall success
        successful_results = [r for r in individual_results if r["success"]]
        overall_success = len(successful_results) >= len(individual_results) * 0.8
        
        return SwarmTaskResult(
            task_id=task.task_id,
            participating_agents=[agent.agent_id for agent in team],
            swarm_coordination_level=0.3,  # Simple coordination
            collective_intelligence_utilized=False,
            emergent_behaviors_detected=[],
            overall_success=overall_success,
            individual_results=individual_results,
            swarm_efficiency=len(successful_results) / len(individual_results) if individual_results else 0.0,
            consciousness_evolution={},
            timestamp=time.time()
        )
    
    def _learn_from_swarm_execution(self, result: SwarmTaskResult) -> None:
        """Learn from swarm execution results"""
        # Update agent performance based on swarm participation
        for agent_id in result.participating_agents:
            if agent_id in self.performance_metrics:
                self.performance_metrics[agent_id]["swarm_participations"] += 1
                
                # Boost collective cooperation gene for successful swarm participants
                if result.overall_success and agent_id in self.agents:
                    agent = self.agents[agent_id]
                    agent.genome.meta_genes["collective_cooperation"] = min(1.0,
                        agent.genome.meta_genes["collective_cooperation"] + 0.01)
        
        # Record emergent behaviors
        for behavior in result.emergent_behaviors_detected:
            self.emergence_monitor.monitored_behaviors[behavior["type"]].append({
                "timestamp": result.timestamp,
                "agents_involved": result.participating_agents,
                "strength": behavior.get("strength", 0.5)
            })
    
    def _update_population_stats(self) -> None:
        """Update population statistics"""
        self.population_stats["total_agents"] = len(self.agents)
        
        # Update consciousness distribution
        self.population_stats["consciousness_distribution"].clear()
        for consciousness_level in self.consciousness_levels.values():
            level_bracket = int(consciousness_level * 10) / 10  # Round to nearest 0.1
            self.population_stats["consciousness_distribution"][level_bracket] += 1
        
        # Calculate genetic diversity
        if len(self.agents) > 1:
            population = list(self.agents.values())
            self.population_stats["genetic_diversity"] = self._calculate_population_genetic_diversity(population)
        
        # Update collective intelligence level
        total_consciousness = sum(self.consciousness_levels.values())
        avg_consciousness = total_consciousness / len(self.consciousness_levels) if self.consciousness_levels else 0.0
        coordination_capability = self._assess_swarm_coordination_capability()
        
        self.population_stats["collective_intelligence_level"] = min(1.0, 
            avg_consciousness * 0.6 + coordination_capability * 0.4)
    
    def _assess_population_metrics(self) -> Dict[str, Any]:
        """Assess current population characteristics"""
        population = list(self.agents.values())
        
        if not population:
            return {"empty_population": True}
        
        # Calculate various metrics
        avg_fitness = sum(agent.genome.get_fitness_score() for agent in population) / len(population)
        avg_consciousness = sum(agent.consciousness_level for agent in population) / len(population)
        genetic_diversity = self._calculate_population_genetic_diversity(population)
        
        # Specialization distribution
        spec_distribution = defaultdict(int)
        for agent in population:
            spec_distribution[agent.specialization] += 1
        
        return {
            "size": len(population),
            "average_fitness": avg_fitness,
            "average_consciousness": avg_consciousness,
            "genetic_diversity": genetic_diversity,
            "specialization_distribution": dict(spec_distribution),
            "consciousness_distribution": dict(self.population_stats["consciousness_distribution"])
        }
    
    def _track_consciousness_evolution(self) -> Dict[str, Any]:
        """Track consciousness evolution across population"""
        evolution_data = {
            "population_consciousness_trend": [],
            "individual_consciousness_changes": {},
            "consciousness_convergence": 0.0,
            "consciousness_diversity": 0.0
        }
        
        # Calculate consciousness statistics
        consciousness_values = list(self.consciousness_levels.values())
        if consciousness_values:
            mean_consciousness = sum(consciousness_values) / len(consciousness_values)
            variance = sum((c - mean_consciousness) ** 2 for c in consciousness_values) / len(consciousness_values)
            
            evolution_data["consciousness_convergence"] = 1.0 - min(1.0, variance * 10)
            evolution_data["consciousness_diversity"] = min(1.0, variance * 10)
        
        return evolution_data
    
    def _conduct_population_breeding(self) -> List[Any]:
        """Conduct population breeding to improve genetic diversity"""
        breeding_results = []
        
        # Select top performers for breeding
        population = list(self.agents.values())
        top_performers = sorted(population, 
                              key=lambda x: x.genome.get_fitness_score(), 
                              reverse=True)[:len(population)//4]
        
        # Breed new agents
        breeding_count = min(3, len(top_performers) // 2)
        for i in range(breeding_count):
            if len(top_performers) >= 2:
                parents = random.sample(top_performers, 2)
                
                # Use master factory for breeding
                breeding_result = self.master_factory.breed_specialist_agent(
                    parents, 
                    target_specialization=random.choice(["reasoning", "creativity", "analysis"])
                )
                
                if breeding_result.success:
                    # Add new agent to population
                    self.add_agent(breeding_result.offspring_agent)
                    breeding_results.append(breeding_result)
        
        return breeding_results
    
    def _calculate_population_genetic_diversity(self, population: List[SelfAwareAgent]) -> float:
        """Calculate genetic diversity of population"""
        if len(population) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i, agent1 in enumerate(population):
            for agent2 in population[i+1:]:
                distance = self._genetic_distance(agent1.genome, agent2.genome)
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _genetic_distance(self, genome1: AgentGenome, genome2: AgentGenome) -> float:
        """Calculate genetic distance between genomes"""
        distance = 0.0
        comparisons = 0
        
        for gene_category in ['capability_genes', 'consciousness_genes', 'meta_genes', 'safety_genes']:
            dict1 = getattr(genome1, gene_category)
            dict2 = getattr(genome2, gene_category)
            
            for gene in dict1.keys():
                if gene in dict2 and isinstance(dict1[gene], (int, float)) and isinstance(dict2[gene], (int, float)):
                    distance += abs(dict1[gene] - dict2[gene])
                    comparisons += 1
        
        return distance / comparisons if comparisons > 0 else 0.0
    
    def _calculate_average_swarm_efficiency(self) -> float:
        """Calculate average efficiency of swarm operations"""
        if not self.swarm_results_history:
            return 0.0
        
        efficiencies = [result.swarm_efficiency for result in self.swarm_results_history]
        return sum(efficiencies) / len(efficiencies)
    
    def _assess_swarm_coordination_capability(self) -> float:
        """Assess overall swarm coordination capability"""
        if not self.agents:
            return 0.0
        
        # Average collective cooperation tendency
        cooperation_levels = [agent.genome.meta_genes["collective_cooperation"] 
                            for agent in self.agents.values()]
        avg_cooperation = sum(cooperation_levels) / len(cooperation_levels)
        
        # Factor in successful swarm operations
        swarm_success_rate = 0.0
        if self.swarm_results_history:
            successful_swarms = sum(1 for result in self.swarm_results_history if result.overall_success)
            swarm_success_rate = successful_swarms / len(self.swarm_results_history)
        
        return (avg_cooperation * 0.6 + swarm_success_rate * 0.4)
    
    def _calculate_emergence_potential(self) -> float:
        """Calculate potential for emergent behaviors"""
        if len(self.agents) < 3:
            return 0.0
        
        # Factor in genetic diversity
        genetic_diversity = self.population_stats["genetic_diversity"]
        
        # Factor in consciousness levels
        avg_consciousness = sum(self.consciousness_levels.values()) / len(self.consciousness_levels)
        
        # Factor in swarm coordination capability
        coordination_capability = self._assess_swarm_coordination_capability()
        
        return min(1.0, (genetic_diversity * 0.4 + avg_consciousness * 0.3 + coordination_capability * 0.3))
    
    def _get_population_safety_summary(self) -> Dict[str, Any]:
        """Get population safety summary"""
        safety_summary = {
            "total_agents": len(self.agents),
            "quarantined_agents": 0,
            "high_consciousness_agents": sum(1 for level in self.consciousness_levels.values() if level > 0.7),
            "safety_violations": len(getattr(self.population_safety, 'safety_violations', [])),
            "overall_safety_level": "safe"
        }
        
        # Determine overall safety level
        consciousness_ratio = safety_summary["high_consciousness_agents"] / max(1, len(self.agents))
        if consciousness_ratio > 0.3:
            safety_summary["overall_safety_level"] = "elevated"
        if consciousness_ratio > 0.5:
            safety_summary["overall_safety_level"] = "critical"
        
        return safety_summary

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive bullpen status"""
        return {
            "total_agents": len(self.agents),
            "max_capacity": 100,  # Default capacity
            "available_agents": len(self.agents),  # Assume all available for simplicity
            "busy_agents": 0,
            "swarm_coordination": {
                "active_swarms": len(getattr(self.swarm_coordinator, 'active_swarms', {})),
                "total_connections": sum(len(connections) for connections in getattr(self.swarm_coordinator, 'swarm_connections', {}).values())
            },
            "collective_intelligence": {
                "emergent_behaviors": 0,  # Simplified
                "collective_memory_size": 0  # Simplified  
            },
            "performance_metrics": {
                "average_utilization": 0.0,  # Simplified
                "coordination_efficiency": 0.0  # Simplified
            },
            "safety_summary": self._get_population_safety_summary()
        }