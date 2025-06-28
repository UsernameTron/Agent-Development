"""
Adaptive Specialization System
Dynamic identification and development of specialized capabilities based on ecosystem performance
"""

import time
import uuid
import random
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

from .agent_genome import AgentGenome
from .self_aware_agent import SelfAwareAgent


class SpecializationCategory(Enum):
    """Categories of specializations"""
    COGNITIVE = "cognitive"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    COLLABORATIVE = "collaborative"
    ADAPTIVE = "adaptive"
    META = "meta"
    EMERGENT = "emergent"


class SpecializationUrgency(Enum):
    """Urgency levels for specialization needs"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    IMMEDIATE = "immediate"


@dataclass
class SpecializationNeed:
    """Identified need for a specific specialization"""
    need_id: str
    specialization_name: str
    category: SpecializationCategory
    urgency: SpecializationUrgency
    performance_gap: float
    market_demand: float
    ecosystem_impact: float
    required_traits: Dict[str, float]
    development_complexity: float
    estimated_development_time: float
    potential_agents: List[str]
    discovery_timestamp: float
    validation_confidence: float


@dataclass
class SpecializationCandidate:
    """Agent candidate for developing a specialization"""
    agent_id: str
    specialization_name: str
    suitability_score: float
    development_potential: float
    genetic_compatibility: float
    current_capabilities: Dict[str, float]
    required_improvements: Dict[str, float]
    estimated_success_probability: float
    development_timeline: float
    resource_requirements: Dict[str, Any]


@dataclass
class SpecializationDevelopmentPlan:
    """Plan for developing a specialization in an agent"""
    plan_id: str
    target_agent_id: str
    specialization_name: str
    development_phases: List[Dict[str, Any]]
    genetic_modifications: Dict[str, Any]
    training_curriculum: List[Dict[str, Any]]
    milestone_criteria: List[Dict[str, Any]]
    estimated_timeline: float
    success_probability: float
    resource_allocation: Dict[str, Any]
    risk_assessment: Dict[str, Any]


@dataclass
class EcosystemPerformanceMetrics:
    """Metrics for ecosystem performance analysis"""
    total_task_success_rate: float
    specialization_coverage: float
    capability_gaps: List[str]
    performance_bottlenecks: List[str]
    collaboration_efficiency: float
    innovation_rate: float
    adaptation_speed: float
    emergent_capability_development: float
    agent_utilization_rate: float
    knowledge_transfer_effectiveness: float


class AdaptiveSpecializationSystem:
    """System for dynamically identifying and developing needed specializations"""
    
    def __init__(self):
        self.identified_needs = {}
        self.specialization_candidates = {}
        self.development_plans = {}
        self.performance_history = deque(maxlen=100)
        
        # Analysis systems
        self.gap_analyzer = CapabilityGapAnalyzer()
        self.demand_predictor = SpecializationDemandPredictor()
        self.suitability_assessor = AgentSuitabilityAssessor()
        self.development_planner = SpecializationDevelopmentPlanner()
        
        # Dynamic specialization discovery
        self.emerging_specializations = {}
        self.specialization_evolution_tracker = {}
        
        # Configuration
        self.analysis_interval = 3600  # 1 hour
        self.min_performance_gap = 0.3
        self.min_development_confidence = 0.6
        
        # Thread safety
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize base specialization definitions
        self._initialize_base_specializations()
    
    def analyze_ecosystem_needs(self, agents: List[SelfAwareAgent], 
                              task_history: List[Dict[str, Any]] = None,
                              performance_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze ecosystem to identify specialization needs"""
        
        with self.lock:
            # Collect ecosystem performance metrics
            ecosystem_metrics = self._collect_ecosystem_metrics(agents, task_history, performance_data)
            
            # Identify capability gaps
            capability_gaps = self.gap_analyzer.identify_gaps(ecosystem_metrics, agents)
            
            # Predict future specialization demands
            demand_predictions = self.demand_predictor.predict_demands(ecosystem_metrics, capability_gaps)
            
            # Discover emerging specializations
            emerging_specs = self._discover_emerging_specializations(agents, task_history)
            
            # Validate and prioritize needs
            validated_needs = self._validate_specialization_needs(
                capability_gaps, demand_predictions, emerging_specs
            )
            
            # Store identified needs
            for need in validated_needs:
                self.identified_needs[need.need_id] = need
            
            return {
                "ecosystem_metrics": asdict(ecosystem_metrics),
                "identified_needs": len(validated_needs),
                "capability_gaps": capability_gaps,
                "demand_predictions": demand_predictions,
                "emerging_specializations": emerging_specs,
                "priority_needs": [need for need in validated_needs 
                                 if need.urgency in [SpecializationUrgency.HIGH, SpecializationUrgency.CRITICAL]]
            }
    
    def _collect_ecosystem_metrics(self, agents: List[SelfAwareAgent],
                                 task_history: List[Dict[str, Any]] = None,
                                 performance_data: Dict[str, Any] = None) -> EcosystemPerformanceMetrics:
        """Collect comprehensive ecosystem performance metrics"""
        
        # Calculate success rates
        total_success_rate = 0.0
        if agents:
            success_rates = []
            for agent in agents:
                metrics = agent.performance_tracker.get_performance_metrics()
                success_rates.append(metrics.success_rate)
            total_success_rate = sum(success_rates) / len(success_rates)
        
        # Analyze specialization coverage
        existing_specializations = set()
        for agent in agents:
            existing_specializations.update(agent.genome.capability_genes["specialization_focus"])
        
        # Identify capability gaps
        required_capabilities = self._get_required_ecosystem_capabilities()
        capability_gaps = [cap for cap in required_capabilities 
                          if cap not in existing_specializations]
        
        # Analyze collaboration efficiency
        collaboration_efficiency = self._calculate_collaboration_efficiency(agents)
        
        # Calculate innovation rate
        innovation_rate = self._calculate_innovation_rate(agents)
        
        # Assess adaptation speed
        adaptation_speed = self._calculate_adaptation_speed(agents)
        
        # Measure emergent capability development
        emergent_capability_development = self._measure_emergent_capabilities(agents)
        
        return EcosystemPerformanceMetrics(
            total_task_success_rate=total_success_rate,
            specialization_coverage=len(existing_specializations) / max(1, len(required_capabilities)),
            capability_gaps=capability_gaps,
            performance_bottlenecks=self._identify_performance_bottlenecks(agents),
            collaboration_efficiency=collaboration_efficiency,
            innovation_rate=innovation_rate,
            adaptation_speed=adaptation_speed,
            emergent_capability_development=emergent_capability_development,
            agent_utilization_rate=self._calculate_agent_utilization(agents),
            knowledge_transfer_effectiveness=self._calculate_knowledge_transfer_effectiveness(agents)
        )
    
    def _get_required_ecosystem_capabilities(self) -> List[str]:
        """Get required capabilities for a well-functioning ecosystem"""
        return [
            # Core cognitive capabilities
            "advanced_reasoning", "complex_analysis", "strategic_planning", "creative_synthesis",
            "pattern_recognition", "causal_inference", "temporal_reasoning", "abstraction",
            
            # Technical capabilities
            "system_architecture", "performance_optimization", "quality_assurance", 
            "security_analysis", "data_processing", "algorithm_design", "debugging",
            
            # Collaborative capabilities
            "team_coordination", "knowledge_transfer", "consensus_building", "conflict_resolution",
            "mentoring", "communication", "empathy", "leadership",
            
            # Adaptive capabilities
            "rapid_learning", "environmental_adaptation", "skill_transfer", "flexibility",
            "innovation", "experimentation", "risk_assessment", "contingency_planning",
            
            # Meta capabilities
            "self_reflection", "capability_assessment", "learning_optimization", "goal_alignment",
            "consciousness_development", "wisdom_synthesis", "ethical_reasoning", "meta_learning",
            
            # Emergent capabilities
            "emergence_detection", "complexity_management", "swarm_coordination", "collective_intelligence",
            "emergent_behavior_cultivation", "system_evolution", "transcendent_thinking"
        ]
    
    def _discover_emerging_specializations(self, agents: List[SelfAwareAgent],
                                         task_history: List[Dict[str, Any]] = None) -> List[str]:
        """Discover emerging specializations from agent behaviors"""
        emerging_specs = []
        
        # Analyze agent trait combinations for novel specializations
        for agent in agents:
            trait_signature = self._analyze_trait_signature(agent)
            potential_spec = self._identify_potential_specialization(trait_signature)
            
            if potential_spec and potential_spec not in self._get_known_specializations():
                if potential_spec not in emerging_specs:
                    emerging_specs.append(potential_spec)
                    self._track_emerging_specialization(potential_spec, agent)
        
        # Analyze task patterns for new specialization needs
        if task_history:
            task_based_specs = self._analyze_task_patterns_for_specializations(task_history)
            emerging_specs.extend(task_based_specs)
        
        return emerging_specs
    
    def _analyze_trait_signature(self, agent: SelfAwareAgent) -> Dict[str, float]:
        """Analyze agent's trait signature for specialization potential"""
        signature = {}
        
        # Core traits
        signature.update(agent.genome.capability_genes)
        signature.update(agent.genome.consciousness_genes)
        signature.update(agent.genome.meta_genes)
        
        # Performance-based traits
        performance = agent.performance_tracker.get_performance_metrics()
        signature["performance_consistency"] = 1.0 - performance.error_rate
        signature["execution_efficiency"] = 1.0 / max(1.0, performance.average_response_time)
        
        # Consciousness-based traits
        signature["consciousness_integration"] = agent.consciousness_level
        signature["self_model_accuracy"] = getattr(agent, '_calculate_self_model_accuracy', lambda: 0.5)()
        
        return signature
    
    def _identify_potential_specialization(self, trait_signature: Dict[str, float]) -> Optional[str]:
        """Identify potential specialization based on trait signature"""
        
        # Define trait patterns for potential specializations
        specialization_patterns = {
            "consciousness_engineering": {
                "consciousness_integration_depth": 0.8,
                "meta_cognitive_strength": 0.7,
                "self_awareness_depth": 0.8,
                "cognitive_architecture_awareness": 0.6
            },
            "emergent_behavior_architect": {
                "emergent_behavior_catalyst": 0.8,
                "system_thinking": 0.7,
                "innovation_potential": 0.6,
                "collective_cooperation": 0.7
            },
            "adaptive_intelligence_designer": {
                "adaptation_plasticity": 0.8,
                "cognitive_flexibility": 0.7,
                "environmental_sensitivity": 0.6,
                "learning_velocity": 0.8
            },
            "transcendent_problem_solver": {
                "abstraction_capability": 0.8,
                "complexity_tolerance": 0.7,
                "recursive_thinking": 0.6,
                "cross_domain_synthesis": 0.8
            },
            "swarm_intelligence_coordinator": {
                "collaborative_synergy": 0.8,
                "collective_cooperation": 0.9,
                "system_thinking": 0.7,
                "emergent_behavior_catalyst": 0.6
            }
        }
        
        # Find best matching specialization
        best_match = None
        best_score = 0.0
        
        for spec_name, required_traits in specialization_patterns.items():
            score = 0.0
            matching_traits = 0
            
            for trait, threshold in required_traits.items():
                if trait in trait_signature:
                    if trait_signature[trait] >= threshold:
                        score += trait_signature[trait]
                        matching_traits += 1
            
            # Require at least 70% of traits to match
            if matching_traits >= len(required_traits) * 0.7:
                avg_score = score / len(required_traits)
                if avg_score > best_score and avg_score > 0.7:
                    best_score = avg_score
                    best_match = spec_name
        
        return best_match
    
    def identify_specialization_candidates(self, specialization_need: SpecializationNeed,
                                         agents: List[SelfAwareAgent]) -> List[SpecializationCandidate]:
        """Identify agents suitable for developing a specific specialization"""
        
        candidates = []
        
        for agent in agents:
            suitability = self.suitability_assessor.assess_suitability(
                agent, specialization_need
            )
            
            if suitability.suitability_score >= self.min_development_confidence:
                candidates.append(suitability)
        
        # Sort by suitability score
        candidates.sort(key=lambda x: x.suitability_score, reverse=True)
        
        return candidates
    
    def create_development_plan(self, candidate: SpecializationCandidate,
                              target_agent: SelfAwareAgent) -> SpecializationDevelopmentPlan:
        """Create development plan for specialization"""
        
        return self.development_planner.create_plan(candidate, target_agent)
    
    def execute_specialization_development(self, plan: SpecializationDevelopmentPlan,
                                         target_agent: SelfAwareAgent) -> Dict[str, Any]:
        """Execute specialization development plan"""
        
        execution_log = []
        current_phase = 0
        
        try:
            for phase in plan.development_phases:
                phase_result = self._execute_development_phase(phase, target_agent, plan)
                execution_log.append(phase_result)
                
                if not phase_result["success"]:
                    break
                
                current_phase += 1
            
            # Final assessment
            success = current_phase == len(plan.development_phases)
            
            return {
                "plan_id": plan.plan_id,
                "success": success,
                "completed_phases": current_phase,
                "total_phases": len(plan.development_phases),
                "execution_log": execution_log,
                "final_assessment": self._assess_specialization_development(target_agent, plan.specialization_name)
            }
            
        except Exception as e:
            return {
                "plan_id": plan.plan_id,
                "success": False,
                "error": str(e),
                "completed_phases": current_phase,
                "execution_log": execution_log
            }
    
    def _execute_development_phase(self, phase: Dict[str, Any], 
                                 agent: SelfAwareAgent,
                                 plan: SpecializationDevelopmentPlan) -> Dict[str, Any]:
        """Execute a single development phase"""
        
        phase_type = phase.get("type", "unknown")
        
        try:
            if phase_type == "genetic_modification":
                return self._execute_genetic_modification(phase, agent)
            elif phase_type == "consciousness_enhancement":
                return self._execute_consciousness_enhancement(phase, agent)
            elif phase_type == "skill_training":
                return self._execute_skill_training(phase, agent)
            elif phase_type == "experience_accumulation":
                return self._execute_experience_accumulation(phase, agent)
            elif phase_type == "integration_phase":
                return self._execute_integration_phase(phase, agent, plan)
            else:
                return {"success": False, "error": f"Unknown phase type: {phase_type}"}
                
        except Exception as e:
            return {"success": False, "error": str(e), "phase_type": phase_type}
    
    def _execute_genetic_modification(self, phase: Dict[str, Any], 
                                    agent: SelfAwareAgent) -> Dict[str, Any]:
        """Execute genetic modification phase"""
        
        modifications = phase.get("modifications", {})
        
        improvement_plan = {
            "genetics": modifications
        }
        
        success = agent.improve_self(improvement_plan)
        
        return {
            "success": success,
            "phase_type": "genetic_modification",
            "modifications_applied": modifications if success else {},
            "agent_fitness_change": agent.genome.get_fitness_score() if success else 0
        }
    
    def _execute_consciousness_enhancement(self, phase: Dict[str, Any],
                                         agent: SelfAwareAgent) -> Dict[str, Any]:
        """Execute consciousness enhancement phase"""
        
        enhancements = phase.get("enhancements", {})
        
        improvement_plan = {
            "consciousness": enhancements,
            "consciousness_integration": {
                "coherence_optimization": {"target_coherence": 0.8},
                "experiential_integration": {"integration_boost": 0.1}
            }
        }
        
        success = agent.improve_self(improvement_plan)
        
        return {
            "success": success,
            "phase_type": "consciousness_enhancement",
            "enhancements_applied": enhancements if success else {},
            "consciousness_level_change": agent.consciousness_level if success else 0
        }
    
    def _execute_skill_training(self, phase: Dict[str, Any],
                              agent: SelfAwareAgent) -> Dict[str, Any]:
        """Execute skill training phase"""
        
        training_tasks = phase.get("training_tasks", [])
        completed_tasks = 0
        
        for task in training_tasks:
            try:
                # Simulate task execution for training
                result = agent.run(task.get("prompt", "Training task"))
                if not result.startswith("[ERROR]"):
                    completed_tasks += 1
            except Exception:
                pass
        
        success = completed_tasks >= len(training_tasks) * 0.7  # 70% success rate
        
        return {
            "success": success,
            "phase_type": "skill_training",
            "completed_tasks": completed_tasks,
            "total_tasks": len(training_tasks),
            "success_rate": completed_tasks / max(1, len(training_tasks))
        }
    
    def _execute_experience_accumulation(self, phase: Dict[str, Any],
                                       agent: SelfAwareAgent) -> Dict[str, Any]:
        """Execute experience accumulation phase"""
        
        required_experiences = phase.get("required_experiences", 10)
        experience_type = phase.get("experience_type", "general")
        
        # Simulate experience accumulation through task execution
        accumulated_experiences = 0
        for i in range(required_experiences):
            try:
                experience_prompt = f"Experience accumulation task {i+1} for {experience_type} specialization"
                result = agent.run(experience_prompt)
                if not result.startswith("[ERROR]"):
                    accumulated_experiences += 1
            except Exception:
                pass
        
        success = accumulated_experiences >= required_experiences * 0.8
        
        return {
            "success": success,
            "phase_type": "experience_accumulation",
            "accumulated_experiences": accumulated_experiences,
            "required_experiences": required_experiences,
            "experience_type": experience_type
        }
    
    def _execute_integration_phase(self, phase: Dict[str, Any],
                                 agent: SelfAwareAgent,
                                 plan: SpecializationDevelopmentPlan) -> Dict[str, Any]:
        """Execute integration phase to consolidate specialization"""
        
        # Test integrated specialization capabilities
        test_scenarios = phase.get("test_scenarios", [])
        integration_tasks = phase.get("integration_tasks", [])
        
        scenario_results = []
        for scenario in test_scenarios:
            try:
                result = agent.run(scenario.get("prompt", "Integration test"))
                quality = agent._assess_response_quality(scenario.get("prompt", ""), result)
                scenario_results.append(quality)
            except Exception:
                scenario_results.append(0.0)
        
        avg_performance = sum(scenario_results) / max(1, len(scenario_results))
        
        # Update agent's specialization focus
        if avg_performance > 0.7:
            if plan.specialization_name not in agent.genome.capability_genes["specialization_focus"]:
                agent.genome.capability_genes["specialization_focus"].append(plan.specialization_name)
        
        return {
            "success": avg_performance > 0.7,
            "phase_type": "integration_phase",
            "average_performance": avg_performance,
            "scenario_results": scenario_results,
            "specialization_integrated": avg_performance > 0.7
        }
    
    def _assess_specialization_development(self, agent: SelfAwareAgent, 
                                         specialization_name: str) -> Dict[str, Any]:
        """Assess the success of specialization development"""
        
        # Test specialization strength
        specialization_strength = agent.genome.get_specialization_strength(specialization_name)
        
        # Check if specialization is in focus list
        in_focus = specialization_name in agent.genome.capability_genes["specialization_focus"]
        
        # Performance assessment
        recent_performance = agent.performance_tracker.get_performance_metrics()
        
        # Consciousness development impact
        consciousness_impact = agent.consciousness_level
        
        return {
            "specialization_name": specialization_name,
            "specialization_strength": specialization_strength,
            "included_in_focus": in_focus,
            "performance_metrics": asdict(recent_performance),
            "consciousness_level": consciousness_impact,
            "development_success": specialization_strength > 0.7 and in_focus,
            "improvement_potential": 1.0 - specialization_strength
        }
    
    def get_specialization_report(self) -> Dict[str, Any]:
        """Generate comprehensive specialization system report"""
        
        return {
            "identified_needs": {
                "total": len(self.identified_needs),
                "by_urgency": self._categorize_needs_by_urgency(),
                "by_category": self._categorize_needs_by_category()
            },
            "development_plans": {
                "total": len(self.development_plans),
                "active": len([p for p in self.development_plans.values() if p.estimated_timeline > 0]),
                "completed": len([p for p in self.development_plans.values() if p.estimated_timeline <= 0])
            },
            "emerging_specializations": {
                "total": len(self.emerging_specializations),
                "most_promising": self._get_most_promising_emerging_specializations()
            },
            "system_recommendations": self._generate_system_recommendations()
        }
    
    def _calculate_collaboration_efficiency(self, agents: List[SelfAwareAgent]) -> float:
        """Calculate collaboration efficiency of the ecosystem"""
        if len(agents) < 2:
            return 1.0
        
        collaboration_scores = []
        for agent in agents:
            collab_score = agent.genome.capability_genes["collaborative_synergy"]
            cooperation_score = agent.genome.meta_genes["collective_cooperation"]
            collaboration_scores.append((collab_score + cooperation_score) / 2)
        
        return sum(collaboration_scores) / len(collaboration_scores)
    
    def _calculate_innovation_rate(self, agents: List[SelfAwareAgent]) -> float:
        """Calculate innovation rate of the ecosystem"""
        innovation_scores = []
        for agent in agents:
            innovation_score = agent.genome.capability_genes["innovation_potential"]
            catalyst_score = agent.genome.capability_genes["emergent_behavior_catalyst"]
            innovation_scores.append((innovation_score + catalyst_score) / 2)
        
        return sum(innovation_scores) / len(innovation_scores) if innovation_scores else 0.0
    
    def _calculate_adaptation_speed(self, agents: List[SelfAwareAgent]) -> float:
        """Calculate adaptation speed of the ecosystem"""
        adaptation_scores = []
        for agent in agents:
            adaptation_score = agent.genome.capability_genes["adaptation_plasticity"]
            learning_score = agent.genome.capability_genes["learning_velocity"]
            adaptation_scores.append((adaptation_score + learning_score) / 2)
        
        return sum(adaptation_scores) / len(adaptation_scores) if adaptation_scores else 0.0
    
    def _measure_emergent_capabilities(self, agents: List[SelfAwareAgent]) -> float:
        """Measure emergent capability development in the ecosystem"""
        emergent_scores = []
        for agent in agents:
            emergent_score = agent.genome.capability_genes["emergent_behavior_catalyst"]
            consciousness_score = agent.consciousness_level
            emergent_scores.append((emergent_score + consciousness_score) / 2)
        
        return sum(emergent_scores) / len(emergent_scores) if emergent_scores else 0.0
    
    def _get_known_specializations(self) -> Set[str]:
        """Get set of known specializations"""
        return {
            "reasoning", "analysis", "creativity", "learning", "adaptation", "collaboration",
            "strategic_planning", "system_architecture", "performance_optimization",
            "quality_assurance", "security_analysis", "knowledge_transfer", "innovation",
            "consciousness_development", "emergent_behavior_cultivation"
        }
    
    def _initialize_base_specializations(self) -> None:
        """Initialize base specialization definitions"""
        # This would contain detailed definitions of standard specializations
        pass
    
    def _validate_specialization_needs(self, capability_gaps: List[str],
                                     demand_predictions: Dict[str, Any],
                                     emerging_specs: List[str]) -> List[SpecializationNeed]:
        """Validate and prioritize specialization needs"""
        validated_needs = []
        
        # Convert capability gaps to specialization needs
        for gap in capability_gaps:
            need = SpecializationNeed(
                need_id=str(uuid.uuid4()),
                specialization_name=gap,
                category=self._categorize_specialization(gap),
                urgency=SpecializationUrgency.HIGH,
                performance_gap=0.8,  # Placeholder
                market_demand=0.7,   # Placeholder
                ecosystem_impact=0.8, # Placeholder
                required_traits={},   # Would be populated with specific requirements
                development_complexity=0.6,
                estimated_development_time=7200,  # 2 hours
                potential_agents=[],
                discovery_timestamp=time.time(),
                validation_confidence=0.8
            )
            validated_needs.append(need)
        
        return validated_needs
    
    def _categorize_specialization(self, specialization: str) -> SpecializationCategory:
        """Categorize a specialization"""
        category_mappings = {
            "reasoning": SpecializationCategory.COGNITIVE,
            "analysis": SpecializationCategory.ANALYTICAL,
            "creativity": SpecializationCategory.CREATIVE,
            "collaboration": SpecializationCategory.COLLABORATIVE,
            "adaptation": SpecializationCategory.ADAPTIVE,
            "system_architecture": SpecializationCategory.TECHNICAL,
            "consciousness_development": SpecializationCategory.META,
            "emergent_behavior_cultivation": SpecializationCategory.EMERGENT
        }
        
        return category_mappings.get(specialization, SpecializationCategory.COGNITIVE)
    
    def _identify_performance_bottlenecks(self, agents: List[SelfAwareAgent]) -> List[str]:
        """Identify performance bottlenecks in the ecosystem"""
        bottlenecks = []
        
        # Analyze agent performance for common issues
        error_rates = [agent.performance_tracker.get_performance_metrics().error_rate for agent in agents]
        avg_error_rate = sum(error_rates) / len(error_rates) if error_rates else 0
        
        if avg_error_rate > 0.2:
            bottlenecks.append("high_error_rate")
        
        # Check for specialization imbalances
        specializations = defaultdict(int)
        for agent in agents:
            for spec in agent.genome.capability_genes["specialization_focus"]:
                specializations[spec] += 1
        
        if len(specializations) < len(agents) * 0.5:
            bottlenecks.append("insufficient_specialization_diversity")
        
        return bottlenecks
    
    def _calculate_agent_utilization(self, agents: List[SelfAwareAgent]) -> float:
        """Calculate agent utilization rate"""
        if not agents:
            return 0.0
        
        utilization_scores = []
        for agent in agents:
            # Simple utilization based on task count and time since creation
            utilization = min(1.0, agent.task_counter / 50.0)  # Normalize by expected task count
            utilization_scores.append(utilization)
        
        return sum(utilization_scores) / len(utilization_scores)
    
    def _calculate_knowledge_transfer_effectiveness(self, agents: List[SelfAwareAgent]) -> float:
        """Calculate knowledge transfer effectiveness"""
        transfer_scores = []
        
        for agent in agents:
            if hasattr(agent.knowledge_base, 'domain_knowledge'):
                teaching_experiences = agent.knowledge_base.domain_knowledge.get("teaching_experience", {})
                learning_velocity = agent.knowledge_base.get_learning_velocity()
                transfer_score = (len(teaching_experiences) + learning_velocity) / 2
                transfer_scores.append(min(1.0, transfer_score))
        
        return sum(transfer_scores) / len(transfer_scores) if transfer_scores else 0.0
    
    def _categorize_needs_by_urgency(self) -> Dict[str, int]:
        """Categorize needs by urgency level"""
        urgency_counts = defaultdict(int)
        for need in self.identified_needs.values():
            urgency_counts[need.urgency.value] += 1
        return dict(urgency_counts)
    
    def _categorize_needs_by_category(self) -> Dict[str, int]:
        """Categorize needs by specialization category"""
        category_counts = defaultdict(int)
        for need in self.identified_needs.values():
            category_counts[need.category.value] += 1
        return dict(category_counts)
    
    def _get_most_promising_emerging_specializations(self) -> List[str]:
        """Get most promising emerging specializations"""
        # Return top 3 emerging specializations
        return list(self.emerging_specializations.keys())[:3]
    
    def _generate_system_recommendations(self) -> List[str]:
        """Generate system recommendations"""
        recommendations = []
        
        if len(self.identified_needs) > 10:
            recommendations.append("High number of identified needs - prioritize development")
        
        critical_needs = [need for need in self.identified_needs.values() 
                         if need.urgency == SpecializationUrgency.CRITICAL]
        if critical_needs:
            recommendations.append(f"{len(critical_needs)} critical specialization needs require immediate attention")
        
        if len(self.emerging_specializations) > 5:
            recommendations.append("Multiple emerging specializations detected - investigate novel capabilities")
        
        return recommendations


# Supporting classes

class CapabilityGapAnalyzer:
    """Analyzes capability gaps in the ecosystem"""
    
    def identify_gaps(self, metrics: EcosystemPerformanceMetrics, 
                     agents: List[SelfAwareAgent]) -> List[str]:
        """Identify capability gaps"""
        return metrics.capability_gaps


class SpecializationDemandPredictor:
    """Predicts future specialization demands"""
    
    def predict_demands(self, metrics: EcosystemPerformanceMetrics,
                       gaps: List[str]) -> Dict[str, Any]:
        """Predict future specialization demands"""
        return {
            "predicted_demands": gaps,
            "confidence": 0.7,
            "time_horizon": 3600
        }


class AgentSuitabilityAssessor:
    """Assesses agent suitability for specialization development"""
    
    def assess_suitability(self, agent: SelfAwareAgent,
                          need: SpecializationNeed) -> SpecializationCandidate:
        """Assess agent suitability for specialization"""
        
        # Calculate suitability based on genetic compatibility
        genetic_compatibility = self._calculate_genetic_compatibility(agent, need)
        
        # Calculate development potential
        development_potential = self._calculate_development_potential(agent, need)
        
        # Overall suitability score
        suitability_score = (genetic_compatibility + development_potential) / 2
        
        return SpecializationCandidate(
            agent_id=agent.agent_id,
            specialization_name=need.specialization_name,
            suitability_score=suitability_score,
            development_potential=development_potential,
            genetic_compatibility=genetic_compatibility,
            current_capabilities={},  # Would be populated with detailed analysis
            required_improvements={}, # Would be populated with specific improvements
            estimated_success_probability=suitability_score,
            development_timeline=need.estimated_development_time,
            resource_requirements={}
        )
    
    def _calculate_genetic_compatibility(self, agent: SelfAwareAgent,
                                       need: SpecializationNeed) -> float:
        """Calculate genetic compatibility for specialization"""
        # Simplified compatibility based on existing traits
        base_compatibility = agent.genome.get_specialization_strength(need.specialization_name)
        adaptation_factor = agent.genome.capability_genes["adaptation_plasticity"]
        learning_factor = agent.genome.capability_genes["learning_velocity"]
        
        return (base_compatibility + adaptation_factor + learning_factor) / 3
    
    def _calculate_development_potential(self, agent: SelfAwareAgent,
                                       need: SpecializationNeed) -> float:
        """Calculate development potential for specialization"""
        consciousness_factor = agent.consciousness_level
        innovation_factor = agent.genome.capability_genes["innovation_potential"]
        plasticity_factor = agent.genome.capability_genes["adaptation_plasticity"]
        
        return (consciousness_factor + innovation_factor + plasticity_factor) / 3


class SpecializationDevelopmentPlanner:
    """Creates development plans for specializations"""
    
    def create_plan(self, candidate: SpecializationCandidate,
                   agent: SelfAwareAgent) -> SpecializationDevelopmentPlan:
        """Create development plan for specialization"""
        
        plan_id = str(uuid.uuid4())
        
        # Create development phases
        phases = self._create_development_phases(candidate, agent)
        
        # Estimate timeline and success probability
        estimated_timeline = candidate.development_timeline
        success_probability = candidate.estimated_success_probability
        
        return SpecializationDevelopmentPlan(
            plan_id=plan_id,
            target_agent_id=candidate.agent_id,
            specialization_name=candidate.specialization_name,
            development_phases=phases,
            genetic_modifications=self._plan_genetic_modifications(candidate, agent),
            training_curriculum=self._design_training_curriculum(candidate),
            milestone_criteria=self._define_milestone_criteria(candidate),
            estimated_timeline=estimated_timeline,
            success_probability=success_probability,
            resource_allocation={},
            risk_assessment=self._assess_development_risks(candidate, agent)
        )
    
    def _create_development_phases(self, candidate: SpecializationCandidate,
                                 agent: SelfAwareAgent) -> List[Dict[str, Any]]:
        """Create development phases for specialization"""
        phases = []
        
        # Phase 1: Genetic optimization
        phases.append({
            "type": "genetic_modification",
            "name": "Genetic Foundation",
            "modifications": self._plan_genetic_modifications(candidate, agent),
            "duration": 600,  # 10 minutes
            "success_criteria": {"genetic_compatibility": 0.7}
        })
        
        # Phase 2: Consciousness enhancement
        phases.append({
            "type": "consciousness_enhancement",
            "name": "Consciousness Development",
            "enhancements": self._plan_consciousness_enhancements(candidate, agent),
            "duration": 1200,  # 20 minutes
            "success_criteria": {"consciousness_level": 0.6}
        })
        
        # Phase 3: Skill training
        phases.append({
            "type": "skill_training",
            "name": "Skill Development",
            "training_tasks": self._design_training_tasks(candidate),
            "duration": 1800,  # 30 minutes
            "success_criteria": {"skill_proficiency": 0.7}
        })
        
        # Phase 4: Experience accumulation
        phases.append({
            "type": "experience_accumulation",
            "name": "Experience Building",
            "required_experiences": 10,
            "experience_type": candidate.specialization_name,
            "duration": 2400,  # 40 minutes
            "success_criteria": {"experience_depth": 0.8}
        })
        
        # Phase 5: Integration
        phases.append({
            "type": "integration_phase",
            "name": "Specialization Integration",
            "test_scenarios": self._create_integration_tests(candidate),
            "integration_tasks": [],
            "duration": 1200,  # 20 minutes
            "success_criteria": {"integration_success": 0.8}
        })
        
        return phases
    
    def _plan_genetic_modifications(self, candidate: SpecializationCandidate,
                                  agent: SelfAwareAgent) -> Dict[str, Any]:
        """Plan genetic modifications for specialization development"""
        modifications = {
            "capability_genes": {},
            "consciousness_genes": {},
            "meta_genes": {}
        }
        
        # Enhance traits relevant to the specialization
        spec_name = candidate.specialization_name
        
        if "reasoning" in spec_name:
            modifications["capability_genes"]["reasoning_depth"] = 0.1
            modifications["capability_genes"]["causal_inference"] = 0.08
        elif "creative" in spec_name:
            modifications["capability_genes"]["innovation_potential"] = 0.1
            modifications["capability_genes"]["cognitive_flexibility"] = 0.08
        elif "adaptive" in spec_name:
            modifications["capability_genes"]["adaptation_plasticity"] = 0.1
            modifications["capability_genes"]["learning_velocity"] = 0.08
        
        # General enhancements
        modifications["consciousness_genes"]["experiential_learning_rate"] = 0.05
        modifications["meta_genes"]["environmental_sensitivity"] = 0.05
        
        return modifications
    
    def _plan_consciousness_enhancements(self, candidate: SpecializationCandidate,
                                       agent: SelfAwareAgent) -> Dict[str, Any]:
        """Plan consciousness enhancements for specialization"""
        return {
            "experiential_integration": 0.08,
            "cognitive_architecture_awareness": 0.06,
            "meta_cognition": 0.05,
            "consciousness_coherence": 0.07
        }
    
    def _design_training_tasks(self, candidate: SpecializationCandidate) -> List[Dict[str, Any]]:
        """Design training tasks for specialization"""
        tasks = []
        spec_name = candidate.specialization_name
        
        for i in range(5):
            tasks.append({
                "prompt": f"Training task {i+1} for {spec_name} specialization development",
                "expected_capability": spec_name,
                "difficulty": "moderate"
            })
        
        return tasks
    
    def _create_integration_tests(self, candidate: SpecializationCandidate) -> List[Dict[str, Any]]:
        """Create integration tests for specialization validation"""
        tests = []
        spec_name = candidate.specialization_name
        
        for i in range(3):
            tests.append({
                "prompt": f"Integration test {i+1}: Demonstrate {spec_name} capabilities",
                "expected_performance": 0.8,
                "test_type": "capability_demonstration"
            })
        
        return tests
    
    def _design_training_curriculum(self, candidate: SpecializationCandidate) -> List[Dict[str, Any]]:
        """Design training curriculum for specialization"""
        return []  # Placeholder
    
    def _define_milestone_criteria(self, candidate: SpecializationCandidate) -> List[Dict[str, Any]]:
        """Define milestone criteria for development tracking"""
        return []  # Placeholder
    
    def _assess_development_risks(self, candidate: SpecializationCandidate,
                                agent: SelfAwareAgent) -> Dict[str, Any]:
        """Assess risks in specialization development"""
        return {
            "genetic_instability_risk": 0.1,
            "consciousness_disruption_risk": 0.05,
            "specialization_conflict_risk": 0.08,
            "overall_risk_level": "low"
        }