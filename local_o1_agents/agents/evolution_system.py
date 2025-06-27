"""
Bullpen Evolution System for Continuous Improvement

This module implements an evolutionary system that continuously improves
the agent population through selection, breeding, and optimization.
"""

import time
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from copy import deepcopy

from .self_aware_agent import SelfAwareAgent
from .agent_bullpen import AgentBullpen, AgentStatus
from .master_agent_factory import MasterAgentFactory, AgentRequirements
from .mentorship_system import AgentMentorshipSystem, TrainingCurriculum


class EvolutionStrategy(Enum):
    """Different evolution strategies"""
    PERFORMANCE_BASED = "performance_based"
    DIVERSITY_FOCUSED = "diversity_focused"
    SPECIALIZATION_DRIVEN = "specialization_driven"
    HYBRID = "hybrid"


@dataclass
class EvolutionReport:
    """Report of evolution cycle results"""
    cycle_number: int
    start_time: datetime
    end_time: datetime
    strategy_used: EvolutionStrategy
    
    # Agent lifecycle changes
    retired_agents: List[str]
    new_agents: List[str]
    promoted_agents: List[str]  # Promoted to master status
    
    # Training and improvement
    training_sessions_conducted: int
    agents_improved: int
    
    # Performance metrics
    population_fitness_before: float
    population_fitness_after: float
    diversity_score_before: float
    diversity_score_after: float
    
    # Breeding statistics
    breeding_attempts: int
    successful_breedings: int
    
    # Overall improvement
    overall_improvement: float
    success: bool
    
    def __post_init__(self):
        if self.end_time and self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()


class PerformanceAnalyzer:
    """Analyzes agent and population performance"""
    
    def __init__(self):
        self.analysis_history: List[Dict[str, Any]] = []
    
    def analyze_all_agents(self, agents: Dict[str, SelfAwareAgent]) -> Dict[str, Any]:
        """Comprehensive analysis of all agents in the population"""
        analysis = {
            'timestamp': datetime.now(),
            'total_agents': len(agents),
            'agent_analyses': {},
            'population_metrics': {},
            'performance_distribution': {},
            'specialization_analysis': {}
        }
        
        # Individual agent analyses
        agent_fitness_scores = []
        specialization_counts = {}
        performance_metrics = []
        
        for agent_id, agent in agents.items():
            agent_analysis = self._analyze_individual_agent(agent)
            analysis['agent_analyses'][agent_id] = agent_analysis
            
            agent_fitness_scores.append(agent_analysis['fitness_score'])
            
            # Track specializations
            spec = agent.specialization or 'general'
            specialization_counts[spec] = specialization_counts.get(spec, 0) + 1
            
            # Collect performance metrics
            performance_metrics.append({
                'success_rate': agent.current_metrics.success_rate,
                'quality_score': agent.current_metrics.quality_score,
                'response_time': agent.current_metrics.average_response_time
            })
        
        # Population-level metrics
        if agent_fitness_scores:
            analysis['population_metrics'] = {
                'average_fitness': sum(agent_fitness_scores) / len(agent_fitness_scores),
                'max_fitness': max(agent_fitness_scores),
                'min_fitness': min(agent_fitness_scores),
                'fitness_std': self._calculate_std(agent_fitness_scores),
                'diversity_score': self._calculate_population_diversity(agents)
            }
        
        # Performance distribution analysis
        if performance_metrics:
            analysis['performance_distribution'] = self._analyze_performance_distribution(performance_metrics)
        
        # Specialization analysis
        analysis['specialization_analysis'] = {
            'distribution': specialization_counts,
            'balance_score': self._calculate_specialization_balance(specialization_counts),
            'coverage_gaps': self._identify_specialization_gaps(specialization_counts)
        }
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _analyze_individual_agent(self, agent: SelfAwareAgent) -> Dict[str, Any]:
        """Analyze individual agent performance and characteristics"""
        # Get agent's self-analysis
        self_analysis = agent.analyze_self()
        
        # Calculate fitness score
        fitness_score = agent.dna.get_fitness_score({
            'success_rate': agent.current_metrics.success_rate,
            'quality_score': agent.current_metrics.quality_score
        })
        
        # Performance trends
        performance_trends = agent.performance_history.get_performance_trends()
        
        return {
            'agent_id': agent.agent_id,
            'name': agent.name,
            'specialization': agent.specialization,
            'generation': agent.dna.generation,
            'fitness_score': fitness_score,
            'performance_metrics': asdict(agent.current_metrics),
            'capability_count': len(agent.knowledge_base.capabilities),
            'knowledge_domain_count': len(agent.knowledge_base.knowledge_domains),
            'task_count': agent.task_counter,
            'specialization_level': self_analysis.specialization_level,
            'strengths': self_analysis.strengths,
            'weaknesses': self_analysis.weaknesses,
            'improvement_opportunities': self_analysis.improvement_opportunities,
            'performance_trends': performance_trends,
            'last_active': agent.current_metrics.last_updated
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_population_diversity(self, agents: Dict[str, SelfAwareAgent]) -> float:
        """Calculate genetic/behavioral diversity of population"""
        if len(agents) < 2:
            return 0.0
        
        diversity_metrics = []
        agent_list = list(agents.values())
        
        # Compare agents pairwise
        for i, agent1 in enumerate(agent_list):
            for j, agent2 in enumerate(agent_list[i+1:], i+1):
                # Compare DNA characteristics
                dna1, dna2 = agent1.dna, agent2.dna
                
                # Specialization diversity
                spec_diff = 1.0 if agent1.specialization != agent2.specialization else 0.0
                
                # Capability diversity
                caps1 = set(dna1.capabilities.keys())
                caps2 = set(dna2.capabilities.keys())
                cap_diversity = len(caps1.symmetric_difference(caps2)) / max(len(caps1.union(caps2)), 1)
                
                # Behavioral trait diversity
                trait_diff = 0.0
                common_traits = set(dna1.behavioral_traits.keys()).intersection(set(dna2.behavioral_traits.keys()))
                if common_traits:
                    trait_diffs = [abs(dna1.behavioral_traits[trait].strength - dna2.behavioral_traits[trait].strength) 
                                  for trait in common_traits]
                    trait_diff = sum(trait_diffs) / len(trait_diffs)
                
                # Combined diversity score
                pair_diversity = (spec_diff * 0.4 + cap_diversity * 0.4 + trait_diff * 0.2)
                diversity_metrics.append(pair_diversity)
        
        return sum(diversity_metrics) / len(diversity_metrics) if diversity_metrics else 0.0
    
    def _analyze_performance_distribution(self, performance_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of performance metrics"""
        if not performance_metrics:
            return {}
        
        success_rates = [m['success_rate'] for m in performance_metrics]
        quality_scores = [m['quality_score'] for m in performance_metrics]
        response_times = [m['response_time'] for m in performance_metrics]
        
        return {
            'success_rate': {
                'mean': sum(success_rates) / len(success_rates),
                'min': min(success_rates),
                'max': max(success_rates),
                'std': self._calculate_std(success_rates)
            },
            'quality_score': {
                'mean': sum(quality_scores) / len(quality_scores),
                'min': min(quality_scores),
                'max': max(quality_scores),
                'std': self._calculate_std(quality_scores)
            },
            'response_time': {
                'mean': sum(response_times) / len(response_times),
                'min': min(response_times),
                'max': max(response_times),
                'std': self._calculate_std(response_times)
            }
        }
    
    def _calculate_specialization_balance(self, specialization_counts: Dict[str, int]) -> float:
        """Calculate how balanced the specialization distribution is"""
        if not specialization_counts:
            return 0.0
        
        total_agents = sum(specialization_counts.values())
        ideal_count = total_agents / len(specialization_counts)
        
        # Calculate deviation from ideal balance
        deviations = [abs(count - ideal_count) for count in specialization_counts.values()]
        avg_deviation = sum(deviations) / len(deviations)
        
        # Convert to balance score (1.0 = perfect balance, 0.0 = completely unbalanced)
        max_possible_deviation = ideal_count
        balance_score = 1.0 - (avg_deviation / max_possible_deviation) if max_possible_deviation > 0 else 1.0
        
        return max(0.0, balance_score)
    
    def _identify_specialization_gaps(self, specialization_counts: Dict[str, int]) -> List[str]:
        """Identify missing or underrepresented specializations"""
        desired_specializations = [
            'strategic_planning', 'code_analysis', 'data_analysis', 
            'performance_optimization', 'testing', 'debugging'
        ]
        
        gaps = []
        total_agents = sum(specialization_counts.values())
        min_threshold = max(1, total_agents // 10)  # At least 10% representation
        
        for spec in desired_specializations:
            current_count = specialization_counts.get(spec, 0)
            if current_count < min_threshold:
                gaps.append(spec)
        
        return gaps


class EvolutionEngine:
    """Core evolution engine for population optimization"""
    
    def __init__(self, agent_factory: MasterAgentFactory, mentorship_system: AgentMentorshipSystem):
        self.agent_factory = agent_factory
        self.mentorship_system = mentorship_system
        self.evolution_history: List[EvolutionReport] = []
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Evolution parameters
        self.population_size_target = 50
        self.retirement_threshold = 0.3  # Retire agents below 30% success rate
        self.breeding_rate = 0.2  # Breed 20% new agents per cycle
        self.mutation_rate = 0.1
        self.elite_preservation_rate = 0.1  # Keep top 10% as elite
    
    def evolve_population(self, bullpen: AgentBullpen, strategy: EvolutionStrategy = EvolutionStrategy.HYBRID) -> EvolutionReport:
        """Execute one evolution cycle on the agent population"""
        start_time = datetime.now()
        cycle_number = len(self.evolution_history) + 1
        
        print(f"Starting evolution cycle {cycle_number} with strategy {strategy.value}")
        
        # Analyze current population
        analysis_before = self.performance_analyzer.analyze_all_agents(bullpen.agents)
        
        # Initialize report
        report = EvolutionReport(
            cycle_number=cycle_number,
            start_time=start_time,
            end_time=None,
            strategy_used=strategy,
            retired_agents=[],
            new_agents=[],
            promoted_agents=[],
            training_sessions_conducted=0,
            agents_improved=0,
            population_fitness_before=analysis_before['population_metrics'].get('average_fitness', 0.0),
            diversity_score_before=analysis_before['population_metrics'].get('diversity_score', 0.0),
            breeding_attempts=0,
            successful_breedings=0,
            population_fitness_after=0.0,
            diversity_score_after=0.0,
            overall_improvement=0.0,
            success=False
        )
        
        try:
            # Step 1: Identify elite agents
            elite_agents = self._identify_elite_agents(bullpen, analysis_before)
            print(f"Identified {len(elite_agents)} elite agents")
            
            # Step 2: Retirement phase
            retired_agents = self._retirement_phase(bullpen, analysis_before, strategy)
            report.retired_agents = retired_agents
            print(f"Retired {len(retired_agents)} underperforming agents")
            
            # Step 3: Promotion phase
            promoted_agents = self._promotion_phase(bullpen, analysis_before)
            report.promoted_agents = promoted_agents
            print(f"Promoted {len(promoted_agents)} agents to master status")
            
            # Step 4: Breeding phase
            breeding_results = self._breeding_phase(bullpen, analysis_before, strategy)
            report.breeding_attempts = breeding_results['attempts']
            report.successful_breedings = breeding_results['successful']
            report.new_agents = breeding_results['new_agents']
            print(f"Breeding: {breeding_results['successful']}/{breeding_results['attempts']} successful")
            
            # Step 5: Training and improvement phase
            training_results = self._training_phase(bullpen, analysis_before, strategy)
            report.training_sessions_conducted = training_results['sessions']
            report.agents_improved = training_results['improved']
            print(f"Training: {training_results['sessions']} sessions, {training_results['improved']} agents improved")
            
            # Step 6: Population optimization
            optimization_results = self._optimization_phase(bullpen, analysis_before, strategy)
            print(f"Optimization completed: {optimization_results}")
            
            # Final analysis
            analysis_after = self.performance_analyzer.analyze_all_agents(bullpen.agents)
            report.population_fitness_after = analysis_after['population_metrics'].get('average_fitness', 0.0)
            report.diversity_score_after = analysis_after['population_metrics'].get('diversity_score', 0.0)
            
            # Calculate overall improvement
            fitness_improvement = report.population_fitness_after - report.population_fitness_before
            diversity_improvement = report.diversity_score_after - report.diversity_score_before
            report.overall_improvement = (fitness_improvement * 0.7 + diversity_improvement * 0.3)
            
            report.success = report.overall_improvement > 0.0
            
        except Exception as e:
            print(f"Evolution cycle failed: {e}")
            report.success = False
        
        report.end_time = datetime.now()
        self.evolution_history.append(report)
        
        print(f"Evolution cycle {cycle_number} completed. Success: {report.success}, "
              f"Improvement: {report.overall_improvement:.3f}")
        
        return report
    
    def _identify_elite_agents(self, bullpen: AgentBullpen, analysis: Dict[str, Any]) -> List[str]:
        """Identify top-performing agents for preservation"""
        agent_scores = []
        
        for agent_id, agent_analysis in analysis['agent_analyses'].items():
            fitness_score = agent_analysis['fitness_score']
            # Add bonus for experience
            experience_bonus = min(agent_analysis['task_count'] * 0.01, 0.2)
            total_score = fitness_score + experience_bonus
            agent_scores.append((agent_id, total_score))
        
        # Sort by score and take top percentage
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        elite_count = max(1, int(len(agent_scores) * self.elite_preservation_rate))
        
        return [agent_id for agent_id, _ in agent_scores[:elite_count]]
    
    def _retirement_phase(self, bullpen: AgentBullpen, analysis: Dict[str, Any], 
                         strategy: EvolutionStrategy) -> List[str]:
        """Retire underperforming or redundant agents"""
        retirement_candidates = []
        
        for agent_id, agent_analysis in analysis['agent_analyses'].items():
            agent = bullpen.get_agent_by_id(agent_id)
            if not agent:
                continue
            
            # Skip if agent doesn't have enough experience
            if agent_analysis['task_count'] < 5:
                continue
            
            should_retire = False
            
            # Performance-based retirement
            success_rate = agent_analysis['performance_metrics']['success_rate']
            quality_score = agent_analysis['performance_metrics']['quality_score']
            
            if success_rate < self.retirement_threshold:
                should_retire = True
                print(f"Retiring {agent_id} due to low success rate: {success_rate:.2f}")
            
            # Strategy-specific retirement criteria
            if strategy == EvolutionStrategy.DIVERSITY_FOCUSED:
                # Retire agents that are too similar to others
                if self._is_redundant_agent(agent, bullpen, analysis):
                    should_retire = True
                    print(f"Retiring {agent_id} due to redundancy")
            
            elif strategy == EvolutionStrategy.SPECIALIZATION_DRIVEN:
                # Retire generalists in favor of specialists
                if agent_analysis['specialization_level'] < 0.3:
                    should_retire = True
                    print(f"Retiring {agent_id} due to low specialization")
            
            if should_retire:
                retirement_candidates.append(agent_id)
        
        # Execute retirements
        for agent_id in retirement_candidates:
            bullpen.agent_status[agent_id] = AgentStatus.RETIRED
        
        return retirement_candidates
    
    def _promotion_phase(self, bullpen: AgentBullpen, analysis: Dict[str, Any]) -> List[str]:
        """Promote high-performing agents to master status"""
        promotion_candidates = []
        
        for agent_id, agent_analysis in analysis['agent_analyses'].items():
            agent = bullpen.get_agent_by_id(agent_id)
            if not agent:
                continue
            
            # Check promotion criteria
            fitness_score = agent_analysis['fitness_score']
            specialization_level = agent_analysis['specialization_level']
            task_count = agent_analysis['task_count']
            success_rate = agent_analysis['performance_metrics']['success_rate']
            
            if (fitness_score > 0.8 and 
                specialization_level > 0.7 and 
                task_count > 20 and 
                success_rate > 0.85):
                
                # Try to register as master agent
                if self.agent_factory.register_master_agent(agent):
                    promotion_candidates.append(agent_id)
                    print(f"Promoted {agent_id} to master status")
        
        return promotion_candidates
    
    def _breeding_phase(self, bullpen: AgentBullpen, analysis: Dict[str, Any], 
                       strategy: EvolutionStrategy) -> Dict[str, Any]:
        """Breed new agents to improve population"""
        breeding_results = {
            'attempts': 0,
            'successful': 0,
            'new_agents': []
        }
        
        # Determine how many new agents to breed
        current_population = len([aid for aid, status in bullpen.agent_status.items() 
                                if status != AgentStatus.RETIRED])
        target_new_agents = max(1, int(current_population * self.breeding_rate))
        
        # Identify breeding opportunities
        breeding_opportunities = self._identify_breeding_opportunities(analysis, strategy)
        
        for opportunity in breeding_opportunities[:target_new_agents]:
            breeding_results['attempts'] += 1
            
            try:
                # Get parent agents
                parent_agents = [bullpen.get_agent_by_id(pid) for pid in opportunity['parent_ids']]
                parent_agents = [a for a in parent_agents if a is not None]
                
                if not parent_agents:
                    continue
                
                # Breed new agent
                breeding_result = self.agent_factory.breed_specialist_agent(
                    parent_agents=parent_agents,
                    target_specialization=opportunity['target_specialization'],
                    requirements=opportunity.get('requirements')
                )
                
                if breeding_result.success and breeding_result.child_agent:
                    # Add to bullpen
                    if bullpen.add_agent(breeding_result.child_agent):
                        breeding_results['successful'] += 1
                        breeding_results['new_agents'].append(breeding_result.child_agent.agent_id)
                        print(f"Successfully bred new {opportunity['target_specialization']} agent")
                
            except Exception as e:
                print(f"Breeding attempt failed: {e}")
        
        return breeding_results
    
    def _training_phase(self, bullpen: AgentBullpen, analysis: Dict[str, Any], 
                       strategy: EvolutionStrategy) -> Dict[str, Any]:
        """Conduct training to improve existing agents"""
        training_results = {
            'sessions': 0,
            'improved': 0
        }
        
        # Identify agents that would benefit from training
        training_candidates = self._identify_training_candidates(analysis, strategy)
        
        for candidate in training_candidates:
            agent = bullpen.get_agent_by_id(candidate['agent_id'])
            if not agent:
                continue
            
            try:
                # Find suitable mentors
                mentors = self._find_mentors_for_agent(agent, bullpen, candidate['improvement_areas'])
                
                if mentors:
                    # Create training curriculum
                    curriculum = TrainingCurriculum.create_default_curriculum(agent.specialization)
                    
                    # Conduct training
                    training_result = self.mentorship_system.train_agent(agent, mentors, curriculum)
                    
                    training_results['sessions'] += training_result.total_sessions
                    if training_result.certification_achieved:
                        training_results['improved'] += 1
                        print(f"Successfully trained agent {agent.agent_id}")
            
            except Exception as e:
                print(f"Training failed for agent {candidate['agent_id']}: {e}")
        
        return training_results
    
    def _optimization_phase(self, bullpen: AgentBullpen, analysis: Dict[str, Any], 
                           strategy: EvolutionStrategy) -> Dict[str, Any]:
        """Final optimization of population structure"""
        optimization_results = {
            'rebalanced_specializations': False,
            'optimized_diversity': False,
            'pruned_redundancy': False
        }
        
        # Specialization rebalancing
        gaps = analysis['specialization_analysis']['coverage_gaps']
        if gaps and strategy in [EvolutionStrategy.SPECIALIZATION_DRIVEN, EvolutionStrategy.HYBRID]:
            optimization_results['rebalanced_specializations'] = self._fill_specialization_gaps(
                bullpen, gaps
            )
        
        # Diversity optimization
        if (analysis['population_metrics'].get('diversity_score', 0.0) < 0.5 and 
            strategy in [EvolutionStrategy.DIVERSITY_FOCUSED, EvolutionStrategy.HYBRID]):
            optimization_results['optimized_diversity'] = self._enhance_population_diversity(bullpen)
        
        return optimization_results
    
    def _is_redundant_agent(self, agent: SelfAwareAgent, bullpen: AgentBullpen, 
                           analysis: Dict[str, Any]) -> bool:
        """Check if agent is too similar to existing agents"""
        similar_agents = 0
        
        for other_id, other_agent in bullpen.agents.items():
            if other_id == agent.agent_id:
                continue
            
            # Compare specializations
            if agent.specialization == other_agent.specialization:
                # Compare capabilities
                agent_caps = set(agent.knowledge_base.capabilities.keys())
                other_caps = set(other_agent.knowledge_base.capabilities.keys())
                
                overlap = len(agent_caps.intersection(other_caps))
                total_unique = len(agent_caps.union(other_caps))
                
                similarity = overlap / total_unique if total_unique > 0 else 0.0
                
                if similarity > 0.8:  # 80% similarity threshold
                    similar_agents += 1
        
        return similar_agents >= 2  # Too many similar agents
    
    def _identify_breeding_opportunities(self, analysis: Dict[str, Any], 
                                       strategy: EvolutionStrategy) -> List[Dict[str, Any]]:
        """Identify opportunities for breeding new agents"""
        opportunities = []
        
        # Get top-performing agents as potential parents
        top_agents = []
        for agent_id, agent_analysis in analysis['agent_analyses'].items():
            if agent_analysis['fitness_score'] > 0.6:
                top_agents.append((agent_id, agent_analysis))
        
        top_agents.sort(key=lambda x: x[1]['fitness_score'], reverse=True)
        
        # Strategy-specific breeding opportunities
        if strategy == EvolutionStrategy.PERFORMANCE_BASED:
            # Breed top performers together
            if len(top_agents) >= 2:
                opportunities.append({
                    'parent_ids': [top_agents[0][0], top_agents[1][0]],
                    'target_specialization': top_agents[0][1]['specialization'],
                    'requirements': None
                })
        
        elif strategy == EvolutionStrategy.SPECIALIZATION_DRIVEN:
            # Breed specialists for missing specializations
            gaps = analysis['specialization_analysis']['coverage_gaps']
            for gap in gaps:
                if top_agents:
                    opportunities.append({
                        'parent_ids': [top_agents[0][0]],
                        'target_specialization': gap,
                        'requirements': AgentRequirements(
                            specialization=gap,
                            required_capabilities=[],
                            minimum_proficiency=0.5
                        )
                    })
        
        elif strategy == EvolutionStrategy.DIVERSITY_FOCUSED:
            # Breed agents with different characteristics
            for i, (agent1_id, agent1_analysis) in enumerate(top_agents[:3]):
                for j, (agent2_id, agent2_analysis) in enumerate(top_agents[i+1:4], i+1):
                    if agent1_analysis['specialization'] != agent2_analysis['specialization']:
                        opportunities.append({
                            'parent_ids': [agent1_id, agent2_id],
                            'target_specialization': random.choice([
                                agent1_analysis['specialization'], 
                                agent2_analysis['specialization']
                            ]),
                            'requirements': None
                        })
        
        else:  # HYBRID strategy
            # Combine all approaches
            opportunities.extend(self._identify_breeding_opportunities(
                analysis, EvolutionStrategy.PERFORMANCE_BASED
            ))
            opportunities.extend(self._identify_breeding_opportunities(
                analysis, EvolutionStrategy.SPECIALIZATION_DRIVEN
            ))
        
        return opportunities[:5]  # Limit to top 5 opportunities
    
    def _identify_training_candidates(self, analysis: Dict[str, Any], 
                                    strategy: EvolutionStrategy) -> List[Dict[str, Any]]:
        """Identify agents that would benefit from training"""
        candidates = []
        
        for agent_id, agent_analysis in analysis['agent_analyses'].items():
            # Skip if agent is performing well
            if agent_analysis['fitness_score'] > 0.8:
                continue
            
            # Identify improvement areas
            improvement_areas = []
            
            if agent_analysis['performance_metrics']['success_rate'] < 0.7:
                improvement_areas.append('reliability')
            
            if agent_analysis['performance_metrics']['quality_score'] < 0.6:
                improvement_areas.append('quality')
            
            if agent_analysis['specialization_level'] < 0.5:
                improvement_areas.append('specialization')
            
            if improvement_areas:
                candidates.append({
                    'agent_id': agent_id,
                    'improvement_areas': improvement_areas,
                    'priority': 1.0 - agent_analysis['fitness_score']  # Lower fitness = higher priority
                })
        
        # Sort by priority
        candidates.sort(key=lambda x: x['priority'], reverse=True)
        
        return candidates[:10]  # Top 10 candidates
    
    def _find_mentors_for_agent(self, agent: SelfAwareAgent, bullpen: AgentBullpen, 
                               improvement_areas: List[str]) -> List[SelfAwareAgent]:
        """Find suitable mentor agents"""
        mentors = []
        
        # Look for master agents first
        for master_agent in self.agent_factory.master_agents.values():
            if master_agent.agent_id in bullpen.agents:
                mentors.append(master_agent)
        
        # Add high-performing agents of same specialization
        for other_agent in bullpen.agents.values():
            if (other_agent.agent_id != agent.agent_id and 
                other_agent.specialization == agent.specialization and
                other_agent.current_metrics.success_rate > 0.8):
                mentors.append(other_agent)
        
        return mentors[:3]  # Limit to 3 mentors
    
    def _fill_specialization_gaps(self, bullpen: AgentBullpen, gaps: List[str]) -> bool:
        """Create agents to fill specialization gaps"""
        for gap in gaps[:2]:  # Fill up to 2 gaps per cycle
            try:
                # Find a suitable parent for specialization breeding
                potential_parents = [agent for agent in bullpen.agents.values() 
                                   if agent.current_metrics.success_rate > 0.7]
                
                if potential_parents:
                    parent = max(potential_parents, key=lambda a: a.current_metrics.quality_score)
                    
                    breeding_result = self.agent_factory.breed_specialist_agent(
                        parent_agents=[parent],
                        target_specialization=gap,
                        requirements=AgentRequirements(
                            specialization=gap,
                            required_capabilities=[],
                            minimum_proficiency=0.4
                        )
                    )
                    
                    if breeding_result.success and breeding_result.child_agent:
                        bullpen.add_agent(breeding_result.child_agent)
                        print(f"Created agent to fill {gap} specialization gap")
                        return True
            except Exception as e:
                print(f"Failed to fill specialization gap {gap}: {e}")
        
        return False
    
    def _enhance_population_diversity(self, bullpen: AgentBullpen) -> bool:
        """Enhance population diversity through targeted breeding"""
        # Find most common specialization
        spec_counts = {}
        for agent in bullpen.agents.values():
            spec = agent.specialization or 'general'
            spec_counts[spec] = spec_counts.get(spec, 0) + 1
        
        if not spec_counts:
            return False
        
        most_common_spec = max(spec_counts.keys(), key=lambda k: spec_counts[k])
        
        # Create agent with different specialization
        alternative_specs = ['strategic_planning', 'code_analysis', 'data_analysis', 
                           'performance_optimization']
        target_spec = random.choice([s for s in alternative_specs if s != most_common_spec])
        
        try:
            # Find good parent
            good_parents = [agent for agent in bullpen.agents.values() 
                           if agent.current_metrics.fitness_score > 0.6]
            
            if good_parents:
                parent = random.choice(good_parents)
                
                breeding_result = self.agent_factory.breed_specialist_agent(
                    parent_agents=[parent],
                    target_specialization=target_spec,
                    requirements=AgentRequirements(
                        specialization=target_spec,
                        required_capabilities=[],
                        minimum_proficiency=0.4
                    )
                )
                
                if breeding_result.success and breeding_result.child_agent:
                    bullpen.add_agent(breeding_result.child_agent)
                    print(f"Enhanced diversity by creating {target_spec} agent")
                    return True
        except Exception as e:
            print(f"Failed to enhance diversity: {e}")
        
        return False


class BullpenEvolutionSystem:
    """Main evolution system coordinator"""
    
    def __init__(self, bullpen: AgentBullpen):
        self.bullpen = bullpen
        self.evolution_engine = EvolutionEngine(bullpen.agent_factory, AgentMentorshipSystem())
        self.auto_evolution_enabled = False
        self.evolution_interval_hours = 24  # Run evolution daily
        self.last_evolution_time = datetime.now()
    
    def run_evolution_cycle(self, strategy: EvolutionStrategy = EvolutionStrategy.HYBRID) -> EvolutionReport:
        """Run a single evolution cycle"""
        return self.evolution_engine.evolve_population(self.bullpen, strategy)
    
    def enable_auto_evolution(self, interval_hours: int = 24):
        """Enable automatic evolution cycles"""
        self.auto_evolution_enabled = True
        self.evolution_interval_hours = interval_hours
        print(f"Auto-evolution enabled with {interval_hours}h interval")
    
    def disable_auto_evolution(self):
        """Disable automatic evolution"""
        self.auto_evolution_enabled = False
        print("Auto-evolution disabled")
    
    def check_auto_evolution(self) -> Optional[EvolutionReport]:
        """Check if auto-evolution should run and execute if needed"""
        if not self.auto_evolution_enabled:
            return None
        
        time_since_last = datetime.now() - self.last_evolution_time
        if time_since_last.total_seconds() >= self.evolution_interval_hours * 3600:
            print("Running scheduled auto-evolution")
            report = self.run_evolution_cycle()
            self.last_evolution_time = datetime.now()
            return report
        
        return None
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics"""
        if not self.evolution_engine.evolution_history:
            return {"no_data": True}
        
        history = self.evolution_engine.evolution_history
        
        # Success rate
        successful_cycles = sum(1 for report in history if report.success)
        success_rate = successful_cycles / len(history)
        
        # Average improvements
        improvements = [report.overall_improvement for report in history if report.success]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0
        
        # Population fitness trend
        fitness_trend = [(report.cycle_number, report.population_fitness_after) for report in history]
        
        # Recent performance (last 5 cycles)
        recent_cycles = history[-5:] if len(history) >= 5 else history
        recent_success_rate = sum(1 for report in recent_cycles if report.success) / len(recent_cycles)
        
        return {
            'total_evolution_cycles': len(history),
            'successful_cycles': successful_cycles,
            'overall_success_rate': success_rate,
            'recent_success_rate': recent_success_rate,
            'average_improvement': avg_improvement,
            'fitness_trend': fitness_trend,
            'auto_evolution_enabled': self.auto_evolution_enabled,
            'last_evolution': self.last_evolution_time.isoformat(),
            'next_evolution_due': (self.last_evolution_time + 
                                 timedelta(hours=self.evolution_interval_hours)).isoformat() if self.auto_evolution_enabled else None
        }