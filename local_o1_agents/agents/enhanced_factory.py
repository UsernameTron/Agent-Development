"""
Enhanced Agent Factory with Breeding Capabilities

This module provides enhanced factory functions that integrate with the
agent breeding and evolution system, allowing for intelligent agent creation.
"""

from typing import Dict, List, Optional, Any
from .agents import Agent, ExecutorWithFallback, TestGeneratorAgent, DependencyAgent
from .self_aware_agent import SelfAwareAgent
from .master_agent_factory import MasterAgentFactory, AgentRequirements
from .agent_bullpen import AgentBullpen

try:
    from config.config import (
        CEO_MODEL, FAST_MODEL, EXECUTOR_MODEL_ORIGINAL, 
        EXECUTOR_MODEL_DISTILLED, USE_DISTILLED_EXECUTOR
    )
except ImportError:
    # Fallback configuration
    CEO_MODEL = 'phi3.5'
    FAST_MODEL = 'phi3.5'
    EXECUTOR_MODEL_ORIGINAL = 'phi3.5'
    EXECUTOR_MODEL_DISTILLED = 'executor-distilled'
    USE_DISTILLED_EXECUTOR = True


class EnhancedAgentFactory:
    """Enhanced factory that can create agents through breeding or traditional methods"""
    
    def __init__(self, bullpen: Optional[AgentBullpen] = None):
        self.bullpen = bullpen
        self.master_factory = MasterAgentFactory()
        self.creation_history: List[Dict[str, Any]] = []
        
        # Register default specialization requirements
        self._register_default_requirements()
    
    def _register_default_requirements(self):
        """Register default requirements for common agent types"""
        requirements = {
            'strategic_planning': AgentRequirements(
                specialization='strategic_planning',
                required_capabilities=['task_decomposition', 'resource_allocation', 'planning'],
                minimum_proficiency=0.6,
                required_knowledge_domains=['business_strategy', 'project_management']
            ),
            'task_execution': AgentRequirements(
                specialization='task_execution',
                required_capabilities=['problem_solving', 'task_completion', 'quality_control'],
                minimum_proficiency=0.5,
                required_knowledge_domains=['task_management', 'execution_strategies']
            ),
            'testing': AgentRequirements(
                specialization='testing',
                required_capabilities=['test_generation', 'bug_reproduction', 'quality_assurance'],
                minimum_proficiency=0.6,
                required_knowledge_domains=['software_testing', 'test_automation']
            ),
            'code_analysis': AgentRequirements(
                specialization='code_analysis',
                required_capabilities=['code_review', 'bug_detection', 'static_analysis'],
                minimum_proficiency=0.7,
                required_knowledge_domains=['software_engineering', 'code_quality']
            ),
            'dependency_analysis': AgentRequirements(
                specialization='dependency_analysis',
                required_capabilities=['dependency_mapping', 'version_analysis', 'compatibility_check'],
                minimum_proficiency=0.6,
                required_knowledge_domains=['package_management', 'software_architecture']
            ),
            'summarization': AgentRequirements(
                specialization='summarization',
                required_capabilities=['content_analysis', 'key_point_extraction', 'concise_writing'],
                minimum_proficiency=0.6,
                required_knowledge_domains=['natural_language_processing', 'content_synthesis']
            )
        }
        
        for spec, req in requirements.items():
            self.master_factory.specialization_templates[spec] = req
    
    def create_agent_intelligent(self, agent_type: str, specialization: str = None, 
                               use_breeding: bool = True, **kwargs) -> Optional[SelfAwareAgent]:
        """
        Intelligently create an agent using breeding if suitable parents exist,
        otherwise create traditionally
        """
        # Determine specialization
        if specialization is None:
            specialization = self._get_default_specialization(agent_type)
        
        creation_record = {
            'timestamp': __import__('datetime').datetime.now(),
            'agent_type': agent_type,
            'specialization': specialization,
            'method': 'traditional',  # Will be updated if breeding is used
            'success': False
        }
        
        try:
            # Try breeding first if enabled and bullpen is available
            if use_breeding and self.bullpen and len(self.bullpen.agents) > 0:
                bred_agent = self._try_breeding(agent_type, specialization, kwargs)
                if bred_agent:
                    creation_record['method'] = 'breeding'
                    creation_record['success'] = True
                    creation_record['agent_id'] = bred_agent.agent_id
                    self.creation_history.append(creation_record)
                    return bred_agent
            
            # Fall back to traditional creation
            traditional_agent = self._create_traditional(agent_type, specialization, kwargs)
            if traditional_agent:
                creation_record['method'] = 'traditional'
                creation_record['success'] = True
                creation_record['agent_id'] = traditional_agent.agent_id
                self.creation_history.append(creation_record)
                
                # Add to bullpen if available
                if self.bullpen:
                    self.bullpen.add_agent(traditional_agent)
                
                return traditional_agent
        
        except Exception as e:
            creation_record['error'] = str(e)
            self.creation_history.append(creation_record)
            print(f"Error creating {agent_type} agent: {e}")
        
        return None
    
    def _get_default_specialization(self, agent_type: str) -> str:
        """Get default specialization for agent type"""
        defaults = {
            'ceo': 'strategic_planning',
            'executor': 'task_execution',
            'test_generator': 'testing',
            'dependency_agent': 'dependency_analysis',
            'summarizer': 'summarization',
            'code_analyzer': 'code_analysis'
        }
        return defaults.get(agent_type, 'general')
    
    def _try_breeding(self, agent_type: str, specialization: str, kwargs: Dict[str, Any]) -> Optional[SelfAwareAgent]:
        """Attempt to breed a new agent"""
        try:
            # Find suitable parents in the bullpen
            potential_parents = self._find_suitable_parents(specialization)
            
            if not potential_parents:
                return None
            
            # Get requirements for this specialization
            requirements = self.master_factory.specialization_templates.get(specialization)
            if not requirements:
                requirements = AgentRequirements(
                    specialization=specialization,
                    required_capabilities=[],
                    minimum_proficiency=0.5
                )
            
            # Attempt breeding
            breeding_result = self.master_factory.breed_specialist_agent(
                parent_agents=potential_parents,
                target_specialization=specialization,
                requirements=requirements
            )
            
            if breeding_result.success and breeding_result.child_agent:
                # Customize based on agent type
                child_agent = breeding_result.child_agent
                child_agent.name = self._generate_agent_name(agent_type, specialization)
                
                return child_agent
        
        except Exception as e:
            print(f"Breeding attempt failed: {e}")
        
        return None
    
    def _find_suitable_parents(self, specialization: str) -> List[SelfAwareAgent]:
        """Find suitable parent agents for breeding"""
        if not self.bullpen:
            return []
        
        # Look for agents with same specialization first
        same_spec_agents = [agent for agent in self.bullpen.agents.values() 
                           if agent.specialization == specialization 
                           and agent.current_metrics.success_rate > 0.6]
        
        if len(same_spec_agents) >= 2:
            # Return top 2 performers
            same_spec_agents.sort(key=lambda a: a.current_metrics.quality_score, reverse=True)
            return same_spec_agents[:2]
        
        # Fall back to any high-performing agents
        high_performers = [agent for agent in self.bullpen.agents.values()
                          if agent.current_metrics.success_rate > 0.7
                          and agent.current_metrics.quality_score > 0.6]
        
        if high_performers:
            high_performers.sort(key=lambda a: a.current_metrics.quality_score, reverse=True)
            return high_performers[:2] if len(high_performers) >= 2 else high_performers[:1]
        
        return []
    
    def _create_traditional(self, agent_type: str, specialization: str, kwargs: Dict[str, Any]) -> Optional[SelfAwareAgent]:
        """Create agent using traditional methods"""
        # Use the enhanced Agent class from agents.py which supports SelfAwareAgent
        if agent_type == 'ceo':
            return Agent('CEO', CEO_MODEL, specialization)
        
        elif agent_type == 'executor':
            executor_id = kwargs.get('executor_id', 1)
            task_complexity = kwargs.get('task_complexity')
            
            # Model selection based on complexity
            if task_complexity == 'high':
                model = EXECUTOR_MODEL_ORIGINAL
            elif task_complexity == 'low':
                model = EXECUTOR_MODEL_DISTILLED
            else:
                model = EXECUTOR_MODEL_DISTILLED if USE_DISTILLED_EXECUTOR else EXECUTOR_MODEL_ORIGINAL
            
            return ExecutorWithFallback(f'Executor_{executor_id}', model, specialization)
        
        elif agent_type == 'test_generator':
            return TestGeneratorAgent('TestGenerator', CEO_MODEL, specialization)
        
        elif agent_type == 'dependency_agent':
            return DependencyAgent('DependencyAgent', CEO_MODEL, specialization)
        
        elif agent_type == 'summarizer':
            return Agent('Summarizer', CEO_MODEL, specialization)
        
        else:
            # Generic agent
            agent_name = kwargs.get('name', f'{agent_type.title()}Agent')
            model = kwargs.get('model', CEO_MODEL)
            return Agent(agent_name, model, specialization)
    
    def _generate_agent_name(self, agent_type: str, specialization: str) -> str:
        """Generate unique name for bred agent"""
        import time
        timestamp = int(time.time())
        spec_short = specialization.split('_')[0][:4].title()
        return f"{agent_type.title()}{spec_short}_{timestamp}"
    
    # Enhanced factory methods with intelligent creation
    def create_ceo(self, specialization: str = 'strategic_planning', use_breeding: bool = True) -> Optional[SelfAwareAgent]:
        """Create CEO agent with strategic planning specialization"""
        return self.create_agent_intelligent('ceo', specialization, use_breeding)
    
    def create_executor(self, executor_id: int = 1, task_complexity: str = None, 
                       specialization: str = 'task_execution', use_breeding: bool = True) -> Optional[SelfAwareAgent]:
        """Create executor agent with enhanced capabilities"""
        return self.create_agent_intelligent('executor', specialization, use_breeding,
                                           executor_id=executor_id, task_complexity=task_complexity)
    
    def create_test_generator(self, specialization: str = 'testing', use_breeding: bool = True) -> Optional[SelfAwareAgent]:
        """Create test generator agent with testing specialization"""
        return self.create_agent_intelligent('test_generator', specialization, use_breeding)
    
    def create_dependency_agent(self, specialization: str = 'dependency_analysis', 
                              use_breeding: bool = True) -> Optional[SelfAwareAgent]:
        """Create dependency agent with dependency analysis specialization"""
        return self.create_agent_intelligent('dependency_agent', specialization, use_breeding)
    
    def create_summarizer(self, specialization: str = 'summarization', use_breeding: bool = True) -> Optional[SelfAwareAgent]:
        """Create summarizer agent with summarization specialization"""
        return self.create_agent_intelligent('summarizer', specialization, use_breeding)
    
    def create_specialized_agent(self, specialization: str, agent_name: str = None, 
                               model: str = None, use_breeding: bool = True) -> Optional[SelfAwareAgent]:
        """Create a specialized agent for any domain"""
        kwargs = {}
        if agent_name:
            kwargs['name'] = agent_name
        if model:
            kwargs['model'] = model
        
        return self.create_agent_intelligent('specialized', specialization, use_breeding, **kwargs)
    
    def create_agent_team(self, team_composition: Dict[str, int], use_breeding: bool = True) -> Dict[str, List[SelfAwareAgent]]:
        """Create a team of agents with specified composition"""
        team = {}
        
        for agent_type, count in team_composition.items():
            team[agent_type] = []
            for i in range(count):
                if agent_type == 'executor':
                    agent = self.create_executor(executor_id=i+1, use_breeding=use_breeding)
                elif agent_type == 'ceo':
                    agent = self.create_ceo(use_breeding=use_breeding)
                elif agent_type == 'test_generator':
                    agent = self.create_test_generator(use_breeding=use_breeding)
                elif agent_type == 'dependency_agent':
                    agent = self.create_dependency_agent(use_breeding=use_breeding)
                elif agent_type == 'summarizer':
                    agent = self.create_summarizer(use_breeding=use_breeding)
                else:
                    # Custom agent type
                    agent = self.create_specialized_agent(agent_type, use_breeding=use_breeding)
                
                if agent:
                    team[agent_type].append(agent)
        
        return team
    
    def get_creation_statistics(self) -> Dict[str, Any]:
        """Get statistics about agent creation"""
        if not self.creation_history:
            return {"no_data": True}
        
        total_creations = len(self.creation_history)
        successful_creations = sum(1 for record in self.creation_history if record['success'])
        breeding_creations = sum(1 for record in self.creation_history 
                               if record.get('method') == 'breeding' and record['success'])
        
        # Method distribution
        methods = {}
        for record in self.creation_history:
            if record['success']:
                method = record.get('method', 'unknown')
                methods[method] = methods.get(method, 0) + 1
        
        # Specialization distribution
        specializations = {}
        for record in self.creation_history:
            if record['success']:
                spec = record.get('specialization', 'unknown')
                specializations[spec] = specializations.get(spec, 0) + 1
        
        return {
            'total_creation_attempts': total_creations,
            'successful_creations': successful_creations,
            'success_rate': successful_creations / total_creations if total_creations > 0 else 0.0,
            'breeding_creations': breeding_creations,
            'breeding_rate': breeding_creations / successful_creations if successful_creations > 0 else 0.0,
            'creation_methods': methods,
            'specialization_distribution': specializations
        }


# Global instance for easy access
_global_factory: Optional[EnhancedAgentFactory] = None

def get_enhanced_factory(bullpen: Optional[AgentBullpen] = None) -> EnhancedAgentFactory:
    """Get the global enhanced factory instance"""
    global _global_factory
    if _global_factory is None:
        _global_factory = EnhancedAgentFactory(bullpen)
    elif bullpen and _global_factory.bullpen != bullpen:
        _global_factory.bullpen = bullpen
    return _global_factory

def set_enhanced_factory(factory: EnhancedAgentFactory):
    """Set the global enhanced factory instance"""
    global _global_factory
    _global_factory = factory

# Convenience functions that use the enhanced factory
def create_intelligent_ceo(bullpen: Optional[AgentBullpen] = None, **kwargs) -> Optional[SelfAwareAgent]:
    """Create CEO using intelligent factory"""
    factory = get_enhanced_factory(bullpen)
    return factory.create_ceo(**kwargs)

def create_intelligent_executor(bullpen: Optional[AgentBullpen] = None, **kwargs) -> Optional[SelfAwareAgent]:
    """Create executor using intelligent factory"""
    factory = get_enhanced_factory(bullpen)
    return factory.create_executor(**kwargs)

def create_intelligent_team(team_composition: Dict[str, int], 
                          bullpen: Optional[AgentBullpen] = None, **kwargs) -> Dict[str, List[SelfAwareAgent]]:
    """Create team using intelligent factory"""
    factory = get_enhanced_factory(bullpen)
    return factory.create_agent_team(team_composition, **kwargs)