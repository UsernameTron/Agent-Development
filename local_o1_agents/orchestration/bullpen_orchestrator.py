"""
Bullpen-Integrated Orchestrator

This module provides an enhanced orchestrator that uses the AgentBullpen system
for intelligent agent selection, breeding, and task execution.
"""

import json
import time
import threading
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Import the bullpen system
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent_bullpen import AgentBullpen, Task, ComplexTask, TaskComplexity, TaskResult
from agents.enhanced_factory import EnhancedAgentFactory
from agents.evolution_system import BullpenEvolutionSystem, EvolutionStrategy
from agents.self_aware_agent import SelfAwareAgent
from memory.vector_memory import vector_memory

# Import task complexity analyzer
try:
    from .task_complexity_analyzer import analyze_task_complexity
except ImportError:
    def analyze_task_complexity(task: str) -> str:
        """Fallback complexity analyzer"""
        if len(task.split()) > 50:
            return 'high'
        elif len(task.split()) > 20:
            return 'medium'
        else:
            return 'low'

# Load configuration
try:
    with open('orchestration_config.json') as f:
        config = json.load(f)
except FileNotFoundError:
    # Default configuration
    config = {
        'workflow_templates': {
            'low': {'max_executors': 2, 'parallel': False, 'agents': ['executor']},
            'medium': {'max_executors': 4, 'parallel': True, 'agents': ['executor', 'summarizer']},
            'high': {'max_executors': 8, 'parallel': True, 'agents': ['executor', 'summarizer', 'test_generator']}
        },
        'resource_limits': {'max_total_executors': 8},
        'feedback': {'enable_feedback_loop': True, 'quality_threshold': 0.7},
        'bullpen': {
            'enable_breeding': True,
            'enable_evolution': True,
            'evolution_interval_hours': 24,
            'max_agents': 50,
            'auto_scaling': True
        }
    }


class BullpenOrchestrator:
    """Advanced orchestrator using AgentBullpen for intelligent task execution"""
    
    def __init__(self, config_path: str = None):
        self.config = config
        if config_path:
            try:
                with open(config_path) as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")
        
        # Initialize bullpen system
        self.bullpen = AgentBullpen(max_agents=self.config['bullpen']['max_agents'])
        self.enhanced_factory = EnhancedAgentFactory(self.bullpen)
        
        # Initialize evolution system if enabled
        self.evolution_system = None
        if self.config['bullpen']['enable_evolution']:
            self.evolution_system = BullpenEvolutionSystem(self.bullpen)
            if self.config['bullpen'].get('auto_evolution', False):
                self.evolution_system.enable_auto_evolution(
                    self.config['bullpen']['evolution_interval_hours']
                )
        
        # Initialize with some basic agents
        self._initialize_bullpen()
        
        # Execution tracking
        self.execution_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def _initialize_bullpen(self):
        """Initialize bullpen with basic agent population"""
        print("Initializing bullpen with basic agents...")
        
        # Create initial population
        initial_agents = [
            self.enhanced_factory.create_ceo(use_breeding=False),
            self.enhanced_factory.create_executor(executor_id=1, use_breeding=False),
            self.enhanced_factory.create_executor(executor_id=2, use_breeding=False),
            self.enhanced_factory.create_summarizer(use_breeding=False),
            self.enhanced_factory.create_test_generator(use_breeding=False),
            self.enhanced_factory.create_dependency_agent(use_breeding=False)
        ]
        
        # Add agents to bullpen
        for agent in initial_agents:
            if agent:
                self.bullpen.add_agent(agent)
        
        print(f"Initialized bullpen with {len(self.bullpen.agents)} agents")
    
    def run_advanced_pipeline(self, task: str, image_path: str = None, 
                            audio_path: str = None, use_evolution: bool = True) -> List[Any]:
        """
        Execute advanced pipeline using bullpen system
        
        Args:
            task: Task description
            image_path: Optional image file path for multimodal tasks
            audio_path: Optional audio file path for multimodal tasks
            use_evolution: Whether to check for evolution opportunities
            
        Returns:
            List of results from task execution
        """
        execution_start = time.time()
        
        # Check for auto-evolution if enabled
        if use_evolution and self.evolution_system:
            evolution_report = self.evolution_system.check_auto_evolution()
            if evolution_report:
                print(f"Evolution cycle completed: {evolution_report.success}")
        
        # Analyze task complexity
        complexity_str = analyze_task_complexity(task)
        complexity = self._map_complexity(complexity_str)
        
        print(f"[BullpenOrchestrator] Task complexity: {complexity_str}")
        print(f"[BullpenOrchestrator] Bullpen status: {len(self.bullpen.get_available_agents())} available agents")
        
        # Handle multimodal content
        multimodal_context = self._process_multimodal_content(image_path, audio_path)
        if multimodal_context:
            task = f"{task}\n[Multimodal context]: {multimodal_context}"
        
        # Check memory cache
        cached_result = vector_memory.retrieve(task)
        if cached_result:
            print("[Memory] Cache hit. Returning cached output.")
            return [cached_result]
        else:
            print("[Memory] Cache miss. Proceeding with agent execution.")
        
        try:
            # Execute task using bullpen
            results = self._execute_with_bullpen(task, complexity)
            
            # Record execution
            execution_record = {
                'timestamp': datetime.now(),
                'task': task[:100] + "..." if len(task) > 100 else task,
                'complexity': complexity_str,
                'execution_time': time.time() - execution_start,
                'results_count': len(results),
                'success': True,
                'agents_used': getattr(self, '_last_agents_used', [])
            }
            
            with self._lock:
                self.execution_history.append(execution_record)
            
            # Cache results
            vector_memory.add(task, "\n".join(str(r) for r in results))
            
            # Save output
            self._save_output(results)
            
            print(f"[BullpenOrchestrator] Total execution time: {time.time() - execution_start:.2f}s")
            print(f"[Memory] Stats: {vector_memory.stats()}")
            
            return results
            
        except Exception as e:
            print(f"[BullpenOrchestrator] Execution failed: {e}")
            
            # Record failed execution
            execution_record = {
                'timestamp': datetime.now(),
                'task': task[:100] + "..." if len(task) > 100 else task,
                'complexity': complexity_str,
                'execution_time': time.time() - execution_start,
                'success': False,
                'error': str(e)
            }
            
            with self._lock:
                self.execution_history.append(execution_record)
            
            # Fallback to traditional orchestration
            print("[BullpenOrchestrator] Falling back to traditional orchestration")
            return self._fallback_execution(task, complexity_str)
    
    def _map_complexity(self, complexity_str: str) -> TaskComplexity:
        """Map string complexity to TaskComplexity enum"""
        mapping = {
            'low': TaskComplexity.LOW,
            'medium': TaskComplexity.MEDIUM,
            'high': TaskComplexity.HIGH
        }
        return mapping.get(complexity_str, TaskComplexity.MEDIUM)
    
    def _process_multimodal_content(self, image_path: str = None, audio_path: str = None) -> Optional[str]:
        """Process multimodal content (images/audio)"""
        multimodal_result = None
        
        try:
            if image_path:
                from agents.agents import ImageAgent
                img_agent = ImageAgent()
                multimodal_result = img_agent.caption(image_path)
                vector_memory.add_image(image_path, multimodal_result)
                
            if audio_path:
                from agents.agents import AudioAgent
                audio_agent = AudioAgent()
                audio_text = audio_agent.transcribe(audio_path)
                multimodal_result = audio_text
                vector_memory.add_audio(audio_path, audio_text)
        except Exception as e:
            print(f"[BullpenOrchestrator] Multimodal processing failed: {e}")
        
        return multimodal_result
    
    def _execute_with_bullpen(self, task: str, complexity: TaskComplexity) -> List[Any]:
        """Execute task using bullpen system"""
        template = self.config['workflow_templates'][complexity.value]
        max_executors = min(template['max_executors'], self.config['resource_limits']['max_total_executors'])
        parallel = template['parallel']
        agents_to_use = template['agents']
        
        self._last_agents_used = []
        results = []
        
        # Step 1: Strategic Planning (CEO)
        if 'ceo' in agents_to_use or len(agents_to_use) == 0:
            planning_task = Task(
                task_id=f"planning_{int(time.time())}",
                description=f"Create an execution plan for: {task}",
                complexity=complexity,
                required_capabilities=['strategic_planning', 'task_decomposition'],
                preferred_specialization='strategic_planning',
                priority=8
            )
            
            planning_result = self.bullpen.execute_task(planning_task)
            if planning_result and planning_result.success:
                plan = planning_result.output
                self._last_agents_used.append(planning_result.agent_id)
                print(f"[BullpenOrchestrator] Plan created by {planning_result.agent_id}")
            else:
                # Fallback: create basic plan
                plan = f"1. Analyze the task: {task}\n2. Execute the main objective\n3. Verify results"
                print("[BullpenOrchestrator] Using fallback planning")
        else:
            plan = f"Execute task: {task}"
        
        # Step 2: Task Execution
        steps = [s.strip() for s in plan.split("\n") if s.strip()][:max_executors]
        
        if parallel and len(steps) > 1:
            # Parallel execution
            execution_results = self._execute_parallel(steps, complexity)
        else:
            # Sequential execution
            execution_results = self._execute_sequential(steps, complexity)
        
        results.extend(execution_results)
        
        # Step 3: Additional processing based on workflow template
        if 'summarizer' in agents_to_use:
            summary_result = self._execute_summarization(results)
            if summary_result:
                results.append(summary_result)
        
        if 'test_generator' in agents_to_use:
            test_results = self._execute_test_generation(steps)
            results.extend(test_results)
        
        if 'dependency_agent' in agents_to_use:
            dep_result = self._execute_dependency_analysis()
            if dep_result:
                results.append(dep_result)
        
        # Step 4: Quality feedback loop
        if self.config['feedback']['enable_feedback_loop']:
            results = self._apply_quality_feedback(results, task, complexity)
        
        return results
    
    def _execute_parallel(self, steps: List[str], complexity: TaskComplexity) -> List[str]:
        """Execute steps in parallel using bullpen"""
        results = []
        
        # Create tasks for parallel execution
        tasks = []
        for i, step in enumerate(steps):
            task = Task(
                task_id=f"exec_{i}_{int(time.time())}",
                description=step,
                complexity=complexity,
                required_capabilities=['task_execution', 'problem_solving'],
                preferred_specialization='task_execution',
                priority=5
            )
            tasks.append(task)
        
        # Execute tasks using bullpen's load balancing
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_task = {executor.submit(self.bullpen.execute_task, task): task for task in tasks}
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result and result.success:
                        results.append(result.output)
                        self._last_agents_used.append(result.agent_id)
                    else:
                        results.append(f"[ERROR] Task {task.task_id} failed")
                except Exception as e:
                    results.append(f"[ERROR] Task {task.task_id} failed: {e}")
        
        return results
    
    def _execute_sequential(self, steps: List[str], complexity: TaskComplexity) -> List[str]:
        """Execute steps sequentially using bullpen"""
        results = []
        
        for i, step in enumerate(steps):
            task = Task(
                task_id=f"exec_{i}_{int(time.time())}",
                description=step,
                complexity=complexity,
                required_capabilities=['task_execution', 'problem_solving'],
                preferred_specialization='task_execution',
                priority=5
            )
            
            result = self.bullpen.execute_task(task)
            if result and result.success:
                results.append(result.output)
                self._last_agents_used.append(result.agent_id)
            else:
                results.append(f"[ERROR] Step {i+1} failed")
        
        return results
    
    def _execute_summarization(self, results: List[str]) -> Optional[str]:
        """Execute summarization task"""
        if not results:
            return None
        
        summary_task = Task(
            task_id=f"summary_{int(time.time())}",
            description=f"Summarize the following results:\n\n" + "\n".join(results),
            complexity=TaskComplexity.MEDIUM,
            required_capabilities=['summarization', 'content_analysis'],
            preferred_specialization='summarization',
            priority=6
        )
        
        result = self.bullpen.execute_task(summary_task)
        if result and result.success:
            self._last_agents_used.append(result.agent_id)
            return result.output
        
        return None
    
    def _execute_test_generation(self, steps: List[str]) -> List[str]:
        """Execute test generation tasks"""
        test_results = []
        
        for i, step in enumerate(steps):
            test_task = Task(
                task_id=f"test_{i}_{int(time.time())}",
                description=f"Generate tests for: {step}",
                complexity=TaskComplexity.MEDIUM,
                required_capabilities=['test_generation', 'quality_assurance'],
                preferred_specialization='testing',
                priority=4
            )
            
            result = self.bullpen.execute_task(test_task)
            if result and result.success:
                test_results.append(result.output)
                self._last_agents_used.append(result.agent_id)
        
        return test_results
    
    def _execute_dependency_analysis(self) -> Optional[str]:
        """Execute dependency analysis task"""
        dep_task = Task(
            task_id=f"deps_{int(time.time())}",
            description="Analyze project dependencies and create requirements",
            complexity=TaskComplexity.LOW,
            required_capabilities=['dependency_analysis', 'system_analysis'],
            preferred_specialization='dependency_analysis',
            priority=3
        )
        
        result = self.bullpen.execute_task(dep_task)
        if result and result.success:
            self._last_agents_used.append(result.agent_id)
            return result.output
        
        return None
    
    def _apply_quality_feedback(self, results: List[str], original_task: str, 
                              complexity: TaskComplexity) -> List[str]:
        """Apply quality feedback loop"""
        quality_threshold = self.config['feedback']['quality_threshold']
        
        # Check result quality
        low_quality_results = []
        for i, result in enumerate(results):
            if isinstance(result, str) and (len(result.strip()) < 10 or result.startswith('[ERROR]')):
                low_quality_results.append(i)
        
        # Retry low quality results with different agents or higher complexity
        if low_quality_results and len(self.bullpen.get_available_agents()) > 1:
            print(f"[BullpenOrchestrator] Retrying {len(low_quality_results)} low-quality results")
            
            for i in low_quality_results:
                retry_task = Task(
                    task_id=f"retry_{i}_{int(time.time())}",
                    description=f"Retry with higher quality: {original_task}",
                    complexity=TaskComplexity.HIGH if complexity != TaskComplexity.HIGH else complexity,
                    required_capabilities=['problem_solving', 'quality_control'],
                    preferred_specialization='task_execution',
                    priority=7
                )
                
                retry_result = self.bullpen.execute_task(retry_task)
                if retry_result and retry_result.success and len(retry_result.output.strip()) > 10:
                    results[i] = retry_result.output
                    self._last_agents_used.append(retry_result.agent_id)
        
        return results
    
    def _fallback_execution(self, task: str, complexity: str) -> List[str]:
        """Fallback to traditional agent creation and execution"""
        try:
            # Import traditional factory functions
            from agents.agents import create_ceo, create_executor, create_summarizer
            
            # Basic execution
            ceo = create_ceo()
            plan = ceo.run(task)
            
            steps = [s.strip() for s in plan.split("\n") if s.strip()][:2]  # Limit to 2 steps
            results = []
            
            for i, step in enumerate(steps):
                executor = create_executor(i, complexity)
                result = executor.run(step)
                results.append(result)
            
            # Add summary
            summarizer = create_summarizer()
            summary = summarizer.run("\n".join(results))
            results.append(summary)
            
            return results
            
        except Exception as e:
            print(f"[BullpenOrchestrator] Fallback execution also failed: {e}")
            return [f"[ERROR] All execution methods failed: {e}"]
    
    def _save_output(self, results: List[Any]):
        """Save execution output to file"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"{output_dir}/bullpen_orchestrator_{timestamp}.txt"
            with open(filename, "w") as f:
                f.write(f"Bullpen Orchestrator Results - {timestamp}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, result in enumerate(results):
                    f.write(f"Result {i+1}:\n{result}\n\n")
                
                # Add bullpen statistics
                stats = self.bullpen.get_bullpen_statistics()
                f.write("Bullpen Statistics:\n")
                f.write(json.dumps(stats, indent=2, default=str))
        
        except Exception as e:
            print(f"[BullpenOrchestrator] Failed to save output: {e}")
    
    def get_orchestrator_statistics(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator statistics"""
        with self._lock:
            execution_stats = {
                'total_executions': len(self.execution_history),
                'successful_executions': sum(1 for e in self.execution_history if e['success']),
                'average_execution_time': 0.0,
                'complexity_distribution': {},
                'recent_performance': []
            }
            
            if self.execution_history:
                execution_stats['success_rate'] = execution_stats['successful_executions'] / len(self.execution_history)
                
                # Calculate average execution time
                successful_executions = [e for e in self.execution_history if e['success']]
                if successful_executions:
                    total_time = sum(e['execution_time'] for e in successful_executions)
                    execution_stats['average_execution_time'] = total_time / len(successful_executions)
                
                # Complexity distribution
                for execution in self.execution_history:
                    complexity = execution.get('complexity', 'unknown')
                    execution_stats['complexity_distribution'][complexity] = \
                        execution_stats['complexity_distribution'].get(complexity, 0) + 1
                
                # Recent performance (last 10 executions)
                execution_stats['recent_performance'] = self.execution_history[-10:]
        
        # Get bullpen statistics
        bullpen_stats = self.bullpen.get_bullpen_statistics()
        
        # Get factory statistics
        factory_stats = self.enhanced_factory.get_creation_statistics()
        
        # Get evolution statistics if available
        evolution_stats = {}
        if self.evolution_system:
            evolution_stats = self.evolution_system.get_evolution_statistics()
        
        return {
            'orchestrator': execution_stats,
            'bullpen': bullpen_stats,
            'factory': factory_stats,
            'evolution': evolution_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def trigger_evolution(self, strategy: EvolutionStrategy = EvolutionStrategy.HYBRID):
        """Manually trigger an evolution cycle"""
        if self.evolution_system:
            print(f"[BullpenOrchestrator] Triggering evolution with strategy: {strategy}")
            report = self.evolution_system.run_evolution_cycle(strategy)
            print(f"[BullpenOrchestrator] Evolution completed: Success={report.success}, "
                  f"Improvement={report.overall_improvement:.3f}")
            return report
        else:
            print("[BullpenOrchestrator] Evolution system not enabled")
            return None
    
    def scale_bullpen(self, target_size: int):
        """Scale bullpen to target size by creating new agents"""
        current_size = len(self.bullpen.agents)
        
        if target_size <= current_size:
            print(f"[BullpenOrchestrator] Bullpen already at or above target size ({current_size} >= {target_size})")
            return
        
        agents_needed = target_size - current_size
        print(f"[BullpenOrchestrator] Scaling bullpen from {current_size} to {target_size} agents")
        
        # Create diverse set of agents
        agent_types = ['executor', 'summarizer', 'test_generator', 'dependency_agent']
        
        for i in range(agents_needed):
            agent_type = agent_types[i % len(agent_types)]
            
            try:
                if agent_type == 'executor':
                    agent = self.enhanced_factory.create_executor(executor_id=current_size + i + 1)
                elif agent_type == 'summarizer':
                    agent = self.enhanced_factory.create_summarizer()
                elif agent_type == 'test_generator':
                    agent = self.enhanced_factory.create_test_generator()
                elif agent_type == 'dependency_agent':
                    agent = self.enhanced_factory.create_dependency_agent()
                
                if agent:
                    self.bullpen.add_agent(agent)
                    print(f"[BullpenOrchestrator] Created {agent_type} agent: {agent.name}")
            
            except Exception as e:
                print(f"[BullpenOrchestrator] Failed to create {agent_type} agent: {e}")
        
        final_size = len(self.bullpen.agents)
        print(f"[BullpenOrchestrator] Bullpen scaling completed: {final_size} agents")


# Global orchestrator instance
_global_orchestrator: Optional[BullpenOrchestrator] = None

def get_bullpen_orchestrator(config_path: str = None) -> BullpenOrchestrator:
    """Get global bullpen orchestrator instance"""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = BullpenOrchestrator(config_path)
    return _global_orchestrator

def run_advanced_pipeline_with_bullpen(task: str, image_path: str = None, 
                                     audio_path: str = None) -> List[Any]:
    """Convenience function using bullpen orchestrator"""
    orchestrator = get_bullpen_orchestrator()
    return orchestrator.run_advanced_pipeline(task, image_path, audio_path)


if __name__ == '__main__':
    import sys
    
    # Get task from command line
    task = sys.argv[1] if len(sys.argv) > 1 else "Design a multi-agent research bootcamp pipeline with dynamic scaling."
    
    # Optional parameters
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    audio_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Run the pipeline
    results = run_advanced_pipeline_with_bullpen(task, image_path, audio_path)
    
    # Print results
    print("\n" + "="*50)
    print("BULLPEN ORCHESTRATOR RESULTS")
    print("="*50)
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(result)
    
    # Print statistics
    orchestrator = get_bullpen_orchestrator()
    stats = orchestrator.get_orchestrator_statistics()
    print(f"\nOrchestrator Statistics:")
    print(f"Total Executions: {stats['orchestrator']['total_executions']}")
    print(f"Success Rate: {stats['orchestrator'].get('success_rate', 0):.2%}")
    print(f"Bullpen Agents: {stats['bullpen']['bullpen_info']['total_agents']}")
    print(f"Available Agents: {stats['bullpen']['bullpen_info']['available_agents']}")