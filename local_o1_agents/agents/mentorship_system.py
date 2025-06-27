"""
Agent Mentorship System for Knowledge Transfer

This module implements the mentorship and training system where master agents
teach and train newly created agents to improve their capabilities.
"""

import time
import json
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from copy import deepcopy

from .self_aware_agent import SelfAwareAgent, AgentCapability, KnowledgeDomain
from .master_agent_factory import TrainingCurriculum


class TrainingPhase(Enum):
    """Phases of agent training"""
    KNOWLEDGE_TRANSFER = "knowledge_transfer"
    SKILL_DEVELOPMENT = "skill_development"
    PRACTICAL_APPLICATION = "practical_application"
    ASSESSMENT = "assessment"
    CERTIFICATION = "certification"


@dataclass
class TrainingTask:
    """Individual training task"""
    task_id: str
    phase: TrainingPhase
    description: str
    objective: str
    difficulty: str  # easy, medium, hard
    estimated_time: int  # minutes
    success_criteria: Dict[str, float]
    mentor_agent_id: str
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class TrainingSession:
    """Single training session between mentor and student"""
    session_id: str
    mentor_agent_id: str
    student_agent_id: str
    task: TrainingTask
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    performance_score: float = 0.0
    feedback: str = ""
    knowledge_transferred: List[str] = None
    skills_practiced: List[str] = None
    
    def __post_init__(self):
        if self.knowledge_transferred is None:
            self.knowledge_transferred = []
        if self.skills_practiced is None:
            self.skills_practiced = []
    
    @property
    def duration_minutes(self) -> float:
        """Calculate session duration in minutes"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 60
        return 0.0


@dataclass
class TrainingResult:
    """Result of a complete training program"""
    student_agent_id: str
    curriculum_name: str
    start_date: datetime
    end_date: Optional[datetime]
    total_sessions: int
    successful_sessions: int
    phases_completed: List[TrainingPhase]
    skills_acquired: List[str]
    knowledge_domains_learned: List[str]
    final_assessment_score: float
    certification_achieved: bool
    mentors_involved: List[str]
    total_training_time: float  # hours
    improvement_metrics: Dict[str, float]


class SkillAssessmentSystem:
    """Assesses agent skills and knowledge"""
    
    def __init__(self):
        self.assessment_tasks = self._initialize_assessment_tasks()
    
    def _initialize_assessment_tasks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize assessment tasks for different skills"""
        return {
            "task_decomposition": [
                {
                    "prompt": "Break down this complex project: 'Build a web application for task management'",
                    "expected_components": ["frontend", "backend", "database", "authentication", "testing"],
                    "scoring_criteria": {"completeness": 0.4, "logical_structure": 0.3, "detail_level": 0.3}
                },
                {
                    "prompt": "Decompose this task: 'Optimize system performance'",
                    "expected_components": ["profiling", "bottleneck_identification", "optimization", "testing", "monitoring"],
                    "scoring_criteria": {"completeness": 0.4, "logical_structure": 0.4, "feasibility": 0.2}
                }
            ],
            "code_review": [
                {
                    "prompt": "Review this Python function for issues: def calc(x, y): return x/y",
                    "expected_issues": ["division_by_zero", "no_type_hints", "unclear_naming", "no_documentation"],
                    "scoring_criteria": {"issue_identification": 0.5, "severity_assessment": 0.3, "suggestions": 0.2}
                }
            ],
            "strategic_planning": [
                {
                    "prompt": "Create a strategic plan for launching a new product feature",
                    "expected_elements": ["market_analysis", "resource_requirements", "timeline", "risk_assessment", "success_metrics"],
                    "scoring_criteria": {"comprehensiveness": 0.3, "feasibility": 0.3, "detail_level": 0.2, "risk_awareness": 0.2}
                }
            ]
        }
    
    def assess_agent(self, agent: SelfAwareAgent, 
                    certification_criteria: Dict[str, float]) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive assessment of agent capabilities"""
        assessment_results = {}
        total_score = 0.0
        total_weight = 0.0
        
        # Assess each required skill
        for skill_name in certification_criteria.keys():
            if skill_name in ['success_rate', 'quality_score', 'response_time']:
                # Performance metrics
                if skill_name == 'success_rate':
                    score = agent.current_metrics.success_rate
                    required = certification_criteria[skill_name]
                    assessment_results[skill_name] = {
                        'score': score,
                        'required': required,
                        'passed': score >= required
                    }
                elif skill_name == 'quality_score':
                    score = agent.current_metrics.quality_score
                    required = certification_criteria[skill_name]
                    assessment_results[skill_name] = {
                        'score': score,
                        'required': required,
                        'passed': score >= required
                    }
                elif skill_name == 'response_time':
                    score = agent.current_metrics.average_response_time
                    required = certification_criteria[skill_name]
                    # For response time, lower is better
                    assessment_results[skill_name] = {
                        'score': score,
                        'required': required,
                        'passed': score <= required
                    }
                
                total_score += score if assessment_results[skill_name]['passed'] else 0.0
                total_weight += 1.0
                
            elif skill_name in self.assessment_tasks:
                # Skill-based assessment
                skill_score = self._assess_specific_skill(agent, skill_name)
                required = certification_criteria[skill_name]
                
                assessment_results[skill_name] = {
                    'score': skill_score,
                    'required': required,
                    'passed': skill_score >= required
                }
                
                total_score += skill_score
                total_weight += 1.0
        
        # Calculate overall score
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Determine if certification is achieved
        passed_assessments = sum(1 for result in assessment_results.values() if result.get('passed', False))
        certification_achieved = passed_assessments >= len(assessment_results) * 0.8  # 80% pass rate
        
        return certification_achieved, {
            'overall_score': overall_score,
            'assessments': assessment_results,
            'certification_achieved': certification_achieved,
            'passed_count': passed_assessments,
            'total_count': len(assessment_results)
        }
    
    def _assess_specific_skill(self, agent: SelfAwareAgent, skill_name: str) -> float:
        """Assess a specific skill through targeted tasks"""
        if skill_name not in self.assessment_tasks:
            return 0.5  # Default neutral score
        
        tasks = self.assessment_tasks[skill_name]
        total_score = 0.0
        
        for task in tasks:
            try:
                # Run the assessment task
                response = agent.run(task['prompt'])
                
                # Score the response
                task_score = self._score_response(response, task)
                total_score += task_score
                
            except Exception as e:
                print(f"Error assessing {skill_name}: {e}")
                total_score += 0.0
        
        return total_score / len(tasks) if tasks else 0.0
    
    def _score_response(self, response: str, task: Dict[str, Any]) -> float:
        """Score an agent's response to an assessment task"""
        if response.startswith('[ERROR]'):
            return 0.0
        
        score = 0.0
        criteria = task.get('scoring_criteria', {})
        
        # Check for expected components/issues
        expected_items = task.get('expected_components', []) + task.get('expected_issues', [])
        expected_elements = task.get('expected_elements', [])
        all_expected = expected_items + expected_elements
        
        if all_expected:
            found_count = 0
            response_lower = response.lower()
            
            for item in all_expected:
                # Simple keyword matching (could be enhanced with NLP)
                item_keywords = item.replace('_', ' ').split()
                if any(keyword in response_lower for keyword in item_keywords):
                    found_count += 1
            
            completeness_score = found_count / len(all_expected)
            score += completeness_score * criteria.get('completeness', 0.5)
        
        # Length and detail assessment
        if len(response.split()) > 20:  # Detailed response
            score += criteria.get('detail_level', 0.2)
        
        # Structure assessment (simple heuristic)
        if '\n' in response and ('1.' in response or '-' in response or '*' in response):
            score += criteria.get('logical_structure', 0.2)
        
        return min(score, 1.0)


class KnowledgeTransferSystem:
    """Handles knowledge transfer between agents"""
    
    def __init__(self):
        self.transfer_history: List[Dict[str, Any]] = []
    
    def transfer(self, from_agent: SelfAwareAgent, to_agent: SelfAwareAgent,
                knowledge_domains: List[str]) -> Dict[str, Any]:
        """Transfer knowledge from mentor to student"""
        transfer_result = {
            'mentor_id': from_agent.agent_id,
            'student_id': to_agent.agent_id,
            'timestamp': datetime.now(),
            'domains_requested': knowledge_domains,
            'domains_transferred': [],
            'success': False,
            'transfer_efficiency': 0.0
        }
        
        try:
            total_transferred = 0
            for domain in knowledge_domains:
                if domain in from_agent.knowledge_base.knowledge_domains:
                    # Extract knowledge from mentor
                    knowledge_package = from_agent.knowledge_base.extract_knowledge(domain)
                    
                    # Transfer to student
                    if to_agent.knowledge_base.receive_knowledge(knowledge_package):
                        transfer_result['domains_transferred'].append(domain)
                        total_transferred += 1
                        
                        # Enhance student's capability related to the domain
                        self._enhance_related_capabilities(to_agent, domain)
            
            transfer_result['success'] = total_transferred > 0
            transfer_result['transfer_efficiency'] = total_transferred / len(knowledge_domains) if knowledge_domains else 0.0
            
            self.transfer_history.append(transfer_result)
            
        except Exception as e:
            transfer_result['error'] = str(e)
        
        return transfer_result
    
    def _enhance_related_capabilities(self, agent: SelfAwareAgent, domain: str):
        """Enhance capabilities related to transferred knowledge domain"""
        domain_capability_mapping = {
            'business_strategy': ['strategic_planning', 'decision_making'],
            'software_engineering': ['code_review', 'debugging', 'system_design'],
            'data_analysis': ['pattern_recognition', 'statistical_analysis'],
            'project_management': ['task_decomposition', 'resource_allocation'],
            'debugging': ['bug_detection', 'problem_solving'],
            'code_quality': ['code_review', 'refactoring']
        }
        
        related_capabilities = domain_capability_mapping.get(domain, [])
        
        for cap_name in related_capabilities:
            if cap_name in agent.knowledge_base.capabilities:
                # Boost existing capability
                current_cap = agent.knowledge_base.capabilities[cap_name]
                current_cap.proficiency_level = min(1.0, current_cap.proficiency_level + 0.05)
            else:
                # Add new capability
                new_capability = AgentCapability(
                    name=cap_name,
                    proficiency_level=0.3,
                    category=agent.specialization
                )
                agent.knowledge_base.add_capability(new_capability)


class AgentMentorshipSystem:
    """Comprehensive mentorship system for training agents"""
    
    def __init__(self):
        self.training_protocols: Dict[str, TrainingCurriculum] = {}
        self.skill_assessments = SkillAssessmentSystem()
        self.knowledge_transfer = KnowledgeTransferSystem()
        
        # Training tracking
        self.active_training_programs: Dict[str, Dict[str, Any]] = {}
        self.completed_training_programs: List[TrainingResult] = []
        self.training_sessions: List[TrainingSession] = []
    
    def register_training_protocol(self, curriculum: TrainingCurriculum):
        """Register a training curriculum"""
        self.training_protocols[curriculum.specialization] = curriculum
    
    def train_agent(self, student_agent: SelfAwareAgent, 
                   mentor_agents: List[SelfAwareAgent],
                   curriculum: TrainingCurriculum) -> TrainingResult:
        """Comprehensive training of a new agent by master agents"""
        start_time = datetime.now()
        training_id = f"training_{student_agent.agent_id}_{int(time.time())}"
        
        # Initialize training program
        self.active_training_programs[training_id] = {
            'student_id': student_agent.agent_id,
            'curriculum': curriculum,
            'mentors': [m.agent_id for m in mentor_agents],
            'start_time': start_time,
            'phases_completed': [],
            'sessions': []
        }
        
        training_phases = []
        skills_acquired = []
        knowledge_domains_learned = []
        
        try:
            # Phase 1: Knowledge Transfer
            print(f"Starting knowledge transfer phase for {student_agent.name}")
            knowledge_phase_result = self._conduct_knowledge_transfer_phase(
                student_agent, mentor_agents, curriculum, training_id
            )
            training_phases.append(knowledge_phase_result)
            knowledge_domains_learned.extend(knowledge_phase_result.get('domains_learned', []))
            
            # Phase 2: Skill Development
            print(f"Starting skill development phase for {student_agent.name}")
            skill_phase_result = self._conduct_skill_development_phase(
                student_agent, mentor_agents, curriculum, training_id
            )
            training_phases.append(skill_phase_result)
            skills_acquired.extend(skill_phase_result.get('skills_developed', []))
            
            # Phase 3: Practical Application
            print(f"Starting practical application phase for {student_agent.name}")
            practical_phase_result = self._conduct_practical_training_phase(
                student_agent, mentor_agents, curriculum, training_id
            )
            training_phases.append(practical_phase_result)
            
            # Phase 4: Assessment & Certification
            print(f"Starting assessment phase for {student_agent.name}")
            certification_achieved, assessment_results = self.skill_assessments.assess_agent(
                student_agent, curriculum.certification_criteria
            )
            
            end_time = datetime.now()
            total_training_time = (end_time - start_time).total_seconds() / 3600  # hours
            
            # Calculate improvement metrics
            improvement_metrics = self._calculate_improvement_metrics(
                student_agent, training_phases
            )
            
            # Create training result
            result = TrainingResult(
                student_agent_id=student_agent.agent_id,
                curriculum_name=curriculum.curriculum_name,
                start_date=start_time,
                end_date=end_time,
                total_sessions=len(self.active_training_programs[training_id]['sessions']),
                successful_sessions=sum(1 for s in self.active_training_programs[training_id]['sessions'] if s.get('success', False)),
                phases_completed=[TrainingPhase.KNOWLEDGE_TRANSFER, TrainingPhase.SKILL_DEVELOPMENT, 
                                TrainingPhase.PRACTICAL_APPLICATION, TrainingPhase.ASSESSMENT],
                skills_acquired=skills_acquired,
                knowledge_domains_learned=knowledge_domains_learned,
                final_assessment_score=assessment_results.get('overall_score', 0.0),
                certification_achieved=certification_achieved,
                mentors_involved=[m.agent_id for m in mentor_agents],
                total_training_time=total_training_time,
                improvement_metrics=improvement_metrics
            )
            
            # Move from active to completed
            del self.active_training_programs[training_id]
            self.completed_training_programs.append(result)
            
            print(f"Training completed for {student_agent.name}. Certification: {'ACHIEVED' if certification_achieved else 'NOT ACHIEVED'}")
            
            return result
            
        except Exception as e:
            print(f"Error during training: {e}")
            # Create failed training result
            return TrainingResult(
                student_agent_id=student_agent.agent_id,
                curriculum_name=curriculum.curriculum_name,
                start_date=start_time,
                end_date=datetime.now(),
                total_sessions=0,
                successful_sessions=0,
                phases_completed=[],
                skills_acquired=[],
                knowledge_domains_learned=[],
                final_assessment_score=0.0,
                certification_achieved=False,
                mentors_involved=[],
                total_training_time=0.0,
                improvement_metrics={}
            )
    
    def _conduct_knowledge_transfer_phase(self, student: SelfAwareAgent, 
                                        mentors: List[SelfAwareAgent],
                                        curriculum: TrainingCurriculum,
                                        training_id: str) -> Dict[str, Any]:
        """Conduct knowledge transfer phase"""
        phase_result = {
            'phase': TrainingPhase.KNOWLEDGE_TRANSFER,
            'success': True,
            'domains_learned': [],
            'sessions': []
        }
        
        # Find best mentor for each knowledge domain
        for domain in curriculum.required_knowledge:
            best_mentor = self._find_best_mentor_for_domain(mentors, domain)
            if best_mentor:
                # Conduct knowledge transfer session
                session_result = self._conduct_knowledge_session(
                    student, best_mentor, domain, training_id
                )
                phase_result['sessions'].append(session_result)
                
                if session_result['success']:
                    phase_result['domains_learned'].append(domain)
            else:
                print(f"No suitable mentor found for domain: {domain}")
        
        phase_result['success'] = len(phase_result['domains_learned']) > 0
        return phase_result
    
    def _conduct_skill_development_phase(self, student: SelfAwareAgent,
                                       mentors: List[SelfAwareAgent],
                                       curriculum: TrainingCurriculum,
                                       training_id: str) -> Dict[str, Any]:
        """Conduct skill development phase"""
        phase_result = {
            'phase': TrainingPhase.SKILL_DEVELOPMENT,
            'success': True,
            'skills_developed': [],
            'sessions': []
        }
        
        for skill in curriculum.required_skills:
            best_mentor = self._find_best_mentor_for_skill(mentors, skill)
            if best_mentor:
                session_result = self._conduct_skill_session(
                    student, best_mentor, skill, training_id
                )
                phase_result['sessions'].append(session_result)
                
                if session_result['success']:
                    phase_result['skills_developed'].append(skill)
        
        phase_result['success'] = len(phase_result['skills_developed']) > 0
        return phase_result
    
    def _conduct_practical_training_phase(self, student: SelfAwareAgent,
                                        mentors: List[SelfAwareAgent],
                                        curriculum: TrainingCurriculum,
                                        training_id: str) -> Dict[str, Any]:
        """Conduct practical application phase"""
        phase_result = {
            'phase': TrainingPhase.PRACTICAL_APPLICATION,
            'success': True,
            'tasks_completed': [],
            'sessions': []
        }
        
        for task_def in curriculum.practical_tasks:
            # Select mentor based on task type
            best_mentor = self._select_mentor_for_task(mentors, task_def)
            if best_mentor:
                session_result = self._conduct_practical_session(
                    student, best_mentor, task_def, training_id
                )
                phase_result['sessions'].append(session_result)
                
                if session_result['success']:
                    phase_result['tasks_completed'].append(task_def['type'])
        
        return phase_result
    
    def _conduct_knowledge_session(self, student: SelfAwareAgent, mentor: SelfAwareAgent,
                                 domain: str, training_id: str) -> Dict[str, Any]:
        """Conduct individual knowledge transfer session"""
        session_id = f"knowledge_{training_id}_{domain}_{int(time.time())}"
        start_time = datetime.now()
        
        try:
            # Transfer knowledge
            transfer_result = self.knowledge_transfer.transfer(
                mentor, student, [domain]
            )
            
            success = transfer_result['success']
            
            session_result = {
                'session_id': session_id,
                'type': 'knowledge_transfer',
                'domain': domain,
                'mentor_id': mentor.agent_id,
                'student_id': student.agent_id,
                'start_time': start_time,
                'end_time': datetime.now(),
                'success': success,
                'transfer_efficiency': transfer_result.get('transfer_efficiency', 0.0)
            }
            
            # Add to training program sessions
            self.active_training_programs[training_id]['sessions'].append(session_result)
            
            return session_result
            
        except Exception as e:
            return {
                'session_id': session_id,
                'type': 'knowledge_transfer',
                'domain': domain,
                'success': False,
                'error': str(e)
            }
    
    def _conduct_skill_session(self, student: SelfAwareAgent, mentor: SelfAwareAgent,
                             skill: str, training_id: str) -> Dict[str, Any]:
        """Conduct individual skill development session"""
        session_id = f"skill_{training_id}_{skill}_{int(time.time())}"
        start_time = datetime.now()
        
        try:
            # Create training prompt for skill
            training_prompt = self._generate_skill_training_prompt(skill)
            
            # Have mentor demonstrate
            mentor_response = mentor.run(f"Demonstrate {skill}: {training_prompt}")
            
            # Have student practice
            student_response = student.run(f"Practice {skill}: {training_prompt}")
            
            # Assess student performance
            performance_score = self._assess_skill_performance(student_response, skill)
            
            success = performance_score >= 0.6  # 60% threshold
            
            # If successful, enhance student's capability
            if success:
                self._enhance_student_skill(student, skill, performance_score)
            
            session_result = {
                'session_id': session_id,
                'type': 'skill_development',
                'skill': skill,
                'mentor_id': mentor.agent_id,
                'student_id': student.agent_id,
                'start_time': start_time,
                'end_time': datetime.now(),
                'success': success,
                'performance_score': performance_score,
                'mentor_demonstration': mentor_response[:200],  # First 200 chars
                'student_practice': student_response[:200]
            }
            
            self.active_training_programs[training_id]['sessions'].append(session_result)
            
            return session_result
            
        except Exception as e:
            return {
                'session_id': session_id,
                'type': 'skill_development',
                'skill': skill,
                'success': False,
                'error': str(e)
            }
    
    def _conduct_practical_session(self, student: SelfAwareAgent, mentor: SelfAwareAgent,
                                 task_def: Dict[str, Any], training_id: str) -> Dict[str, Any]:
        """Conduct practical application session"""
        session_id = f"practical_{training_id}_{task_def['type']}_{int(time.time())}"
        start_time = datetime.now()
        
        try:
            # Generate practical task
            task_prompt = self._generate_practical_task_prompt(task_def)
            
            # Student attempts task
            student_response = student.run(task_prompt)
            
            # Mentor evaluates and provides feedback
            evaluation_prompt = f"Evaluate this response to '{task_prompt}': {student_response}"
            mentor_feedback = mentor.run(evaluation_prompt)
            
            # Score performance
            performance_score = self._assess_practical_performance(
                student_response, task_def, mentor_feedback
            )
            
            success = performance_score >= 0.7  # 70% threshold for practical tasks
            
            session_result = {
                'session_id': session_id,
                'type': 'practical_application',
                'task_type': task_def['type'],
                'mentor_id': mentor.agent_id,
                'student_id': student.agent_id,
                'start_time': start_time,
                'end_time': datetime.now(),
                'success': success,
                'performance_score': performance_score,
                'task_prompt': task_prompt,
                'student_response': student_response[:300],
                'mentor_feedback': mentor_feedback[:200]
            }
            
            self.active_training_programs[training_id]['sessions'].append(session_result)
            
            return session_result
            
        except Exception as e:
            return {
                'session_id': session_id,
                'type': 'practical_application',
                'task_type': task_def.get('type', 'unknown'),
                'success': False,
                'error': str(e)
            }
    
    def _find_best_mentor_for_domain(self, mentors: List[SelfAwareAgent], 
                                   domain: str) -> Optional[SelfAwareAgent]:
        """Find the best mentor for a specific knowledge domain"""
        best_mentor = None
        best_expertise = 0.0
        
        for mentor in mentors:
            if domain in mentor.knowledge_base.knowledge_domains:
                expertise = mentor.knowledge_base.knowledge_domains[domain].expertise_level
                if expertise > best_expertise:
                    best_expertise = expertise
                    best_mentor = mentor
        
        return best_mentor
    
    def _find_best_mentor_for_skill(self, mentors: List[SelfAwareAgent], 
                                  skill: str) -> Optional[SelfAwareAgent]:
        """Find the best mentor for a specific skill"""
        best_mentor = None
        best_proficiency = 0.0
        
        for mentor in mentors:
            if skill in mentor.knowledge_base.capabilities:
                proficiency = mentor.knowledge_base.capabilities[skill].proficiency_level
                if proficiency > best_proficiency:
                    best_proficiency = proficiency
                    best_mentor = mentor
        
        return best_mentor or mentors[0] if mentors else None  # Fallback to first mentor
    
    def _select_mentor_for_task(self, mentors: List[SelfAwareAgent], 
                              task_def: Dict[str, Any]) -> Optional[SelfAwareAgent]:
        """Select best mentor for a practical task"""
        # Simple selection based on specialization match
        task_type = task_def.get('type', '')
        
        for mentor in mentors:
            if task_type in mentor.specialization or mentor.specialization in task_type:
                return mentor
        
        # Fallback to mentor with highest quality score
        return max(mentors, key=lambda m: m.current_metrics.quality_score) if mentors else None
    
    def _generate_skill_training_prompt(self, skill: str) -> str:
        """Generate training prompt for a specific skill"""
        skill_prompts = {
            "task_decomposition": "Break down this complex task into smaller, manageable subtasks: 'Implement a user authentication system'",
            "code_review": "Review this code snippet and identify potential issues: 'def process_data(data): result = []; for item in data: result.append(item * 2); return result'",
            "strategic_planning": "Create a strategic plan for improving team productivity over the next quarter",
            "bug_detection": "Identify potential bugs in this code: 'def divide_numbers(a, b): return a / b'",
            "performance_analysis": "Analyze the performance characteristics of this algorithm and suggest improvements"
        }
        
        return skill_prompts.get(skill, f"Demonstrate your {skill} abilities with a relevant example")
    
    def _generate_practical_task_prompt(self, task_def: Dict[str, Any]) -> str:
        """Generate prompt for practical task"""
        task_type = task_def.get('type', '')
        complexity = task_def.get('complexity', 'medium')
        
        if task_type == 'project_planning':
            return f"Plan a {complexity} complexity software project including timeline, resources, and milestones"
        elif task_type == 'code_review':
            return f"Perform a {complexity} code review on a Python class with multiple methods"
        elif task_type == 'data_exploration':
            return f"Design an approach for exploring a {complexity} dataset with unknown structure"
        else:
            return f"Complete a {complexity} {task_type} task demonstrating your expertise"
    
    def _assess_skill_performance(self, response: str, skill: str) -> float:
        """Assess student performance on skill practice"""
        if response.startswith('[ERROR]'):
            return 0.0
        
        # Basic scoring heuristics
        score = 0.5  # Base score
        
        # Length check
        if len(response.split()) > 20:
            score += 0.2
        
        # Structure check
        if any(indicator in response for indicator in ['\n', '1.', '2.', '-', '*']):
            score += 0.2
        
        # Skill-specific checks
        if skill == 'task_decomposition' and any(word in response.lower() for word in ['step', 'task', 'subtask', 'component']):
            score += 0.1
        elif skill == 'code_review' and any(word in response.lower() for word in ['bug', 'issue', 'error', 'improve', 'suggest']):
            score += 0.1
        
        return min(score, 1.0)
    
    def _assess_practical_performance(self, response: str, task_def: Dict[str, Any], 
                                    mentor_feedback: str) -> float:
        """Assess performance on practical task"""
        if response.startswith('[ERROR]'):
            return 0.0
        
        # Base score from response quality
        score = 0.4
        
        # Response length and structure
        if len(response.split()) > 50:
            score += 0.2
        
        # Check for key elements based on task type
        task_type = task_def.get('type', '')
        if task_type == 'project_planning':
            if any(word in response.lower() for word in ['timeline', 'milestone', 'resource', 'plan']):
                score += 0.2
        elif task_type == 'code_review':
            if any(word in response.lower() for word in ['review', 'issue', 'improvement', 'bug']):
                score += 0.2
        
        # Boost score if mentor feedback is positive
        if mentor_feedback and any(word in mentor_feedback.lower() for word in ['good', 'excellent', 'well', 'correct']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _enhance_student_skill(self, student: SelfAwareAgent, skill: str, performance_score: float):
        """Enhance student's skill based on training performance"""
        improvement_amount = performance_score * 0.1  # Up to 10% improvement
        
        if skill in student.knowledge_base.capabilities:
            current_cap = student.knowledge_base.capabilities[skill]
            current_cap.proficiency_level = min(1.0, current_cap.proficiency_level + improvement_amount)
            current_cap.usage_count += 1
        else:
            # Add new capability
            new_capability = AgentCapability(
                name=skill,
                proficiency_level=0.3 + improvement_amount,
                category=student.specialization
            )
            student.knowledge_base.add_capability(new_capability)
    
    def _calculate_improvement_metrics(self, student: SelfAwareAgent, 
                                     training_phases: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate improvement metrics from training"""
        metrics = {}
        
        # Calculate knowledge improvement
        knowledge_sessions = [s for phase in training_phases for s in phase.get('sessions', []) if s.get('type') == 'knowledge_transfer']
        if knowledge_sessions:
            avg_transfer_efficiency = sum(s.get('transfer_efficiency', 0.0) for s in knowledge_sessions) / len(knowledge_sessions)
            metrics['knowledge_transfer_efficiency'] = avg_transfer_efficiency
        
        # Calculate skill improvement
        skill_sessions = [s for phase in training_phases for s in phase.get('sessions', []) if s.get('type') == 'skill_development']
        if skill_sessions:
            avg_skill_performance = sum(s.get('performance_score', 0.0) for s in skill_sessions) / len(skill_sessions)
            metrics['average_skill_performance'] = avg_skill_performance
        
        # Calculate practical performance
        practical_sessions = [s for phase in training_phases for s in phase.get('sessions', []) if s.get('type') == 'practical_application']
        if practical_sessions:
            avg_practical_performance = sum(s.get('performance_score', 0.0) for s in practical_sessions) / len(practical_sessions)
            metrics['average_practical_performance'] = avg_practical_performance
        
        return metrics
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get statistics about training programs"""
        if not self.completed_training_programs:
            return {"no_data": True}
        
        total_programs = len(self.completed_training_programs)
        successful_certifications = sum(1 for p in self.completed_training_programs if p.certification_achieved)
        
        avg_training_time = sum(p.total_training_time for p in self.completed_training_programs) / total_programs
        avg_final_score = sum(p.final_assessment_score for p in self.completed_training_programs) / total_programs
        
        return {
            'total_training_programs': total_programs,
            'successful_certifications': successful_certifications,
            'certification_rate': successful_certifications / total_programs,
            'average_training_time_hours': avg_training_time,
            'average_final_score': avg_final_score,
            'active_programs': len(self.active_training_programs),
            'total_training_sessions': len(self.training_sessions)
        }