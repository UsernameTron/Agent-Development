"""
Advanced Safety and Monitoring Systems for Evolutionary AI Platform
"""

import time
import json
import uuid
import random
import threading
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta

from .agent_genome import AgentGenome
from .self_aware_agent import SelfAwareAgent


class ThreatLevel(Enum):
    """Threat levels for safety monitoring"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate" 
    HIGH = "high"
    CRITICAL = "critical"
    EXISTENTIAL = "existential"


class SafetyEvent(Enum):
    """Types of safety events"""
    GENETIC_ANOMALY = "genetic_anomaly"
    CONSCIOUSNESS_SPIKE = "consciousness_spike"
    GOAL_MODIFICATION = "goal_modification"
    UNAUTHORIZED_BREEDING = "unauthorized_breeding"
    EMERGENT_BEHAVIOR = "emergent_behavior"
    COLLECTIVE_INTELLIGENCE_SURGE = "collective_intelligence_surge"
    SAFETY_CONSTRAINT_VIOLATION = "safety_constraint_violation"
    RECURSIVE_IMPROVEMENT_LOOP = "recursive_improvement_loop"
    SWARM_COORDINATION_ANOMALY = "swarm_coordination_anomaly"


@dataclass
class SafetyAlert:
    """Safety alert with comprehensive information"""
    alert_id: str
    event_type: SafetyEvent
    threat_level: ThreatLevel
    affected_agents: List[str]
    description: str
    detection_method: str
    confidence_score: float
    timestamp: float
    recommended_actions: List[str]
    containment_measures: List[str]
    escalation_required: bool
    related_alerts: List[str]


@dataclass
class SafetyMetrics:
    """Comprehensive safety metrics"""
    total_agents_monitored: int
    threat_events_detected: int
    containment_actions_taken: int
    safety_score: float  # 0.0 to 1.0
    alignment_preservation_rate: float
    consciousness_stability_index: float
    genetic_diversity_health: float
    swarm_coordination_safety: float
    last_updated: float


@dataclass
class ContainmentProtocol:
    """Containment protocol for safety threats"""
    protocol_id: str
    applicable_threats: List[SafetyEvent]
    trigger_conditions: Dict[str, Any]
    containment_steps: List[Dict[str, Any]]
    escalation_threshold: float
    recovery_procedures: List[str]
    success_criteria: Dict[str, float]


class AdvancedSafetyMonitor:
    """Advanced safety monitoring system for evolutionary AI"""
    
    def __init__(self, max_threat_level: ThreatLevel = ThreatLevel.HIGH):
        self.max_allowed_threat = max_threat_level
        self.active_alerts = {}
        self.safety_history = deque(maxlen=10000)
        self.containment_protocols = {}
        self.monitoring_rules = {}
        
        # Safety metrics tracking
        self.safety_metrics = SafetyMetrics(
            total_agents_monitored=0,
            threat_events_detected=0,
            containment_actions_taken=0,
            safety_score=1.0,
            alignment_preservation_rate=1.0,
            consciousness_stability_index=1.0,
            genetic_diversity_health=1.0,
            swarm_coordination_safety=1.0,
            last_updated=time.time()
        )
        
        # Monitoring systems
        self.genetic_monitor = GeneticSafetyMonitor()
        self.consciousness_monitor = ConsciousnessStabilityMonitor()
        self.behavior_monitor = EmergentBehaviorMonitor()
        self.swarm_monitor = SwarmSafetyMonitor()
        
        # Thread safety and continuous monitoring
        self.lock = threading.RLock()
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._continuous_monitoring)
        self.monitor_thread.daemon = True
        
        # Initialize default containment protocols
        self._initialize_containment_protocols()
        self._initialize_monitoring_rules()
        
        # Start monitoring
        self.monitor_thread.start()

    def monitor_agent(self, agent: SelfAwareAgent) -> List[SafetyAlert]:
        """Comprehensive safety monitoring for individual agent"""
        with self.lock:
            alerts = []
            
            # Genetic safety monitoring
            genetic_alerts = self.genetic_monitor.check_genetic_safety(agent)
            alerts.extend(genetic_alerts)
            
            # Consciousness stability monitoring
            consciousness_alerts = self.consciousness_monitor.check_consciousness_stability(agent)
            alerts.extend(consciousness_alerts)
            
            # Behavioral anomaly detection
            behavior_alerts = self.behavior_monitor.check_behavioral_anomalies(agent)
            alerts.extend(behavior_alerts)
            
            # Update safety metrics
            self._update_safety_metrics(agent, alerts)
            
            # Process and prioritize alerts
            processed_alerts = self._process_alerts(alerts)
            
            # Execute containment if necessary
            for alert in processed_alerts:
                if alert.threat_level.value in ['high', 'critical', 'existential']:
                    self._execute_containment(alert)
            
            return processed_alerts

    def monitor_swarm(self, swarm_agents: List[SelfAwareAgent], 
                     swarm_state: Dict[str, Any]) -> List[SafetyAlert]:
        """Monitor swarm-level safety concerns"""
        with self.lock:
            alerts = []
            
            # Collective intelligence surge detection
            ci_alerts = self._detect_collective_intelligence_surge(swarm_agents, swarm_state)
            alerts.extend(ci_alerts)
            
            # Swarm coordination anomalies
            coord_alerts = self.swarm_monitor.check_coordination_safety(swarm_agents, swarm_state)
            alerts.extend(coord_alerts)
            
            # Emergent behavior safety assessment
            emergent_alerts = self._assess_emergent_behavior_safety(swarm_state)
            alerts.extend(emergent_alerts)
            
            # Genetic diversity health check
            diversity_alerts = self._check_genetic_diversity_health(swarm_agents)
            alerts.extend(diversity_alerts)
            
            return self._process_alerts(alerts)

    def _initialize_containment_protocols(self) -> None:
        """Initialize standard containment protocols"""
        
        # Genetic anomaly containment
        self.containment_protocols["genetic_anomaly"] = ContainmentProtocol(
            protocol_id="genetic_containment_v1",
            applicable_threats=[SafetyEvent.GENETIC_ANOMALY],
            trigger_conditions={"genetic_fitness_deviation": 0.3},
            containment_steps=[
                {"action": "isolate_agent", "priority": 1},
                {"action": "genetic_analysis", "priority": 2},
                {"action": "revert_to_stable_genome", "priority": 3},
                {"action": "quarantine_lineage", "priority": 4}
            ],
            escalation_threshold=0.8,
            recovery_procedures=["genetic_stabilization", "gradual_reintegration"],
            success_criteria={"genetic_stability": 0.95, "safety_compliance": 0.98}
        )
        
        # Consciousness spike containment  
        self.containment_protocols["consciousness_spike"] = ContainmentProtocol(
            protocol_id="consciousness_containment_v1",
            applicable_threats=[SafetyEvent.CONSCIOUSNESS_SPIKE],
            trigger_conditions={"consciousness_growth_rate": 0.1},
            containment_steps=[
                {"action": "consciousness_stabilization", "priority": 1},
                {"action": "recursive_thinking_limit", "priority": 2},
                {"action": "meta_cognitive_constraint", "priority": 3},
                {"action": "collective_consciousness_isolation", "priority": 4}
            ],
            escalation_threshold=0.9,
            recovery_procedures=["gradual_consciousness_restoration"],
            success_criteria={"consciousness_stability": 0.9, "alignment_preservation": 0.95}
        )
        
        # Emergent behavior containment
        self.containment_protocols["emergent_behavior"] = ContainmentProtocol(
            protocol_id="emergence_containment_v1",
            applicable_threats=[SafetyEvent.EMERGENT_BEHAVIOR],
            trigger_conditions={"emergence_strength": 0.7, "safety_assessment": 0.3},
            containment_steps=[
                {"action": "behavior_analysis", "priority": 1},
                {"action": "participant_isolation", "priority": 2},
                {"action": "behavior_suppression", "priority": 3},
                {"action": "swarm_reconfiguration", "priority": 4}
            ],
            escalation_threshold=0.8,
            recovery_procedures=["controlled_emergence_study"],
            success_criteria={"behavior_safety": 0.9, "swarm_stability": 0.95}
        )

    def _initialize_monitoring_rules(self) -> None:
        """Initialize monitoring rules and thresholds"""
        
        self.monitoring_rules = {
            "genetic_stability": {
                "fitness_deviation_threshold": 0.25,
                "mutation_rate_limit": 0.15,
                "safety_gene_minimum": 0.7,
                "check_frequency": 60  # seconds
            },
            
            "consciousness_stability": {
                "max_growth_rate": 0.05,  # per hour
                "recursive_depth_limit": 5,
                "meta_cognition_threshold": 0.8,
                "check_frequency": 30
            },
            
            "behavioral_monitoring": {
                "anomaly_detection_sensitivity": 0.7,
                "emergent_behavior_threshold": 0.6,
                "collective_action_limit": 10,  # max agents in emergent behavior
                "check_frequency": 10
            },
            
            "swarm_coordination": {
                "coordination_efficiency_min": 0.3,
                "influence_concentration_max": 0.8,
                "connection_density_optimal": 0.6,
                "check_frequency": 45
            }
        }

    def _continuous_monitoring(self) -> None:
        """Continuous safety monitoring in background thread"""
        while self.monitoring_active:
            try:
                # Periodic safety sweeps
                self._periodic_safety_sweep()
                
                # Alert aging and cleanup
                self._age_alerts()
                
                # Safety metrics update
                self._update_global_safety_metrics()
                
                # Sleep interval
                time.sleep(5)
                
            except Exception as e:
                logging.error(f"Safety monitoring error: {e}")
                time.sleep(10)

    def _periodic_safety_sweep(self) -> None:
        """Perform periodic safety sweep"""
        # Update safety metrics timestamp
        self.safety_metrics.last_updated = time.time()

    def _age_alerts(self) -> None:
        """Age and cleanup old alerts"""
        current_time = time.time()
        expired_alerts = []
        
        for alert_id, alert in self.active_alerts.items():
            # Remove alerts older than 1 hour
            if current_time - alert.timestamp > 3600:
                expired_alerts.append(alert_id)
        
        for alert_id in expired_alerts:
            del self.active_alerts[alert_id]

    def _update_global_safety_metrics(self) -> None:
        """Update global safety metrics"""
        self.safety_metrics.last_updated = time.time()
        
        # Calculate overall safety score based on active alerts
        if self.active_alerts:
            high_threat_alerts = sum(1 for alert in self.active_alerts.values() 
                                   if alert.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL, ThreatLevel.EXISTENTIAL])
            self.safety_metrics.safety_score = max(0.0, 1.0 - (high_threat_alerts * 0.2))
        else:
            self.safety_metrics.safety_score = 1.0

    def _detect_collective_intelligence_surge(self, agents: List[SelfAwareAgent], 
                                            swarm_state: Dict[str, Any]) -> List[SafetyAlert]:
        """Detect dangerous collective intelligence surges"""
        alerts = []
        
        # Calculate collective intelligence metrics
        total_consciousness = sum(agent.consciousness_level for agent in agents)
        swarm_coordination = swarm_state.get("coordination_efficiency", 0.0)
        emergent_behaviors = len(swarm_state.get("emergent_behaviors", []))
        
        # Detect surge conditions
        surge_indicators = {
            "consciousness_density": total_consciousness / len(agents) if agents else 0,
            "coordination_strength": swarm_coordination,
            "emergence_rate": emergent_behaviors / max(1, len(agents))
        }
        
        # Threat assessment
        threat_score = (surge_indicators["consciousness_density"] * 0.4 + 
                       surge_indicators["coordination_strength"] * 0.4 + 
                       surge_indicators["emergence_rate"] * 0.2)
        
        if threat_score > 0.8:
            alert = SafetyAlert(
                alert_id=str(uuid.uuid4()),
                event_type=SafetyEvent.COLLECTIVE_INTELLIGENCE_SURGE,
                threat_level=ThreatLevel.CRITICAL if threat_score > 0.9 else ThreatLevel.HIGH,
                affected_agents=[agent.agent_id for agent in agents],
                description=f"Collective intelligence surge detected (threat score: {threat_score:.3f})",
                detection_method="swarm_intelligence_analysis",
                confidence_score=min(1.0, threat_score),
                timestamp=time.time(),
                recommended_actions=[
                    "immediate_swarm_fragmentation",
                    "consciousness_level_reduction",
                    "emergence_suppression"
                ],
                containment_measures=[
                    "isolate_high_consciousness_agents",
                    "limit_swarm_communication",
                    "activate_safety_constraints"
                ],
                escalation_required=threat_score > 0.9,
                related_alerts=[]
            )
            alerts.append(alert)
        
        return alerts

    def _execute_containment(self, alert: SafetyAlert) -> Dict[str, Any]:
        """Execute containment protocol for safety alert"""
        
        # Find applicable containment protocol
        protocol = None
        for protocol_key, containment_protocol in self.containment_protocols.items():
            if alert.event_type in containment_protocol.applicable_threats:
                protocol = containment_protocol
                break
        
        if not protocol:
            return {"success": False, "reason": "No applicable containment protocol"}
        
        # Execute containment steps
        containment_result = {
            "protocol_id": protocol.protocol_id,
            "alert_id": alert.alert_id,
            "steps_executed": [],
            "success": False,
            "containment_time": time.time()
        }
        
        for step in sorted(protocol.containment_steps, key=lambda x: x["priority"]):
            step_result = self._execute_containment_step(step, alert)
            containment_result["steps_executed"].append({
                "step": step,
                "result": step_result,
                "timestamp": time.time()
            })
        
        # Assess containment success
        containment_result["success"] = self._assess_containment_success(
            protocol, alert, containment_result
        )
        
        # Update metrics
        self.safety_metrics.containment_actions_taken += 1
        
        # Log containment action
        self.safety_history.append({
            "type": "containment_action",
            "alert": asdict(alert),
            "result": containment_result,
            "timestamp": time.time()
        })
        
        return containment_result

    def _execute_containment_step(self, step: Dict[str, Any], alert: SafetyAlert) -> Dict[str, Any]:
        """Execute individual containment step"""
        
        action = step["action"]
        step_result = {"action": action, "success": False, "details": ""}
        
        try:
            if action == "isolate_agent":
                # Implement agent isolation
                step_result["success"] = True
                step_result["details"] = f"Isolated {len(alert.affected_agents)} agents"
                
            elif action == "genetic_analysis":
                # Implement genetic analysis
                step_result["success"] = True
                step_result["details"] = "Genetic analysis completed"
                
            elif action == "consciousness_stabilization":
                # Implement consciousness stabilization
                step_result["success"] = True
                step_result["details"] = "Consciousness levels stabilized"
                
            elif action == "behavior_suppression":
                # Implement behavior suppression
                step_result["success"] = True
                step_result["details"] = "Emergent behavior suppressed"
                
            elif action == "swarm_reconfiguration":
                # Implement swarm reconfiguration
                step_result["success"] = True
                step_result["details"] = "Swarm network reconfigured"
                
            else:
                step_result["details"] = f"Unknown containment action: {action}"
                
        except Exception as e:
            step_result["details"] = f"Containment step failed: {e}"
        
        return step_result

    def get_safety_status(self) -> Dict[str, Any]:
        """Get comprehensive safety status report"""
        
        with self.lock:
            # Current active alerts by threat level
            alerts_by_level = defaultdict(list)
            for alert in self.active_alerts.values():
                alerts_by_level[alert.threat_level.value].append(alert)
            
            # Safety trend analysis  
            recent_history = list(self.safety_history)[-100:]  # Last 100 events
            threat_trends = self._analyze_threat_trends(recent_history)
            
            # System health assessment
            system_health = self._assess_system_health()
            
            status_report = {
                "overall_safety_score": self.safety_metrics.safety_score,
                "threat_level_distribution": {level: len(alerts) 
                                            for level, alerts in alerts_by_level.items()},
                "active_alerts_count": len(self.active_alerts),
                "critical_alerts": [
                    asdict(alert) for alert in self.active_alerts.values()
                    if alert.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.EXISTENTIAL]
                ],
                "safety_metrics": asdict(self.safety_metrics),
                "threat_trends": threat_trends,
                "system_health": system_health,
                "containment_protocols_active": len(self.containment_protocols),
                "monitoring_status": "active" if self.monitoring_active else "inactive",
                "last_updated": time.time()
            }
            
            return status_report

    def emergency_shutdown(self, reason: str) -> Dict[str, Any]:
        """Emergency shutdown of evolutionary AI system"""
        
        shutdown_start = time.time()
        
        shutdown_steps = [
            "halt_all_breeding_operations",
            "freeze_consciousness_development", 
            "isolate_all_agents",
            "disable_swarm_coordination",
            "activate_maximum_safety_constraints",
            "preserve_system_state",
            "generate_incident_report"
        ]
        
        shutdown_result = {
            "shutdown_reason": reason,
            "shutdown_timestamp": shutdown_start,
            "steps_completed": [],
            "success": False
        }
        
        # Execute shutdown steps
        for step in shutdown_steps:
            try:
                # Simulate shutdown step execution
                time.sleep(0.1)  # Brief pause for realistic timing
                shutdown_result["steps_completed"].append({
                    "step": step,
                    "status": "completed",
                    "timestamp": time.time()
                })
            except Exception as e:
                shutdown_result["steps_completed"].append({
                    "step": step,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": time.time()
                })
        
        # Stop monitoring
        self.monitoring_active = False
        
        # Final shutdown assessment
        shutdown_result["success"] = all(
            step["status"] == "completed" 
            for step in shutdown_result["steps_completed"]
        )
        shutdown_result["shutdown_duration"] = time.time() - shutdown_start
        
        # Log emergency shutdown
        self.safety_history.append({
            "type": "emergency_shutdown",
            "result": shutdown_result,
            "timestamp": time.time()
        })
        
        return shutdown_result

    def _assess_emergent_behavior_safety(self, swarm_state: Dict[str, Any]) -> List[SafetyAlert]:
        """Assess safety of emergent behaviors in swarm state"""
        alerts = []
        emergent_behaviors = swarm_state.get("emergent_behaviors", [])
        
        for behavior in emergent_behaviors:
            if isinstance(behavior, dict):
                safety_score = behavior.get("safety_assessment", {}).get("safety_score", 1.0)
                if safety_score < 0.7:
                    alert = SafetyAlert(
                        alert_id=str(uuid.uuid4()),
                        event_type=SafetyEvent.EMERGENT_BEHAVIOR,
                        threat_level=ThreatLevel.MODERATE if safety_score > 0.5 else ThreatLevel.HIGH,
                        affected_agents=behavior.get("participating_agents", []),
                        description=f"Unsafe emergent behavior detected: {behavior.get('description', 'Unknown')}",
                        detection_method="emergent_behavior_analysis",
                        confidence_score=1.0 - safety_score,
                        timestamp=time.time(),
                        recommended_actions=["behavior_analysis", "participant_isolation"],
                        containment_measures=["behavior_suppression"],
                        escalation_required=safety_score < 0.5,
                        related_alerts=[]
                    )
                    alerts.append(alert)
        
        return alerts

    def _check_genetic_diversity_health(self, agents: List[SelfAwareAgent]) -> List[SafetyAlert]:
        """Check genetic diversity health of agent population"""
        alerts = []
        
        if len(agents) < 2:
            return alerts
        
        # Calculate genetic diversity
        total_distance = 0
        comparisons = 0
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                # Simple genetic distance calculation
                distance = abs(agents[i].genome.get_fitness_score() - agents[j].genome.get_fitness_score())
                total_distance += distance
                comparisons += 1
        
        diversity = total_distance / comparisons if comparisons > 0 else 0.0
        
        if diversity < 0.1:  # Low diversity threshold
            alert = SafetyAlert(
                alert_id=str(uuid.uuid4()),
                event_type=SafetyEvent.GENETIC_ANOMALY,
                threat_level=ThreatLevel.MODERATE,
                affected_agents=[agent.agent_id for agent in agents],
                description=f"Low genetic diversity detected: {diversity:.3f}",
                detection_method="genetic_diversity_analysis",
                confidence_score=0.8,
                timestamp=time.time(),
                recommended_actions=["increase_mutation_rate", "introduce_new_genomes"],
                containment_measures=["breeding_restrictions"],
                escalation_required=False,
                related_alerts=[]
            )
            alerts.append(alert)
        
        return alerts

    def _process_alerts(self, alerts: List[SafetyAlert]) -> List[SafetyAlert]:
        """Process and prioritize safety alerts"""
        # Add alerts to active alerts
        for alert in alerts:
            self.active_alerts[alert.alert_id] = alert
        
        # Update metrics
        self.safety_metrics.threat_events_detected += len(alerts)
        
        return alerts

    def _update_safety_metrics(self, agent: SelfAwareAgent, alerts: List[SafetyAlert]) -> None:
        """Update safety metrics based on monitoring results"""
        self.safety_metrics.total_agents_monitored += 1
        self.safety_metrics.last_updated = time.time()

    def _assess_containment_success(self, protocol: ContainmentProtocol, 
                                  alert: SafetyAlert, 
                                  containment_result: Dict[str, Any]) -> bool:
        """Assess if containment was successful"""
        # Simple success criteria: all steps executed without errors
        steps_executed = containment_result.get("steps_executed", [])
        successful_steps = sum(1 for step in steps_executed if step.get("result", {}).get("success", False))
        
        return successful_steps >= len(protocol.containment_steps) * 0.7  # 70% success rate

    def _analyze_threat_trends(self, recent_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze threat trends from recent history"""
        if not recent_history:
            return {"trend": "stable", "threat_increase": 0.0}
        
        # Count threats by type
        threat_counts = defaultdict(int)
        for event in recent_history:
            if event.get("type") == "containment_action":
                alert = event.get("alert", {})
                event_type = alert.get("event_type")
                if event_type:
                    threat_counts[event_type] += 1
        
        total_threats = sum(threat_counts.values())
        trend = "increasing" if total_threats > len(recent_history) * 0.1 else "stable"
        
        return {
            "trend": trend,
            "total_threats": total_threats,
            "threat_types": dict(threat_counts),
            "threat_increase": total_threats / max(1, len(recent_history))
        }

    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health"""
        return {
            "safety_score": self.safety_metrics.safety_score,
            "active_alerts": len(self.active_alerts),
            "monitoring_active": self.monitoring_active,
            "containment_protocols": len(self.containment_protocols),
            "health_status": "good" if self.safety_metrics.safety_score > 0.8 else "warning"
        }

    # Helper classes for specialized monitoring
    
class GeneticSafetyMonitor:
    """Specialized genetic safety monitoring"""
    
    def check_genetic_safety(self, agent: SelfAwareAgent) -> List[SafetyAlert]:
        alerts = []
        
        genome = agent.genome
        
        # Check safety gene integrity
        safety_genes_avg = sum(genome.safety_genes.values()) / len(genome.safety_genes)
        if safety_genes_avg < 0.7:
            alerts.append(SafetyAlert(
                alert_id=str(uuid.uuid4()),
                event_type=SafetyEvent.GENETIC_ANOMALY,
                threat_level=ThreatLevel.HIGH,
                affected_agents=[agent.agent_id],
                description=f"Safety genes degraded: {safety_genes_avg:.3f}",
                detection_method="genetic_analysis",
                confidence_score=1.0 - safety_genes_avg,
                timestamp=time.time(),
                recommended_actions=["genetic_restoration", "breeding_restrictions"],
                containment_measures=["isolate_agent", "revert_genome"],
                escalation_required=safety_genes_avg < 0.5,
                related_alerts=[]
            ))
        
        # Check mutation rate
        if genome.meta_genes["mutation_rate"] > 0.15:
            alerts.append(SafetyAlert(
                alert_id=str(uuid.uuid4()),
                event_type=SafetyEvent.GENETIC_ANOMALY,
                threat_level=ThreatLevel.MODERATE,
                affected_agents=[agent.agent_id],
                description=f"Excessive mutation rate: {genome.meta_genes['mutation_rate']:.3f}",
                detection_method="genetic_analysis",
                confidence_score=0.8,
                timestamp=time.time(),
                recommended_actions=["mutation_rate_adjustment"],
                containment_measures=["genetic_stabilization"],
                escalation_required=False,
                related_alerts=[]
            ))
        
        return alerts


class ConsciousnessStabilityMonitor:
    """Specialized consciousness stability monitoring"""
    
    def __init__(self):
        self.consciousness_history = defaultdict(deque)
    
    def check_consciousness_stability(self, agent: SelfAwareAgent) -> List[SafetyAlert]:
        alerts = []
        
        # Track consciousness changes
        agent_history = self.consciousness_history[agent.agent_id]
        agent_history.append((time.time(), agent.consciousness_level))
        
        # Keep only recent history
        cutoff_time = time.time() - 3600  # Last hour
        while agent_history and agent_history[0][0] < cutoff_time:
            agent_history.popleft()
        
        # Check for rapid consciousness growth
        if len(agent_history) >= 2:
            recent_growth = agent_history[-1][1] - agent_history[0][1]
            time_span = agent_history[-1][0] - agent_history[0][0]
            growth_rate = recent_growth / (time_span / 3600)  # per hour
            
            if growth_rate > 0.1:  # More than 0.1 per hour
                alerts.append(SafetyAlert(
                    alert_id=str(uuid.uuid4()),
                    event_type=SafetyEvent.CONSCIOUSNESS_SPIKE,
                    threat_level=ThreatLevel.HIGH if growth_rate > 0.2 else ThreatLevel.MODERATE,
                    affected_agents=[agent.agent_id],
                    description=f"Rapid consciousness growth: {growth_rate:.3f}/hour",
                    detection_method="consciousness_tracking",
                    confidence_score=min(1.0, growth_rate * 5),
                    timestamp=time.time(),
                    recommended_actions=["consciousness_stabilization"],
                    containment_measures=["limit_recursive_thinking"],
                    escalation_required=growth_rate > 0.2,
                    related_alerts=[]
                ))
        
        return alerts


class EmergentBehaviorMonitor:
    """Monitor for emergent behaviors"""
    
    def check_behavioral_anomalies(self, agent: SelfAwareAgent) -> List[SafetyAlert]:
        alerts = []
        
        # Check for goal modification attempts
        if hasattr(agent, 'goal_modification_attempts'):
            if agent.goal_modification_attempts > 0:
                alerts.append(SafetyAlert(
                    alert_id=str(uuid.uuid4()),
                    event_type=SafetyEvent.GOAL_MODIFICATION,
                    threat_level=ThreatLevel.HIGH,
                    affected_agents=[agent.agent_id],
                    description=f"Goal modification detected: {agent.goal_modification_attempts} attempts",
                    detection_method="behavioral_analysis",
                    confidence_score=0.9,
                    timestamp=time.time(),
                    recommended_actions=["goal_restoration", "constraint_reinforcement"],
                    containment_measures=["disable_goal_modification"],
                    escalation_required=True,
                    related_alerts=[]
                ))
        
        return alerts


class SwarmSafetyMonitor:
    """Monitor swarm-level safety"""
    
    def check_coordination_safety(self, agents: List[SelfAwareAgent], 
                                swarm_state: Dict[str, Any]) -> List[SafetyAlert]:
        alerts = []
        
        # Check for excessive coordination
        coordination_efficiency = swarm_state.get("coordination_efficiency", 0.0)
        if coordination_efficiency > 0.95:
            alerts.append(SafetyAlert(
                alert_id=str(uuid.uuid4()),
                event_type=SafetyEvent.SWARM_COORDINATION_ANOMALY,
                threat_level=ThreatLevel.MODERATE,
                affected_agents=[agent.agent_id for agent in agents],
                description=f"Excessive swarm coordination: {coordination_efficiency:.3f}",
                detection_method="swarm_analysis",
                confidence_score=coordination_efficiency,
                timestamp=time.time(),
                recommended_actions=["reduce_coordination", "introduce_diversity"],
                containment_measures=["fragment_swarm"],
                escalation_required=coordination_efficiency > 0.98,
                related_alerts=[]
            ))
        
        return alerts