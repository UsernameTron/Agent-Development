"""
Agent DNA System for Inheritable Agent Traits

This module implements the genetic system for agent breeding and evolution.
Each agent has DNA that defines its inheritable characteristics.
"""

import random
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from copy import deepcopy


@dataclass
class BehavioralTrait:
    """Defines a behavioral trait with inheritance properties"""
    name: str
    strength: float  # 0.0 to 1.0
    dominance: float  # How likely this trait is to be inherited (0.0 to 1.0)
    category: str
    mutation_rate: float = 0.1
    
    def mutate(self) -> 'BehavioralTrait':
        """Apply beneficial mutation to trait"""
        mutated = deepcopy(self)
        if random.random() < self.mutation_rate:
            # Small random adjustment
            adjustment = random.uniform(-0.1, 0.1)
            mutated.strength = max(0.0, min(1.0, self.strength + adjustment))
        return mutated


@dataclass
class AgentCapabilityGene:
    """Represents a capability as genetic information"""
    capability_name: str
    base_proficiency: float  # 0.0 to 1.0
    learning_rate: float  # How quickly this capability improves
    specialization_bonus: float  # Bonus when used in specialization context
    inheritance_weight: float = 1.0  # How strongly this gene is passed on
    
    def express(self, environmental_factors: Dict[str, float] = None) -> float:
        """Express the gene based on environmental factors"""
        expressed_proficiency = self.base_proficiency
        
        if environmental_factors:
            # Apply environmental bonuses/penalties
            for factor, multiplier in environmental_factors.items():
                if factor == 'specialization_match':
                    expressed_proficiency += self.specialization_bonus * multiplier
                elif factor == 'training_bonus':
                    expressed_proficiency += self.learning_rate * multiplier
        
        return max(0.0, min(1.0, expressed_proficiency))


@dataclass
class KnowledgeGene:
    """Represents knowledge as genetic information"""
    domain: str
    base_expertise: float  # 0.0 to 1.0
    concepts: List[str] = field(default_factory=list)
    transferability: float = 0.8  # How well this knowledge transfers to offspring
    decay_rate: float = 0.01  # How quickly knowledge degrades without use
    
    def transfer_to_offspring(self, inheritance_factor: float = 1.0) -> 'KnowledgeGene':
        """Create offspring version of this knowledge gene"""
        offspring_gene = deepcopy(self)
        
        # Apply inheritance factor and transferability
        offspring_gene.base_expertise *= (self.transferability * inheritance_factor)
        
        # Some concepts might not transfer
        concepts_transferred = []
        for concept in self.concepts:
            if random.random() < self.transferability:
                concepts_transferred.append(concept)
        
        offspring_gene.concepts = concepts_transferred
        return offspring_gene


@dataclass
class PerformanceGene:
    """Genetic factors affecting performance characteristics"""
    response_speed_modifier: float = 1.0  # Multiplier for response speed
    accuracy_modifier: float = 1.0  # Multiplier for accuracy
    creativity_modifier: float = 1.0  # Multiplier for creative problem solving
    persistence_modifier: float = 1.0  # Multiplier for retry behavior
    
    def apply_to_metrics(self, base_metrics: Dict[str, float]) -> Dict[str, float]:
        """Apply genetic modifiers to base performance metrics"""
        modified_metrics = deepcopy(base_metrics)
        
        if 'response_time' in modified_metrics:
            modified_metrics['response_time'] /= self.response_speed_modifier
        
        if 'accuracy' in modified_metrics:
            modified_metrics['accuracy'] *= self.accuracy_modifier
        
        if 'creativity_score' in modified_metrics:
            modified_metrics['creativity_score'] *= self.creativity_modifier
        
        return modified_metrics


class AgentDNA:
    """Defines the core characteristics that can be inherited/modified"""
    
    def __init__(self):
        # Core genetic components
        self.capabilities: Dict[str, AgentCapabilityGene] = {}
        self.specializations: List[str] = []
        self.behavioral_traits: Dict[str, BehavioralTrait] = {}
        self.knowledge_domains: Dict[str, KnowledgeGene] = {}
        self.performance_genes: PerformanceGene = PerformanceGene()
        self.interaction_styles: Dict[str, float] = {}
        
        # Genetic metadata
        self.generation: int = 1
        self.lineage: List[str] = []  # Parent agent IDs
        self.mutation_rate: float = 0.05
        self.dna_id: str = self._generate_dna_id()
        self.created_at: datetime = datetime.now()
        
        # Initialize with default traits
        self._initialize_default_traits()
    
    def _generate_dna_id(self) -> str:
        """Generate unique DNA identifier"""
        unique_string = f"{random.random()}_{datetime.now().isoformat()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def _initialize_default_traits(self):
        """Initialize with basic default behavioral traits"""
        default_traits = [
            BehavioralTrait("adaptability", 0.5, 0.7, "cognitive"),
            BehavioralTrait("persistence", 0.6, 0.8, "performance"),
            BehavioralTrait("collaboration", 0.5, 0.6, "social"),
            BehavioralTrait("precision", 0.5, 0.7, "performance"),
            BehavioralTrait("creativity", 0.4, 0.5, "cognitive")
        ]
        
        for trait in default_traits:
            self.behavioral_traits[trait.name] = trait
        
        # Default interaction styles
        self.interaction_styles = {
            "cooperative": 0.6,
            "competitive": 0.3,
            "supportive": 0.7,
            "independent": 0.5
        }
    
    def add_capability_gene(self, capability_gene: AgentCapabilityGene):
        """Add a capability gene to the DNA"""
        self.capabilities[capability_gene.capability_name] = capability_gene
    
    def add_knowledge_gene(self, knowledge_gene: KnowledgeGene):
        """Add a knowledge gene to the DNA"""
        self.knowledge_domains[knowledge_gene.domain] = knowledge_gene
    
    def add_behavioral_trait(self, trait: BehavioralTrait):
        """Add a behavioral trait to the DNA"""
        self.behavioral_traits[trait.name] = trait
    
    def crossover(self, other_dna: 'AgentDNA') -> 'AgentDNA':
        """Combine DNA from two parent agents"""
        child_dna = AgentDNA()
        child_dna.generation = max(self.generation, other_dna.generation) + 1
        child_dna.lineage = [self.dna_id, other_dna.dna_id]
        
        # Merge capabilities (take best from each parent)
        child_dna.capabilities = self._merge_capabilities(
            self.capabilities, other_dna.capabilities
        )
        
        # Combine specializations
        child_dna.specializations = self._combine_specializations(
            self.specializations, other_dna.specializations
        )
        
        # Merge behavioral traits
        child_dna.behavioral_traits = self._merge_behavioral_traits(
            self.behavioral_traits, other_dna.behavioral_traits
        )
        
        # Merge knowledge domains
        child_dna.knowledge_domains = self._merge_knowledge_domains(
            self.knowledge_domains, other_dna.knowledge_domains
        )
        
        # Combine performance genes
        child_dna.performance_genes = self._combine_performance_genes(
            self.performance_genes, other_dna.performance_genes
        )
        
        # Merge interaction styles
        child_dna.interaction_styles = self._merge_interaction_styles(
            self.interaction_styles, other_dna.interaction_styles
        )
        
        # Apply beneficial mutations
        child_dna = self._apply_beneficial_mutations(child_dna)
        
        return child_dna
    
    def _merge_capabilities(self, caps1: Dict[str, AgentCapabilityGene], 
                          caps2: Dict[str, AgentCapabilityGene]) -> Dict[str, AgentCapabilityGene]:
        """Merge capability genes from two parents"""
        merged = {}
        
        # Get all unique capability names
        all_caps = set(caps1.keys()) | set(caps2.keys())
        
        for cap_name in all_caps:
            if cap_name in caps1 and cap_name in caps2:
                # Both parents have this capability - take the better one
                gene1, gene2 = caps1[cap_name], caps2[cap_name]
                if gene1.base_proficiency >= gene2.base_proficiency:
                    merged[cap_name] = deepcopy(gene1)
                    # But inherit some traits from the other parent
                    merged[cap_name].learning_rate = (gene1.learning_rate + gene2.learning_rate) / 2
                else:
                    merged[cap_name] = deepcopy(gene2)
                    merged[cap_name].learning_rate = (gene1.learning_rate + gene2.learning_rate) / 2
            elif cap_name in caps1:
                # Only parent 1 has this capability
                merged[cap_name] = deepcopy(caps1[cap_name])
                # Reduce inheritance weight since only one parent had it
                merged[cap_name].inheritance_weight *= 0.8
            else:
                # Only parent 2 has this capability
                merged[cap_name] = deepcopy(caps2[cap_name])
                merged[cap_name].inheritance_weight *= 0.8
        
        return merged
    
    def _combine_specializations(self, specs1: List[str], specs2: List[str]) -> List[str]:
        """Combine specializations from two parents"""
        # Take unique specializations from both parents
        combined = list(set(specs1 + specs2))
        
        # Limit to maximum 3 specializations to maintain focus
        if len(combined) > 3:
            # Prioritize specializations that appear in both parents
            shared = [spec for spec in specs1 if spec in specs2]
            unique = [spec for spec in combined if spec not in shared]
            
            # Take all shared, then fill up to 3 with unique ones
            result = shared[:]
            remaining_slots = 3 - len(shared)
            if remaining_slots > 0:
                result.extend(random.sample(unique, min(remaining_slots, len(unique))))
            return result
        
        return combined
    
    def _merge_behavioral_traits(self, traits1: Dict[str, BehavioralTrait], 
                               traits2: Dict[str, BehavioralTrait]) -> Dict[str, BehavioralTrait]:
        """Merge behavioral traits from two parents"""
        merged = {}
        
        all_traits = set(traits1.keys()) | set(traits2.keys())
        
        for trait_name in all_traits:
            if trait_name in traits1 and trait_name in traits2:
                # Average the trait strengths
                trait1, trait2 = traits1[trait_name], traits2[trait_name]
                merged_trait = deepcopy(trait1)
                merged_trait.strength = (trait1.strength + trait2.strength) / 2
                merged_trait.dominance = max(trait1.dominance, trait2.dominance)
                merged[trait_name] = merged_trait
            elif trait_name in traits1:
                merged[trait_name] = deepcopy(traits1[trait_name])
                # Reduce dominance since only one parent had it
                merged[trait_name].dominance *= 0.7
            else:
                merged[trait_name] = deepcopy(traits2[trait_name])
                merged[trait_name].dominance *= 0.7
        
        return merged
    
    def _merge_knowledge_domains(self, domains1: Dict[str, KnowledgeGene], 
                               domains2: Dict[str, KnowledgeGene]) -> Dict[str, KnowledgeGene]:
        """Merge knowledge domains from two parents"""
        merged = {}
        
        all_domains = set(domains1.keys()) | set(domains2.keys())
        
        for domain_name in all_domains:
            if domain_name in domains1 and domain_name in domains2:
                # Combine knowledge from both parents
                gene1, gene2 = domains1[domain_name], domains2[domain_name]
                merged_gene = deepcopy(gene1)
                
                # Average expertise levels
                merged_gene.base_expertise = (gene1.base_expertise + gene2.base_expertise) / 2
                
                # Combine concepts (union of both sets)
                all_concepts = set(gene1.concepts) | set(gene2.concepts)
                merged_gene.concepts = list(all_concepts)
                
                # Average transferability
                merged_gene.transferability = (gene1.transferability + gene2.transferability) / 2
                
                merged[domain_name] = merged_gene
            elif domain_name in domains1:
                merged[domain_name] = domains1[domain_name].transfer_to_offspring(0.8)
            else:
                merged[domain_name] = domains2[domain_name].transfer_to_offspring(0.8)
        
        return merged
    
    def _combine_performance_genes(self, genes1: PerformanceGene, 
                                 genes2: PerformanceGene) -> PerformanceGene:
        """Combine performance genes from two parents"""
        return PerformanceGene(
            response_speed_modifier=(genes1.response_speed_modifier + genes2.response_speed_modifier) / 2,
            accuracy_modifier=max(genes1.accuracy_modifier, genes2.accuracy_modifier),  # Take best accuracy
            creativity_modifier=(genes1.creativity_modifier + genes2.creativity_modifier) / 2,
            persistence_modifier=max(genes1.persistence_modifier, genes2.persistence_modifier)  # Take best persistence
        )
    
    def _merge_interaction_styles(self, styles1: Dict[str, float], 
                                styles2: Dict[str, float]) -> Dict[str, float]:
        """Merge interaction styles from two parents"""
        merged = {}
        all_styles = set(styles1.keys()) | set(styles2.keys())
        
        for style in all_styles:
            val1 = styles1.get(style, 0.5)  # Default neutral value
            val2 = styles2.get(style, 0.5)
            merged[style] = (val1 + val2) / 2
        
        return merged
    
    def _apply_beneficial_mutations(self, dna: 'AgentDNA') -> 'AgentDNA':
        """Apply controlled mutations that can be beneficial"""
        mutated_dna = deepcopy(dna)
        
        # Mutate behavioral traits
        for trait_name, trait in mutated_dna.behavioral_traits.items():
            if random.random() < mutated_dna.mutation_rate:
                mutated_dna.behavioral_traits[trait_name] = trait.mutate()
        
        # Mutate capability genes
        for cap_name, gene in mutated_dna.capabilities.items():
            if random.random() < mutated_dna.mutation_rate:
                # Small beneficial mutation
                if random.random() < 0.7:  # 70% chance of positive mutation
                    gene.base_proficiency = min(1.0, gene.base_proficiency + random.uniform(0.01, 0.05))
                gene.learning_rate = max(0.01, min(1.0, gene.learning_rate + random.uniform(-0.02, 0.05)))
        
        # Mutate performance genes
        if random.random() < mutated_dna.mutation_rate:
            perf = mutated_dna.performance_genes
            mutations = [
                lambda: setattr(perf, 'response_speed_modifier', 
                              max(0.5, min(2.0, perf.response_speed_modifier + random.uniform(-0.1, 0.1)))),
                lambda: setattr(perf, 'accuracy_modifier', 
                              max(0.5, min(2.0, perf.accuracy_modifier + random.uniform(-0.05, 0.1)))),
                lambda: setattr(perf, 'creativity_modifier', 
                              max(0.5, min(2.0, perf.creativity_modifier + random.uniform(-0.1, 0.15)))),
            ]
            random.choice(mutations)()
        
        return mutated_dna
    
    def mutate(self, mutation_rate: float = None) -> 'AgentDNA':
        """Introduce controlled variations for evolution"""
        if mutation_rate is not None:
            self.mutation_rate = mutation_rate
        
        return self._apply_beneficial_mutations(self)
    
    def get_fitness_score(self, performance_metrics: Dict[str, float] = None) -> float:
        """Calculate fitness score for evolutionary selection"""
        fitness = 0.0
        
        # Base fitness from capabilities
        if self.capabilities:
            avg_capability = sum(gene.base_proficiency for gene in self.capabilities.values()) / len(self.capabilities)
            fitness += avg_capability * 0.3
        
        # Fitness from behavioral traits
        if self.behavioral_traits:
            avg_trait_strength = sum(trait.strength for trait in self.behavioral_traits.values()) / len(self.behavioral_traits)
            fitness += avg_trait_strength * 0.2
        
        # Fitness from knowledge domains
        if self.knowledge_domains:
            avg_knowledge = sum(gene.base_expertise for gene in self.knowledge_domains.values()) / len(self.knowledge_domains)
            fitness += avg_knowledge * 0.2
        
        # Fitness from performance genes
        perf_fitness = (
            self.performance_genes.response_speed_modifier +
            self.performance_genes.accuracy_modifier +
            self.performance_genes.creativity_modifier +
            self.performance_genes.persistence_modifier
        ) / 4
        fitness += (perf_fitness - 1.0) * 0.1  # Normalized around 1.0
        
        # Bonus for specialization (focused agents are often better)
        if len(self.specializations) > 0:
            fitness += 0.1
        
        # Bonus for being later generation (evolved agents)
        fitness += min(self.generation * 0.01, 0.1)
        
        # Apply performance metrics if provided
        if performance_metrics:
            success_rate = performance_metrics.get('success_rate', 0.5)
            quality_score = performance_metrics.get('quality_score', 0.5)
            fitness += (success_rate + quality_score) * 0.1
        
        return max(0.0, min(1.0, fitness))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert DNA to dictionary for serialization"""
        return {
            'dna_id': self.dna_id,
            'generation': self.generation,
            'lineage': self.lineage,
            'mutation_rate': self.mutation_rate,
            'created_at': self.created_at.isoformat(),
            'specializations': self.specializations,
            'capabilities': {k: asdict(v) for k, v in self.capabilities.items()},
            'behavioral_traits': {k: asdict(v) for k, v in self.behavioral_traits.items()},
            'knowledge_domains': {k: asdict(v) for k, v in self.knowledge_domains.items()},
            'performance_genes': asdict(self.performance_genes),
            'interaction_styles': self.interaction_styles
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentDNA':
        """Create DNA from dictionary"""
        dna = cls()
        dna.dna_id = data['dna_id']
        dna.generation = data['generation']
        dna.lineage = data['lineage']
        dna.mutation_rate = data['mutation_rate']
        dna.created_at = datetime.fromisoformat(data['created_at'])
        dna.specializations = data['specializations']
        dna.interaction_styles = data['interaction_styles']
        
        # Reconstruct capabilities
        dna.capabilities = {}
        for cap_name, cap_data in data['capabilities'].items():
            dna.capabilities[cap_name] = AgentCapabilityGene(**cap_data)
        
        # Reconstruct behavioral traits
        dna.behavioral_traits = {}
        for trait_name, trait_data in data['behavioral_traits'].items():
            dna.behavioral_traits[trait_name] = BehavioralTrait(**trait_data)
        
        # Reconstruct knowledge domains
        dna.knowledge_domains = {}
        for domain_name, domain_data in data['knowledge_domains'].items():
            dna.knowledge_domains[domain_name] = KnowledgeGene(**domain_data)
        
        # Reconstruct performance genes
        dna.performance_genes = PerformanceGene(**data['performance_genes'])
        
        return dna
    
    def __str__(self) -> str:
        """String representation of DNA"""
        return f"AgentDNA(id={self.dna_id}, gen={self.generation}, specs={self.specializations})"