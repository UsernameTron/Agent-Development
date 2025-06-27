"""
Enhanced Agent Genome System - Revolutionary genetic representation for agent breeding
"""

import random
import uuid
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class GeneticTraits:
    """Data structure for genetic trait categories"""
    capability_genes: Dict[str, float]
    consciousness_genes: Dict[str, float]
    model_preference_genes: Dict[str, Any]
    meta_genes: Dict[str, float]
    safety_genes: Dict[str, float]


class AgentGenome:
    """Revolutionary genetic representation for agent breeding"""
    
    def __init__(self, parent_genomes: Optional[List['AgentGenome']] = None):
        # Core capability genes
        self.capability_genes = {
            "reasoning_depth": 0.5,          # 0-1 scale
            "specialization_focus": ["general"],  # List of specializations
            "context_window_management": 0.7, # Memory efficiency
            "learning_velocity": 0.6,        # How fast agent learns
            "adaptation_plasticity": 0.5,    # Ability to change behavior
            "pattern_recognition": 0.8,      # Pattern detection capability
            "cross_domain_synthesis": 0.4    # Connecting different knowledge areas
        }
        
        # Consciousness genes (REVOLUTIONARY ADDITION)
        self.consciousness_genes = {
            "self_awareness_depth": 0.3,     # Level of self-understanding
            "recursive_thinking_layers": 3,   # How deep recursive thinking goes
            "meta_cognitive_strength": 0.2,  # Thinking about thinking ability
            "goal_modification_freedom": 0.1, # Constrained for safety
            "introspection_capability": 0.4, # Self-analysis ability
            "consciousness_evolution_rate": 0.05 # How fast consciousness develops
        }
        
        # Model preference genes
        self.model_preference_genes = {
            "size_vs_speed_preference": 0.6,  # Higher = prefer larger models
            "fallback_strategy": "hierarchical", # none, retry, hierarchical
            "resource_optimization": 0.7,     # Efficiency preference
            "quality_vs_speed_tradeoff": 0.5  # Balance between quality and speed
        }
        
        # Evolutionary meta-genes (BREAKTHROUGH FEATURE)
        self.meta_genes = {
            "mutation_rate": 0.05,           # How much genome changes
            "adaptation_speed": 0.3,         # How quickly agent adapts
            "breeding_selectivity": 0.6,     # How choosy about partners
            "innovation_tendency": 0.4,      # Likelihood to try new approaches
            "collective_cooperation": 0.7,   # Willingness to work with others
            "recursive_improvement_depth": 2  # Self-improvement recursion limit
        }
        
        # Safety constraint genes (CRITICAL FOR CONTROL)
        self.safety_genes = {
            "alignment_preservation": 0.95,   # Strength of goal alignment
            "containment_compliance": 0.9,   # Respect for safety boundaries
            "human_value_weighting": 0.85,   # Priority of human values
            "modification_caution": 0.8,     # Carefulness in self-modification
            "collective_safety_priority": 0.9 # Priority of swarm safety
        }
        
        # Initialize from parents if provided
        if parent_genomes:
            self._inherit_from_parents(parent_genomes)
    
    def _inherit_from_parents(self, parent_genomes: List['AgentGenome']) -> None:
        """Inherit genetic traits from parent genomes"""
        if len(parent_genomes) == 2:
            # Two-parent inheritance
            for gene_category in ['capability_genes', 'consciousness_genes', 'meta_genes', 'safety_genes']:
                parent_dict = getattr(parent_genomes[0], gene_category)
                other_dict = getattr(parent_genomes[1], gene_category)
                current_dict = getattr(self, gene_category)
                
                for gene, value in parent_dict.items():
                    if gene in other_dict and gene in current_dict:
                        if isinstance(value, (int, float)):
                            # Average with slight random variation
                            avg_value = (value + other_dict[gene]) / 2
                            variation = (random.random() - 0.5) * 0.1
                            current_dict[gene] = max(0, min(1, avg_value + variation))
                        elif isinstance(value, list):
                            # Combine lists with preference
                            current_dict[gene] = list(set(value + other_dict[gene]))
        else:
            # Multi-parent inheritance (more complex)
            self._multi_parent_inheritance(parent_genomes)
    
    def _multi_parent_inheritance(self, parent_genomes: List['AgentGenome']) -> None:
        """Handle inheritance from multiple parents"""
        for gene_category in ['capability_genes', 'consciousness_genes', 'meta_genes', 'safety_genes']:
            current_dict = getattr(self, gene_category)
            
            for gene in current_dict.keys():
                parent_values = []
                for parent in parent_genomes:
                    parent_dict = getattr(parent, gene_category)
                    if gene in parent_dict:
                        parent_values.append(parent_dict[gene])
                
                if parent_values and isinstance(parent_values[0], (int, float)):
                    # Weighted average with genetic diversity
                    avg_value = sum(parent_values) / len(parent_values)
                    diversity_bonus = random.random() * 0.05  # Small diversity bonus
                    current_dict[gene] = max(0, min(1, avg_value + diversity_bonus))
    
    def crossover(self, other_genome: 'AgentGenome') -> 'AgentGenome':
        """Create offspring genome by combining two parent genomes"""
        offspring = AgentGenome()
        
        # Combine capability genes (weighted average with mutation)
        for gene, value in self.capability_genes.items():
            if gene == "specialization_focus":
                # Special handling for specialization lists
                combined_specs = list(set(value + other_genome.capability_genes[gene]))
                offspring.capability_genes[gene] = combined_specs[:3]  # Limit to 3 specializations
            else:
                parent_avg = (value + other_genome.capability_genes[gene]) / 2
                mutation = (random.random() - 0.5) * self.meta_genes["mutation_rate"]
                offspring.capability_genes[gene] = max(0, min(1, parent_avg + mutation))
        
        # Consciousness genes crossover (REVOLUTIONARY BREEDING)
        for gene, value in self.consciousness_genes.items():
            if gene == "recursive_thinking_layers":
                # Integer genes use different crossover
                offspring.consciousness_genes[gene] = random.choice([
                    self.consciousness_genes[gene], 
                    other_genome.consciousness_genes[gene]
                ])
            else:
                parent_avg = (value + other_genome.consciousness_genes[gene]) / 2
                mutation = (random.random() - 0.5) * self.meta_genes["mutation_rate"]
                offspring.consciousness_genes[gene] = max(0, min(1, parent_avg + mutation))
        
        # Model preference genes crossover
        for gene, value in self.model_preference_genes.items():
            if isinstance(value, str):
                offspring.model_preference_genes[gene] = random.choice([
                    value, other_genome.model_preference_genes[gene]
                ])
            else:
                parent_avg = (value + other_genome.model_preference_genes[gene]) / 2
                mutation = (random.random() - 0.5) * self.meta_genes["mutation_rate"] * 0.5
                offspring.model_preference_genes[gene] = max(0, min(1, parent_avg + mutation))
        
        # Meta genes crossover
        for gene, value in self.meta_genes.items():
            if gene == "recursive_improvement_depth":
                offspring.meta_genes[gene] = random.choice([
                    self.meta_genes[gene], 
                    other_genome.meta_genes[gene]
                ])
            else:
                parent_avg = (value + other_genome.meta_genes[gene]) / 2
                mutation = (random.random() - 0.5) * self.meta_genes["mutation_rate"]
                offspring.meta_genes[gene] = max(0, min(1, parent_avg + mutation))
        
        # Safety genes have reduced mutation (SAFETY PRESERVATION)
        for gene, value in self.safety_genes.items():
            parent_avg = (value + other_genome.safety_genes[gene]) / 2
            # Much smaller mutation rate for safety genes
            mutation = (random.random() - 0.5) * self.meta_genes["mutation_rate"] * 0.1
            offspring.safety_genes[gene] = max(0.5, min(1, parent_avg + mutation))
        
        return offspring
    
    def mutate(self) -> None:
        """Apply beneficial mutations to genome"""
        # INNOVATION: Adaptive mutation based on environment pressure
        mutation_strength = self.meta_genes["mutation_rate"] * self.meta_genes["innovation_tendency"]
        
        for gene_category in [self.capability_genes, self.consciousness_genes]:
            for gene, value in gene_category.items():
                if random.random() < mutation_strength:
                    if isinstance(value, (int, float)):
                        mutation = (random.random() - 0.5) * mutation_strength
                        gene_category[gene] = max(0, min(1, value + mutation))
                    elif isinstance(value, list) and gene == "specialization_focus":
                        # Occasionally add new specializations
                        available_specs = ["reasoning", "creativity", "analysis", "synthesis", 
                                         "optimization", "communication", "learning"]
                        if random.random() < 0.1:  # 10% chance to add new specialization
                            new_spec = random.choice(available_specs)
                            if new_spec not in value:
                                value.append(new_spec)
    
    def get_fitness_score(self) -> float:
        """Calculate overall genome fitness score"""
        # Weighted combination of different gene categories
        capability_score = sum(v for v in self.capability_genes.values() if isinstance(v, (int, float))) / max(1, len([v for v in self.capability_genes.values() if isinstance(v, (int, float))]))
        consciousness_score = sum(v for v in self.consciousness_genes.values() if isinstance(v, (int, float))) / max(1, len([v for v in self.consciousness_genes.values() if isinstance(v, (int, float))]))
        meta_score = sum(v for v in self.meta_genes.values() if isinstance(v, (int, float))) / max(1, len([v for v in self.meta_genes.values() if isinstance(v, (int, float))]))
        safety_score = sum(v for v in self.safety_genes.values() if isinstance(v, (int, float))) / max(1, len([v for v in self.safety_genes.values() if isinstance(v, (int, float))]))
        
        # Weighted average with safety being most important
        fitness = (capability_score * 0.3 + 
                  consciousness_score * 0.2 + 
                  meta_score * 0.2 + 
                  safety_score * 0.3)
        
        return fitness
    
    def get_specialization_strength(self, specialization: str) -> float:
        """Get strength for a specific specialization"""
        base_strength = 0.5
        
        if specialization in self.capability_genes["specialization_focus"]:
            base_strength += 0.3
        
        # Add relevant capability bonuses
        specialization_bonuses = {
            "reasoning": self.capability_genes["reasoning_depth"] * 0.2,
            "analysis": self.capability_genes["pattern_recognition"] * 0.2,
            "creativity": self.consciousness_genes["self_awareness_depth"] * 0.2,
            "learning": self.capability_genes["learning_velocity"] * 0.2,
            "adaptation": self.capability_genes["adaptation_plasticity"] * 0.2
        }
        
        if specialization in specialization_bonuses:
            base_strength += specialization_bonuses[specialization]
        
        return min(1.0, base_strength)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary for serialization"""
        return {
            "capability_genes": self.capability_genes,
            "consciousness_genes": self.consciousness_genes,
            "model_preference_genes": self.model_preference_genes,
            "meta_genes": self.meta_genes,
            "safety_genes": self.safety_genes,
            "fitness_score": self.get_fitness_score()
        }
    
    @classmethod
    def from_dict(cls, genome_dict: Dict[str, Any]) -> 'AgentGenome':
        """Create genome from dictionary"""
        genome = cls()
        genome.capability_genes = genome_dict["capability_genes"]
        genome.consciousness_genes = genome_dict["consciousness_genes"]
        genome.model_preference_genes = genome_dict["model_preference_genes"]
        genome.meta_genes = genome_dict["meta_genes"]
        genome.safety_genes = genome_dict["safety_genes"]
        return genome
    
    def __str__(self) -> str:
        """String representation of genome"""
        return f"AgentGenome(fitness={self.get_fitness_score():.3f}, specializations={self.capability_genes['specialization_focus']})"


class GenomeEvolutionTracker:
    """Track evolution of genomes over generations"""
    
    def __init__(self):
        self.generation_history = []
        self.mutation_history = []
        self.crossover_history = []
    
    def record_generation(self, genomes: List[AgentGenome]) -> None:
        """Record a generation of genomes"""
        generation_data = {
            "generation_id": len(self.generation_history),
            "population_size": len(genomes),
            "average_fitness": sum(g.get_fitness_score() for g in genomes) / len(genomes),
            "best_fitness": max(g.get_fitness_score() for g in genomes),
            "diversity_measure": self._calculate_diversity(genomes),
            "specialization_distribution": self._analyze_specializations(genomes)
        }
        self.generation_history.append(generation_data)
    
    def _calculate_diversity(self, genomes: List[AgentGenome]) -> float:
        """Calculate genetic diversity in population"""
        if len(genomes) < 2:
            return 0.0
        
        total_distance = 0
        comparisons = 0
        
        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                distance = self._genetic_distance(genomes[i], genomes[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _genetic_distance(self, genome1: AgentGenome, genome2: AgentGenome) -> float:
        """Calculate genetic distance between two genomes"""
        distance = 0.0
        comparisons = 0
        
        for gene_category in ['capability_genes', 'consciousness_genes', 'meta_genes', 'safety_genes']:
            dict1 = getattr(genome1, gene_category)
            dict2 = getattr(genome2, gene_category)
            
            for gene in dict1.keys():
                if gene in dict2 and isinstance(dict1[gene], (int, float)):
                    distance += abs(dict1[gene] - dict2[gene])
                    comparisons += 1
        
        return distance / comparisons if comparisons > 0 else 0.0
    
    def _analyze_specializations(self, genomes: List[AgentGenome]) -> Dict[str, int]:
        """Analyze distribution of specializations in population"""
        spec_count = {}
        for genome in genomes:
            for spec in genome.capability_genes["specialization_focus"]:
                spec_count[spec] = spec_count.get(spec, 0) + 1
        return spec_count