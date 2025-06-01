import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import networkx as nx

class GenealogyGraph:
    def __init__(self, individuals: Dict[str, 'Individual'], families: Dict[str, 'Family']):
        self.individuals = individuals
        self.families = families
        self.graph = nx.DiGraph()
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.edge_types = {
            'parent_child': 0,
            'spouse': 1,
            'sibling': 2
        }
        
    def build_graph(self):
        """Build NetworkX graph from GEDCOM data."""
        # Add nodes
        for idx, (ind_id, individual) in enumerate(self.individuals.items()):
            self.graph.add_node(ind_id, **self._get_node_features(individual))
            self.node_to_idx[ind_id] = idx
            self.idx_to_node[idx] = ind_id
        
        # Add edges
        for fam_id, family in self.families.items():
            # Parent-child relationships
            parents = []
            if family.husband:
                parents.append(family.husband)
            if family.wife:
                parents.append(family.wife)
            
            for parent in parents:
                for child in family.children:
                    if parent in self.individuals and child in self.individuals:
                        self.graph.add_edge(parent, child, 
                                          type='parent_child',
                                          weight=1.0)
            
            # Spouse relationships
            if family.husband and family.wife:
                if family.husband in self.individuals and family.wife in self.individuals:
                    self.graph.add_edge(family.husband, family.wife, 
                                      type='spouse',
                                      weight=1.0)
                    self.graph.add_edge(family.wife, family.husband, 
                                      type='spouse',
                                      weight=1.0)
            
            # Sibling relationships
            for i, child1 in enumerate(family.children):
                for child2 in family.children[i+1:]:
                    if child1 in self.individuals and child2 in self.individuals:
                        self.graph.add_edge(child1, child2, 
                                          type='sibling',
                                          weight=1.0)
                        self.graph.add_edge(child2, child1, 
                                          type='sibling',
                                          weight=1.0)
    
    def _get_node_features(self, individual: 'Individual') -> Dict:
        """Extract features from an individual."""
        features = {
            'name': individual.name or '',
            'gender': individual.gender or 'U',
            'birth_year': individual.birth_date.year if individual.birth_date else -1,
            'death_year': individual.death_date.year if individual.death_date else -1,
            'has_parents': bool(individual.family_child),
            'num_marriages': len(individual.family_spouse),
            'is_missing_parents': individual.family_child is None
        }
        return features
    
    def to_pytorch_geometric(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert graph to PyTorch Geometric format."""
        # Node features
        num_nodes = len(self.individuals)
        node_features = []
        
        for idx in range(num_nodes):
            ind_id = self.idx_to_node[idx]
            ind = self.individuals[ind_id]
            
            # Create feature vector
            features = [
                1.0 if ind.gender == 'M' else 0.0,
                1.0 if ind.gender == 'F' else 0.0,
                1.0 if ind.gender == 'U' else 0.0,
                float(ind.birth_date.year) / 2000.0 if ind.birth_date else 0.0,
                float(ind.death_date.year) / 2000.0 if ind.death_date else 0.0,
                1.0 if ind.family_child else 0.0,
                float(len(ind.family_spouse)) / 10.0,
            ]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Edge indices and types
        edge_index = []
        edge_type = []
        
        for u, v, data in self.graph.edges(data=True):
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            edge_index.append([u_idx, v_idx])
            edge_type.append(self.edge_types[data['type']])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        
        return x, edge_index, edge_type
    
    def get_missing_parent_candidates(self) -> List[int]:
        """Get indices of individuals missing parents."""
        candidates = []
        for idx in range(len(self.individuals)):
            ind_id = self.idx_to_node[idx]
            ind = self.individuals[ind_id]
            if not ind.family_child:
                candidates.append(idx)
        return candidates