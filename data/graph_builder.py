"""
Graph Builder for ʻŌhanaGPT
Converts genealogical data into graph structures for GNN processing
"""

import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import torch_geometric
from torch_geometric.data import Data, HeteroData
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

from .gedcom_parser import Individual, Family


@dataclass
class GraphConfig:
    """Configuration for graph building"""
    use_heterogeneous: bool = True
    max_generations: int = 5
    include_spouse_edges: bool = True
    include_sibling_edges: bool = True
    include_temporal_features: bool = True
    node_feature_dim: int = 64
    edge_feature_dim: int = 16


class GraphBuilder:
    """Builds graph representations from genealogical data"""
    
    def __init__(self, config: GraphConfig = None):
        self.config = config or GraphConfig()
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: Dict[int, str] = {}
        self.feature_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        
    def build_homogeneous_graph(self, 
                               individuals: Dict[str, Individual], 
                               families: Dict[str, Family]) -> Data:
        """Build a homogeneous graph representation"""
        # Create node mappings
        self._create_node_mappings(individuals)
        
        # Extract features
        node_features = self._extract_node_features(individuals)
        
        # Build edges
        edge_index, edge_attr = self._build_edges(individuals, families)
        
        # Create masks for missing parents
        train_mask, val_mask, test_mask = self._create_masks(individuals, families)
        
        # Create labels (parent indices)
        labels = self._create_parent_labels(individuals, families)
        
        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=len(individuals)
        )
        
        return data
    
    def build_heterogeneous_graph(self,
                                 individuals: Dict[str, Individual],
                                 families: Dict[str, Family]) -> HeteroData:
        """Build a heterogeneous graph with different node and edge types"""
        data = HeteroData()
        
        # Create individual nodes
        self._create_node_mappings(individuals)
        individual_features = self._extract_node_features(individuals)
        data['individual'].x = individual_features
        data['individual'].num_nodes = len(individuals)
        
        # Create family nodes
        family_to_idx = {fam_id: idx for idx, fam_id in enumerate(families.keys())}
        family_features = self._extract_family_features(families, individuals)
        data['family'].x = family_features
        data['family'].num_nodes = len(families)
        
        # Create edges
        # Parent-child relationships
        parent_child_edges = self._build_parent_child_edges(individuals, families, family_to_idx)
        data['individual', 'parent_of', 'individual'].edge_index = parent_child_edges
        
        # Spouse relationships
        if self.config.include_spouse_edges:
            spouse_edges = self._build_spouse_edges(families)
            data['individual', 'spouse_of', 'individual'].edge_index = spouse_edges
        
        # Individual-family relationships
        ind_fam_edges = self._build_individual_family_edges(individuals, families, family_to_idx)
        data['individual', 'belongs_to', 'family'].edge_index = ind_fam_edges[0]
        data['family', 'contains', 'individual'].edge_index = ind_fam_edges[1]
        
        # Add temporal edges if configured
        if self.config.include_temporal_features:
            temporal_edges = self._build_temporal_edges(individuals)
            data['individual', 'contemporary_of', 'individual'].edge_index = temporal_edges
        
        return data
    
    def _create_node_mappings(self, individuals: Dict[str, Individual]):
        """Create bidirectional mappings between node IDs and indices"""
        self.node_to_idx = {ind_id: idx for idx, ind_id in enumerate(individuals.keys())}
        self.idx_to_node = {idx: ind_id for ind_id, idx in self.node_to_idx.items()}
    
    def _extract_node_features(self, individuals: Dict[str, Individual]) -> torch.Tensor:
        """Extract feature vectors for individual nodes"""
        features = []
        
        for ind_id in sorted(individuals.keys()):
            ind = individuals[ind_id]
            feature_vec = []
            
            # Basic features
            feature_vec.append(1 if ind.sex == 'M' else 0 if ind.sex == 'F' else 0.5)
            
            # Name features
            feature_vec.append(len(ind.given_name) if ind.given_name else 0)
            feature_vec.append(len(ind.surname) if ind.surname else 0)
            feature_vec.append(1 if ind.given_name else 0)
            feature_vec.append(1 if ind.surname else 0)
            
            # Date features
            birth_year = self._extract_year(ind.birth_date)
            death_year = self._extract_year(ind.death_date)
            
            feature_vec.append(birth_year if birth_year else -1)
            feature_vec.append(death_year if death_year else -1)
            feature_vec.append(death_year - birth_year if birth_year and death_year else -1)
            
            # Location features (encoded)
            birth_place_enc = self._encode_location(ind.birth_place)
            death_place_enc = self._encode_location(ind.death_place)
            feature_vec.extend(birth_place_enc)
            feature_vec.extend(death_place_enc)
            
            # Relationship features
            feature_vec.append(len(ind.parent_family_ids))
            feature_vec.append(len(ind.spouse_family_ids))
            
            # Attribute features
            feature_vec.append(1 if ind.occupation else 0)
            feature_vec.append(1 if ind.religion else 0)
            feature_vec.append(1 if ind.education else 0)
            
            # Generation estimate
            gen_estimate = self._estimate_generation(ind, individuals)
            feature_vec.append(gen_estimate)
            
            features.append(feature_vec)
        
        # Convert to tensor and normalize
        features_array = np.array(features, dtype=np.float32)
        features_array = self.scaler.fit_transform(features_array)
        
        return torch.tensor(features_array, dtype=torch.float32)
    
    def _extract_family_features(self, 
                                families: Dict[str, Family], 
                                individuals: Dict[str, Individual]) -> torch.Tensor:
        """Extract features for family nodes"""
        features = []
        
        for fam_id in sorted(families.keys()):
            fam = families[fam_id]
            feature_vec = []
            
            # Family structure features
            feature_vec.append(1 if fam.husband_id else 0)
            feature_vec.append(1 if fam.wife_id else 0)
            feature_vec.append(len(fam.child_ids))
            
            # Marriage features
            marriage_year = self._extract_year(fam.marriage_date)
            feature_vec.append(marriage_year if marriage_year else -1)
            feature_vec.append(1 if fam.divorce_date else 0)
            
            # Parent age at marriage
            if fam.husband_id and fam.husband_id in individuals:
                husband = individuals[fam.husband_id]
                h_birth = self._extract_year(husband.birth_date)
                feature_vec.append(marriage_year - h_birth if marriage_year and h_birth else -1)
            else:
                feature_vec.append(-1)
                
            if fam.wife_id and fam.wife_id in individuals:
                wife = individuals[fam.wife_id]
                w_birth = self._extract_year(wife.birth_date)
                feature_vec.append(marriage_year - w_birth if marriage_year and w_birth else -1)
            else:
                feature_vec.append(-1)
            
            features.append(feature_vec)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _build_edges(self, 
                    individuals: Dict[str, Individual], 
                    families: Dict[str, Family]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edge index and attributes for homogeneous graph"""
        edges = []
        edge_attrs = []
        
        # Parent-child edges
        for fam in families.values():
            parents = []
            if fam.husband_id and fam.husband_id in self.node_to_idx:
                parents.append(self.node_to_idx[fam.husband_id])
            if fam.wife_id and fam.wife_id in self.node_to_idx:
                parents.append(self.node_to_idx[fam.wife_id])
                
            for child_id in fam.child_ids:
                if child_id in self.node_to_idx:
                    child_idx = self.node_to_idx[child_id]
                    for parent_idx in parents:
                        # Parent -> Child
                        edges.append([parent_idx, child_idx])
                        edge_attrs.append([1, 0, 0])  # [is_parent, is_spouse, is_sibling]
                        # Child -> Parent (bidirectional)
                        edges.append([child_idx, parent_idx])
                        edge_attrs.append([1, 0, 0])
        
        # Spouse edges
        if self.config.include_spouse_edges:
            for fam in families.values():
                if fam.husband_id in self.node_to_idx and fam.wife_id in self.node_to_idx:
                    h_idx = self.node_to_idx[fam.husband_id]
                    w_idx = self.node_to_idx[fam.wife_id]
                    edges.append([h_idx, w_idx])
                    edge_attrs.append([0, 1, 0])
                    edges.append([w_idx, h_idx])
                    edge_attrs.append([0, 1, 0])
        
        # Sibling edges
        if self.config.include_sibling_edges:
            for fam in families.values():
                child_indices = [self.node_to_idx[c] for c in fam.child_ids 
                               if c in self.node_to_idx]
                for i in range(len(child_indices)):
                    for j in range(i+1, len(child_indices)):
                        edges.append([child_indices[i], child_indices[j]])
                        edge_attrs.append([0, 0, 1])
                        edges.append([child_indices[j], child_indices[i]])
                        edge_attrs.append([0, 0, 1])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float32)
        
        return edge_index, edge_attr
    
    def _build_parent_child_edges(self,
                                 individuals: Dict[str, Individual],
                                 families: Dict[str, Family],
                                 family_to_idx: Dict[str, int]) -> torch.Tensor:
        """Build parent-child edges for heterogeneous graph"""
        edges = []
        
        for fam in families.values():
            parents = []
            if fam.husband_id and fam.husband_id in self.node_to_idx:
                parents.append(self.node_to_idx[fam.husband_id])
            if fam.wife_id and fam.wife_id in self.node_to_idx:
                parents.append(self.node_to_idx[fam.wife_id])
                
            for child_id in fam.child_ids:
                if child_id in self.node_to_idx:
                    child_idx = self.node_to_idx[child_id]
                    for parent_idx in parents:
                        edges.append([parent_idx, child_idx])
        
        if edges:
            return torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            return torch.empty((2, 0), dtype=torch.long)
    
    def _build_spouse_edges(self, families: Dict[str, Family]) -> torch.Tensor:
        """Build spouse edges"""
        edges = []
        
        for fam in families.values():
            if fam.husband_id in self.node_to_idx and fam.wife_id in self.node_to_idx:
                h_idx = self.node_to_idx[fam.husband_id]
                w_idx = self.node_to_idx[fam.wife_id]
                edges.append([h_idx, w_idx])
                edges.append([w_idx, h_idx])
        
        if edges:
            return torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            return torch.empty((2, 0), dtype=torch.long)
    
    def _build_individual_family_edges(self,
                                     individuals: Dict[str, Individual],
                                     families: Dict[str, Family],
                                     family_to_idx: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges between individuals and families"""
        ind_to_fam_edges = []
        fam_to_ind_edges = []
        
        for ind_id, ind in individuals.items():
            if ind_id not in self.node_to_idx:
                continue
                
            ind_idx = self.node_to_idx[ind_id]
            
            # Individual belongs to parent families
            for fam_id in ind.parent_family_ids:
                if fam_id in family_to_idx:
                    fam_to_ind_edges.append([fam_idx, ind_idx])
            
            # Individual belongs to spouse families
            for fam_id in ind.spouse_family_ids:
                if fam_id in family_to_idx:
                    fam_idx = family_to_idx[fam_id]
                    ind_to_fam_edges.append([ind_idx, fam_idx])
                    fam_to_ind_edges.append([fam_idx, ind_idx])
        
        ind_to_fam = torch.tensor(ind_to_fam_edges, dtype=torch.long).t().contiguous() if ind_to_fam_edges else torch.empty((2, 0), dtype=torch.long)
        fam_to_ind = torch.tensor(fam_to_ind_edges, dtype=torch.long).t().contiguous() if fam_to_ind_edges else torch.empty((2, 0), dtype=torch.long)
        
        return ind_to_fam, fam_to_ind
    
    def _build_temporal_edges(self, individuals: Dict[str, Individual]) -> torch.Tensor:
        """Build edges between individuals who lived in the same time period"""
        edges = []
        ind_list = list(individuals.values())
        
        for i in range(len(ind_list)):
            for j in range(i+1, len(ind_list)):
                if self._are_contemporary(ind_list[i], ind_list[j]):
                    if ind_list[i].id in self.node_to_idx and ind_list[j].id in self.node_to_idx:
                        idx_i = self.node_to_idx[ind_list[i].id]
                        idx_j = self.node_to_idx[ind_list[j].id]
                        edges.append([idx_i, idx_j])
                        edges.append([idx_j, idx_i])
        
        if edges:
            return torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            return torch.empty((2, 0), dtype=torch.long)
    
    def _are_contemporary(self, ind1: Individual, ind2: Individual, threshold: int = 50) -> bool:
        """Check if two individuals lived in overlapping time periods"""
        birth1 = self._extract_year(ind1.birth_date)
        death1 = self._extract_year(ind1.death_date)
        birth2 = self._extract_year(ind2.birth_date)
        death2 = self._extract_year(ind2.death_date)
        
        if not birth1 or not birth2:
            return False
        
        # Use 80 as default lifespan if death date missing
        death1 = death1 or (birth1 + 80)
        death2 = death2 or (birth2 + 80)
        
        # Check if lifespans overlap
        return not (death1 < birth2 - threshold or death2 < birth1 - threshold)
    
    def _create_masks(self, 
                     individuals: Dict[str, Individual], 
                     families: Dict[str, Family]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create train/val/test masks for missing parent prediction"""
        n_nodes = len(individuals)
        
        # Identify individuals with known vs unknown parents
        has_parents = torch.zeros(n_nodes, dtype=torch.bool)
        missing_parents = torch.zeros(n_nodes, dtype=torch.bool)
        
        for ind_id, ind in individuals.items():
            idx = self.node_to_idx[ind_id]
            if ind.parent_family_ids:
                # Check if parents are actually known
                has_known_parents = False
                for fam_id in ind.parent_family_ids:
                    if fam_id in families:
                        fam = families[fam_id]
                        if fam.husband_id or fam.wife_id:
                            has_known_parents = True
                            break
                has_parents[idx] = has_known_parents
            else:
                missing_parents[idx] = True
        
        # Split into train/val/test
        known_indices = torch.where(has_parents)[0]
        unknown_indices = torch.where(missing_parents)[0]
        
        # Use 60/20/20 split for known parents
        n_known = len(known_indices)
        perm = torch.randperm(n_known)
        
        train_known = known_indices[perm[:int(0.6 * n_known)]]
        val_known = known_indices[perm[int(0.6 * n_known):int(0.8 * n_known)]]
        test_known = known_indices[perm[int(0.8 * n_known):]]
        
        # All unknown parents go to test set (these are what we want to predict)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        train_mask[train_known] = True
        val_mask[val_known] = True
        test_mask[test_known] = True
        test_mask[unknown_indices] = True
        
        return train_mask, val_mask, test_mask
    
    def _create_parent_labels(self,
                            individuals: Dict[str, Individual],
                            families: Dict[str, Family]) -> torch.Tensor:
        """Create labels for parent prediction task"""
        n_nodes = len(individuals)
        # Labels will be (father_idx, mother_idx), using -1 for unknown
        labels = torch.full((n_nodes, 2), -1, dtype=torch.long)
        
        for ind_id, ind in individuals.items():
            idx = self.node_to_idx[ind_id]
            
            for fam_id in ind.parent_family_ids:
                if fam_id in families:
                    fam = families[fam_id]
                    if fam.husband_id and fam.husband_id in self.node_to_idx:
                        labels[idx, 0] = self.node_to_idx[fam.husband_id]
                    if fam.wife_id and fam.wife_id in self.node_to_idx:
                        labels[idx, 1] = self.node_to_idx[fam.wife_id]
        
        return labels
    
    def _extract_year(self, date_str: Optional[str]) -> Optional[int]:
        """Extract year from various date formats"""
        if not date_str:
            return None
            
        # Try to find 4-digit year
        import re
        year_match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', date_str)
        if year_match:
            return int(year_match.group(1))
            
        # Try to find 3-digit year (before 1000)
        year_match = re.search(r'\b([1-9][0-9]{2})\b', date_str)
        if year_match:
            return int(year_match.group(1))
            
        return None
    
    def _encode_location(self, location: Optional[str]) -> List[float]:
        """Encode location string into feature vector"""
        if not location:
            return [0, 0, 0, 0]
        
        # Simple encoding: country, state/region, city, has_location
        features = [0, 0, 0, 1]
        
        # Extract location components (simplified)
        parts = location.split(',')
        if len(parts) >= 3:
            features[0] = 1  # Has country
            features[1] = 1  # Has state/region
            features[2] = 1  # Has city
        elif len(parts) == 2:
            features[1] = 1  # Has state/region
            features[2] = 1  # Has city
        elif len(parts) == 1:
            features[2] = 1  # Has city
            
        return features
    
    def _estimate_generation(self, 
                           individual: Individual, 
                           all_individuals: Dict[str, Individual]) -> float:
        """Estimate generation number based on birth year and relationships"""
        birth_year = self._extract_year(individual.birth_date)
        if not birth_year:
            return 0
        
        # Assume ~25 years per generation
        # Normalize to 1900 as generation 0
        generation = (birth_year - 1900) / 25.0
        
        return generation
    
    def add_graph_statistics(self, graph_data) -> Dict:
        """Add statistics about the graph"""
        if isinstance(graph_data, Data):
            stats = {
                'num_nodes': graph_data.num_nodes,
                'num_edges': graph_data.edge_index.shape[1],
                'num_features': graph_data.x.shape[1],
                'avg_degree': graph_data.edge_index.shape[1] / graph_data.num_nodes,
                'has_isolated_nodes': self._check_isolated_nodes(graph_data)
            }
        else:  # HeteroData
            stats = {
                'num_individual_nodes': graph_data['individual'].num_nodes,
                'num_family_nodes': graph_data['family'].num_nodes,
                'num_parent_child_edges': graph_data['individual', 'parent_of', 'individual'].edge_index.shape[1] if hasattr(graph_data['individual', 'parent_of', 'individual'], 'edge_index') else 0,
                'num_spouse_edges': graph_data['individual', 'spouse_of', 'individual'].edge_index.shape[1] if hasattr(graph_data['individual', 'spouse_of', 'individual'], 'edge_index') else 0
            }
        
        return stats
    
    def _check_isolated_nodes(self, data: Data) -> bool:
        """Check if graph has isolated nodes"""
        if data.edge_index.shape[1] == 0:
            # No edges means all nodes are isolated (if any exist)
            return data.num_nodes > 0
        
        edge_index = data.edge_index
        # Get all unique node indices that appear in edges
        connected_nodes = torch.unique(torch.cat([edge_index[0], edge_index[1]]))
        
        # Check if the number of connected nodes is less than total nodes
        return len(connected_nodes) < data.num_nodes
    
    def _build_temporal_edges(self, individuals: Dict[str, Individual]) -> torch.Tensor:
        """Build edges between individuals who lived in the same time period"""
        edges = []
        ind_list = list(individuals.values())
        
        for i in range(len(ind_list)):
            for j in range(i+1, len(ind_list)):
                if self._are_contemporary(ind_list[i], ind_list[j]):
                    if ind_list[i].id in self.node_to_idx and ind_list[j].id in self.node_to_idx:
                        idx_i = self.node_to_idx[ind_list[i].id]
                        idx_j = self.node_to_idx[ind_list[j].id]
                        edges.append([idx_i, idx_j])
                        edges.append([idx_j, idx_i])
        
        if edges:
            return torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            return torch.empty((2, 0), dtype=torch.long)
    
    def _are_contemporary(self, ind1: Individual, ind2: Individual, threshold: int = 50) -> bool:
        """Check if two individuals lived in overlapping time periods"""
        birth1 = self._extract_year(ind1.birth_date)
        death1 = self._extract_year(ind1.death_date)
        birth2 = self._extract_year(ind2.birth_date)
        death2 = self._extract_year(ind2.death_date)
        
        if not birth1 or not birth2:
            return False
        
        # Use 80 as default lifespan if death date missing
        death1 = death1 or (birth1 + 80)
        death2 = death2 or (birth2 + 80)
        
        # Check if lifespans overlap
        return not (death1 < birth2 - threshold or death2 < birth1 - threshold)
    
    def _create_masks(self, 
                     individuals: Dict[str, Individual], 
                     families: Dict[str, Family]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create train/val/test masks for missing parent prediction"""
        n_nodes = len(individuals)
        
        # Identify individuals with known vs unknown parents
        has_parents = torch.zeros(n_nodes, dtype=torch.bool)
        missing_parents = torch.zeros(n_nodes, dtype=torch.bool)
        
        for ind_id, ind in individuals.items():
            idx = self.node_to_idx[ind_id]
            if ind.parent_family_ids:
                # Check if parents are actually known
                has_known_parents = False
                for fam_id in ind.parent_family_ids:
                    if fam_id in families:
                        fam = families[fam_id]
                        if fam.husband_id or fam.wife_id:
                            has_known_parents = True
                            break
                has_parents[idx] = has_known_parents
            else:
                missing_parents[idx] = True
        
        # Split into train/val/test
        known_indices = torch.where(has_parents)[0]
        unknown_indices = torch.where(missing_parents)[0]
        
        # Use 60/20/20 split for known parents
        n_known = len(known_indices)
        perm = torch.randperm(n_known)
        
        train_known = known_indices[perm[:int(0.6 * n_known)]]
        val_known = known_indices[perm[int(0.6 * n_known):int(0.8 * n_known)]]
        test_known = known_indices[perm[int(0.8 * n_known):]]
        
        # All unknown parents go to test set (these are what we want to predict)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        train_mask[train_known] = True
        val_mask[val_known] = True
        test_mask[test_known] = True
        test_mask[unknown_indices] = True
        
        return train_mask, val_mask, test_mask
    
    def _create_parent_labels(self,
                            individuals: Dict[str, Individual],
                            families: Dict[str, Family]) -> torch.Tensor:
        """Create labels for parent prediction task"""
        n_nodes = len(individuals)
        # Labels will be (father_idx, mother_idx), using -1 for unknown
        labels = torch.full((n_nodes, 2), -1, dtype=torch.long)
        
        for ind_id, ind in individuals.items():
            idx = self.node_to_idx[ind_id]
            
            for fam_id in ind.parent_family_ids:
                if fam_id in families:
                    fam = families[fam_id]
                    if fam.husband_id and fam.husband_id in self.node_to_idx:
                        labels[idx, 0] = self.node_to_idx[fam.husband_id]
                    if fam.wife_id and fam.wife_id in self.node_to_idx:
                        labels[idx, 1] = self.node_to_idx[fam.wife_id]
        
        return labels
    
    def _extract_year(self, date_str: Optional[str]) -> Optional[int]:
        """Extract year from various date formats"""
        if not date_str:
            return None
            
        # Try to find 4-digit year
        import re
        year_match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', date_str)
        if year_match:
            return int(year_match.group(1))
            
        # Try to find 3-digit year (before 1000)
        year_match = re.search(r'\b([1-9][0-9]{2})\b', date_str)
        if year_match:
            return int(year_match.group(1))
            
        return None
    
    def _encode_location(self, location: Optional[str]) -> List[float]:
        """Encode location string into feature vector"""
        if not location:
            return [0, 0, 0, 0]
        
        # Simple encoding: country, state/region, city, has_location
        features = [0, 0, 0, 1]
        
        # Extract location components (simplified)
        parts = location.split(',')
        if len(parts) >= 3:
            features[0] = 1  # Has country
            features[1] = 1  # Has state/region
            features[2] = 1  # Has city
        elif len(parts) == 2:
            features[1] = 1  # Has state/region
            features[2] = 1  # Has city
        elif len(parts) == 1:
            features[2] = 1  # Has city
            
        return features
    
    def _estimate_generation(self, 
                           individual: Individual, 
                           all_individuals: Dict[str, Individual]) -> float:
        """Estimate generation number based on birth year and relationships"""
        birth_year = self._extract_year(individual.birth_date)
        if not birth_year:
            return 0
        
        # Assume ~25 years per generation
        # Normalize to 1900 as generation 0
        generation = (birth_year - 1900) / 25.0
        
        return generation
    
    def add_graph_statistics(self, graph_data) -> Dict:
        """Add statistics about the graph"""
        if isinstance(graph_data, Data):
            stats = {
                'num_nodes': graph_data.num_nodes,
                'num_edges': graph_data.edge_index.shape[1],
                'num_features': graph_data.x.shape[1],
                'avg_degree': graph_data.edge_index.shape[1] / graph_data.num_nodes,
                'has_isolated_nodes': self._check_isolated_nodes(graph_data)
            }
        else:  # HeteroData
            stats = {
                'num_individual_nodes': graph_data['individual'].num_nodes,
                'num_family_nodes': graph_data['family'].num_nodes,
                'num_parent_child_edges': graph_data['individual', 'parent_of', 'individual'].edge_index.shape[1] if hasattr(graph_data['individual', 'parent_of', 'individual'], 'edge_index') else 0,
                'num_spouse_edges': graph_data['individual', 'spouse_of', 'individual'].edge_index.shape[1] if hasattr(graph_data['individual', 'spouse_of', 'individual'], 'edge_index') else 0
            }
        
        return stats
    
    # def _check_isolated_nodes(self, data: Data) -> bool:
    #     """Check if graph has isolated nodes"""
    #     edge_index = data.edge_index
    #     connected_nodes = torch.unique(edge_index)
    #     return len(connected_nodes) < data.num_nodesidx = family_to_idx[fam_id]
    #                 ind_to_fam_edges.append([ind_idx, fam_idx])
    #                 fam_