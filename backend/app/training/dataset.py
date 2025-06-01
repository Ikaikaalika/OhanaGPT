import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

class GenealogyDataset(Dataset):
    """Dataset for genealogy parent prediction."""
    
    def __init__(self, graph_data: Dict, negative_samples: int = 5):
        self.graph_data = graph_data
        self.negative_samples = negative_samples
        
        # Create training examples
        self.examples = self._create_examples()
    
    def _create_examples(self) -> List[Dict]:
        """Create positive and negative training examples."""
        examples = []
        
        # Get all parent-child relationships
        for fam_id, family in self.graph_data['families'].items():
            parents = []
            if family.husband:
                parents.append(family.husband)
            if family.wife:
                parents.append(family.wife)
            
            for child in family.children:
                if child in self.graph_data['individuals']:
                    # Positive examples
                    for parent in parents:
                        if parent in self.graph_data['individuals']:
                            examples.append({
                                'child': child,
                                'parent': parent,
                                'label': 1.0
                            })
                    
                    # Negative examples
                    non_parents = self._sample_non_parents(child, parents)
                    for non_parent in non_parents:
                        examples.append({
                            'child': child,
                            'parent': non_parent,
                            'label': 0.0
                        })
        
        return examples
    
    def _sample_non_parents(self, child: str, parents: List[str]) -> List[str]:
        """Sample individuals who are not parents of the child."""
        all_individuals = set(self.graph_data['individuals'].keys())
        non_parents = all_individuals - set(parents) - {child}
        
        # Sample negative examples
        sampled = np.random.choice(
            list(non_parents),
            size=min(self.negative_samples, len(non_parents)),
            replace=False
        )
        
        return sampled.tolist()
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a training example."""
        example = self.examples[idx]
        
        # Get features for child and parent
        child_features = self._get_features(example['child'])
        parent_features = self._get_features(example['parent'])
        
        return {
            'child_features': torch.tensor(child_features, dtype=torch.float),
            'parent_features': torch.tensor(parent_features, dtype=torch.float),
            'label': torch.tensor(example['label'], dtype=torch.float)
        }
    
    def _get_features(self, individual_id: str) -> List[float]:
        """Extract features for an individual."""
        ind = self.graph_data['individuals'][individual_id]
        
        features = [
            1.0 if ind.gender == 'M' else 0.0,
            1.0 if ind.gender == 'F' else 0.0,
            float(ind.birth_date.year) / 2000.0 if ind.birth_date else 0.0,
            float(ind.death_date.year) / 2000.0 if ind.death_date else 0.0,
            1.0 if ind.family_child else 0.0,
            float(len(ind.family_spouse)) / 10.0
        ]
        
        return features

