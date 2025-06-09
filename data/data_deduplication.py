"""
Data Deduplication for ʻŌhanaGPT
Handles duplicate detection and merging across multiple GEDCOM files
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import hashlib
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import faiss
import pickle
import json
from datetime import datetime
import re

from .gedcom_parser import Individual, Family


@dataclass
class DuplicateCandidate:
    """Represents a potential duplicate pair"""
    id1: str
    id2: str
    similarity_score: float
    match_type: str  # 'exact', 'fuzzy_name', 'relationship', 'ml_based'
    supporting_evidence: Dict


class DeduplicationEngine:
    """Advanced deduplication engine for genealogical data"""
    
    def __init__(self, 
                 name_threshold: float = 0.85,
                 date_tolerance: int = 2,  # years
                 ml_threshold: float = 0.9):
        self.name_threshold = name_threshold
        self.date_tolerance = date_tolerance
        self.ml_threshold = ml_threshold
        
        # Caches for performance
        self.fingerprint_index: Dict[str, List[str]] = {}
        self.name_index: Dict[str, List[str]] = {}
        self.vector_index: Optional[faiss.IndexFlatL2] = None
        self.individual_vectors: Dict[str, np.ndarray] = {}
        
        # Duplicate mappings
        self.duplicate_clusters: List[Set[str]] = []
        self.canonical_ids: Dict[str, str] = {}  # Maps duplicate IDs to canonical ID
        
    def find_duplicates(self, 
                       all_individuals: Dict[str, Individual],
                       all_families: Dict[str, Family]) -> List[DuplicateCandidate]:
        """Find all duplicate individuals across the dataset"""
        print(f"Starting deduplication for {len(all_individuals)} individuals...")
        
        duplicates = []
        
        # 1. Exact matching based on fingerprints
        print("Phase 1: Exact matching...")
        exact_duplicates = self._find_exact_duplicates(all_individuals)
        duplicates.extend(exact_duplicates)
        
        # 2. Fuzzy name matching with date constraints
        print("Phase 2: Fuzzy name matching...")
        fuzzy_duplicates = self._find_fuzzy_duplicates(all_individuals)
        duplicates.extend(fuzzy_duplicates)
        
        # 3. Relationship-based matching
        print("Phase 3: Relationship-based matching...")
        relationship_duplicates = self._find_relationship_duplicates(
            all_individuals, all_families
        )
        duplicates.extend(relationship_duplicates)
        
        # 4. ML-based matching using embeddings
        print("Phase 4: ML-based matching...")
        ml_duplicates = self._find_ml_duplicates(all_individuals)
        duplicates.extend(ml_duplicates)
        
        # Deduplicate the duplicate list itself
        duplicates = self._deduplicate_candidates(duplicates)
        
        # Build duplicate clusters
        self._build_duplicate_clusters(duplicates)
        
        print(f"Found {len(duplicates)} duplicate pairs forming {len(self.duplicate_clusters)} clusters")
        
        return duplicates
    
    def _find_exact_duplicates(self, individuals: Dict[str, Individual]) -> List[DuplicateCandidate]:
        """Find exact duplicates based on fingerprints"""
        duplicates = []
        
        # Build fingerprint index
        for ind_id, ind in individuals.items():
            fp = ind.fingerprint
            if fp not in self.fingerprint_index:
                self.fingerprint_index[fp] = []
            self.fingerprint_index[fp].append(ind_id)
        
        # Find duplicates
        for fp, ind_ids in self.fingerprint_index.items():
            if len(ind_ids) > 1:
                for i in range(len(ind_ids)):
                    for j in range(i + 1, len(ind_ids)):
                        duplicates.append(DuplicateCandidate(
                            id1=ind_ids[i],
                            id2=ind_ids[j],
                            similarity_score=1.0,
                            match_type='exact',
                            supporting_evidence={'fingerprint': fp}
                        ))
        
        return duplicates
    
    def _find_fuzzy_duplicates(self, individuals: Dict[str, Individual]) -> List[DuplicateCandidate]:
        """Find duplicates using fuzzy name matching"""
        duplicates = []
        
        # Build name index for efficiency
        for ind_id, ind in individuals.items():
            if ind.given_name and ind.surname:
                full_name = f"{ind.given_name} {ind.surname}".lower()
                name_key = full_name[:3]  # First 3 chars for blocking
                if name_key not in self.name_index:
                    self.name_index[name_key] = []
                self.name_index[name_key].append(ind_id)
        
        # Compare within blocks
        for name_key, ind_ids in self.name_index.items():
            if len(ind_ids) < 2:
                continue
                
            for i in range(len(ind_ids)):
                for j in range(i + 1, len(ind_ids)):
                    ind1 = individuals[ind_ids[i]]
                    ind2 = individuals[ind_ids[j]]
                    
                    # Skip if already exact match
                    if ind1.fingerprint == ind2.fingerprint:
                        continue
                    
                    # Compare names
                    name1 = f"{ind1.given_name} {ind1.surname}".lower()
                    name2 = f"{ind2.given_name} {ind2.surname}".lower()
                    
                    name_sim = fuzz.ratio(name1, name2) / 100.0
                    
                    if name_sim >= self.name_threshold:
                        # Check date compatibility
                        if self._are_dates_compatible(ind1, ind2):
                            duplicates.append(DuplicateCandidate(
                                id1=ind_ids[i],
                                id2=ind_ids[j],
                                similarity_score=name_sim,
                                match_type='fuzzy_name',
                                supporting_evidence={
                                    'name_similarity': name_sim,
                                    'name1': name1,
                                    'name2': name2
                                }
                            ))
        
        return duplicates
    
    def _find_relationship_duplicates(self, 
                                    individuals: Dict[str, Individual],
                                    families: Dict[str, Family]) -> List[DuplicateCandidate]:
        """Find duplicates based on family relationships"""
        duplicates = []
        
        # Build relationship signatures
        rel_signatures = {}
        for ind_id, ind in individuals.items():
            signature = self._build_relationship_signature(ind, individuals, families)
            if signature:
                sig_key = self._hash_signature(signature)
                if sig_key not in rel_signatures:
                    rel_signatures[sig_key] = []
                rel_signatures[sig_key].append((ind_id, signature))
        
        # Find matches
        for sig_key, ind_list in rel_signatures.items():
            if len(ind_list) < 2:
                continue
                
            for i in range(len(ind_list)):
                for j in range(i + 1, len(ind_list)):
                    ind1_id, sig1 = ind_list[i]
                    ind2_id, sig2 = ind_list[j]
                    
                    ind1 = individuals[ind1_id]
                    ind2 = individuals[ind2_id]
                    
                    # Additional verification
                    if self._verify_relationship_match(ind1, ind2, sig1, sig2):
                        similarity = self._calculate_relationship_similarity(sig1, sig2)
                        
                        duplicates.append(DuplicateCandidate(
                            id1=ind1_id,
                            id2=ind2_id,
                            similarity_score=similarity,
                            match_type='relationship',
                            supporting_evidence={
                                'relationship_signature': sig_key,
                                'common_relations': len(sig1['parents']) + len(sig1['children'])
                            }
                        ))
        
        return duplicates
    
    def _find_ml_duplicates(self, individuals: Dict[str, Individual]) -> List[DuplicateCandidate]:
        """Find duplicates using machine learning embeddings"""
        duplicates = []
        
        # Create embeddings for all individuals
        embeddings = []
        ind_ids = []
        
        for ind_id, ind in individuals.items():
            embedding = self._create_individual_embedding(ind)
            embeddings.append(embedding)
            ind_ids.append(ind_id)
            self.individual_vectors[ind_id] = embedding
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Build FAISS index for efficient similarity search
        dimension = embeddings_array.shape[1]
        self.vector_index = faiss.IndexFlatL2(dimension)
        self.vector_index.add(embeddings_array)
        
        # Search for similar individuals
        k = 10  # Find top-k similar
        distances, indices = self.vector_index.search(embeddings_array, k)
        
        # Process results
        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            for j, (dist, idx) in enumerate(zip(dist_row, idx_row)):
                if i != idx and dist < (1 - self.ml_threshold):
                    # Convert distance to similarity
                    similarity = 1 - dist
                    
                    if similarity >= self.ml_threshold:
                        duplicates.append(DuplicateCandidate(
                            id1=ind_ids[i],
                            id2=ind_ids[idx],
                            similarity_score=similarity,
                            match_type='ml_based',
                            supporting_evidence={
                                'embedding_distance': float(dist),
                                'embedding_similarity': similarity
                            }
                        ))
        
        return duplicates
    
    def _are_dates_compatible(self, ind1: Individual, ind2: Individual) -> bool:
        """Check if two individuals have compatible dates"""
        # Extract years
        birth1 = self._extract_year(ind1.birth_date)
        birth2 = self._extract_year(ind2.birth_date)
        death1 = self._extract_year(ind1.death_date)
        death2 = self._extract_year(ind2.death_date)
        
        # Check birth dates
        if birth1 and birth2:
            if abs(birth1 - birth2) > self.date_tolerance:
                return False
        
        # Check death dates
        if death1 and death2:
            if abs(death1 - death2) > self.date_tolerance:
                return False
        
        # Check logical consistency
        if birth1 and death2 and birth1 > death2 + self.date_tolerance:
            return False
        if birth2 and death1 and birth2 > death1 + self.date_tolerance:
            return False
        
        return True
    
    def _build_relationship_signature(self, 
                                    individual: Individual,
                                    all_individuals: Dict[str, Individual],
                                    all_families: Dict[str, Family]) -> Optional[Dict]:
        """Build a signature based on family relationships"""
        signature = {
            'parents': set(),
            'children': set(),
            'spouses': set(),
            'siblings': set()
        }
        
        # Get parents
        for fam_id in individual.parent_family_ids:
            if fam_id in all_families:
                fam = all_families[fam_id]
                if fam.husband_id:
                    signature['parents'].add(self._get_name_key(all_individuals.get(fam.husband_id)))
                if fam.wife_id:
                    signature['parents'].add(self._get_name_key(all_individuals.get(fam.wife_id)))
                    
                # Get siblings
                for child_id in fam.child_ids:
                    if child_id != individual.id and child_id in all_individuals:
                        signature['siblings'].add(self._get_name_key(all_individuals[child_id]))
        
        # Get spouses and children
        for fam_id in individual.spouse_family_ids:
            if fam_id in all_families:
                fam = all_families[fam_id]
                
                # Get spouse
                if individual.id == fam.husband_id and fam.wife_id:
                    signature['spouses'].add(self._get_name_key(all_individuals.get(fam.wife_id)))
                elif individual.id == fam.wife_id and fam.husband_id:
                    signature['spouses'].add(self._get_name_key(all_individuals.get(fam.husband_id)))
                
                # Get children
                for child_id in fam.child_ids:
                    if child_id in all_individuals:
                        signature['children'].add(self._get_name_key(all_individuals[child_id]))
        
        # Only return if we have some relationships
        total_relations = sum(len(v) for v in signature.values())
        return signature if total_relations > 0 else None
    
    def _get_name_key(self, individual: Optional[Individual]) -> str:
        """Get a normalized name key for an individual"""
        if not individual:
            return "unknown"
        
        name_parts = []
        if individual.given_name:
            name_parts.append(individual.given_name.lower())
        if individual.surname:
            name_parts.append(individual.surname.lower())
        
        return " ".join(name_parts) if name_parts else "unknown"
    
    def _hash_signature(self, signature: Dict) -> str:
        """Create a hash of a relationship signature"""
        # Sort all components for consistent hashing
        components = []
        for key in sorted(signature.keys()):
            for item in sorted(signature[key]):
                components.append(f"{key}:{item}")
        
        sig_string = "|".join(components)
        return hashlib.md5(sig_string.encode()).hexdigest()
    
    def _verify_relationship_match(self, 
                                 ind1: Individual, 
                                 ind2: Individual,
                                 sig1: Dict, 
                                 sig2: Dict) -> bool:
        """Verify that two individuals with similar signatures are likely the same"""
        # Check name similarity
        name1 = self._get_name_key(ind1)
        name2 = self._get_name_key(ind2)
        
        if name1 != "unknown" and name2 != "unknown":
            name_sim = fuzz.ratio(name1, name2) / 100.0
            if name_sim < 0.6:  # Too different
                return False
        
        # Check date compatibility
        return self._are_dates_compatible(ind1, ind2)
    
    def _calculate_relationship_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Calculate similarity between two relationship signatures"""
        total_score = 0
        total_weight = 0
        
        weights = {
            'parents': 2.0,
            'children': 1.5,
            'spouses': 2.0,
            'siblings': 1.0
        }
        
        for key, weight in weights.items():
            set1 = sig1.get(key, set())
            set2 = sig2.get(key, set())
            
            if set1 or set2:
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                
                if union > 0:
                    jaccard = intersection / union
                    total_score += jaccard * weight
                    total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _create_individual_embedding(self, individual: Individual) -> np.ndarray:
        """Create a numerical embedding for an individual"""
        features = []
        
        # Name features
        name_vec = self._encode_name(individual)
        features.extend(name_vec)
        
        # Date features
        birth_year = self._extract_year(individual.birth_date) or 0
        death_year = self._extract_year(individual.death_date) or 0
        features.extend([
            birth_year / 2000.0,  # Normalize
            death_year / 2000.0,
            (death_year - birth_year) / 100.0 if birth_year and death_year else 0
        ])
        
        # Location features
        birth_loc_vec = self._encode_location(individual.birth_place)
        death_loc_vec = self._encode_location(individual.death_place)
        features.extend(birth_loc_vec)
        features.extend(death_loc_vec)
        
        # Gender
        features.append(1 if individual.sex == 'M' else 0 if individual.sex == 'F' else 0.5)
        
        # Relationship counts
        features.extend([
            len(individual.parent_family_ids) / 5.0,  # Normalize
            len(individual.spouse_family_ids) / 5.0
        ])
        
        # Additional attributes
        features.extend([
            1 if individual.occupation else 0,
            1 if individual.religion else 0,
            1 if individual.education else 0
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _encode_name(self, individual: Individual) -> List[float]:
        """Encode name into numerical features"""
        features = []
        
        # Length features
        given_len = len(individual.given_name) if individual.given_name else 0
        surname_len = len(individual.surname) if individual.surname else 0
        features.extend([given_len / 20.0, surname_len / 20.0])
        
        # Character features (first and last char codes)
        if individual.given_name:
            features.extend([
                ord(individual.given_name[0].lower()) / 127.0,
                ord(individual.given_name[-1].lower()) / 127.0
            ])
        else:
            features.extend([0, 0])
            
        if individual.surname:
            features.extend([
                ord(individual.surname[0].lower()) / 127.0,
                ord(individual.surname[-1].lower()) / 127.0
            ])
        else:
            features.extend([0, 0])
        
        return features
    
    def _encode_location(self, location: Optional[str]) -> List[float]:
        """Encode location into numerical features"""
        if not location:
            return [0, 0, 0, 0]
        
        features = []
        
        # Length
        features.append(len(location) / 100.0)
        
        # Number of components (commas)
        features.append(location.count(',') / 5.0)
        
        # Contains common place indicators
        indicators = ['county', 'state', 'usa', 'england', 'germany']
        has_indicator = any(ind in location.lower() for ind in indicators)
        features.append(1 if has_indicator else 0)
        
        # First character
        features.append(ord(location[0].lower()) / 127.0)
        
        return features
    
    def _extract_year(self, date_str: Optional[str]) -> Optional[int]:
        """Extract year from date string"""
        if not date_str:
            return None
            
        year_match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', date_str)
        if year_match:
            return int(year_match.group(1))
        return None
    
    def _deduplicate_candidates(self, candidates: List[DuplicateCandidate]) -> List[DuplicateCandidate]:
        """Remove duplicate entries from candidate list"""
        seen = set()
        unique_candidates = []
        
        for candidate in candidates:
            # Create a canonical pair representation
            pair = tuple(sorted([candidate.id1, candidate.id2]))
            
            if pair not in seen:
                seen.add(pair)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _build_duplicate_clusters(self, duplicates: List[DuplicateCandidate]):
        """Build clusters of duplicate individuals"""
        # Create a graph of duplicates
        G = nx.Graph()
        
        for dup in duplicates:
            G.add_edge(dup.id1, dup.id2, weight=dup.similarity_score)
        
        # Find connected components (clusters)
        self.duplicate_clusters = [set(cluster) for cluster in nx.connected_components(G)]
        
        # Assign canonical IDs (choose the ID that appears in most files or has most data)
        for cluster in self.duplicate_clusters:
            canonical_id = self._choose_canonical_id(cluster)
            for ind_id in cluster:
                self.canonical_ids[ind_id] = canonical_id
    
    def _choose_canonical_id(self, cluster: Set[str]) -> str:
        """Choose the best ID to represent a cluster"""
        # For now, just choose the first one alphabetically
        # In production, you'd want to choose based on data completeness
        return sorted(cluster)[0]
    
    def merge_individuals(self, 
                         individuals: Dict[str, Individual],
                         families: Dict[str, Family]) -> Tuple[Dict[str, Individual], Dict[str, Family]]:
        """Merge duplicate individuals and update family references"""
        merged_individuals = {}
        merged_families = {}
        
        # Process individuals
        processed = set()
        for ind_id, individual in individuals.items():
            canonical_id = self.canonical_ids.get(ind_id, ind_id)
            
            if canonical_id in processed:
                continue
                
            # Get all individuals in this cluster
            if ind_id in self.canonical_ids:
                cluster = next(c for c in self.duplicate_clusters if canonical_id in c)
                cluster_individuals = [individuals[id] for id in cluster if id in individuals]
                
                # Merge all individuals in cluster
                merged_individual = self._merge_individual_data(cluster_individuals)
                merged_individual.id = canonical_id
                merged_individual.alternate_ids = cluster - {canonical_id}
            else:
                merged_individual = individual
            
            merged_individuals[canonical_id] = merged_individual
            processed.add(canonical_id)
        
        # Update family references
        for fam_id, family in families.items():
            new_family = Family(
                id=fam_id,
                husband_id=self.canonical_ids.get(family.husband_id, family.husband_id) if family.husband_id else None,
                wife_id=self.canonical_ids.get(family.wife_id, family.wife_id) if family.wife_id else None,
                child_ids=[self.canonical_ids.get(child_id, child_id) for child_id in family.child_ids],
                marriage_date=family.marriage_date,
                marriage_place=family.marriage_place,
                divorce_date=family.divorce_date
            )
            merged_families[fam_id] = new_family
        
        return merged_individuals, merged_families
    
    def _merge_individual_data(self, individuals: List[Individual]) -> Individual:
        """Merge data from multiple individual records"""
        # Start with the first individual
        merged = Individual(id=individuals[0].id)
        
        # Merge names (prefer longest)
        merged.given_name = max((ind.given_name for ind in individuals if ind.given_name), 
                               key=lambda x: len(x) if x else 0, default=None)
        merged.surname = max((ind.surname for ind in individuals if ind.surname), 
                            key=lambda x: len(x) if x else 0, default=None)
        
        # Merge sex (majority vote)
        sex_votes = [ind.sex for ind in individuals if ind.sex]
        if sex_votes:
            merged.sex = max(set(sex_votes), key=sex_votes.count)
        
        # Merge dates (prefer most specific)
        merged.birth_date = self._merge_dates([ind.birth_date for ind in individuals])
        merged.birth_place = self._merge_places([ind.birth_place for ind in individuals])
        merged.death_date = self._merge_dates([ind.death_date for ind in individuals])
        merged.death_place = self._merge_places([ind.death_place for ind in individuals])
        
        # Merge attributes (union)
        for ind in individuals:
            if ind.occupation and not merged.occupation:
                merged.occupation = ind.occupation
            if ind.religion and not merged.religion:
                merged.religion = ind.religion
            if ind.education and not merged.education:
                merged.education = ind.education
        
        # Merge family relationships (union)
        merged.parent_family_ids = list(set().union(*[ind.parent_family_ids for ind in individuals]))
        merged.spouse_family_ids = list(set().union(*[ind.spouse_family_ids for ind in individuals]))
        
        # Merge attributes dictionary
        for ind in individuals:
            merged.attributes.update(ind.attributes)
        
        return merged
    
    def _merge_dates(self, dates: List[Optional[str]]) -> Optional[str]:
        """Merge multiple date values, preferring most specific"""
        valid_dates = [d for d in dates if d]
        if not valid_dates:
            return None
        
        # Prefer dates with full year-month-day
        for date in valid_dates:
            if re.search(r'\d{1,2}\s+\w+\s+\d{4}', date):  # e.g., "12 JAN 1850"
                return date
        
        # Otherwise, prefer dates with years
        for date in valid_dates:
            if re.search(r'\d{4}', date):
                return date
        
        # Return first non-empty
        return valid_dates[0]
    
    def _merge_places(self, places: List[Optional[str]]) -> Optional[str]:
        """Merge multiple place values, preferring most detailed"""
        valid_places = [p for p in places if p]
        if not valid_places:
            return None
        
        # Prefer places with most components (commas)
        return max(valid_places, key=lambda x: x.count(','))
    
    def save_deduplication_index(self, filepath: str):
        """Save deduplication index for future use"""
        index_data = {
            'duplicate_clusters': [list(cluster) for cluster in self.duplicate_clusters],
            'canonical_ids': self.canonical_ids,
            'fingerprint_index': self.fingerprint_index,
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def load_deduplication_index(self, filepath: str):
        """Load previously saved deduplication index"""
        with open(filepath, 'r') as f:
            index_data = json.load(f)
        
        self.duplicate_clusters = [set(cluster) for cluster in index_data['duplicate_clusters']]
        self.canonical_ids = index_data['canonical_ids']
        self.fingerprint_index = index_data['fingerprint_index']
    
    def get_statistics(self) -> Dict:
        """Get deduplication statistics"""
        total_duplicates = sum(len(cluster) - 1 for cluster in self.duplicate_clusters)
        
        return {
            'total_duplicate_pairs': total_duplicates,
            'duplicate_clusters': len(self.duplicate_clusters),
            'largest_cluster_size': max(len(cluster) for cluster in self.duplicate_clusters) if self.duplicate_clusters else 0,
            'deduplication_rate': total_duplicates / (total_duplicates + len(self.canonical_ids)) if self.canonical_ids else 0
        }