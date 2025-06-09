"""
GEDCOM Parser for ʻŌhanaGPT
Handles parsing of GEDCOM genealogical files with enhanced error handling
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from gedcom.element.individual import IndividualElement
from gedcom.element.family import FamilyElement
from gedcom.parser import Parser
import hashlib
import json


@dataclass
class Individual:
    """Represents an individual in the genealogy"""
    id: str
    given_name: Optional[str] = None
    surname: Optional[str] = None
    sex: Optional[str] = None
    birth_date: Optional[str] = None
    birth_place: Optional[str] = None
    death_date: Optional[str] = None
    death_place: Optional[str] = None
    occupation: Optional[str] = None
    religion: Optional[str] = None
    nationality: Optional[str] = None
    education: Optional[str] = None
    
    # Family relationships
    parent_family_ids: List[str] = field(default_factory=list)
    spouse_family_ids: List[str] = field(default_factory=list)
    
    # Additional attributes for ML features
    attributes: Dict[str, any] = field(default_factory=dict)
    
    # For deduplication
    fingerprint: Optional[str] = None
    alternate_ids: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        self.fingerprint = self._generate_fingerprint()
    
    def _generate_fingerprint(self) -> str:
        """Generate a unique fingerprint for deduplication"""
        key_parts = [
            self.given_name or "",
            self.surname or "",
            self.birth_date or "",
            self.birth_place or "",
            self.sex or ""
        ]
        key_string = "|".join(key_parts).lower().strip()
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'given_name': self.given_name,
            'surname': self.surname,
            'sex': self.sex,
            'birth_date': self.birth_date,
            'birth_place': self.birth_place,
            'death_date': self.death_date,
            'death_place': self.death_place,
            'occupation': self.occupation,
            'parent_family_ids': self.parent_family_ids,
            'spouse_family_ids': self.spouse_family_ids,
            'attributes': self.attributes,
            'fingerprint': self.fingerprint
        }


@dataclass
class Family:
    """Represents a family unit"""
    id: str
    husband_id: Optional[str] = None
    wife_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    marriage_date: Optional[str] = None
    marriage_place: Optional[str] = None
    divorce_date: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'husband_id': self.husband_id,
            'wife_id': self.wife_id,
            'child_ids': self.child_ids,
            'marriage_date': self.marriage_date,
            'marriage_place': self.marriage_place,
            'divorce_date': self.divorce_date
        }


class GedcomParser:
    """Enhanced GEDCOM parser for ʻŌhanaGPT"""
    
    def __init__(self):
        self.individuals: Dict[str, Individual] = {}
        self.families: Dict[str, Family] = {}
        self.source_file: Optional[str] = None
        self.parsing_errors: List[str] = []
        
    def parse_file(self, filepath: str) -> Tuple[Dict[str, Individual], Dict[str, Family]]:
        """Parse a GEDCOM file and extract individuals and families"""
        self.source_file = filepath
        
        try:
            gedcom_parser = Parser()
            gedcom_parser.parse_file(filepath)
            
            # Parse individuals
            for element in gedcom_parser.get_element_list():
                if isinstance(element, IndividualElement):
                    individual = self._parse_individual(element)
                    self.individuals[individual.id] = individual
                    
            # Parse families
            for element in gedcom_parser.get_element_list():
                if isinstance(element, FamilyElement):
                    family = self._parse_family(element)
                    self.families[family.id] = family
                    
            # Update family relationships
            self._update_relationships()
            
        except Exception as e:
            self.parsing_errors.append(f"Error parsing {filepath}: {str(e)}")
            
        return self.individuals, self.families
    
    def _parse_individual(self, element: IndividualElement) -> Individual:
        """Extract individual information from GEDCOM element"""
        individual = Individual(id=element.get_pointer())
        
        # Parse name
        names = element.get_name()
        if names:
            name_parts = names[0].split('/')
            individual.given_name = name_parts[0].strip() if name_parts else None
            individual.surname = name_parts[1].strip() if len(name_parts) > 1 else None
        
        # Basic attributes
        individual.sex = element.get_gender()
        
        # Birth information
        birth_data = element.get_birth_data()
        if birth_data:
            individual.birth_date = birth_data[0] if birth_data[0] else None
            individual.birth_place = birth_data[1] if len(birth_data) > 1 else None
            
        # Death information
        death_data = element.get_death_data()
        if death_data:
            individual.death_date = death_data[0] if death_data[0] else None
            individual.death_place = death_data[1] if len(death_data) > 1 else None
        
        # Occupation
        for child in element.get_child_elements():
            if child.get_tag() == "OCCU":
                individual.occupation = child.get_value()
            elif child.get_tag() == "RELI":
                individual.religion = child.get_value()
            elif child.get_tag() == "NATI":
                individual.nationality = child.get_value()
            elif child.get_tag() == "EDUC":
                individual.education = child.get_value()
                
        # Store additional attributes for ML features
        individual.attributes = {
            'num_names': len(element.get_name()) if element.get_name() else 0,
            'has_birth_date': bool(individual.birth_date),
            'has_death_date': bool(individual.death_date),
            'has_occupation': bool(individual.occupation),
            'source_file': self.source_file
        }
        
        return individual
    
    def _parse_family(self, element: FamilyElement) -> Family:
        """Extract family information from GEDCOM element"""
        family = Family(id=element.get_pointer())
        
        # Get parents
        for member in element.get_members():
            if member.get_tag() == "HUSB":
                family.husband_id = member.get_value()
            elif member.get_tag() == "WIFE":
                family.wife_id = member.get_value()
            elif member.get_tag() == "CHIL":
                family.child_ids.append(member.get_value())
        
        # Marriage information
        for child in element.get_child_elements():
            if child.get_tag() == "MARR":
                for marr_child in child.get_child_elements():
                    if marr_child.get_tag() == "DATE":
                        family.marriage_date = marr_child.get_value()
                    elif marr_child.get_tag() == "PLAC":
                        family.marriage_place = marr_child.get_value()
            elif child.get_tag() == "DIV":
                for div_child in child.get_child_elements():
                    if div_child.get_tag() == "DATE":
                        family.divorce_date = div_child.get_value()
                        
        return family
    
    def _update_relationships(self):
        """Update individual objects with family relationship IDs"""
        for family_id, family in self.families.items():
            # Update children's parent family
            for child_id in family.child_ids:
                if child_id in self.individuals:
                    self.individuals[child_id].parent_family_ids.append(family_id)
                    
            # Update spouses' family
            if family.husband_id and family.husband_id in self.individuals:
                self.individuals[family.husband_id].spouse_family_ids.append(family_id)
            if family.wife_id and family.wife_id in self.individuals:
                self.individuals[family.wife_id].spouse_family_ids.append(family_id)
    
    def get_statistics(self) -> Dict:
        """Get parsing statistics"""
        stats = {
            'total_individuals': len(self.individuals),
            'total_families': len(self.families),
            'individuals_with_parents': sum(1 for ind in self.individuals.values() 
                                          if ind.parent_family_ids),
            'individuals_with_birth_dates': sum(1 for ind in self.individuals.values() 
                                              if ind.birth_date),
            'complete_families': sum(1 for fam in self.families.values() 
                                   if fam.husband_id and fam.wife_id),
            'parsing_errors': len(self.parsing_errors)
        }
        return stats
    
    def export_to_json(self, output_path: str):
        """Export parsed data to JSON format"""
        data = {
            'individuals': {id: ind.to_dict() for id, ind in self.individuals.items()},
            'families': {id: fam.to_dict() for id, fam in self.families.items()},
            'statistics': self.get_statistics(),
            'source_file': self.source_file
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)