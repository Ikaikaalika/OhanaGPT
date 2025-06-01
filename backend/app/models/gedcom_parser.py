import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Individual:
    id: str
    name: Optional[str] = None
    birth_date: Optional[datetime] = None
    death_date: Optional[datetime] = None
    gender: Optional[str] = None
    family_child: Optional[str] = None  # FAMC
    family_spouse: List[str] = None  # FAMS
    attributes: Dict[str, str] = None

    def __post_init__(self):
        if self.family_spouse is None:
            self.family_spouse = []
        if self.attributes is None:
            self.attributes = {}

@dataclass
class Family:
    id: str
    husband: Optional[str] = None
    wife: Optional[str] = None
    children: List[str] = None
    marriage_date: Optional[datetime] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

class GedcomParser:
    def __init__(self):
        self.individuals: Dict[str, Individual] = {}
        self.families: Dict[str, Family] = {}
        
    def parse(self, gedcom_content: str) -> Tuple[Dict[str, Individual], Dict[str, Family]]:
        """Parse GEDCOM content and extract individuals and families."""
        lines = gedcom_content.strip().split('\n')
        current_record = None
        current_type = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            match = re.match(r'^(\d+)\s+(@\w+@)?\s*(\w+)?\s*(.*)$', line)
            if not match:
                continue
                
            level, xref, tag, value = match.groups()
            level = int(level)
            
            if level == 0:
                if xref:
                    if tag == 'INDI':
                        current_record = Individual(id=xref)
                        current_type = 'INDI'
                        self.individuals[xref] = current_record
                    elif tag == 'FAM':
                        current_record = Family(id=xref)
                        current_type = 'FAM'
                        self.families[xref] = current_record
                else:
                    current_record = None
                    current_type = None
            
            elif current_record:
                if current_type == 'INDI':
                    self._parse_individual_tag(current_record, level, tag, value)
                elif current_type == 'FAM':
                    self._parse_family_tag(current_record, level, tag, value)
        
        return self.individuals, self.families
    
    def _parse_individual_tag(self, individual: Individual, level: int, tag: str, value: str):
        """Parse individual-specific tags."""
        if level == 1:
            if tag == 'NAME':
                individual.name = value.strip('/')
            elif tag == 'SEX':
                individual.gender = value
            elif tag == 'FAMC':
                individual.family_child = value
            elif tag == 'FAMS':
                individual.family_spouse.append(value)
            elif tag in ['BIRT', 'DEAT']:
                individual.attributes[f'current_{tag.lower()}'] = True
        elif level == 2:
            if tag == 'DATE':
                if individual.attributes.get('current_birt'):
                    individual.birth_date = self._parse_date(value)
                    del individual.attributes['current_birt']
                elif individual.attributes.get('current_deat'):
                    individual.death_date = self._parse_date(value)
                    del individual.attributes['current_deat']
    
    def _parse_family_tag(self, family: Family, level: int, tag: str, value: str):
        """Parse family-specific tags."""
        if level == 1:
            if tag == 'HUSB':
                family.husband = value
            elif tag == 'WIFE':
                family.wife = value
            elif tag == 'CHIL':
                family.children.append(value)
            elif tag == 'MARR':
                family.attributes = {'current_marr': True}
        elif level == 2 and tag == 'DATE' and hasattr(family, 'attributes') and family.attributes.get('current_marr'):
            family.marriage_date = self._parse_date(value)
            delattr(family, 'attributes')
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse GEDCOM date format."""
        # Simple date parsing - can be enhanced
        try:
            # Handle various date formats
            date_str = date_str.upper().replace('ABT', '').replace('EST', '').strip()
            if len(date_str) == 4:  # Year only
                return datetime(int(date_str), 1, 1)
            # Add more date parsing logic as needed
            return None
        except:
            return None