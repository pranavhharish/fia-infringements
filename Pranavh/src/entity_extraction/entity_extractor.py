"""
Entity Extraction Module
Extracts key entities from FIA decision documents
"""

import re
from typing import Dict, List, Optional
import logging


class EntityExtractor:
    """Extract entities from FIA decision documents"""

    def __init__(self):
        """Initialize entity extractor"""
        self.logger = logging.getLogger(__name__)

        # Infraction type patterns
        self.infraction_patterns = {
            'track_limits': [
                r'leaving the track',
                r'track limits',
                r'exceeded track limits',
                r'left the track'
            ],
            'collision': [
                r'causing a collision',
                r'causing collision',
                r'collision with',
                r'incident with car'
            ],
            'unsafe_release': [
                r'unsafe release',
                r'unsafe pit release',
                r'released unsafely'
            ],
            'speeding': [
                r'pit lane speeding',
                r'pit lane speed',
                r'exceeded speed limit',
                r'speeding in pit lane'
            ],
            'yellow_flags': [
                r'yellow flags',
                r'failure to slow',
                r'not slowing for yellow',
                r'double yellow'
            ],
            'impeding': [
                r'impeding',
                r'blocking',
                r'unnecessarily slowly'
            ],
            'technical': [
                r'technical non-compliance',
                r'technical infringement',
                r'plank',
                r'PU element',
                r'power unit'
            ],
            'procedure': [
                r'pre-race procedure',
                r'parc ferme',
                r'scrutineering',
                r'incorrect use of tyres'
            ],
            'other': []
        }

        # Penalty patterns
        self.penalty_patterns = {
            'time_penalty': r'(\d+)\s*second[s]?\s*(?:time\s*)?penalty',
            'grid_penalty': r'(\d+)\s*(?:grid\s*)?(?:place|position)[s]?\s*(?:grid\s*)?penalty',
            'reprimand': r'reprimand',
            'fine': r'(?:fine|fined)\s*(?:of\s*)?(?:€|EUR|USD|\$)?\s*(\d+)',
            'disqualification': r'disqualif',
            'no_action': r'no\s*(?:further\s*)?action'
        }

    def extract_car_numbers(self, text: str) -> List[int]:
        """
        Extract car numbers from text

        Args:
            text: Document text

        Returns:
            List of car numbers
        """
        car_numbers = []

        # Pattern for "Car XX" or "Car XX-"
        pattern = r'\bcar\s*(\d{1,2})\b'
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            car_num = int(match.group(1))
            if 1 <= car_num <= 99 and car_num not in car_numbers:
                car_numbers.append(car_num)

        return sorted(car_numbers)

    def extract_infraction_type(self, text: str) -> List[str]:
        """
        Extract infraction types from text

        Args:
            text: Document text

        Returns:
            List of infraction types
        """
        infractions = []
        text_lower = text.lower()

        for infraction_type, patterns in self.infraction_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    infractions.append(infraction_type)
                    break

        return infractions if infractions else ['other']

    def extract_penalties(self, text: str) -> Dict[str, any]:
        """
        Extract penalty information from text

        Args:
            text: Document text

        Returns:
            Dictionary with penalty details
        """
        penalties = {
            'type': None,
            'value': None,
            'description': None
        }

        text_lower = text.lower()

        # Check for no action
        if re.search(self.penalty_patterns['no_action'], text_lower):
            penalties['type'] = 'no_action'
            penalties['description'] = 'No further action'
            return penalties

        # Check for time penalty
        time_match = re.search(self.penalty_patterns['time_penalty'], text_lower)
        if time_match:
            penalties['type'] = 'time_penalty'
            penalties['value'] = int(time_match.group(1))
            penalties['description'] = f"{time_match.group(1)} second time penalty"
            return penalties

        # Check for grid penalty
        grid_match = re.search(self.penalty_patterns['grid_penalty'], text_lower)
        if grid_match:
            penalties['type'] = 'grid_penalty'
            penalties['value'] = int(grid_match.group(1))
            penalties['description'] = f"{grid_match.group(1)} place grid penalty"
            return penalties

        # Check for reprimand
        if re.search(self.penalty_patterns['reprimand'], text_lower):
            penalties['type'] = 'reprimand'
            penalties['description'] = 'Reprimand'
            return penalties

        # Check for fine
        fine_match = re.search(self.penalty_patterns['fine'], text_lower)
        if fine_match:
            penalties['type'] = 'fine'
            penalties['value'] = int(fine_match.group(1))
            penalties['description'] = f"Fine of €{fine_match.group(1)}"
            return penalties

        # Check for disqualification
        if re.search(self.penalty_patterns['disqualification'], text_lower):
            penalties['type'] = 'disqualification'
            penalties['description'] = 'Disqualification'
            return penalties

        # Unknown penalty
        penalties['type'] = 'unknown'
        return penalties

    def extract_decision_section(self, text: str) -> Optional[str]:
        """
        Extract the decision/reason section from document

        Args:
            text: Document text

        Returns:
            Decision section text or None
        """
        # Common decision section markers
        decision_patterns = [
            r'Decision\s*[:.]?\s*(.{50,500})',
            r'Reason\s*[:.]?\s*(.{50,500})',
            r'The Stewards.*?(?:decide|determined|concluded)\s*[:.]?\s*(.{50,500})',
        ]

        for pattern in decision_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                decision_text = match.group(1).strip()
                # Clean up
                decision_text = re.sub(r'\s+', ' ', decision_text)
                return decision_text

        return None

    def extract_all_entities(self, text: str, team_name: str = None) -> Dict:
        """
        Extract all entities from document

        Args:
            text: Document text
            team_name: Optional team name (from team detector)

        Returns:
            Dictionary with all extracted entities
        """
        entities = {
            'team': team_name,
            'car_numbers': self.extract_car_numbers(text),
            'infraction_types': self.extract_infraction_type(text),
            'penalty': self.extract_penalties(text),
            'decision_text': self.extract_decision_section(text)
        }

        return entities

    def create_structured_record(
        self,
        filename: str,
        text: str,
        team_name: str,
        year: str,
        driver_name: str = None,
        metadata: Dict = None
    ) -> Dict:
        """
        Create a structured record from document

        Args:
            filename: PDF filename
            text: Document text
            team_name: Team name
            year: Year
            driver_name: Driver name (optional)
            metadata: Additional metadata (optional)

        Returns:
            Structured record dictionary
        """
        entities = self.extract_all_entities(text, team_name)

        record = {
            'filename': filename,
            'year': year,
            'team': team_name,
            'driver': driver_name,
            'car_numbers': entities['car_numbers'],
            'primary_car_number': entities['car_numbers'][0] if entities['car_numbers'] else None,
            'infraction_types': entities['infraction_types'],
            'primary_infraction': entities['infraction_types'][0] if entities['infraction_types'] else None,
            'penalty_type': entities['penalty']['type'],
            'penalty_value': entities['penalty']['value'],
            'penalty_description': entities['penalty']['description'],
            'decision_text': entities['decision_text'],
            'text_length': len(text)
        }

        # Add metadata if provided
        if metadata:
            record.update({
                'document_number': metadata.get('document_number'),
                'date': metadata.get('date'),
                'time': metadata.get('time'),
                'session': metadata.get('session'),
                'race': metadata.get('race')
            })

        return record
