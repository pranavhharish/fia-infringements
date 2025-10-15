"""
Team Detection Module for F1 Infringement Analysis
Detects team affiliations from FIA decision documents
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TeamDetector:
    """Detects F1 team affiliations from document text"""

    def __init__(self, team_mappings_path: str = "config/team_mappings.json"):
        """
        Initialize team detector with team mappings

        Args:
            team_mappings_path: Path to team mappings JSON file
        """
        self.team_mappings_path = Path(team_mappings_path)
        self.teams = self._load_team_mappings()

    def _load_team_mappings(self) -> Dict:
        """Load team mappings from JSON file"""
        with open(self.team_mappings_path, 'r') as f:
            data = json.load(f)
        return data['teams']

    def detect_team(self, text: str, year: str = None) -> Optional[Tuple[str, float]]:
        """
        Detect team from document text

        Args:
            text: Document text
            year: Year of the document (for car number validation)

        Returns:
            Tuple of (team_name, confidence_score) or None
        """
        if not text:
            return None

        text_lower = text.lower()
        header_section = text_lower[:2000]  # Focus on header

        team_scores = {}

        for team_name, team_info in self.teams.items():
            score = 0

            # Check official name
            if team_info['official_name'].lower() in header_section:
                score += 10

            # Check aliases
            for alias in team_info['aliases']:
                # Look for "To: [Team Name]" pattern
                to_pattern = rf'to:\s*(?:the\s+)?(?:team\s+manager,?\s*)?{re.escape(alias)}'
                if re.search(to_pattern, header_section, re.IGNORECASE):
                    score += 15
                elif alias.lower() in header_section:
                    score += 5

            # Check car numbers if year is provided
            if year and year in team_info['car_numbers']:
                car_numbers = team_info['car_numbers'][year]
                for car_num in car_numbers:
                    car_pattern = rf'\bcar\s*{car_num}\b'
                    if re.search(car_pattern, header_section, re.IGNORECASE):
                        score += 8

            # Check driver names if year is provided
            if year and year in team_info['drivers']:
                drivers = team_info['drivers'][year]
                for driver_name in drivers.values():
                    if driver_name.lower() in text_lower[:3000]:
                        score += 3

            if score > 0:
                team_scores[team_name] = score

        if not team_scores:
            return None

        # Get team with highest score
        best_team = max(team_scores, key=team_scores.get)
        confidence = min(team_scores[best_team] / 20.0, 1.0)  # Normalize to 0-1

        return (best_team, confidence)

    def detect_all_teams(self, text: str, year: str = None) -> List[Tuple[str, float]]:
        """
        Detect all potential teams from document text

        Args:
            text: Document text
            year: Year of the document

        Returns:
            List of tuples (team_name, confidence_score) sorted by confidence
        """
        if not text:
            return []

        text_lower = text.lower()
        team_scores = {}

        for team_name, team_info in self.teams.items():
            score = 0

            # Check all team identifiers
            for alias in [team_info['official_name']] + team_info['aliases']:
                if alias.lower() in text_lower:
                    score += 3

            # Check car numbers
            if year and year in team_info['car_numbers']:
                car_numbers = team_info['car_numbers'][year]
                for car_num in car_numbers:
                    if f'car {car_num}' in text_lower or f'car{car_num}' in text_lower:
                        score += 5

            if score > 0:
                confidence = min(score / 15.0, 1.0)
                team_scores[team_name] = confidence

        # Sort by confidence
        sorted_teams = sorted(team_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_teams

    def get_team_info(self, team_name: str) -> Optional[Dict]:
        """
        Get team information by name

        Args:
            team_name: Name of the team

        Returns:
            Team information dictionary or None
        """
        return self.teams.get(team_name)

    def get_driver_name(self, team_name: str, car_number: int, year: str) -> Optional[str]:
        """
        Get driver name from team, car number, and year

        Args:
            team_name: Name of the team
            car_number: Car number
            year: Year

        Returns:
            Driver name or None
        """
        team_info = self.get_team_info(team_name)
        if not team_info:
            return None

        if year not in team_info['drivers']:
            return None

        drivers = team_info['drivers'][year]
        return drivers.get(str(car_number))

    def extract_year_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract year from filename

        Args:
            filename: Name of the file

        Returns:
            Year as string or None
        """
        # Try to match year patterns
        year_patterns = [
            r'202[0-4]',  # Direct year match
            r'(2020|2021|2022|2023|2024)-infr',  # Year-infringement pattern
        ]

        for pattern in year_patterns:
            match = re.search(pattern, filename)
            if match:
                year = match.group(0)[:4]
                return year

        return None
