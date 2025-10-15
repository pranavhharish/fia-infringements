"""
Team-Year Summary Generation Module
Generates comprehensive summaries for each team per year
"""

import logging
from typing import List, Dict
from collections import defaultdict
import pandas as pd


class TeamYearSummarizer:
    """Generate team-year summaries from infringement data"""

    def __init__(self, abstractive_summarizer=None):
        """
        Initialize team-year summarizer

        Args:
            abstractive_summarizer: Optional AbstractiveSummarizer instance
        """
        self.abstractive_summarizer = abstractive_summarizer
        self.logger = logging.getLogger(__name__)

    def aggregate_team_data(self, records: List[Dict], team: str, year: str) -> Dict:
        """
        Aggregate data for a specific team and year

        Args:
            records: List of infringement records
            team: Team name
            year: Year

        Returns:
            Aggregated statistics
        """
        # Filter records
        team_year_records = [
            r for r in records
            if r.get('team') == team and r.get('year') == year
        ]

        if not team_year_records:
            return None

        # Aggregate statistics
        stats = {
            'team': team,
            'year': year,
            'total_infractions': len(team_year_records),
            'infraction_types': {},
            'penalty_types': {},
            'drivers': set(),
            'races': set(),
            'documents': []
        }

        for record in team_year_records:
            # Count infraction types
            for infraction in record.get('infraction_types', []):
                stats['infraction_types'][infraction] = stats['infraction_types'].get(infraction, 0) + 1

            # Count penalty types
            penalty_type = record.get('penalty_type')
            if penalty_type:
                stats['penalty_types'][penalty_type] = stats['penalty_types'].get(penalty_type, 0) + 1

            # Collect drivers
            if record.get('driver'):
                stats['drivers'].add(record['driver'])

            # Collect races
            if record.get('race'):
                stats['races'].add(record['race'])

            # Store document info
            stats['documents'].append({
                'filename': record.get('filename'),
                'text': record.get('decision_text', ''),
                'infraction': record.get('primary_infraction'),
                'penalty': record.get('penalty_description')
            })

        # Convert sets to lists
        stats['drivers'] = list(stats['drivers'])
        stats['races'] = list(stats['races'])

        return stats

    def generate_factual_summary(self, stats: Dict) -> List[str]:
        """
        Generate factual summary from statistics

        Args:
            stats: Aggregated statistics

        Returns:
            List of summary statements
        """
        summary = []

        # Total infractions
        summary.append(
            f"{stats['team']} received {stats['total_infractions']} infringement(s) in {stats['year']}."
        )

        # Most common infraction
        if stats['infraction_types']:
            top_infraction = max(stats['infraction_types'].items(), key=lambda x: x[1])
            infraction_name = top_infraction[0].replace('_', ' ')
            summary.append(
                f"The most common infraction type was {infraction_name} ({top_infraction[1]} occurrence(s))."
            )

        # Most common penalty
        if stats['penalty_types']:
            top_penalty = max(stats['penalty_types'].items(), key=lambda x: x[1])
            penalty_name = top_penalty[0].replace('_', ' ')
            summary.append(
                f"The most frequent penalty was {penalty_name} ({top_penalty[1]} time(s))."
            )

        # Drivers involved
        if stats['drivers']:
            if len(stats['drivers']) == 1:
                summary.append(f"Driver involved: {stats['drivers'][0]}.")
            else:
                summary.append(f"Drivers involved: {', '.join(stats['drivers'])}.")

        # Number of races affected
        if stats['races']:
            summary.append(f"Infractions occurred across {len(stats['races'])} race(s).")

        return summary

    def generate_abstractive_summary(self, stats: Dict, num_insights: int = 5) -> List[str]:
        """
        Generate abstractive summary using transformer model

        Args:
            stats: Aggregated statistics
            num_insights: Number of insights to generate

        Returns:
            List of insights
        """
        if not self.abstractive_summarizer:
            return self.generate_factual_summary(stats)

        # Combine all document texts
        all_texts = [doc['text'] for doc in stats['documents'] if doc['text']]

        if not all_texts:
            return self.generate_factual_summary(stats)

        # Generate insights using abstractive summarizer
        insights = self.abstractive_summarizer.generate_insights(all_texts, num_insights)

        return insights

    def generate_team_year_summary(
        self,
        records: List[Dict],
        team: str,
        year: str,
        style: str = 'factual',
        num_insights: int = 5
    ) -> Dict:
        """
        Generate complete team-year summary

        Args:
            records: List of infringement records
            team: Team name
            year: Year
            style: Summary style ('factual' or 'abstractive')
            num_insights: Number of insights

        Returns:
            Complete summary dictionary
        """
        # Aggregate data
        stats = self.aggregate_team_data(records, team, year)

        if not stats:
            return None

        # Generate summary based on style
        if style == 'abstractive' and self.abstractive_summarizer:
            insights = self.generate_abstractive_summary(stats, num_insights)
        else:
            insights = self.generate_factual_summary(stats)

        summary = {
            'team': team,
            'year': year,
            'statistics': {
                'total_infractions': stats['total_infractions'],
                'infraction_types': stats['infraction_types'],
                'penalty_types': stats['penalty_types'],
                'drivers': stats['drivers'],
                'races_affected': len(stats['races'])
            },
            'insights': insights,
            'summary_text': ' '.join(insights)
        }

        return summary

    def generate_all_summaries(
        self,
        records: List[Dict],
        style: str = 'factual',
        num_insights: int = 5
    ) -> List[Dict]:
        """
        Generate summaries for all team-year combinations

        Args:
            records: List of all infringement records
            style: Summary style
            num_insights: Number of insights per summary

        Returns:
            List of all team-year summaries
        """
        # Group records by team and year
        team_year_groups = defaultdict(list)

        for record in records:
            team = record.get('team')
            year = record.get('year')
            if team and year:
                key = (team, year)
                team_year_groups[key].append(record)

        # Generate summaries
        all_summaries = []

        for (team, year), group_records in team_year_groups.items():
            self.logger.info(f"Generating summary for {team} - {year}")

            summary = self.generate_team_year_summary(
                records,
                team,
                year,
                style,
                num_insights
            )

            if summary:
                all_summaries.append(summary)

        return all_summaries

    def export_summaries_to_csv(self, summaries: List[Dict], output_path: str):
        """
        Export summaries to CSV

        Args:
            summaries: List of summaries
            output_path: Output file path
        """
        rows = []

        for summary in summaries:
            row = {
                'team': summary['team'],
                'year': summary['year'],
                'total_infractions': summary['statistics']['total_infractions'],
                'races_affected': summary['statistics']['races_affected'],
                'drivers': ', '.join(summary['statistics']['drivers']),
                'summary': summary['summary_text']
            }

            # Add top 3 infraction types
            top_infractions = sorted(
                summary['statistics']['infraction_types'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            for i, (infraction, count) in enumerate(top_infractions, 1):
                row[f'top_infraction_{i}'] = infraction
                row[f'top_infraction_{i}_count'] = count

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Summaries exported to {output_path}")
