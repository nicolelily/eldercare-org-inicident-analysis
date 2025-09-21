"""
Customer Insights Analysis for Healthcare Communities
Data Analyst Take-Home Assignment

This script analyzes resident incident and weight data across healthcare communities
to provide insights and flag communities that may need closer observation.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
# NOTE: actual file in repo is 'Data Analyst Case - Dataset.xlsx' (single hyphen). Use exact filename.
incidents_df = pd.read_excel('Data Analyst Case - Dataset.xlsx', sheet_name='Incidents')
weight_df = pd.read_excel('Data Analyst Case - Dataset.xlsx', sheet_name='Weight')

print(f"Incidents data: {len(incidents_df)} rows")
print(f"Weight data: {len(weight_df)} rows")

# ============================================================================
# DATA CLEANING & PREPROCESSING
# ============================================================================

def parse_json_field(field):
    """Parse JSON string fields into lists"""
    try:
        return json.loads(field.replace("'", '"'))
    except:
        return []

# Parse incident types (stored as JSON strings)
incidents_df['type_list'] = incidents_df['type'].apply(parse_json_field)
incidents_df['occurred_at'] = pd.to_datetime(incidents_df['occurred_at'])

# Parse weight measurements (stored as JSON strings)
weight_df['dates_list'] = weight_df['dates_recorded'].apply(parse_json_field)
weight_df['weights_list'] = weight_df['weight_lbs'].apply(parse_json_field)

# Convert weights to numeric
weight_df['weights_numeric'] = weight_df['weights_list'].apply(
    lambda x: [float(w) for w in x] if isinstance(x, list) else []
)

print("\nData cleaning complete!")

# ============================================================================
# INCIDENT ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("INCIDENT ANALYSIS")
print("="*70)

# Expand incident types for analysis (defensive: handle missing or non-string fields)
incident_types_expanded = []
for idx, row in incidents_df.iterrows():
    types = row.get('type_list') if 'type_list' in row.index else []
    if not isinstance(types, list):
        types = []
    for inc_type in types:
        # Skip empty or null types
        if not inc_type:
            continue
        # Ensure strings before replace
        inc_type_str = str(inc_type)
        location_str = str(row['location']) if not pd.isna(row.get('location')) else ''

        incident_types_expanded.append({
            'resident_id': row['resident_id'],
            'organization': row['organization'],
            'community': row['community'],
            'occurred_at': row['occurred_at'],
            'type': inc_type_str.replace('INCIDENT_TYPE_', ''),
            'location': location_str.replace('INCIDENT_LOCATION_', '')
        })

incidents_expanded_df = pd.DataFrame(incident_types_expanded)

# Overall incident summary
print("\n1. INCIDENT TYPE DISTRIBUTION")
print("-" * 40)
incident_type_counts = incidents_expanded_df['type'].value_counts()
for inc_type, count in incident_type_counts.items():
    pct = (count / len(incidents_expanded_df)) * 100
    print(f"{inc_type:20s}: {count:4d} ({pct:5.1f}%)")

# Incidents by community
print("\n2. INCIDENTS BY COMMUNITY")
print("-" * 40)
community_incidents = incidents_df.groupby('community').agg({
    'resident_id': 'count',
    'occurred_at': lambda x: x.min().strftime('%Y-%m-%d')
}).rename(columns={'resident_id': 'incident_count', 'occurred_at': 'earliest_incident'})

# Get unique residents per community
unique_residents = incidents_df.groupby('community')['resident_id'].nunique().rename('unique_residents')
community_incidents = community_incidents.join(unique_residents)
community_incidents['incidents_per_resident'] = (
    community_incidents['incident_count'] / community_incidents['unique_residents']
).round(2)

community_incidents = community_incidents.sort_values('incident_count', ascending=False)
print(community_incidents.head(10))

# High-risk incident types
print("\n3. HIGH-RISK INCIDENT ANALYSIS (Falls, Medical Emergencies, Hospitalizations)")
print("-" * 40)
high_risk_types = ['FALL', 'MEDICAL_EMERGENCY', 'HOSPITALIZATION', 'ON_FLOOR']
high_risk_incidents = incidents_expanded_df[incidents_expanded_df['type'].isin(high_risk_types)]

high_risk_by_community = high_risk_incidents.groupby('community').agg({
    'resident_id': 'count',
    'type': lambda x: x.value_counts().to_dict()
}).rename(columns={'resident_id': 'high_risk_count'})

high_risk_by_community = high_risk_by_community.sort_values('high_risk_count', ascending=False)
print(high_risk_by_community.head(10))

# ============================================================================
# WEIGHT ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("WEIGHT ANALYSIS")
print("="*70)

# Expand weight measurements for analysis
weight_measurements = []
for idx, row in weight_df.iterrows():
    if len(row['dates_list']) > 0 and len(row['weights_numeric']) > 0:
        for date, weight in zip(row['dates_list'], row['weights_numeric']):
            weight_measurements.append({
                'resident_id': row['resident_id'],
                'organization': row['organization'],
                'community': row['community'],
                'date': pd.to_datetime(date),
                'weight_lbs': weight
            })

weights_expanded_df = pd.DataFrame(weight_measurements)

# Calculate weight changes per resident
print("\n1. WEIGHT CHANGE ANALYSIS")
print("-" * 40)

weight_changes = []
for resident_id in weights_expanded_df['resident_id'].unique():
    resident_weights = weights_expanded_df[weights_expanded_df['resident_id'] == resident_id].sort_values('date')
    
    if len(resident_weights) >= 2:
        first_weight = resident_weights.iloc[0]['weight_lbs']
        last_weight = resident_weights.iloc[-1]['weight_lbs']
        weight_change = last_weight - first_weight
        pct_change = (weight_change / first_weight) * 100
        
        weight_changes.append({
            'resident_id': resident_id,
            'community': resident_weights.iloc[0]['community'],
            'first_weight': first_weight,
            'last_weight': last_weight,
            'weight_change_lbs': weight_change,
            'pct_change': pct_change,
            'num_measurements': len(resident_weights),
            'days_tracked': (resident_weights.iloc[-1]['date'] - resident_weights.iloc[0]['date']).days
        })

weight_changes_df = pd.DataFrame(weight_changes)

# Flag significant weight loss (>5% loss or >10 lbs loss)
significant_loss = weight_changes_df[
    (weight_changes_df['pct_change'] < -5) | (weight_changes_df['weight_change_lbs'] < -10)
]

print(f"Residents with significant weight loss: {len(significant_loss)}")
print(f"Average weight loss in this group: {significant_loss['weight_change_lbs'].mean():.1f} lbs")
print(f"Average percentage loss: {significant_loss['pct_change'].mean():.1f}%")

# Communities with most weight loss concerns
print("\n2. COMMUNITIES WITH WEIGHT LOSS CONCERNS")
print("-" * 40)
community_weight_concerns = significant_loss.groupby('community').agg({
    'resident_id': 'count',
    'weight_change_lbs': 'mean',
    'pct_change': 'mean'
}).rename(columns={'resident_id': 'residents_with_loss'})

community_weight_concerns = community_weight_concerns.sort_values('residents_with_loss', ascending=False)
print(community_weight_concerns.head(10))

# ============================================================================
# COMBINED ANALYSIS: INCIDENTS + WEIGHT
# ============================================================================

print("\n" + "="*70)
print("COMBINED ANALYSIS: COMMUNITIES REQUIRING ATTENTION")
print("="*70)

# Merge incident and weight data by community
community_analysis = community_incidents.copy()
community_analysis = community_analysis.join(
    community_weight_concerns[['residents_with_loss', 'pct_change']],
    how='outer'
).fillna(0)

# Create risk score (normalized)
community_analysis['incident_score'] = (
    community_analysis['incidents_per_resident'] / community_analysis['incidents_per_resident'].max()
) * 100

community_analysis['weight_score'] = (
    community_analysis['residents_with_loss'] / community_analysis['unique_residents'].fillna(1)
) * 100

community_analysis['combined_risk_score'] = (
    community_analysis['incident_score'] * 0.6 + 
    community_analysis['weight_score'] * 0.4
).round(1)

# Sort by combined risk score
community_analysis = community_analysis.sort_values('combined_risk_score', ascending=False)

print("\nTOP 10 COMMUNITIES REQUIRING CLOSE OBSERVATION")
print("-" * 70)
print(community_analysis[[
    'incident_count', 'unique_residents', 'incidents_per_resident',
    'residents_with_loss', 'combined_risk_score'
]].head(10))

# ============================================================================
# KEY INSIGHTS & RECOMMENDATIONS
# ============================================================================

print("\n" + "="*70)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*70)

print("\n1. HIGH-PRIORITY COMMUNITIES:")
top_5_communities = community_analysis.head(5).index.tolist()
print(f"   Communities {top_5_communities} require immediate attention")
print(f"   - High incident rates combined with weight loss concerns")
print(f"   - Recommend weekly monitoring and staff intervention review")

print("\n2. INCIDENT PATTERNS:")
print(f"   - Most common incident: {incident_type_counts.index[0]}")
print(f"   - {(incident_type_counts['FALL'] / incident_type_counts.sum() * 100):.1f}% of incidents are falls")
print(f"   - Recommend fall prevention program implementation")

print("\n3. WEIGHT MONITORING:")
print(f"   - {len(significant_loss)} residents show concerning weight loss")
print(f"   - Average loss: {significant_loss['weight_change_lbs'].mean():.1f} lbs over {significant_loss['days_tracked'].mean():.0f} days")
print(f"   - Recommend nutritional assessment for affected residents")

print("\n4. ORGANIZATIONAL STRUCTURE:")
orgs_count = incidents_df['organization'].nunique()
communities_per_org = incidents_df.groupby('organization')['community'].nunique().mean()
print(f"   - {orgs_count} organizations managing an average of {communities_per_org:.1f} communities")
print(f"   - Consider regional coordination for best practices sharing")

# Save summary results
print("\n" + "="*70)
print("Saving results to CSV files...")
print("="*70)

community_analysis.to_csv('community_risk_analysis.csv')
incidents_expanded_df.to_csv('incidents_detailed.csv', index=False)
weights_expanded_df.to_csv('weights_detailed.csv', index=False)
if len(weight_changes_df) > 0:
    weight_changes_df.to_csv('weight_changes_summary.csv', index=False)

print("\nAnalysis complete! Files saved:")
print("  - community_risk_analysis.csv")
print("  - incidents_detailed.csv")
print("  - weights_detailed.csv")
print("  - weight_changes_summary.csv")
