"""
Customer Insights Analysis - Organization 179

This script analyzes resident incident and weight data across elder care communities
to provide insights and flag communities that may need closer observation.

NOTE: Each row in the Incidents table represents ONE unique incident, even though
      the 'type' field may contain multiple incident types (e.g., both FALL and INJURY).
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
# The expected filename contains a single space between 'Case' and '-'
EXCEL_FILENAME = 'Data Analyst Case - Dataset.xlsx'
incidents_df = pd.read_excel(EXCEL_FILENAME, sheet_name='Incidents')
weight_df = pd.read_excel(EXCEL_FILENAME, sheet_name='Weight')

print(f"Incidents data: {len(incidents_df)} rows")
print(f"Weight data: {len(weight_df)} rows")

# ============================================================================
# DATA CLEANING & PREPROCESSING
# ============================================================================

def parse_json_field(field):
    """Parse JSON string fields into lists"""
    # Defensive: field may be NaN, already a list, or a string with single quotes
    if field is None or (isinstance(field, float) and np.isnan(field)):
        return []
    if isinstance(field, list):
        return field
    try:
        s = str(field)
        return json.loads(s.replace("'", '"'))
    except Exception:
        return []


def normalize_datetime(value):
    """Coerce various datetime inputs to a timezone-naive pandas Timestamp.

    Returns pd.NaT when parsing fails. This ensures sorting won't fail when
    mixing tz-aware and tz-naive datetimes in the same column.
    """
    if value is None:
        return pd.NaT
    try:
        ts = pd.to_datetime(value, errors='coerce', utc=True)
        if pd.isna(ts):
            return pd.NaT
        # Convert to python datetime and drop tzinfo to get a naive Timestamp
        py = ts.to_pydatetime()
        py = py.replace(tzinfo=None)
        return pd.Timestamp(py)
    except Exception:
        return pd.NaT

# Create unique incident ID for each row (each row = one incident)
incidents_df['incident_id'] = range(1, len(incidents_df) + 1)

# Parse incident types (stored as JSON strings)
# NOTE: One incident can have multiple types (e.g., FALL + INJURY)
incidents_df['type_list'] = incidents_df['type'].apply(parse_json_field)
# Normalize occurred_at to timezone-naive Timestamps
incidents_df['occurred_at'] = incidents_df['occurred_at'].apply(normalize_datetime)

# Categorize falls: both FALL and ON_FLOOR are types of falls
def categorize_fall_incident(type_list):
    """Determine if incident involves a fall (FALL or ON_FLOOR)"""
    fall_types = ['INCIDENT_TYPE_FALL', 'INCIDENT_TYPE_ON_FLOOR']
    return any(t in fall_types for t in type_list)

incidents_df['is_fall_related'] = incidents_df['type_list'].apply(categorize_fall_incident)

# Parse weight measurements (stored as JSON strings)
weight_df['dates_list'] = weight_df['dates_recorded'].apply(parse_json_field)
weight_df['weights_list'] = weight_df['weight_lbs'].apply(parse_json_field)

# Convert weights to numeric
weight_df['weights_numeric'] = weight_df['weights_list'].apply(
    lambda x: [float(w) for w in x] if isinstance(x, list) else []
)

print("\nData cleaning complete!")

# ============================================================================
# ORGANIZATION ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("ORGANIZATION ANALYSIS")
print("="*70)

# Overall statistics
total_orgs = incidents_df['organization'].nunique()
total_communities = incidents_df['community'].nunique()
total_residents = incidents_df['resident_id'].nunique()

print("\n1. OVERALL STRUCTURE")
print("-" * 40)
print(f"Total Organizations: {total_orgs}")
print(f"Total Communities: {total_communities}")
print(f"Total Residents (with incidents): {total_residents}")

# Residents by community
print("\n2. RESIDENTS PER COMMUNITY")
print("-" * 40)
residents_by_community = incidents_df.groupby('community')['resident_id'].nunique().sort_values(ascending=False)
print(f"Average residents per community: {residents_by_community.mean():.1f}")
print(f"Median residents per community: {residents_by_community.median():.1f}")
print(f"Range: {residents_by_community.min()} to {residents_by_community.max()} residents")
print(f"\nTop 10 communities by resident count:")
print(residents_by_community.head(10))

# Check for residents who moved between communities
print("\n3. RESIDENT MOBILITY ANALYSIS (Community Transfers)")
print("-" * 40)

# Combine incidents and weight data to track all observations
all_observations = []

# Add incident observations
for idx, row in incidents_df.iterrows():
    all_observations.append({
        'resident_id': row['resident_id'],
        'community': row['community'],
        'date': row['occurred_at'],
        'source': 'incident'
    })

# Add weight observations
for idx, row in weight_df.iterrows():
    if len(row['dates_list']) > 0:
        for date_str in row['dates_list']:
            all_observations.append({
                'resident_id': row['resident_id'],
                'community': row['community'],
                'date': normalize_datetime(date_str),
                'source': 'weight'
            })

# Create observations dataframe
obs_df = pd.DataFrame(all_observations)
obs_df = obs_df.sort_values(['resident_id', 'date'])

# Identify residents with multiple communities
residents_communities = obs_df.groupby('resident_id')['community'].nunique()
residents_who_moved = residents_communities[residents_communities > 1].index.tolist()

if len(residents_who_moved) > 0:
    print(f"⚠️  {len(residents_who_moved)} resident(s) appear in multiple communities")
    print(f"   This may indicate community transfers/moves")
    print(f"\n   Details of residents with multiple communities:")
    
    for res_id in residents_who_moved[:10]:  # Show first 10
        res_obs = obs_df[obs_df['resident_id'] == res_id].sort_values('date')
        communities = res_obs['community'].unique()
        first_obs = res_obs.iloc[0]
        last_obs = res_obs.iloc[-1]
        
        print(f"\n   Resident {res_id}:")
        print(f"      Communities: {list(communities)}")
        print(f"      First observation: Community {first_obs['community']} on {first_obs['date'].strftime('%Y-%m-%d')}")
        print(f"      Last observation: Community {last_obs['community']} on {last_obs['date'].strftime('%Y-%m-%d')}")
        print(f"      Total observations: {len(res_obs)}")
    
    if len(residents_who_moved) > 10:
        print(f"\n   ... and {len(residents_who_moved) - 10} more residents with multiple communities")
else:
    print("✓ No residents appear to have moved between communities")
    print("  All residents consistently observed in a single community")

print("\n" + "-" * 40)

# ============================================================================
# INCIDENT ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("INCIDENT ANALYSIS")
print("="*70)

print(f"\n📊 TOTAL UNIQUE INCIDENTS: {len(incidents_df)}")
print(f"📊 FALL-RELATED INCIDENTS: {incidents_df['is_fall_related'].sum()} ({incidents_df['is_fall_related'].sum()/len(incidents_df)*100:.1f}%)")

# Expand incident types for detailed analysis
# NOTE: This creates multiple rows per incident if an incident has multiple types
incident_types_expanded = []
for idx, row in incidents_df.iterrows():
    for inc_type in row['type_list']:
        # Defensive string operations: some fields may be NaN or non-strings
        safe_type = str(inc_type).replace('INCIDENT_TYPE_', '') if pd.notna(inc_type) else ''
        raw_location = row['location'] if pd.notna(row['location']) else ''
        safe_location = str(raw_location).replace('INCIDENT_LOCATION_', '')

        incident_types_expanded.append({
            'incident_id': row['incident_id'],
            'resident_id': row['resident_id'],
            'organization': row['organization'],
            'community': row['community'],
            'occurred_at': row['occurred_at'],
            'type': safe_type,
            'location': safe_location,
            'is_fall_related': row['is_fall_related']
        })

incidents_expanded_df = pd.DataFrame(incident_types_expanded)

# Overall incident type summary
print("\n1. INCIDENT TYPE DISTRIBUTION")
print("-" * 40)
print("Note: Incidents can have multiple types, so percentages may exceed 100%")
print()
incident_type_counts = incidents_expanded_df['type'].value_counts()
for inc_type, count in incident_type_counts.items():
    pct = (count / len(incidents_df)) * 100  # Percentage of unique incidents
    is_fall = "🔴 FALL TYPE" if inc_type in ['FALL', 'ON_FLOOR'] else ""
    print(f"{inc_type:20s}: {count:4d} ({pct:5.1f}% of incidents) {is_fall}")

# Incidents by community (counting unique incidents)
print("\n2. INCIDENTS BY COMMUNITY (Unique Incident Count)")
print("-" * 40)
community_incidents = incidents_df.groupby('community').agg({
    'incident_id': 'count',  # Count unique incidents
    'is_fall_related': 'sum',  # Count fall-related incidents
    'occurred_at': lambda x: x.min().strftime('%Y-%m-%d')
}).rename(columns={
    'incident_id': 'total_incidents',
    'is_fall_related': 'fall_incidents',
    'occurred_at': 'earliest_incident'
})

# Get unique residents per community
unique_residents = incidents_df.groupby('community')['resident_id'].nunique().rename('unique_residents')
community_incidents = community_incidents.join(unique_residents)

# Calculate rates
community_incidents['incidents_per_resident'] = (
    community_incidents['total_incidents'] / community_incidents['unique_residents']
).round(2)
community_incidents['fall_rate'] = (
    community_incidents['fall_incidents'] / community_incidents['total_incidents'] * 100
).round(1)

community_incidents = community_incidents.sort_values('total_incidents', ascending=False)
print(community_incidents.head(10))

# High-risk incident types (focusing on falls and serious events)
print("\n3. FALL ANALYSIS (FALL + ON_FLOOR combined)")
print("-" * 40)
fall_incidents = incidents_df[incidents_df['is_fall_related'] == True]
print(f"Total fall-related incidents: {len(fall_incidents)}")
print(f"Percentage of all incidents: {len(fall_incidents)/len(incidents_df)*100:.1f}%")
print(f"Communities affected: {fall_incidents['community'].nunique()}")
print(f"Residents affected: {fall_incidents['resident_id'].nunique()}")

# Fall breakdown by specific type
fall_type_breakdown = incidents_expanded_df[
    incidents_expanded_df['type'].isin(['FALL', 'ON_FLOOR'])
]['type'].value_counts()
print(f"\nBreakdown:")
for fall_type, count in fall_type_breakdown.items():
    print(f"  {fall_type}: {count} occurrences")

# Other high-risk incidents
print("\n4. OTHER HIGH-RISK INCIDENTS")
print("-" * 40)
high_risk_types = ['MEDICAL_EMERGENCY', 'HOSPITALIZATION', 'ER_VISIT', 'INJURY']
high_risk_incidents = incidents_df[
    incidents_df['type_list'].apply(
        lambda x: any(f'INCIDENT_TYPE_{t}' in x for t in high_risk_types)
    )
]

print(f"Medical emergencies/hospitalizations: {len(high_risk_incidents)}")
print(f"Percentage of all incidents: {len(high_risk_incidents)/len(incidents_df)*100:.1f}%")

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
                'date': normalize_datetime(date),
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
print("COMBINED ANALYSIS: INCIDENTS + WEIGHT BY COMMUNITY")
print("="*70)

# Merge incident and weight data by community
community_analysis = community_incidents.copy()
community_analysis = community_analysis.join(
    community_weight_concerns[['residents_with_loss', 'pct_change']],
    how='outer'
).fillna(0)

# Sort by total incidents (can be re-sorted later based on your risk model)
community_analysis = community_analysis.sort_values('total_incidents', ascending=False)

print("\nCOMMUNITY SUMMARY (Top 15 by Incident Count)")
print("-" * 80)
display_cols = [
    'total_incidents', 'fall_incidents', 'unique_residents', 
    'incidents_per_resident', 'fall_rate', 'residents_with_loss'
]
print(community_analysis[display_cols].head(15))

print("\n\nCOMMUNITY SUMMARY (Top 15 by Incidents Per Resident)")
print("-" * 80)
print(community_analysis[display_cols].sort_values('incidents_per_resident', ascending=False).head(15))

# ============================================================================
# KEY INSIGHTS & RECOMMENDATIONS
# ============================================================================

print("\n" + "="*70)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*70)

print("\n1. INCIDENT OVERVIEW:")
print(f"   - Total unique incidents: {len(incidents_df)}")
print(f"   - Communities affected: {incidents_df['community'].nunique()}")
print(f"   - Residents involved: {incidents_df['resident_id'].nunique()}")
print(f"   - Average incidents per community: {len(incidents_df) / incidents_df['community'].nunique():.1f}")

print("\n2. FALL INCIDENT PATTERNS:")
fall_pct = (incidents_df['is_fall_related'].sum() / len(incidents_df)) * 100
print(f"   - {incidents_df['is_fall_related'].sum()} fall-related incidents (FALL + ON_FLOOR)")
print(f"   - Represents {fall_pct:.1f}% of all incidents")
print(f"   - {fall_type_breakdown.get('FALL', 0)} witnessed falls, {fall_type_breakdown.get('ON_FLOOR', 0)} found on floor")
print(f"   - Recommend comprehensive fall prevention program")

print("\n3. WEIGHT MONITORING:")
print(f"   - {len(significant_loss)} residents show concerning weight loss")
if len(significant_loss) > 0:
    print(f"   - Average loss: {significant_loss['weight_change_lbs'].mean():.1f} lbs over {significant_loss['days_tracked'].mean():.0f} days")
print(f"   - Recommend nutritional assessment for affected residents")

print("\n4. ORGANIZATIONAL STRUCTURE:")
orgs_count = incidents_df['organization'].nunique()
communities_count = incidents_df['community'].nunique()
communities_per_org = incidents_df.groupby('organization')['community'].nunique().mean()
print(f"   - {orgs_count} organizations managing {communities_count} communities")
print(f"   - Average of {communities_per_org:.1f} communities per organization")
print(f"   - Consider regional coordination for best practices sharing")

print("\n5. DATA METHODOLOGY NOTES:")
print(f"   ✓ Each row in incidents data = 1 unique incident (Total: {len(incidents_df)} incidents)")
print(f"   ✓ Some incidents have multiple types (e.g., FALL + INJURY)")
print(f"   ✓ FALL and ON_FLOOR both classified as fall-related incidents")
print(f"   ✓ Analysis provides raw metrics for custom risk scoring")

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