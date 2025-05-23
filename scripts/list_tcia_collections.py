#!/usr/bin/env python3
"""List all available TCIA collections to find COVID datasets"""

from tcia_utils import nbia
import pandas as pd

# Get all collections
print("Fetching all TCIA collections...")
collections_df = nbia.getCollections(format="df")

# Filter for COVID or lung-related collections
print("\nAll available collections:")
for idx, row in collections_df.iterrows():
    collection_name = row['Collection']
    print(f"  - {collection_name}")

print("\nCOVID-related collections:")
covid_collections = collections_df[collections_df['Collection'].str.contains('COVID', case=False, na=False)]
for idx, row in covid_collections.iterrows():
    print(f"  - {row['Collection']}")

print("\nLung/Chest-related collections:")
lung_keywords = ['LUNG', 'CHEST', 'NSCLC', 'THORAX']
for keyword in lung_keywords:
    matching = collections_df[collections_df['Collection'].str.contains(keyword, case=False, na=False)]
    if not matching.empty:
        print(f"\n  {keyword}:")
        for idx, row in matching.iterrows():
            print(f"    - {row['Collection']}")

# Check specific COVID collections for CT data
print("\n\nChecking COVID collections for CT data:")
for collection in covid_collections['Collection']:
    print(f"\nCollection: {collection}")
    try:
        modalities = nbia.getModalityCounts(collection=collection, format="df")
        if 'CT' in modalities['criteria'].values:
            ct_count = modalities[modalities['criteria'] == 'CT']['count'].values[0]
            print(f"  ✓ Has CT data: {ct_count} series")
        else:
            print("  ✗ No CT data")
    except Exception as e:
        print(f"  Error checking modalities: {e}")