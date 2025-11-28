"""
Compute and save location frequency statistics for enhanced modeling.
"""
import os
import pickle
import numpy as np
from collections import Counter
import json

def compute_location_stats():
    """Compute location frequency statistics from training data."""
    
    # Load training data
    with open('data/geolife/geolife_transformer_7_train.pk', 'rb') as f:
        train_data = pickle.load(f)
    
    # Compute location frequencies
    all_locs = []
    for sample in train_data:
        all_locs.extend(sample['X'].tolist())
        all_locs.append(sample['Y'])
    
    loc_freq = Counter(all_locs)
    
    # Create frequency mapping (location_id -> log frequency)
    max_freq = max(loc_freq.values())
    freq_map = {}
    for loc_id, freq in loc_freq.items():
        # Use log frequency to reduce dynamic range
        freq_map[int(loc_id)] = np.log(freq + 1) / np.log(max_freq + 1)
    
    # Save to JSON
    with open('data/location_freq.json', 'w') as f:
        json.dump(freq_map, f)
    
    print(f"Computed frequency statistics for {len(freq_map)} locations")
    print(f"Max frequency: {max_freq}")
    print(f"Total samples: {len(all_locs)}")
    
    # Compute user-location frequencies
    user_loc_freq = {}
    for sample in train_data:
        user = int(sample['user_X'][0])
        if user not in user_loc_freq:
            user_loc_freq[user] = Counter()
        
        user_loc_freq[user].update(sample['X'].tolist())
        user_loc_freq[user].update([sample['Y']])
    
    # Convert to normalized format
    user_freq_map = {}
    for user, counter in user_loc_freq.items():
        max_user_freq = max(counter.values())
        user_freq_map[user] = {}
        for loc_id, freq in counter.items():
            user_freq_map[user][int(loc_id)] = np.log(freq + 1) / np.log(max_user_freq + 1)
    
    with open('data/user_location_freq.json', 'w') as f:
        json.dump(user_freq_map, f)
    
    print(f"Computed user-location frequencies for {len(user_freq_map)} users")
    
    return freq_map, user_freq_map

if __name__ == '__main__':
    compute_location_stats()
