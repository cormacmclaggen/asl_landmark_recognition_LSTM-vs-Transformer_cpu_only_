#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Create sign to prediction index mapping JSON
import pandas as pd
import json

# Load the ASL training CSV from Kaggle
train = pd.read_csv("/home/lananh/GISLR/train.csv")

# Get the list of unique signs sorted alphabetically
signs = sorted(train['sign'].unique())

# Create mapping: sign → index
sign2idx = {sign: i for i, sign in enumerate(signs)}

# Save to JSON
with open("sign_to_prediction_index_map.json", "w") as f:
    json.dump(sign2idx, f, indent=4)

print("Created sign_to_prediction_index_map.json with", len(sign2idx), "signs.")
