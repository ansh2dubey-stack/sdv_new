import numpy as np


class InfluenceTracker:
    """
    Tracks influence of each training row.
    """

    def __init__(self):
        self.row_influence = {}

    def compute(self, data):

        import numpy as np
        import pandas as pd

    # convert categorical columns to numeric codes
        numeric_data = data.copy()

        for col in numeric_data.columns:
            if numeric_data[col].dtype == "object":
                numeric_data[col] = numeric_data[col].astype("category").cat.codes

        influence_map = {}

        for idx, row in numeric_data.iterrows():
            vector = np.array(row.values, dtype=float)
            influence_map[idx] = vector

        return influence_map