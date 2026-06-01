import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import json

class AdvancedAnalytics:
    def __init__(self, df, numeric_cols, date_cols):
        self.df = df
        self.numeric_cols = numeric_cols
        self.date_cols = date_cols

    def profile_data(self):
        profile = {
            'basic': {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'duplicates': int(self.df.duplicated().sum()),
                'missing_cells': int(self.df.isnull().sum().sum())
            },
            'numeric_stats': {},
            'categorical_stats': {},
            'correlations': {}
        }
        for col in self.numeric_cols:
            profile['numeric_stats'][col] = {
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'missing': int(self.df[col].isnull().sum())
            }
        cat_cols = [c for c in self.df.columns if c not in self.numeric_cols and c not in self.date_cols]
        for col in cat_cols[:5]:
            top = self.df[col].value_counts().head(3).to_dict()
            profile['categorical_stats'][col] = {'top_values': top, 'unique': int(self.df[col].nunique())}
        return profile

    def perform_clustering(self, n_clusters=3):
        if len(self.numeric_cols) < 2:
            return None
        data = self.df[self.numeric_cols].dropna()
        if len(data) < n_clusters:
            return None
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        return {
            'labels': clusters.tolist(),
            'centers': {f'Cluster_{i}': {self.numeric_cols[j]: float(centers[i][j]) for j in range(len(self.numeric_cols))} for i in range(n_clusters)},
            'inertia': float(kmeans.inertia_)
        }

    def forecast_simple(self, target_col, periods=3):
        if not self.date_cols or target_col not in self.numeric_cols:
            return None
        date_col = self.date_cols[0]
        try:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
        except:
            return None
        df_sorted = self.df[[date_col, target_col]].dropna().sort_values(date_col)
        if len(df_sorted) < 3:
            return None
        df_sorted['time_idx'] = range(len(df_sorted))
        X = df_sorted[['time_idx']].values
        y = df_sorted[target_col].values
        model = LinearRegression()
        model.fit(X, y)
        last_idx = df_sorted['time_idx'].iloc[-1]
        future_idx = np.array([[last_idx + i+1] for i in range(periods)])
        forecast = model.predict(future_idx)
        last_date = df_sorted[date_col].iloc[-1]
        future_dates = [last_date + timedelta(days=30*(i+1)) for i in range(periods)]
        return {
            'target': target_col,
            'forecast_dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'forecast_values': [float(v) for v in forecast],
            'r2': float(model.score(X, y))
        }