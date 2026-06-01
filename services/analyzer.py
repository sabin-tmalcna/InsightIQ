import pandas as pd
import numpy as np
from scipy import stats

class DataAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        try:
            if self.filepath.endswith('.csv'):
                self.df = pd.read_csv(self.filepath)
            else:
                self.df = pd.read_excel(self.filepath)
            return self.df
        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    def detect_column_types(self):
        if self.df is None:
            return {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        date_cols = []
        for col in self.df.columns:
            if col not in numeric_cols:
                try:
                    pd.to_datetime(self.df[col], errors='raise')
                    date_cols.append(col)
                except:
                    pass
        categorical_cols = [col for col in self.df.columns if col not in numeric_cols and col not in date_cols]
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'date': date_cols,
            'all': self.df.columns.tolist()
        }

    def get_statistics(self, numeric_cols):
        if self.df is None or not numeric_cols:
            return {}
        stats_dict = {}
        for col in numeric_cols:
            series = self.df[col].dropna()
            if len(series) == 0:
                continue
            stats_dict[col] = {
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'q1': float(series.quantile(0.25)),
                'q3': float(series.quantile(0.75)),
                'skew': float(series.skew()),
                'kurtosis': float(series.kurtosis())
            }
        return stats_dict

    def get_correlation_matrix(self, numeric_cols):
        if self.df is None or len(numeric_cols) < 2:
            return {}
        corr_matrix = self.df[numeric_cols].corr()
        correlations = {}
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    correlations[f"{col1} vs {col2}"] = float(round(corr_val, 3))
        correlations = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
        return correlations

    def detect_anomalies(self, numeric_cols, method='iqr'):
        if self.df is None or not numeric_cols:
            return {}
        anomalies = {}
        for col in numeric_cols:
            series = self.df[col].dropna()
            if len(series) == 0:
                continue
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = series[(series < lower_bound) | (series > upper_bound)]
            else:
                z_scores = np.abs(stats.zscore(series))
                outliers = series[z_scores > 3]
            anomalies[col] = len(outliers)  # Python int
        return anomalies