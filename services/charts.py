import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

class ChartGenerator:
    def __init__(self, df, numeric_cols, categorical_cols, date_cols):
        self.df = df
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.date_cols = date_cols

    def create_histograms(self, max_cols=6):
        """Create histograms for numeric columns (returns HTML div)."""
        if not self.numeric_cols:
            return None
        cols_to_plot = self.numeric_cols[:max_cols]
        fig = px.histogram(self.df, x=cols_to_plot, facet_col='variable', facet_col_wrap=3,
                           title='Distribution of Numeric Features', 
                           labels={'value': 'Value', 'count': 'Frequency'})
        fig.update_layout(height=500)
        return fig.to_html(full_html=False)

    def create_correlation_heatmap(self, correlations):
        """Create correlation heatmap."""
        if not correlations:
            return None
        # Build matrix from correlations dict (simplified)
        # For full matrix we'd need original df, but we'll use the stored pairs
        # Instead generate a simple bar chart of top correlations
        if not correlations:
            return None
        pairs = list(correlations.keys())
        values = list(correlations.values())
        fig = px.bar(x=values, y=pairs, orientation='h', 
                     title='Top Feature Correlations',
                     labels={'x': 'Correlation Coefficient', 'y': 'Feature Pair'},
                     color=values, color_continuous_scale='RdBu')
        fig.update_layout(height=max(400, len(pairs)*30))
        return fig.to_html(full_html=False)

    def create_boxplots(self, max_cols=8):
        """Create boxplots for numeric columns."""
        if not self.numeric_cols:
            return None
        cols_to_plot = self.numeric_cols[:max_cols]
        fig = px.box(self.df, y=cols_to_plot, title='Boxplots for Numeric Features')
        fig.update_layout(height=500)
        return fig.to_html(full_html=False)

    def create_time_series(self):
        """Create time series line chart if date column present."""
        if not self.date_cols or not self.numeric_cols:
            return None
        date_col = self.date_cols[0]
        numeric_col = self.numeric_cols[0] if self.numeric_cols else None
        if numeric_col:
            fig = px.line(self.df, x=date_col, y=numeric_col, 
                          title=f'Trend: {numeric_col} over Time')
            return fig.to_html(full_html=False)
        return None