import os
import requests
from dotenv import load_dotenv

load_dotenv()

class InsightGenerator:
    def __init__(self, df, stats, correlations, anomalies, summary):
        self.df = df
        self.stats = stats
        self.correlations = correlations
        self.anomalies = anomalies
        self.summary = summary
        self.nvidia_api_key = os.getenv('NVIDIA_API_KEY')
        self.nvidia_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.model = "moonshotai/kimi-k2.6"

    def _call_nvidia_api(self, prompt, max_tokens=800):
        if not self.nvidia_api_key:
            return None
        headers = {"Authorization": f"Bearer {self.nvidia_api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.7, "max_tokens": max_tokens}
        try:
            response = requests.post(self.nvidia_url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"NVIDIA API exception: {e}")
        return None

    def generate_executive_summary(self):
        # same as before (plain text)
        dataset_summary = f"Rows: {self.summary['rows']}, Columns: {self.summary['columns']}, Missing: {self.summary['missing_values']}"
        if self.stats:
            stats_summary = "\n".join([f"{col}: mean={stat['mean']:.2f}, std={stat['std']:.2f}" for col, stat in list(self.stats.items())[:3]])
        else:
            stats_summary = "No numeric statistics."
        prompt = f"Write a concise executive summary (2-3 sentences, plain text) for a dataset with: {dataset_summary}. Key stats: {stats_summary}"
        ai = self._call_nvidia_api(prompt, 300)
        if ai:
            return ai.strip()
        # fallback
        txt = f"This dataset has {self.summary['rows']} rows and {self.summary['columns']} columns. "
        txt += "No missing values. " if self.summary['missing_values']==0 else f"{self.summary['missing_values']} missing values. "
        if self.correlations:
            top = list(self.correlations.items())[0]
            txt += f"Strongest correlation: {top[0]} ({top[1]:.2f}). "
        outliers = sum(self.anomalies.values())
        txt += f"{outliers} outliers detected." if outliers else "No outliers detected."
        return txt

    def generate_kpi_insights(self):
        # same as before (plain text, no markdown)
        if not self.stats:
            return "No numeric columns for KPI analysis."
        top_col = max(self.stats.items(), key=lambda x: x[1].get('std',0))[0]
        prompt = f"Suggest 2 KPIs from column '{top_col}' (mean={self.stats[top_col]['mean']:.2f}, std={self.stats[top_col]['std']:.2f}). Provide formula, business value, interpretation. Plain text only."
        ai = self._call_nvidia_api(prompt, 400)
        if ai:
            return ai.strip()
        return f"Potential KPI: {top_col} (mean={self.stats[top_col]['mean']:.2f}). Track its variance."

    def generate_recommendations(self):
        recs = []
        for pair, corr in self.correlations.items():
            if abs(corr)>0.7:
                recs.append(f"Investigate strong correlation between {pair} (r={corr:.2f}) for business leverage.")
        if sum(self.anomalies.values())>0:
            recs.append(f"Review {sum(self.anomalies.values())} outliers in {', '.join(self.anomalies.keys())}.")
        if not recs:
            recs.append("Data appears clean. Consider segmentation for deeper insights.")
        prompt = f"Convert these into professional business recommendations (plain text): {' '.join(recs)}"
        ai = self._call_nvidia_api(prompt, 400)
        return ai.strip() if ai else "\n".join(recs)

    def answer_question(self, question):
        q = question.lower()
        # --- Improved fallback using actual data ---
        if 'anomaly' in q or 'outlier' in q:
            if self.anomalies:
                total = sum(self.anomalies.values())
                details = ", ".join([f"{col}: {cnt}" for col, cnt in self.anomalies.items() if cnt>0])
                return f"Detected {total} anomalies total. Details: {details}."
            else:
                return "No anomalies detected in the dataset."
        if 'correlation' in q:
            if self.correlations:
                top = list(self.correlations.items())[0]
                return f"The strongest correlation is {top[0]} with coefficient {top[1]:.2f}."
            else:
                return "No strong correlations found."
        if 'row' in q or 'record' in q:
            return f"The dataset contains {self.summary['rows']} rows."
        if 'column' in q or 'field' in q:
            return f"There are {self.summary['columns']} columns."
        if 'missing' in q or 'null' in q:
            return f"Missing values: {self.summary['missing_values']} ({(self.summary['missing_values']/self.summary['rows'])*100:.1f}% of rows)."
        if 'trend' in q or 'pattern' in q:
            return "No time-series detected. To identify trends, upload data with a date column."
        # Try AI if available
        context = f"Summary: {self.summary}, Correlations: {self.correlations}, Anomalies: {self.anomalies}, Stats: {self.stats}"
        prompt = f"Answer concisely (1 sentence) based on: {context}\nQuestion: {question}"
        ai = self._call_nvidia_api(prompt, 200)
        if ai:
            return ai.strip()
        return "Please ask about rows, columns, missing values, correlations, anomalies, or numeric statistics."