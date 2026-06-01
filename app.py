import os
import json
import uuid
import numpy as np
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from services.analyzer import DataAnalyzer
from services.charts import ChartGenerator
from services.insights import InsightGenerator
from services.advanced import AdvancedAnalytics

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-me')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///insightiq.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ------------------ Helper: JSON serialization for numpy types ------------------
def make_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

# ------------------ Database Models ------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    analyses = db.relationship('Analysis', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    analysis_results = db.Column(db.Text)
    summary_text = db.Column(db.Text)
    kpi_insights = db.Column(db.Text)
    recommendations = db.Column(db.Text)
    is_saved = db.Column(db.Boolean, default=False)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ------------------ Helper Functions ------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def perform_analysis(filepath, original_filename):
    analyzer = DataAnalyzer(filepath)
    df = analyzer.load_data()
    if df is None or df.empty:
        return None
    col_types = analyzer.detect_column_types()
    numeric_cols = col_types['numeric']
    categorical_cols = col_types['categorical']
    date_cols = col_types['date']

    stats = analyzer.get_statistics(numeric_cols) if numeric_cols else {}
    correlations = analyzer.get_correlation_matrix(numeric_cols) if len(numeric_cols) >= 2 else {}
    anomalies = analyzer.detect_anomalies(numeric_cols) if numeric_cols else {}

    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'numeric_cols': len(numeric_cols),
        'categorical_cols': len(categorical_cols),
        'date_cols': len(date_cols),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum())
    }

    chart_gen = ChartGenerator(df, numeric_cols, categorical_cols, date_cols)
    charts = {
        'histograms': chart_gen.create_histograms(),
        'correlation_heatmap': chart_gen.create_correlation_heatmap(correlations) if correlations else None,
        'boxplots': chart_gen.create_boxplots(),
        'time_series': chart_gen.create_time_series() if date_cols else None
    }

    insight_gen = InsightGenerator(df, stats, correlations, anomalies, summary)
    summary_text = insight_gen.generate_executive_summary()
    kpi_insights = insight_gen.generate_kpi_insights()
    recommendations = insight_gen.generate_recommendations()

    results_json = {
        'column_types': col_types,
        'statistics': stats,
        'correlations': correlations,
        'anomalies': anomalies,
        'summary': summary,
        'charts': {k: v for k, v in charts.items() if v is not None}
    }
    return {
        'df': df,
        'results_json': results_json,
        'summary_text': summary_text,
        'kpi_insights': kpi_insights,
        'recommendations': recommendations,
        'charts': charts,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'date_cols': date_cols,
        'stats': stats,
        'correlations': correlations,
        'anomalies': anomalies,
        'summary_stats': summary
    }

# ------------------ Routes ------------------
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')
        if not all([username, email, password]):
            flash('All fields required.', 'danger')
            return render_template('register.html')
        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
        if User.query.filter_by(username=username).first():
            flash('Username exists.', 'danger')
            return render_template('register.html')
        if User.query.filter_by(email=email).first():
            flash('Email registered.', 'danger')
            return render_template('register.html')
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash(f'Welcome back, {username}!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    recent_analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.uploaded_at.desc()).limit(10).all()
    saved_reports = Analysis.query.filter_by(user_id=current_user.id, is_saved=True).order_by(Analysis.uploaded_at.desc()).all()
    latest = recent_analyses[0] if recent_analyses else None
    kpi_data = None
    if latest:
        try:
            results = json.loads(latest.analysis_results)
            summary = results.get('summary', {})
            kpi_data = {'rows': summary.get('rows',0), 'columns': summary.get('columns',0),
                        'missing': summary.get('missing_values',0), 'numeric_cols': summary.get('numeric_cols',0)}
        except:
            pass
    return render_template('dashboard.html', analyses=recent_analyses, saved_reports=saved_reports, kpi_data=kpi_data)

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file selected.', 'danger')
        return redirect(url_for('dashboard'))
    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('dashboard'))
    if not allowed_file(file.filename):
        flash('Invalid file type. Upload CSV or Excel.', 'danger')
        return redirect(url_for('dashboard'))
    original_filename = secure_filename(file.filename)
    stored_filename = f"{uuid.uuid4().hex}_{original_filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)
    file.save(filepath)
    try:
        analysis_data = perform_analysis(filepath, original_filename)
        if analysis_data is None:
            flash('Failed to process file.', 'danger')
            os.remove(filepath)
            return redirect(url_for('dashboard'))
        serializable_results = make_json_serializable(analysis_data['results_json'])
        analysis = Analysis(
            user_id=current_user.id,
            filename=stored_filename,
            original_filename=original_filename,
            analysis_results=json.dumps(serializable_results),
            summary_text=analysis_data['summary_text'],
            kpi_insights=analysis_data['kpi_insights'],
            recommendations=analysis_data['recommendations'],
            is_saved=False
        )
        db.session.add(analysis)
        db.session.commit()
        flash('Analysis completed successfully!', 'success')
        return redirect(url_for('analysis_result', analysis_id=analysis.id))
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f'Error: {str(e)}', 'danger')
        if os.path.exists(filepath):
            os.remove(filepath)
        return redirect(url_for('dashboard'))

@app.route('/analysis/<int:analysis_id>')
@login_required
def analysis_result(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    if analysis.user_id != current_user.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))
    results = json.loads(analysis.analysis_results)
    charts_data = results.get('charts', {})
    stats = results.get('statistics', {})
    correlations = results.get('correlations', {})
    anomalies = results.get('anomalies', {})
    summary = results.get('summary', {})
    col_types = results.get('column_types', {})
    return render_template('analysis_result.html', analysis=analysis, summary=summary, stats=stats,
                           correlations=correlations, anomalies=anomalies, col_types=col_types,
                           charts_data=charts_data)

@app.route('/profile/<int:analysis_id>')
@login_required
def data_profile(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    if analysis.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], analysis.filename)
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
    results = json.loads(analysis.analysis_results)
    numeric_cols = results.get('column_types', {}).get('numeric', [])
    date_cols = results.get('column_types', {}).get('date', [])
    adv = AdvancedAnalytics(df, numeric_cols, date_cols)
    profile = adv.profile_data()
    return jsonify(make_json_serializable(profile))

@app.route('/cluster/<int:analysis_id>')
@login_required
def clustering(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    if analysis.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], analysis.filename)
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
    results = json.loads(analysis.analysis_results)
    numeric_cols = results.get('column_types', {}).get('numeric', [])
    date_cols = results.get('column_types', {}).get('date', [])
    adv = AdvancedAnalytics(df, numeric_cols, date_cols)
    clusters = adv.perform_clustering(n_clusters=3)
    if clusters is None:
        return jsonify({'error': 'Not enough data for clustering (need at least 2 numeric columns and 3 rows).'}), 400
    return jsonify(make_json_serializable(clusters))

@app.route('/forecast/<int:analysis_id>')
@login_required
def forecast(analysis_id):
    """Forecast a numeric column using simple linear regression."""
    analysis = Analysis.query.get_or_404(analysis_id)
    if analysis.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403

    # Get query parameters
    target = request.args.get('target', '')
    periods = request.args.get('periods', 3)
    try:
        periods = int(periods)
    except:
        periods = 3

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], analysis.filename)
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
    results = json.loads(analysis.analysis_results)
    numeric_cols = results.get('column_types', {}).get('numeric', [])
    date_cols = results.get('column_types', {}).get('date', [])

    if not date_cols:
        return jsonify({'error': 'No date column found. Forecasting requires a date column.'}), 400
    if not numeric_cols:
        return jsonify({'error': 'No numeric columns found for forecasting.'}), 400

    if target not in numeric_cols:
        # If target not provided or invalid, use first numeric column
        target = numeric_cols[0]

    adv = AdvancedAnalytics(df, numeric_cols, date_cols)
    forecast_result = adv.forecast_simple(target, periods=periods)
    if forecast_result is None:
        return jsonify({'error': f'Could not generate forecast for {target}. Need at least 3 data points with valid dates.'}), 400

    return jsonify(make_json_serializable(forecast_result))

@app.route('/save_report/<int:analysis_id>', methods=['POST'])
@login_required
def save_report(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    if analysis.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    analysis.is_saved = True
    db.session.commit()
    flash('Report saved.', 'success')
    return redirect(url_for('analysis_result', analysis_id=analysis_id))

@app.route('/history')
@login_required
def history():
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.uploaded_at.desc()).all()
    return render_template('history.html', analyses=analyses)

@app.route('/delete_analysis/<int:analysis_id>', methods=['POST'])
@login_required
def delete_analysis(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    if analysis.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], analysis.filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    db.session.delete(analysis)
    db.session.commit()
    flash('Analysis deleted.', 'success')
    return redirect(url_for('history'))

@app.route('/export_pdf/<int:analysis_id>')
@login_required
def export_pdf(analysis_id):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from io import BytesIO
    analysis = Analysis.query.get_or_404(analysis_id)
    if analysis.user_id != current_user.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))
    results = json.loads(analysis.analysis_results)
    summary = results.get('summary', {})
    stats = results.get('statistics', {})
    correlations = results.get('correlations', {})
    anomalies = results.get('anomalies', {})
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"InsightIQ Pro Report: {analysis.original_filename}", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Dataset Summary", styles['Heading2']))
    story.append(Paragraph(f"Rows: {summary.get('rows',0)}<br/>Columns: {summary.get('columns',0)}<br/>Missing: {summary.get('missing_values',0)}", styles['Normal']))
    story.append(Spacer(1,0.2*inch))
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Paragraph(analysis.summary_text or "None", styles['Normal']))
    story.append(Spacer(1,0.1*inch))
    story.append(Paragraph("KPIs", styles['Heading2']))
    story.append(Paragraph(analysis.kpi_insights or "None", styles['Normal']))
    story.append(Spacer(1,0.1*inch))
    story.append(Paragraph("Recommendations", styles['Heading2']))
    story.append(Paragraph(analysis.recommendations or "None", styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"insightiq_report_{analysis.id}.pdf", mimetype='application/pdf')

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.get_json()
    question = data.get('question', '').strip()
    analysis_id = data.get('analysis_id')
    if not question:
        return jsonify({'answer': 'Please ask a question.'})
    if not analysis_id:
        return jsonify({'answer': 'No analysis selected.'})
    analysis = Analysis.query.get(analysis_id)
    if not analysis or analysis.user_id != current_user.id:
        return jsonify({'answer': 'Invalid analysis.'})
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], analysis.filename)
    if not os.path.exists(filepath):
        return jsonify({'answer': 'Dataset not found.'})
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
    results = json.loads(analysis.analysis_results)
    stats = results.get('statistics', {})
    correlations = results.get('correlations', {})
    anomalies = results.get('anomalies', {})
    summary = results.get('summary', {})
    insight_gen = InsightGenerator(df, stats, correlations, anomalies, summary)
    answer = insight_gen.answer_question(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)