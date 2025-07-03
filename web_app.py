"""
Flask Frontend for RAG System

This Flask application provides a web interface for interacting with the RAG system backend.
It includes functionality for:
- Document search with similarity scoring
- Document upload from URLs or local files
- Journal summarization
- Paper comparison
- Real-time results display with citations
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import requests
import json
from typing import Dict, Any, List, Optional
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Configuration
API_BASE_URL = "http://localhost:8000"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'json'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if uploaded file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def make_api_request(endpoint: str, method: str = 'GET', data: Dict = None, files: Dict = None) -> Dict[str, Any]:
    """Make API request to FastAPI backend with error handling."""
    try:
        url = f"{API_BASE_URL}{endpoint}"

        if method == 'POST':
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        elif method == 'PUT':
            if files:
                response = requests.put(url, data=data, files=files)
            else:
                response = requests.put(url, json=data)
        else:
            response = requests.get(url)

        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


@app.route('/')
def index():
    """Main page with search interface."""
    return render_template('flask-templates_complete.html')


@app.route('/search', methods=['POST'])
def search():
    """Handle similarity search requests."""
    query = request.form.get('query', '').strip()
    k = int(request.form.get('k', 5))
    min_score = float(request.form.get('min_score', 0.4))

    if not query:
        flash('Please enter a search query.', 'error')
        return redirect(url_for('index'))

    # Make API request
    result = make_api_request('/api/similarity_search', 'POST', {
        'query': query,
        'k': k,
        'min_score': min_score
    })

    if result['success']:
        search_results = result['data']
        return render_template('search_results.html',
                               query=query,
                               results=search_results,
                               k=k,
                               min_score=min_score)
    else:
        flash(f'Search failed: {result["error"]}', 'error')
        return redirect(url_for('index'))


@app.route('/upload')
def upload_page():
    """Upload page for documents."""
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_document():
    """Handle document upload requests."""
    schema_version = request.form.get('schema_version', '1.0')
    file_url = request.form.get('file_url', '').strip()
    file_path = request.form.get('file_path', '').strip()

    # Check if file was uploaded
    uploaded_file = request.files.get('file')

    if not file_url and not file_path and not uploaded_file:
        flash('Please provide a file URL, file path, or upload a file.', 'error')
        return redirect(url_for('upload_page'))

    # Handle file upload
    if uploaded_file and uploaded_file.filename != '':
        if allowed_file(uploaded_file.filename):
            filename = uploaded_file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            uploaded_file.save(filepath)
            file_path = filepath
        else:
            flash('Invalid file type. Please upload a JSON file.', 'error')
            return redirect(url_for('upload_page'))

    # Prepare data for API
    data = {'schema_version': schema_version}
    if file_url:
        data['file_url'] = file_url
    if file_path:
        data['file'] = file_path

    # Make API request
    result = make_api_request('/api/upload', 'PUT', data)

    if result['success']:
        flash('Document uploaded successfully!', 'success')
        return render_template('upload_success.html', result=result['data'])
    else:
        flash(f'Upload failed: {result["error"]}', 'error')
        return redirect(url_for('upload_page'))


@app.route('/summarize')
def summarize_page():
    """Summarization page."""
    return render_template('summarize.html')


@app.route('/summarize', methods=['POST'])
def summarize_journal():
    """Handle journal summarization requests."""
    journal = request.form.get('journal', '').strip()

    if not journal:
        flash('Please enter a journal name.', 'error')
        return redirect(url_for('summarize_page'))

    # Make API request
    result = make_api_request('/api/summary', 'POST', {'journal': journal})

    if result['success']:
        summary_data = result['data']
        return render_template('summary_results.html',
                               journal=journal,
                               summary=summary_data)
    else:
        flash(f'Summarization failed: {result["error"]}', 'error')
        return redirect(url_for('summarize_page'))


@app.route('/compare')
def compare_page():
    """Paper comparison page."""
    return render_template('compare.html')


@app.route('/compare', methods=['POST'])
def compare_papers():
    """Handle paper comparison requests."""
    doc_id_1 = request.form.get('doc_id_1', '').strip()
    doc_id_2 = request.form.get('doc_id_2', '').strip()

    if not doc_id_1 or not doc_id_2:
        flash('Please enter both document IDs.', 'error')
        return redirect(url_for('compare_page'))

    # Make API request
    result = make_api_request('/api/compare_papers', 'POST', {
        'doc_id_1': doc_id_1,
        'doc_id_2': doc_id_2
    })

    if result['success']:
        comparison_data = result['data']
        return render_template('comparison_results.html',
                               doc_id_1=doc_id_1,
                               doc_id_2=doc_id_2,
                               comparison=comparison_data)
    else:
        flash(f'Comparison failed: {result["error"]}', 'error')
        return redirect(url_for('compare_page'))


@app.route('/api/health')
def health_check():
    """Check if the FastAPI backend is running."""
    result = make_api_request('/docs')
    return jsonify({"backend_available": result['success']})


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)