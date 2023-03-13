import os
import pandas as pd
import traceback
from flask import Flask, render_template, request, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, DATA
from ..config import csv_files, UPLOAD_FILE_DEST

from flask import Blueprint, jsonify, make_response, request
from .utils import text_to_sql_with_retry, generate_summary_dataframe

bp = Blueprint('api_bp', __name__)

@bp.route('/upload', methods=['POST'])
def upload():
    """
    Upload CSV files to process later
    """
    if 'file' in request.files:
        csv_file = request.files['file']
        if csv_file.content_type == 'text/csv':
            df = pd.read_csv(csv_file)

            filename = csv_file.filename

            # Generates the summary that is used to pass in as a prompt
            summary_df = generate_summary_dataframe(df)

            # Some extra error checking and type coercion will be required here eventually
            df.to_parquet(os.path.join(UPLOAD_FILE_DEST, 'uploaded_file.parquet'))

            # Save the summary file as it will be easier to read and easier to construct prompt with
            summary_df.to_parquet(os.path.join(UPLOAD_FILE_DEST, 'summary_file.parquet'))
            return 'File saved successfully'
        return 'Only CSV files are allowed'
    return 'No file uploaded'

@bp.route('/text_to_sql', methods=['POST'])
def text_to_sql():
    """
    Convert natural language query to SQL
    """
    request_body = request.get_json()
    natural_language_query = request_body.get('natural_language_query')

    if not natural_language_query:
        error_msg = 'natural_language_query is missing from request body'
        return make_response(jsonify({'error': error_msg}), 400)

    try:
        # LM outputs are non-deterministic, so same natural language query may result in different SQL queries (some of which may be invalid)
        # Generate queries in parallel and use the first one that works
        result, sql_query = text_to_sql_with_retry(natural_language_query)
    except Exception as e:
        error_msg = f'Error processing request: {str(e)}'
        traceback_str = traceback.format_exc()
        print(traceback_str)
        print(error_msg)
        return make_response(jsonify({'error': error_msg}), 500)

    return make_response(jsonify({'result': result, 'sql_query': sql_query}), 200)
