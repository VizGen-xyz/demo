from flask import Flask
from flask_cors import CORS
from .config import csv_files
from flask_uploads import configure_uploads

from app.api.ping import bp as ping_bp
from app.api.routes import bp as api_bp
from app.config import FlaskAppConfig


def create_app(config_object=FlaskAppConfig):
    app = Flask(__name__)
    app.config.from_object(config_object)

    # Set up saving of CSV files
    configure_uploads(app, csv_files)

    CORS(app)

    # Register the ping
    app.register_blueprint(ping_bp)

    # Register the api
    app.register_blueprint(api_bp, url_prefix='/api')

    return app