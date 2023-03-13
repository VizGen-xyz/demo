from os import getenv
from dotenv import load_dotenv
from flask_uploads import UploadSet, configure_uploads, DATA

load_dotenv()

UPLOAD_FILE_DEST = './csv_uploads/'

OPENAI_KEY = getenv("OPENAI_KEY")

csv_files = UploadSet('csv', DATA)

class FlaskAppConfig:
    CORS_HEADERS = 'Content-Type'
    UPLOADED_CSV_DEST = UPLOAD_FILE_DEST
    # Maximum allowed upload size
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
