from flask import Blueprint
bp = Blueprint('ping_bp', __name__)

# This is used as a health check, necessary to keep the server running
@bp.route('/ping', methods=['GET'])
def ping():
    return 'pong'