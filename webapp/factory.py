from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify
from werkzeug.exceptions import RequestEntityTooLarge

from webapp.config import load_config_from_env
from webapp.routes import register_routes
from webapp.state import AppState

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # type: ignore[assignment]


def create_app() -> Flask:
    if load_dotenv is not None:
        load_dotenv()

    config = load_config_from_env()
    root_dir = Path(__file__).resolve().parent.parent
    upload_dir = root_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    app = Flask(
        __name__,
        template_folder=str(root_dir / "templates"),
        static_folder=str(root_dir / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = config.max_content_length

    state = AppState()
    app.extensions["app_config"] = config
    app.extensions["app_state"] = state

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_too_large(_err):
        return jsonify({"error": f"File too large. Max upload size is {config.max_upload_mb} MB."}), 413

    register_routes(app, state, config, upload_dir)
    return app
