from webapp import create_app


app = create_app()


if __name__ == "__main__":
    config = app.extensions["app_config"]
    app.run(
        host=config.flask_host,
        port=config.flask_port,
        debug=config.flask_debug,
    )
