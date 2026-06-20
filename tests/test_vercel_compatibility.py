import importlib

def test_app_import_exposes_app():
    module=importlib.import_module("app")
    assert module.app is not None

def test_no_database_dependencies():
    import app
    assert "sqlalchemy" not in app.__dict__ and "sqlite3" not in app.__dict__
