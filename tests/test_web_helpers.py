import importlib.util
from pathlib import Path


WEB_PATH = Path(__file__).resolve().parents[1] / "nothotdog" / "web.py"


def load_module():
    spec = importlib.util.spec_from_file_location("hotdog_web", WEB_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_label():
    mod = load_module()
    assert mod.normalize_label("Hot Dog") == "hot_dog"
    assert mod.normalize_label("hot-dog") == "hot_dog"


def test_health_endpoint():
    mod = load_module()
    client = mod.app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "status" in body
    assert "stats" in body


def test_metrics_endpoint():
    mod = load_module()
    client = mod.app.test_client()
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "requests_total" in body
