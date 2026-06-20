def test_pages(client):
    for path in ["/","/predict","/dashboard","/insights","/methodology","/about"]:
        assert client.get(path).status_code == 200

def test_health(client):
    response=client.get("/health")
    assert response.status_code == 200 and response.json["models_available"] is True

def test_static(client):
    assert client.get("/css/style.css").status_code == 200

def test_404(client):
    assert client.get("/does-not-exist").status_code == 404
