import pytest

from app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(str(tmp_path / "test.db"))
    app.config["TESTING"] = True
    return app.test_client()


def make_seller(client, verify=True, city="Lahore", phone="03001234567"):
    sid = client.post("/api/sellers", json={"name": "Test Seller", "phone": phone,
                                            "city": city}).get_json()["id"]
    if verify:
        client.post(f"/api/sellers/{sid}/verify")
    return sid


def make_product(client, seller_id, **over):
    data = {"seller_id": seller_id, "title": "Cotton Kurta", "category": "clothing",
            "price": 2000, "stock": 10, "sizes": ["S", "M", "L"],
            "description": "Soft breathable cotton kurta for daily wear.",
            "image_url": "https://example.com/a.jpg"}
    data.update(over)
    return client.post("/api/products", json=data)


ORDER = {"customer_name": "Ali", "phone": "03111234567", "city": "Lahore",
         "address": "House 12, Street 4, Gulberg, Lahore", "size": "M"}


def place_order(client, product_id, **over):
    return client.post("/api/orders", json={**ORDER, "product_id": product_id, **over})


def deliver(client, order_id):
    for status in ("confirmed", "shipped", "out_for_delivery", "delivered"):
        assert client.post(f"/api/orders/{order_id}/status", json={"status": status}).status_code == 200


# ---- basics -------------------------------------------------------------
def test_health_and_home(client):
    assert client.get("/health").get_json()["status"] == "ok"
    assert b"SmartShop AI" in client.get("/").data


def test_unknown_route_is_json_404(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.get_json()["error"] == "not_found"


def test_invalid_json_rejected(client):
    r = client.post("/api/sellers", data="not json", content_type="text/plain")
    assert r.status_code == 400 and r.get_json()["error"] == "invalid_json"


# ---- sellers & trust ----------------------------------------------------
def test_seller_phone_validation(client):
    r = client.post("/api/sellers", json={"name": "X", "phone": "12345", "city": "Lahore"})
    assert r.status_code == 400


def test_verification_raises_trust_score(client):
    sid = make_seller(client, verify=False)
    before = client.get(f"/api/sellers/{sid}").get_json()["trust_score"]
    client.post(f"/api/sellers/{sid}/verify")
    after = client.get(f"/api/sellers/{sid}").get_json()
    assert after["verified"] is True and after["trust_score"] == before + 20


def test_unverified_seller_cannot_list(client):
    sid = make_seller(client, verify=False)
    r = make_product(client, sid)
    assert r.status_code == 403 and r.get_json()["error"] == "seller_not_verified"


# ---- listings & fake detection -----------------------------------------
def test_clean_listing_is_active(client):
    r = make_product(client, make_seller(client))
    assert r.status_code == 201
    body = r.get_json()
    assert body["status"] == "active" and body["risk_score"] == 0


def test_counterfeit_keywords_flagged_and_held(client):
    sid = make_seller(client)
    r = make_product(client, sid, title="Nike master copy shoes", description="1:1 replica, AAA quality",
                     image_url="")
    body = r.get_json()
    assert "counterfeit_keywords" in body["risk_flags"]
    assert body["status"] == "pending_review"
    # held listings are not searchable
    assert client.get("/api/products").get_json()["count"] == 0


def test_price_far_below_market_flagged(client):
    sid = make_seller(client)
    make_product(client, sid, price=2000)
    make_product(client, sid, price=2200)
    body = make_product(client, sid, price=500).get_json()
    assert "price_far_below_market" in body["risk_flags"]


def test_invalid_listing_fields(client):
    sid = make_seller(client)
    assert make_product(client, sid, price=-5).status_code == 400
    assert make_product(client, sid, category="weapons").status_code == 400
    assert make_product(client, sid, stock=-1).status_code == 400


# ---- final price & ETA --------------------------------------------------
def test_final_price_includes_delivery(client):
    pid = make_product(client, make_seller(client), price=2000).get_json()["id"]
    p = client.get(f"/api/products/{pid}?city=Lahore").get_json()
    assert p["delivery_fee"] == 250 and p["final_price"] == 2250


def test_free_delivery_above_threshold_and_remote_surcharge(client):
    sid = make_seller(client)
    cheap = make_product(client, sid, price=1000).get_json()["id"]
    rich = make_product(client, sid, price=3500).get_json()["id"]
    assert client.get(f"/api/products/{rich}?city=Multan").get_json()["delivery_fee"] == 0
    assert client.get(f"/api/products/{cheap}?city=Multan").get_json()["delivery_fee"] == 350


def test_eta_longer_for_other_city(client):
    pid = make_product(client, make_seller(client, city="Lahore")).get_json()["id"]
    local = client.get(f"/api/products/{pid}?city=Lahore").get_json()["eta_days"]
    far = client.get(f"/api/products/{pid}?city=Multan").get_json()["eta_days"]
    assert local == {"min": 2, "max": 4} and far == {"min": 5, "max": 7}


# ---- search & compare ---------------------------------------------------
def test_search_filter_and_sort(client):
    sid = make_seller(client)
    make_product(client, sid, title="Red Kurta", price=3000)
    make_product(client, sid, title="Blue Kurta", price=1500)
    make_product(client, sid, title="Running Shoes", category="shoes", price=6000,
                 description="Lightweight running shoes with cushioned sole.")
    r = client.get("/api/products?q=kurta&sort=price_asc").get_json()
    assert [i["title"] for i in r["items"]] == ["Blue Kurta", "Red Kurta"]
    assert client.get("/api/products?category=shoes").get_json()["count"] == 1
    assert client.get("/api/products?max_price=2000").get_json()["count"] == 1
    assert client.get("/api/products?max_price=abc").status_code == 400


def test_compare_labels_prices(client):
    sid = make_seller(client)
    for price in (1000, 2000, 3000):
        make_product(client, sid, title="Kurta", price=price)
    items = client.get("/api/compare?q=Kurta").get_json()["items"]
    assert items[0]["cheapest"] is True
    assert items[0]["price_label"] == "below_market"
    assert items[1]["price_label"] == "fair"
    assert items[2]["price_label"] == "above_market"


def test_compare_needs_query_and_handles_little_data(client):
    assert client.get("/api/compare").status_code == 400
    make_product(client, make_seller(client), title="Lamp", category="other")
    items = client.get("/api/compare?q=Lamp").get_json()["items"]
    assert items[0]["price_label"] == "not_enough_data"


# ---- size advisor -------------------------------------------------------
def test_clothing_size_by_body(client):
    small = client.post("/api/size-advisor", json={"category": "clothing", "height_cm": 160, "weight_kg": 50})
    big = client.post("/api/size-advisor", json={"category": "clothing", "height_cm": 185, "weight_kg": 95})
    assert small.get_json()["recommended_size"] == "S"
    assert big.get_json()["recommended_size"] == "XXL"


def test_shoe_size(client):
    r = client.post("/api/size-advisor", json={"category": "shoes", "foot_length_cm": 26})
    assert r.get_json()["recommended_size"] == "41"


def test_size_advisor_validates(client):
    assert client.post("/api/size-advisor", json={"category": "clothing", "height_cm": 20, "weight_kg": 50}).status_code == 400
    assert client.post("/api/size-advisor", json={"category": "hats"}).status_code == 400
    assert client.post("/api/size-advisor", json={"category": "clothing"}).status_code == 400


def test_size_advisor_learns_from_returns(client):
    sid = make_seller(client)
    pid = make_product(client, sid, stock=10).get_json()["id"]
    base = {"category": "clothing", "height_cm": 170, "weight_kg": 60, "product_id": pid}
    assert client.post("/api/size-advisor", json=base).get_json()["recommended_size"] == "M"
    for n in range(2):
        oid = place_order(client, pid, phone=f"0311123456{n}").get_json()["id"]
        deliver(client, oid)
        rid = client.post("/api/returns", json={"order_id": oid, "reason": "wrong_size_small"}).get_json()["id"]
        client.post(f"/api/returns/{rid}/resolve", json={"decision": "approve"})
    adjusted = client.post("/api/size-advisor", json=base).get_json()
    assert adjusted["recommended_size"] == "L" and adjusted["confidence"] == "high"
    assert "runs small" in adjusted["note"]


# ---- orders -------------------------------------------------------------
def test_order_totals_stock_and_tracking(client):
    pid = make_product(client, make_seller(client), price=1000, stock=5).get_json()["id"]
    r = place_order(client, pid, qty=2)
    assert r.status_code == 201
    o = r.get_json()
    assert o["subtotal"] == 2000 and o["delivery_fee"] == 250 and o["total"] == 2250
    assert o["tracking"][0]["status"] == "placed"
    assert client.get(f"/api/products/{pid}").get_json()["stock"] == 3


def test_order_validation(client):
    pid = make_product(client, make_seller(client), stock=1).get_json()["id"]
    assert place_order(client, pid, size="XXXL").status_code == 400
    assert place_order(client, pid, qty=0).status_code == 400
    assert place_order(client, pid, payment_method="bitcoin").status_code == 400
    assert place_order(client, pid, qty=2).status_code == 409
    assert client.post("/api/orders", json={**ORDER, "product_id": 999}).status_code == 404


def test_risky_cod_requires_advance(client):
    pid = make_product(client, make_seller(client), price=9000, stock=50).get_json()["id"]
    risky = {"phone": "123", "address": "short", "qty": 5}
    r = place_order(client, pid, **risky)
    assert r.status_code == 402
    body = r.get_json()
    assert body["error"] == "advance_required" and body["risk_score"] >= 60
    assert "invalid_phone" in body["risk_flags"]
    ok = place_order(client, pid, advance_paid=True, **risky)
    assert ok.status_code == 201


def test_online_payment_skips_cod_risk(client):
    pid = make_product(client, make_seller(client), price=9000, stock=50).get_json()["id"]
    r = place_order(client, pid, phone="123", address="short", qty=5, payment_method="online")
    assert r.status_code == 201 and r.get_json()["risk_score"] == 0


def test_previous_refusal_raises_risk(client):
    pid = make_product(client, make_seller(client), stock=10).get_json()["id"]
    oid = place_order(client, pid).get_json()["id"]
    for s in ("confirmed", "shipped", "refused"):
        client.post(f"/api/orders/{oid}/status", json={"status": s})
    assert client.get(f"/api/products/{pid}").get_json()["stock"] == 10  # restocked
    r = place_order(client, pid)
    assert r.status_code == 402 and "previous_refusals" in r.get_json()["risk_flags"]


def test_invalid_status_transition(client):
    pid = make_product(client, make_seller(client)).get_json()["id"]
    oid = place_order(client, pid).get_json()["id"]
    r = client.post(f"/api/orders/{oid}/status", json={"status": "delivered"})
    assert r.status_code == 409 and "confirmed" in r.get_json()["allowed"]
    assert client.get("/api/orders/9999").status_code == 404


def test_cancel_restocks(client):
    pid = make_product(client, make_seller(client), stock=4).get_json()["id"]
    oid = place_order(client, pid, qty=2).get_json()["id"]
    client.post(f"/api/orders/{oid}/status", json={"status": "cancelled"})
    assert client.get(f"/api/products/{pid}").get_json()["stock"] == 4


# ---- returns & trust ----------------------------------------------------
def test_return_requires_delivery_and_photo(client):
    pid = make_product(client, make_seller(client)).get_json()["id"]
    oid = place_order(client, pid).get_json()["id"]
    assert client.post("/api/returns", json={"order_id": oid, "reason": "other"}).status_code == 409
    deliver(client, oid)
    assert client.post("/api/returns", json={"order_id": oid, "reason": "damaged"}).status_code == 400
    assert client.post("/api/returns", json={"order_id": oid, "reason": "bogus"}).status_code == 400
    ok = client.post("/api/returns", json={"order_id": oid, "reason": "damaged", "photo_count": 2})
    assert ok.status_code == 201
    assert client.post("/api/returns", json={"order_id": oid, "reason": "other"}).status_code == 409


def test_refund_method_depends_on_payment(client):
    pid = make_product(client, make_seller(client)).get_json()["id"]
    cod = place_order(client, pid).get_json()["id"]
    online = place_order(client, pid, payment_method="online", phone="03111234568").get_json()["id"]
    results = {}
    for oid in (cod, online):
        deliver(client, oid)
        rid = client.post("/api/returns", json={"order_id": oid, "reason": "other"}).get_json()["id"]
        results[oid] = client.post(f"/api/returns/{rid}/resolve", json={"decision": "approve"}).get_json()
    assert results[cod]["refund_method"] == "wallet"
    assert results[online]["refund_method"] == "original_payment"
    assert results[cod]["refund_amount"] == 2250
    assert client.get(f"/api/orders/{cod}").get_json()["status"] == "refunded"


def test_fake_return_suspends_product(client):
    pid = make_product(client, make_seller(client)).get_json()["id"]
    oid = place_order(client, pid).get_json()["id"]
    deliver(client, oid)
    rid = client.post("/api/returns", json={"order_id": oid, "reason": "fake_product",
                                            "photo_count": 1}).get_json()["id"]
    client.post(f"/api/returns/{rid}/resolve", json={"decision": "approve"})
    assert client.get(f"/api/products/{pid}").get_json()["status"] == "suspended"
    assert client.get("/api/products").get_json()["count"] == 0
    assert place_order(client, pid).status_code == 404


def test_resolve_twice_and_bad_decision(client):
    pid = make_product(client, make_seller(client)).get_json()["id"]
    oid = place_order(client, pid).get_json()["id"]
    deliver(client, oid)
    rid = client.post("/api/returns", json={"order_id": oid, "reason": "other"}).get_json()["id"]
    assert client.post(f"/api/returns/{rid}/resolve", json={"decision": "maybe"}).status_code == 400
    assert client.post(f"/api/returns/{rid}/resolve", json={"decision": "reject"}).status_code == 200
    assert client.post(f"/api/returns/{rid}/resolve", json={"decision": "approve"}).status_code == 409


def test_trust_score_rises_with_deliveries_and_falls_with_returns(client):
    sid = make_seller(client)
    pid = make_product(client, sid, stock=20).get_json()["id"]
    start = client.get(f"/api/sellers/{sid}").get_json()["trust_score"]
    oids = []
    for n in range(3):
        oid = place_order(client, pid, phone=f"0311123450{n}").get_json()["id"]
        deliver(client, oid)
        oids.append(oid)
    after_deliveries = client.get(f"/api/sellers/{sid}").get_json()["trust_score"]
    assert after_deliveries > start
    rid = client.post("/api/returns", json={"order_id": oids[0], "reason": "other"}).get_json()["id"]
    assert client.get(f"/api/sellers/{sid}").get_json()["trust_score"] < after_deliveries
    assert rid
