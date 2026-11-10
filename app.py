"""SmartShop AI - MVP prototype backend.

Shop Smart, Shop Safe, Shop Easy.

A small Flask + SQLite API that demonstrates the core trust features from the
business blueprint:

  * seller verification and a live trust score
  * fake-listing risk detection (rule based; swap for ML later)
  * all-inclusive final price (no surprise delivery charges)
  * delivery ETA estimate
  * price comparison with a fair-price label
  * AI-style size advisor that learns from return reasons
  * COD risk scoring with an advance-payment rule for risky orders
  * order tracking and an easy photo-based return / refund flow

Run:
    python app.py seed      # optional demo data
    python app.py           # start on http://127.0.0.1:5000
"""
import json
import os
import re
import sqlite3
import statistics
import sys
from datetime import datetime, timezone

from flask import Flask, Response, g, jsonify, request

# ----------------------------------------------------------------------------
# Business rules (tune these; they are hypotheses, not market data)
# ----------------------------------------------------------------------------
PHONE_RE = re.compile(r"^(\+92|0)3\d{9}$")  # Pakistani mobile numbers
MAJOR_CITIES = {"karachi", "lahore", "islamabad", "rawalpindi"}
FREE_DELIVERY_FROM = 3000      # PKR subtotal
BASE_DELIVERY = 250            # PKR
REMOTE_SURCHARGE = 100         # PKR, cities outside the major list
FAKE_KEYWORDS = ("replica", "master copy", "first copy", "1:1", "aaa quality",
                 "clone", "copy of", "fake")
RISK_REVIEW_THRESHOLD = 60     # listings at/above this wait for human review
COD_ADVANCE_THRESHOLD = 60     # COD orders at/above this need an advance
CATEGORIES = {"clothing", "shoes", "cosmetics", "electronics", "other"}
RETURN_REASONS = {"wrong_size_small", "wrong_size_large", "fake_product",
                  "damaged", "not_as_described", "other"}
CLOTHING_SIZES = ["S", "M", "L", "XL", "XXL"]

ORDER_FLOW = {
    "placed": {"confirmed", "cancelled"},
    "confirmed": {"shipped", "cancelled"},
    "shipped": {"out_for_delivery", "refused"},
    "out_for_delivery": {"delivered", "refused"},
    "delivered": set(),
    "cancelled": set(),
    "refused": set(),
    "refunded": set(),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS sellers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    phone TEXT NOT NULL,
    city TEXT NOT NULL,
    verified INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seller_id INTEGER NOT NULL REFERENCES sellers(id),
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL,
    price REAL NOT NULL,
    stock INTEGER NOT NULL DEFAULT 0,
    sizes TEXT NOT NULL DEFAULT '[]',
    image_url TEXT NOT NULL DEFAULT '',
    risk_score INTEGER NOT NULL DEFAULT 0,
    risk_flags TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL REFERENCES products(id),
    customer_name TEXT NOT NULL,
    phone TEXT NOT NULL,
    city TEXT NOT NULL,
    address TEXT NOT NULL,
    size TEXT,
    qty INTEGER NOT NULL,
    payment_method TEXT NOT NULL,
    advance_paid INTEGER NOT NULL DEFAULT 0,
    subtotal REAL NOT NULL,
    delivery_fee REAL NOT NULL,
    total REAL NOT NULL,
    risk_score INTEGER NOT NULL DEFAULT 0,
    eta_min_days INTEGER NOT NULL,
    eta_max_days INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'placed',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    status TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS returns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL UNIQUE REFERENCES orders(id),
    reason TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    photo_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'requested',
    refund_amount REAL,
    refund_method TEXT,
    created_at TEXT NOT NULL
);
"""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ApiError(Exception):
    def __init__(self, code, message, status=400, **extra):
        super().__init__(message)
        self.code, self.message, self.status, self.extra = code, message, status, extra


def body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError("invalid_json", "Send a JSON object in the request body.")
    return data


def need(data, field, kind=str):
    value = data.get(field)
    if value is None or (kind is str and not str(value).strip()):
        raise ApiError("missing_field", f"'{field}' is required.")
    if kind is str:
        return str(value).strip()
    try:
        return kind(value)
    except (TypeError, ValueError):
        raise ApiError("invalid_field", f"'{field}' must be a {kind.__name__}.")


def is_major(city):
    return city.strip().lower() in MAJOR_CITIES


def delivery_fee(subtotal, city):
    if subtotal >= FREE_DELIVERY_FROM:
        return 0.0
    return float(BASE_DELIVERY + (0 if is_major(city) else REMOTE_SURCHARGE))


def estimate_eta(seller_city, customer_city):
    days = 2 if is_major(customer_city) else 4
    if seller_city.strip().lower() != customer_city.strip().lower():
        days += 1
    return days, days + 2


def row(r):
    return dict(r) if r is not None else None


# ----------------------------------------------------------------------------
# AI-style services (rule based now; each is a single function to replace
# with a trained model later)
# ----------------------------------------------------------------------------
def trust_score(db, seller_id):
    seller = db.execute("SELECT verified FROM sellers WHERE id=?", (seller_id,)).fetchone()
    delivered = db.execute(
        "SELECT COUNT(*) FROM orders o JOIN products p ON p.id=o.product_id "
        "WHERE p.seller_id=? AND o.status IN ('delivered','refunded')", (seller_id,)).fetchone()[0]
    returns = db.execute(
        "SELECT COUNT(*) FROM returns r JOIN orders o ON o.id=r.order_id "
        "JOIN products p ON p.id=o.product_id WHERE p.seller_id=?", (seller_id,)).fetchone()[0]
    return_rate = min(returns / max(delivered, 1), 1.0)
    score = 40 + (20 if seller["verified"] else 0) + min(delivered, 20) * 1.5 - return_rate * 50
    return int(round(max(0, min(100, score))))


def analyze_listing(db, category, price, title, description, image_url):
    """Return (risk_score 0-100, flags) for a new listing."""
    score, flags = 0, []
    text = f"{title} {description}".lower()
    if any(word in text for word in FAKE_KEYWORDS):
        score += 60  # counterfeit wording alone is enough to hold the listing
        flags.append("counterfeit_keywords")
    prices = [r[0] for r in db.execute(
        "SELECT price FROM products WHERE category=? AND status='active'", (category,))]
    if len(prices) >= 2 and price < 0.5 * statistics.median(prices):
        score += 30
        flags.append("price_far_below_market")
    if not image_url:
        score += 10
        flags.append("no_image")
    if len(description.strip()) < 20:
        score += 10
        flags.append("short_description")
    return min(score, 100), flags


def cod_risk(db, phone, address, qty, total):
    score, flags = 0, []
    if not PHONE_RE.match(phone):
        score += 30
        flags.append("invalid_phone")
    if len(address.strip()) < 10:
        score += 20
        flags.append("short_address")
    if qty >= 5:
        score += 25
        flags.append("large_quantity")
    if total > 20000:
        score += 25
        flags.append("high_value")
    prior = db.execute("SELECT COUNT(*) FROM orders WHERE phone=?", (phone,)).fetchone()[0]
    if prior == 0:
        score += 10
        flags.append("first_order")
    refused = db.execute(
        "SELECT COUNT(*) FROM orders WHERE phone=? AND status='refused'", (phone,)).fetchone()[0]
    if refused:
        score += 60 * refused  # one refused COD order is enough to require an advance
        flags.append("previous_refusals")
    return min(score, 100), flags


def recommend_size(db, category, data):
    """Recommend a size from body data, adjusted by this product's return reasons."""
    if category == "shoes":
        foot = need(data, "foot_length_cm", float)
        if not 15 <= foot <= 35:
            raise ApiError("invalid_field", "'foot_length_cm' must be between 15 and 35.")
        return {"recommended_size": str(round((foot + 1.5) * 1.5)),
                "system": "EU", "confidence": "medium", "note": None}
    if category != "clothing":
        raise ApiError("invalid_field", "category must be 'clothing' or 'shoes'.")
    height = need(data, "height_cm", float)
    weight = need(data, "weight_kg", float)
    if not (100 <= height <= 230 and 25 <= weight <= 250):
        raise ApiError("invalid_field", "height_cm or weight_kg is outside a realistic range.")
    index_score = weight + (height - 170) * 0.4
    idx = next((i for i, cap in enumerate((55, 68, 80, 92)) if index_score < cap), 4)
    confidence, note = "medium", None
    product_id = data.get("product_id")
    if product_id is not None:
        small = db.execute(
            "SELECT COUNT(*) FROM returns r JOIN orders o ON o.id=r.order_id "
            "WHERE o.product_id=? AND r.reason='wrong_size_small'", (product_id,)).fetchone()[0]
        large = db.execute(
            "SELECT COUNT(*) FROM returns r JOIN orders o ON o.id=r.order_id "
            "WHERE o.product_id=? AND r.reason='wrong_size_large'", (product_id,)).fetchone()[0]
        if small - large >= 2:
            idx, confidence = min(idx + 1, 4), "high"
            note = "Past buyers found this item runs small, so we sized up."
        elif large - small >= 2:
            idx, confidence = max(idx - 1, 0), "high"
            note = "Past buyers found this item runs large, so we sized down."
    return {"recommended_size": CLOTHING_SIZES[idx], "system": "letter",
            "confidence": confidence, "note": note}


# ----------------------------------------------------------------------------
# Serialisers
# ----------------------------------------------------------------------------
def product_view(db, p, city="Lahore"):
    seller = db.execute("SELECT * FROM sellers WHERE id=?", (p["seller_id"],)).fetchone()
    fee = delivery_fee(p["price"], city)
    eta_min, eta_max = estimate_eta(seller["city"], city)
    return {
        "id": p["id"], "title": p["title"], "description": p["description"],
        "category": p["category"], "price": p["price"], "stock": p["stock"],
        "sizes": json.loads(p["sizes"]), "image_url": p["image_url"],
        "status": p["status"], "risk_score": p["risk_score"],
        "risk_flags": json.loads(p["risk_flags"]),
        "delivery_fee": fee, "final_price": p["price"] + fee,
        "delivery_city": city, "eta_days": {"min": eta_min, "max": eta_max},
        "seller": {"id": seller["id"], "name": seller["name"], "city": seller["city"],
                   "verified": bool(seller["verified"]),
                   "trust_score": trust_score(db, seller["id"])},
    }


def order_view(db, o):
    d = row(o)
    d["advance_paid"] = bool(d["advance_paid"])
    d["eta_days"] = {"min": d.pop("eta_min_days"), "max": d.pop("eta_max_days")}
    d["tracking"] = [row(t) for t in db.execute(
        "SELECT status, at FROM tracking WHERE order_id=? ORDER BY id", (o["id"],))]
    return d


# ----------------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------------
def create_app(db_path=None):
    app = Flask(__name__)
    app.config["DATABASE"] = db_path or os.environ.get("SMARTSHOP_DB", "smartshop.db")

    def get_db():
        if "db" not in g:
            g.db = sqlite3.connect(app.config["DATABASE"])
            g.db.row_factory = sqlite3.Row
            g.db.execute("PRAGMA foreign_keys=ON")
        return g.db

    @app.teardown_appcontext
    def close_db(_exc):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    with sqlite3.connect(app.config["DATABASE"]) as conn:
        conn.executescript(SCHEMA)

    @app.errorhandler(ApiError)
    def handle_api_error(e):
        return jsonify({"error": e.code, "message": e.message, **e.extra}), e.status

    @app.errorhandler(404)
    def handle_404(_e):
        return jsonify({"error": "not_found", "message": "Resource not found."}), 404

    # ---- basics ------------------------------------------------------------
    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "app": "SmartShop AI"})

    @app.get("/")
    def home():
        return Response(DEMO_PAGE, mimetype="text/html")

    # ---- sellers -----------------------------------------------------------
    @app.post("/api/sellers")
    def create_seller():
        data = body()
        name, city = need(data, "name"), need(data, "city")
        phone = need(data, "phone")
        if not PHONE_RE.match(phone):
            raise ApiError("invalid_field", "Use a valid Pakistani mobile number, e.g. 03001234567.")
        db = get_db()
        cur = db.execute("INSERT INTO sellers (name, phone, city, created_at) VALUES (?,?,?,?)",
                         (name, phone, city, now()))
        db.commit()
        return jsonify({"id": cur.lastrowid, "name": name, "verified": False,
                        "trust_score": trust_score(db, cur.lastrowid)}), 201

    @app.post("/api/sellers/<int:seller_id>/verify")
    def verify_seller(seller_id):
        db = get_db()
        if not db.execute("SELECT 1 FROM sellers WHERE id=?", (seller_id,)).fetchone():
            raise ApiError("not_found", "Seller not found.", 404)
        db.execute("UPDATE sellers SET verified=1 WHERE id=?", (seller_id,))
        db.commit()
        return jsonify({"id": seller_id, "verified": True,
                        "trust_score": trust_score(db, seller_id)})

    @app.get("/api/sellers/<int:seller_id>")
    def get_seller(seller_id):
        db = get_db()
        s = db.execute("SELECT * FROM sellers WHERE id=?", (seller_id,)).fetchone()
        if not s:
            raise ApiError("not_found", "Seller not found.", 404)
        d = row(s)
        d["verified"] = bool(d["verified"])
        d["trust_score"] = trust_score(db, seller_id)
        return jsonify(d)

    # ---- products ----------------------------------------------------------
    @app.post("/api/products")
    def create_product():
        data = body()
        db = get_db()
        seller = db.execute("SELECT * FROM sellers WHERE id=?",
                            (need(data, "seller_id", int),)).fetchone()
        if not seller:
            raise ApiError("not_found", "Seller not found.", 404)
        if not seller["verified"]:
            raise ApiError("seller_not_verified",
                           "Only verified sellers can list products.", 403)
        title = need(data, "title")
        category = need(data, "category").lower()
        if category not in CATEGORIES:
            raise ApiError("invalid_field", f"category must be one of {sorted(CATEGORIES)}.")
        price = need(data, "price", float)
        if price <= 0:
            raise ApiError("invalid_field", "'price' must be greater than zero.")
        stock = int(data.get("stock", 0))
        if stock < 0:
            raise ApiError("invalid_field", "'stock' cannot be negative.")
        sizes = data.get("sizes", [])
        if not isinstance(sizes, list):
            raise ApiError("invalid_field", "'sizes' must be a list.")
        description = str(data.get("description", ""))
        image_url = str(data.get("image_url", ""))
        risk, flags = analyze_listing(db, category, price, title, description, image_url)
        status = "pending_review" if risk >= RISK_REVIEW_THRESHOLD else "active"
        cur = db.execute(
            "INSERT INTO products (seller_id,title,description,category,price,stock,sizes,"
            "image_url,risk_score,risk_flags,status,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (seller["id"], title, description, category, price, stock,
             json.dumps([str(s) for s in sizes]), image_url, risk, json.dumps(flags),
             status, now()))
        db.commit()
        p = db.execute("SELECT * FROM products WHERE id=?", (cur.lastrowid,)).fetchone()
        return jsonify(product_view(db, p)), 201

    @app.get("/api/products")
    def list_products():
        db = get_db()
        city = request.args.get("city", "Lahore")
        sql, params = "SELECT * FROM products WHERE status='active'", []
        if q := request.args.get("q"):
            sql += " AND (title LIKE ? OR description LIKE ?)"
            params += [f"%{q}%", f"%{q}%"]
        if cat := request.args.get("category"):
            sql += " AND category=?"
            params.append(cat.lower())
        if mp := request.args.get("max_price"):
            try:
                params_val = float(mp)
            except ValueError:
                raise ApiError("invalid_field", "'max_price' must be a number.")
            sql += " AND price<=?"
            params.append(params_val)
        items = [product_view(db, p, city) for p in db.execute(sql + " ORDER BY id DESC", params)]
        sort = request.args.get("sort")
        if sort == "price_asc":
            items.sort(key=lambda x: x["final_price"])
        elif sort == "price_desc":
            items.sort(key=lambda x: -x["final_price"])
        elif sort == "trust":
            items.sort(key=lambda x: -x["seller"]["trust_score"])
        return jsonify({"count": len(items), "items": items})

    @app.get("/api/products/<int:product_id>")
    def get_product(product_id):
        db = get_db()
        p = db.execute("SELECT * FROM products WHERE id=?", (product_id,)).fetchone()
        if not p:
            raise ApiError("not_found", "Product not found.", 404)
        return jsonify(product_view(db, p, request.args.get("city", "Lahore")))

    @app.get("/api/compare")
    def compare():
        db = get_db()
        q = request.args.get("q", "").strip()
        if not q:
            raise ApiError("missing_field", "'q' is required.")
        city = request.args.get("city", "Lahore")
        items = [product_view(db, p, city) for p in db.execute(
            "SELECT * FROM products WHERE status='active' AND title LIKE ?", (f"%{q}%",))]
        items.sort(key=lambda x: x["final_price"])
        if len(items) >= 3:
            median = statistics.median(i["final_price"] for i in items)
            for i in items:
                i["price_label"] = ("below_market" if i["final_price"] < 0.85 * median
                                    else "above_market" if i["final_price"] > 1.15 * median
                                    else "fair")
        else:
            for i in items:
                i["price_label"] = "not_enough_data"
        if items:
            items[0]["cheapest"] = True
        return jsonify({"query": q, "count": len(items), "items": items})

    @app.post("/api/size-advisor")
    def size_advisor():
        data = body()
        return jsonify(recommend_size(get_db(), need(data, "category").lower(), data))

    # ---- orders ------------------------------------------------------------
    @app.post("/api/orders")
    def create_order():
        data = body()
        db = get_db()
        p = db.execute("SELECT * FROM products WHERE id=?",
                       (need(data, "product_id", int),)).fetchone()
        if not p or p["status"] != "active":
            raise ApiError("product_unavailable", "This product is not available.", 404)
        customer = need(data, "customer_name")
        phone, city, address = need(data, "phone"), need(data, "city"), need(data, "address")
        qty = int(data.get("qty", 1))
        if not 1 <= qty <= 20:
            raise ApiError("invalid_field", "'qty' must be between 1 and 20.")
        payment = str(data.get("payment_method", "cod")).lower()
        if payment not in {"cod", "online"}:
            raise ApiError("invalid_field", "payment_method must be 'cod' or 'online'.")
        sizes = json.loads(p["sizes"])
        size = data.get("size")
        if sizes and size not in sizes:
            raise ApiError("invalid_field", f"Choose a size from {sizes}.")
        if p["stock"] < qty:
            raise ApiError("out_of_stock", f"Only {p['stock']} left in stock.", 409)
        subtotal = p["price"] * qty
        fee = delivery_fee(subtotal, city)
        total = subtotal + fee
        risk, flags = (cod_risk(db, phone, address, qty, total) if payment == "cod" else (0, []))
        advance = bool(data.get("advance_paid", False))
        if payment == "cod" and risk >= COD_ADVANCE_THRESHOLD and not advance:
            raise ApiError("advance_required",
                           "This COD order needs a small advance payment to be confirmed.",
                           402, risk_score=risk, risk_flags=flags)
        seller_city = db.execute("SELECT city FROM sellers WHERE id=?",
                                 (p["seller_id"],)).fetchone()["city"]
        eta_min, eta_max = estimate_eta(seller_city, city)
        cur = db.execute(
            "INSERT INTO orders (product_id,customer_name,phone,city,address,size,qty,"
            "payment_method,advance_paid,subtotal,delivery_fee,total,risk_score,"
            "eta_min_days,eta_max_days,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p["id"], customer, phone, city, address, size, qty, payment,
             int(advance), subtotal, fee, total, risk, eta_min, eta_max, now()))
        db.execute("UPDATE products SET stock=stock-? WHERE id=?", (qty, p["id"]))
        db.execute("INSERT INTO tracking (order_id,status,at) VALUES (?,?,?)",
                   (cur.lastrowid, "placed", now()))
        db.commit()
        o = db.execute("SELECT * FROM orders WHERE id=?", (cur.lastrowid,)).fetchone()
        out = order_view(db, o)
        out["risk_flags"] = flags
        return jsonify(out), 201

    @app.get("/api/orders/<int:order_id>")
    def get_order(order_id):
        db = get_db()
        o = db.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
        if not o:
            raise ApiError("not_found", "Order not found.", 404)
        return jsonify(order_view(db, o))

    @app.post("/api/orders/<int:order_id>/status")
    def update_order_status(order_id):
        new = need(body(), "status")
        db = get_db()
        o = db.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
        if not o:
            raise ApiError("not_found", "Order not found.", 404)
        if new not in ORDER_FLOW.get(o["status"], set()):
            raise ApiError("invalid_transition",
                           f"Cannot move an order from '{o['status']}' to '{new}'.", 409,
                           allowed=sorted(ORDER_FLOW.get(o["status"], set())))
        db.execute("UPDATE orders SET status=? WHERE id=?", (new, order_id))
        db.execute("INSERT INTO tracking (order_id,status,at) VALUES (?,?,?)",
                   (order_id, new, now()))
        if new in {"cancelled", "refused"}:
            db.execute("UPDATE products SET stock=stock+? WHERE id=?",
                       (o["qty"], o["product_id"]))
        db.commit()
        return jsonify(order_view(db, db.execute(
            "SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()))

    # ---- returns -----------------------------------------------------------
    @app.post("/api/returns")
    def create_return():
        data = body()
        db = get_db()
        o = db.execute("SELECT * FROM orders WHERE id=?",
                       (need(data, "order_id", int),)).fetchone()
        if not o:
            raise ApiError("not_found", "Order not found.", 404)
        if o["status"] != "delivered":
            raise ApiError("not_returnable", "Only delivered orders can be returned.", 409)
        reason = need(data, "reason")
        if reason not in RETURN_REASONS:
            raise ApiError("invalid_field", f"reason must be one of {sorted(RETURN_REASONS)}.")
        if db.execute("SELECT 1 FROM returns WHERE order_id=?", (o["id"],)).fetchone():
            raise ApiError("already_requested", "A return was already requested.", 409)
        photos = int(data.get("photo_count", 0))
        if reason in {"fake_product", "damaged", "not_as_described"} and photos < 1:
            raise ApiError("photo_required", "Add at least one photo for this return reason.")
        cur = db.execute(
            "INSERT INTO returns (order_id,reason,description,photo_count,created_at) "
            "VALUES (?,?,?,?,?)",
            (o["id"], reason, str(data.get("description", "")), photos, now()))
        db.commit()
        return jsonify(row(db.execute("SELECT * FROM returns WHERE id=?",
                                      (cur.lastrowid,)).fetchone())), 201

    @app.post("/api/returns/<int:return_id>/resolve")
    def resolve_return(return_id):
        decision = need(body(), "decision")
        if decision not in {"approve", "reject"}:
            raise ApiError("invalid_field", "decision must be 'approve' or 'reject'.")
        db = get_db()
        r = db.execute("SELECT * FROM returns WHERE id=?", (return_id,)).fetchone()
        if not r:
            raise ApiError("not_found", "Return not found.", 404)
        if r["status"] != "requested":
            raise ApiError("already_resolved", "This return was already resolved.", 409)
        if decision == "reject":
            db.execute("UPDATE returns SET status='rejected' WHERE id=?", (return_id,))
        else:
            o = db.execute("SELECT * FROM orders WHERE id=?", (r["order_id"],)).fetchone()
            method = "original_payment" if o["payment_method"] == "online" else "wallet"
            db.execute("UPDATE returns SET status='approved', refund_amount=?, refund_method=? "
                       "WHERE id=?", (o["total"], method, return_id))
            db.execute("UPDATE orders SET status='refunded' WHERE id=?", (o["id"],))
            db.execute("INSERT INTO tracking (order_id,status,at) VALUES (?,?,?)",
                       (o["id"], "refunded", now()))
            if r["reason"] == "fake_product":
                db.execute("UPDATE products SET status='suspended' WHERE id=?",
                           (o["product_id"],))
        db.commit()
        return jsonify(row(db.execute("SELECT * FROM returns WHERE id=?",
                                      (return_id,)).fetchone()))

    return app


# ----------------------------------------------------------------------------
# Tiny demo front-end
# ----------------------------------------------------------------------------
DEMO_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartShop AI</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f6f7f9;color:#1c1f23}
header{background:#0b6e4f;color:#fff;padding:20px 16px}
header h1{margin:0;font-size:22px}header p{margin:4px 0 0;opacity:.9}
main{max-width:900px;margin:0 auto;padding:16px}
input,select,button{font:inherit;padding:8px 10px;border-radius:8px;border:1px solid #c9ced6}
button{background:#0b6e4f;color:#fff;border:0;cursor:pointer}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.card{background:#fff;border-radius:12px;padding:14px;margin-bottom:12px;box-shadow:0 1px 3px #0001}
.badge{display:inline-block;font-size:12px;padding:2px 8px;border-radius:99px;background:#e3f4ec;color:#0b6e4f;margin-right:6px}
.price{font-size:20px;font-weight:700}.muted{color:#5b6470;font-size:14px}
</style></head><body>
<header><h1>SmartShop AI</h1><p>Shop Smart, Shop Safe, Shop Easy.</p></header>
<main>
<div class="row">
<input id="q" placeholder="Search products">
<select id="city"><option>Lahore</option><option>Karachi</option><option>Islamabad</option><option>Multan</option><option>Peshawar</option></select>
<select id="sort"><option value="">Newest</option><option value="price_asc">Lowest final price</option><option value="trust">Most trusted seller</option></select>
<button onclick="load()">Search</button>
</div>
<div class="card"><b>Size advisor</b><div class="row" style="margin:8px 0 0">
<input id="h" type="number" placeholder="Height cm"><input id="w" type="number" placeholder="Weight kg">
<button onclick="size()">Find my size</button><span id="sz" class="muted"></span></div></div>
<div id="list"></div></main>
<script>
async function load(){
 const p=new URLSearchParams({q:q.value,city:city.value,sort:sort.value});
 const r=await (await fetch('/api/products?'+p)).json();
 list.innerHTML=r.items.map(i=>`<div class="card"><div><span class="badge">${i.seller.verified?'Verified seller':'Unverified'}</span><span class="badge">Trust ${i.seller.trust_score}</span></div>
 <h3 style="margin:8px 0">${i.title}</h3><div class="price">Rs ${i.final_price.toLocaleString()}</div>
 <div class="muted">Item Rs ${i.price.toLocaleString()} + delivery Rs ${i.delivery_fee} (final price, no surprises)<br>
 Arrives in ${i.eta_days.min}-${i.eta_days.max} days to ${i.delivery_city} &middot; ${i.stock} in stock</div></div>`).join('')||'<p class="muted">No products yet. Run: python app.py seed</p>';
}
async function size(){
 const r=await fetch('/api/size-advisor',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({category:'clothing',height_cm:h.value,weight_kg:w.value})});
 const j=await r.json();sz.textContent=r.ok?'Recommended size: '+j.recommended_size:j.message;
}
load();
</script></body></html>"""


def seed(app):
    """Insert a small demo catalogue."""
    with app.test_client() as c:
        sellers = [("Lahore Fashion House", "03001234567", "Lahore"),
                   ("Karachi Kicks", "03211234567", "Karachi")]
        ids = []
        for name, phone, city in sellers:
            sid = c.post("/api/sellers", json={"name": name, "phone": phone, "city": city}).get_json()["id"]
            c.post(f"/api/sellers/{sid}/verify")
            ids.append(sid)
        demo = [
            (ids[0], "Cotton Kurta", "clothing", 2200, ["S", "M", "L", "XL"], "Soft breathable cotton kurta, stitched in Lahore."),
            (ids[0], "Cotton Kurta Premium", "clothing", 2800, ["S", "M", "L", "XL"], "Premium cotton kurta with embroidered collar."),
            (ids[1], "Running Shoes", "shoes", 6500, ["40", "41", "42", "43"], "Lightweight running shoes with cushioned sole."),
            (ids[1], "Casual Sneakers", "shoes", 4200, ["40", "41", "42", "43"], "Everyday sneakers, durable rubber outsole."),
        ]
        for sid, title, cat, price, sizes, desc in demo:
            c.post("/api/products", json={"seller_id": sid, "title": title, "category": cat,
                                          "price": price, "stock": 25, "sizes": sizes,
                                          "description": desc, "image_url": "https://example.com/p.jpg"})


app = None

if __name__ == "__main__":
    app = create_app()
    if len(sys.argv) > 1 and sys.argv[1] == "seed":
        seed(app)
        print("Demo data added.")
    else:
        app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
