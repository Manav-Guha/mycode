"""Expense Tracker — Demo Flask + SQLite application.

A minimal REST API for tracking personal expenses.  Used as a test
target for the myCode stress-testing pipeline.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import Flask, g, jsonify, request

app = Flask(__name__)

DATABASE = os.environ.get(
    "EXPENSE_DB",
    str(Path(__file__).parent / "expenses.db"),
)


# ── Database helpers ──


def get_db():
    """Return a per-request database connection (stored on ``g``)."""
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create the expenses table if it does not exist."""
    db = get_db()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL NOT NULL,
            category TEXT NOT NULL DEFAULT 'uncategorized',
            description TEXT,
            date TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    db.commit()


# ── Routes ──


@app.route("/expenses", methods=["GET"])
def list_expenses():
    """List all expenses, optionally filtered by category."""
    db = get_db()
    category = request.args.get("category")
    if category:
        rows = db.execute(
            "SELECT * FROM expenses WHERE category = ? ORDER BY date DESC",
            (category,),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM expenses ORDER BY date DESC"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/expenses", methods=["POST"])
def create_expense():
    """Create a new expense entry."""
    data = request.get_json()
    if not data or "amount" not in data:
        return jsonify({"error": "amount is required"}), 400

    amount = data["amount"]
    category = data.get("category", "uncategorized")
    description = data.get("description", "")
    date = data.get("date", datetime.now().strftime("%Y-%m-%d"))
    created_at = datetime.now().isoformat()

    db = get_db()
    cursor = db.execute(
        "INSERT INTO expenses (amount, category, description, date, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (amount, category, description, date, created_at),
    )
    db.commit()
    return jsonify({"id": cursor.lastrowid}), 201


@app.route("/expenses/<int:expense_id>", methods=["GET"])
def get_expense(expense_id):
    """Retrieve a single expense by ID."""
    db = get_db()
    row = db.execute(
        "SELECT * FROM expenses WHERE id = ?", (expense_id,)
    ).fetchone()
    if row is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(dict(row))


@app.route("/expenses/<int:expense_id>", methods=["DELETE"])
def delete_expense(expense_id):
    """Delete an expense by ID."""
    db = get_db()
    db.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
    db.commit()
    return "", 204


@app.route("/summary", methods=["GET"])
def expense_summary():
    """Return spending summary grouped by category."""
    db = get_db()
    rows = db.execute(
        "SELECT category, COUNT(*) as count, SUM(amount) as total "
        "FROM expenses GROUP BY category ORDER BY total DESC"
    ).fetchall()
    return jsonify([dict(r) for r in rows])


# ── Startup ──


with app.app_context():
    init_db()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
