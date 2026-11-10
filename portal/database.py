"""
Database Layer for Quantellix Chatbot & Order Fulfillment Portal.
Uses Neon PostgreSQL (DATABASE_URL) as the single authoritative database
for all Licenses, Authentication, and Solution Credits across the platform.
"""

import os
import json
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse, unquote
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
import threading
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv(override=False)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"), override=False)

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

def _clean_str(val: Optional[str]) -> str:
    if not val:
        return ""
    return str(val).strip().strip('"').strip("'")

# Global Neon Connection Pool
_pool = None
_pool_lock = threading.Lock()

def _create_fresh_connection():
    p = urlparse(DATABASE_URL)
    conn = psycopg2.connect(
        host=p.hostname,
        user=unquote(p.username or ''),
        password=unquote(p.password or ''),
        port=p.port or 5432,
        dbname=unquote((p.path or '/').lstrip('/') or ''),
        sslmode='require',
        connect_timeout=15,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5
    )
    conn.autocommit = True
    return conn

def get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                p = urlparse(DATABASE_URL)
                _pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    host=p.hostname,
                    user=unquote(p.username or ''),
                    password=unquote(p.password or ''),
                    port=p.port or 5432,
                    dbname=unquote((p.path or '/').lstrip('/') or ''),
                    sslmode='require',
                    connect_timeout=15
                )
    return _pool

class NeonConnContext:
    def __enter__(self):
        self.conn = None
        self.from_pool = False
        try:
            self.conn = get_pool().getconn()
            if self.conn.closed != 0:
                raise psycopg2.OperationalError("Connection is closed")
            # Verify socket is actually alive against Neon serverless
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1;")
            self.from_pool = True
            self.conn.autocommit = True
            return self.conn
        except Exception:
            # Pooled connection was dropped by Neon during idle or long tasks; discard and open fresh
            if self.conn:
                try:
                    get_pool().putconn(self.conn, close=True)
                except Exception:
                    pass
            self.conn = _create_fresh_connection()
            self.from_pool = False
            return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is not None or self.conn.closed != 0:
                # On connection error, close socket and do not put bad connection back into pool
                if self.from_pool:
                    try:
                        get_pool().putconn(self.conn, close=True)
                    except Exception:
                        pass
                else:
                    try:
                        self.conn.close()
                    except Exception:
                        pass
            else:
                if self.from_pool:
                    try:
                        get_pool().putconn(self.conn)
                    except Exception:
                        try:
                            self.conn.close()
                        except Exception:
                            pass
                else:
                    try:
                        self.conn.close()
                    except Exception:
                        pass

def get_neon_connection():
    return NeonConnContext()

def init_db():
    try:
        with get_neon_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id VARCHAR(255) PRIMARY KEY,
                        user_name VARCHAR(255) DEFAULT '',
                        email VARCHAR(255) DEFAULT '',
                        stage VARCHAR(100) DEFAULT 'awaiting_name',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS user_leads (
                        id SERIAL PRIMARY KEY,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        user_name VARCHAR(255) DEFAULT '',
                        last_session_id VARCHAR(255) DEFAULT '',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS uploaded_files (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255),
                        filename VARCHAR(500),
                        file_path TEXT,
                        file_size_bytes BIGINT,
                        row_count INTEGER,
                        col_count INTEGER,
                        uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS eda_reports (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255),
                        file_id INTEGER,
                        summary_json TEXT,
                        report_html_path TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS agent_approvals (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        user_name VARCHAR(255) NOT NULL,
                        dossier_version VARCHAR(100) NOT NULL,
                        action_id VARCHAR(100) NOT NULL,
                        decision VARCHAR(50) NOT NULL,
                        reason TEXT DEFAULT '',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_agent_approvals_session ON agent_approvals (session_id, action_id, id DESC);
                """)
    except Exception as e:
        print(f"[database] Note on Neon database init: {e}")

# --- Neon PostgreSQL License & Credits Functions ---

def generate_license_key() -> str:
    parts = ["SRAI"]
    for _ in range(3):
        parts.append(secrets.token_hex(2).upper())
    return "-".join(parts)

def create_or_get_license(email: str, initial_credits: int = 0) -> Tuple[Dict[str, Any], bool]:
    """
    Creates or retrieves a lifetime license key.
    Follows SriyaAI-Unified standard: Free Lifetime License, 0 starter credits.
    """
    email = _clean_str(email).lower()
    now = datetime.utcnow()
    
    with get_neon_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check if user already has an active license
            cur.execute("SELECT * FROM licenses_licensekey WHERE LOWER(owner_email) = LOWER(%s) AND is_active = TRUE ORDER BY id DESC LIMIT 1", (email,))
            existing = cur.fetchone()
            if existing:
                return dict(existing), False

            # Create new lifetime license with 0 starter credits (matching SriyaAI-Unified)
            key = generate_license_key()
            cur.execute("""
                INSERT INTO licenses_licensekey (
                    key, owner_email, is_active, created_at, 
                    included_scan_credits, scan_credits_granted, 
                    notes, activation_email_sent, term_months
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
            """, (key, email, True, now, initial_credits, initial_credits > 0, f"Free lifetime open-account · {email}", False, 0))
            new_lic = cur.fetchone()
            
            # If explicit initial credits were requested (>0), grant them safely
            if initial_credits > 0:
                lic_id = new_lic.get("id") or 1
                expires_at = now + timedelta(days=30)
                for idx in range(initial_credits):
                    cur.execute("""
                        INSERT INTO portal_solutioncredit (
                            user_email, status, solution_slug, amount_usd, stripe_checkout_session_id,
                            purchased_at, expires_at, pack_sku, consumed_run_id
                        )
                        VALUES (%s, 'paid', '', 0.0, %s, %s, %s, 'license', '')
                    """, (email, f"lic-{lic_id}-{idx}", now, expires_at))
                
            return dict(new_lic), True

def verify_license(key: str, email: str) -> Dict[str, Any]:
    key = _clean_str(key).upper()
    email = _clean_str(email).lower()
    now = datetime.utcnow()

    if not key or not email:
        return {"valid": False, "message": "Invalid license key. Please check and try again."}

    try:
        with get_neon_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM licenses_licensekey WHERE key = %s LIMIT 1", (key,))
                lic = cur.fetchone()

                if not lic:
                    return {"valid": False, "message": "Invalid license key. Please check and try again."}

                owner = (lic.get("owner_email") or "").strip().lower()
                if owner and owner != email:
                    return {"valid": False, "message": "Invalid license key. Please check and try again."}

                if not lic.get("is_active"):
                    return {"valid": False, "message": "This license key has been deactivated. Please contact support."}

                first_use = False
                if not lic.get("first_used_at"):
                    cur.execute("UPDATE licenses_licensekey SET first_used_at = %s WHERE key = %s", (now, key))
                    lic["first_used_at"] = now
                    first_use = True

                return {"valid": True, "message": "License verified.", "license": lic, "first_use": first_use}
    except Exception as e:
        print(f"[database] Neon verify_license exception: {e}")
        return {"valid": False, "message": f"Database error: {e}"}

_USER_CREDITS_CACHE = {}

def get_user_credits(email: str) -> int:
    email = _clean_str(email).lower()
    if not email:
        return 0
    import time
    now = time.time()
    # Cache hit within 60s
    if email in _USER_CREDITS_CACHE:
        ts, cached_val = _USER_CREDITS_CACHE[email]
        if (now - ts) < 60.0:
            return cached_val

    for attempt in range(2):
        try:
            with get_neon_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT COUNT(*) 
                        FROM portal_solutioncredit 
                        WHERE LOWER(user_email) = LOWER(%s) AND status IN ('paid', 'active', 'available')
                    """, (email,))
                    row = cur.fetchone()
                    count = int(row[0]) if row else 0
                    _USER_CREDITS_CACHE[email] = (now, count)
                    return count
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            if attempt == 0:
                continue
            print(f"[database] Neon get_user_credits error: {e}")
            return 0
        except Exception as e:
            print(f"[database] Neon get_user_credits error: {e}")
            return 0
    return 0

def deduct_credit(email: str, solution_slug: str = "", run_id: str = "") -> bool:
    """Deducts 1 credit from the user's available credits in Neon PostgreSQL."""
    email = _clean_str(email).lower()
    _USER_CREDITS_CACHE.pop(email, None)
    if not email:
        return False
    for attempt in range(2):
        try:
            with get_neon_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE portal_solutioncredit
                        SET status = 'consumed',
                            consumed_at = CURRENT_TIMESTAMP,
                            solution_slug = CASE WHEN %s != '' THEN %s ELSE solution_slug END,
                            consumed_run_id = CASE WHEN %s != '' THEN %s ELSE consumed_run_id END
                        WHERE id = (
                            SELECT id FROM portal_solutioncredit
                            WHERE LOWER(user_email) = LOWER(%s) AND status IN ('paid', 'active', 'available')
                            ORDER BY id ASC
                            LIMIT 1
                        )
                    """, (solution_slug, solution_slug, run_id, run_id, email))
                    if cur.rowcount > 0:
                        print(f"[database] Successfully deducted 1 credit for {email} (solution: {solution_slug}, run_id: {run_id})")
                        return True
                    else:
                        print(f"[database] No available credits found to deduct for {email}")
                        return False
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            print(f"[database] Neon deduct_credit connection retry (attempt {attempt+1}/2): {e}")
            if attempt == 0:
                continue
            return False
        except Exception as e:
            print(f"[database] Neon deduct_credit error: {e}")
            return False
    return False

def get_credit_ledger(email: str) -> List[Dict[str, Any]]:
    email = _clean_str(email).lower()
    try:
        with get_neon_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, user_email, status, pack_sku, purchased_at as created_at, expires_at, amount_usd 
                    FROM portal_solutioncredit 
                    WHERE LOWER(user_email) = LOWER(%s) 
                    ORDER BY id DESC LIMIT 50
                """, (email,))
                records = [dict(r) for r in cur.fetchall()]
                for rec in records:
                    if rec.get("created_at"):
                        rec["created_at"] = str(rec["created_at"])
                    if rec.get("expires_at"):
                        rec["expires_at"] = str(rec["expires_at"])
                    rec["pack_name"] = rec.get("pack_sku") or "Solution Credit"
                    rec["credits_granted"] = 1
                    rec["credits_remaining"] = 1 if rec.get("status") in ('paid', 'active', 'available') else 0
                return records
    except Exception as e:
        print(f"[database] Neon get_credit_ledger error: {e}")
        return []

def record_credit_purchase(email: str, pack_name: str, credits: int, amount_paid: float) -> int:
    email = _clean_str(email).lower()
    now = datetime.utcnow()
    expires_at = now + timedelta(days=30)
    per_unit = round(amount_paid / max(credits, 1), 2)
    sku = str(pack_name or "pack")[:8]
    
    with get_neon_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO portal_creditpurchase (
                        user_email, pack_sku, credits_qty, list_usd, discount_pct, paid_usd,
                        terms_version, terms_accepted_at, stripe_checkout_session_id, expires_at, created_at
                    ) VALUES (%s, %s, %s, %s, 0.0, %s, '2026-03-04', %s, %s, %s, %s)
                """, (email, str(credits), credits, amount_paid, amount_paid, now, f"dev-grant-{int(now.timestamp())}", expires_at, now))
            except Exception as pe:
                print(f"[database] portal_creditpurchase insert error: {pe}")

            for idx in range(credits):
                session_id = f"buy-{int(now.timestamp())}-{idx}"
                cur.execute("""
                    INSERT INTO portal_solutioncredit (
                        user_email, status, solution_slug, amount_usd, stripe_checkout_session_id,
                        purchased_at, expires_at, pack_sku, consumed_run_id
                    )
                    VALUES (%s, 'paid', '', %s, %s, %s, %s, %s, '')
                """, (email, per_unit, session_id, now, expires_at, sku))
            return credits

def _format_acct_date(dt) -> str:
    if not dt:
        return "—"
    return dt.strftime("%b %d, %Y").replace(" 0", " ")

def _format_acct_datetime(dt) -> str:
    if not dt:
        return "—"
    s = dt.strftime("%b %d, %Y, %I:%M %p").replace(" 0", " ")
    return s.replace("AM", "a.m.").replace("PM", "p.m.")

def _format_solution_title(slug: str) -> str:
    if not slug:
        return "Solution Analysis"
    mapping = {
        "order-validation": "Order Validation",
        "customer-order": "Customer Order",
        "last-mile-delivery": "Last Mile Delivery",
        "logistics-tracking": "Logistics Tracking",
        "transportation": "Transportation",
        "dispatch": "Dispatch",
        "forecast-sales-demand": "Forecast Sales Demand",
        "minimize-forecast-variance": "Minimize Forecast Variance",
        "reduce-stockouts": "Reduce Stockouts",
        "reduce-delivery-lead-time": "Reduce Delivery Lead Time",
        "reduce-margin-leakage": "Reduce Margin Leakage",
        "reduce-inbound-defect-risk": "Reduce Inbound Defect Risk",
        "improve-supplier-quality-protection": "Reduce Inbound Defect Risk",
        "optimize-inventory-holding": "Optimize Inventory Holding",
        "improve-order-fill-rate": "Improve Order Fill Rate",
        "ensure-regulatory-compliance": "Ensure Regulatory Compliance",
        "automate-cross-border-trade": "Automate Cross-Border Trade"
    }
    cleaned = slug.strip().lower()
    return mapping.get(cleaned, cleaned.replace("-", " ").replace("_", " ").title())

def get_account_data(email: str, key: str = "") -> Dict[str, Any]:
    email = _clean_str(email).lower()
    key = _clean_str(key).upper()
    now = datetime.utcnow()
    
    result = {
        "email": email,
        "license_key": key,
        "is_active": True,
        "term": "Lifetime",
        "activated_str": "Oct 5, 2026, 6:46 p.m.",
        "expiry_str": "No expiry",
        "days_remaining_str": "Lifetime",
        "available_credits": 0,
        "earliest_expiry_str": "—",
        "lots": [],
        "recent_purchases": [],
        "used_credits": []
    }
    
    try:
        with get_neon_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # 1. License info
                if key:
                    cur.execute("SELECT * FROM licenses_licensekey WHERE key = %s LIMIT 1", (key,))
                else:
                    cur.execute("SELECT * FROM licenses_licensekey WHERE LOWER(owner_email) = %s ORDER BY id DESC LIMIT 1", (email,))
                lic = cur.fetchone()
                if lic:
                    result["license_key"] = lic.get("key") or key
                    result["is_active"] = bool(lic.get("is_active", True))
                    first_used = lic.get("first_used_at")
                    if first_used:
                        result["activated_str"] = _format_acct_datetime(first_used)
                    else:
                        result["activated_str"] = "Active"
                    
                    if lic.get("expires_at"):
                        result["expiry_str"] = _format_acct_datetime(lic.get("expires_at"))
                        days = (lic.get("expires_at") - now).days
                        result["days_remaining_str"] = f"{max(days, 0)} days"
                    else:
                        result["expiry_str"] = "No expiry"
                        result["days_remaining_str"] = "Lifetime"
                
                # 2. Available credits count
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM portal_solutioncredit 
                    WHERE LOWER(user_email) = %s AND status IN ('paid', 'active', 'available')
                """, (email,))
                result["available_credits"] = int(cur.fetchone()["count"])
                
                # 3. Available lots grouped by purchase lot / expiry day
                cur.execute("""
                    SELECT id, pack_sku, purchase_id, purchased_at, expires_at
                    FROM portal_solutioncredit
                    WHERE LOWER(user_email) = %s AND status IN ('paid', 'active', 'available')
                    ORDER BY expires_at ASC, purchased_at ASC, id ASC
                """, (email,))
                raw_credits = cur.fetchall()
                
                lots_map = {}
                for r in raw_credits:
                    p_id = r.get("purchase_id")
                    p_date = r.get("purchased_at")
                    p_day = p_date.date() if p_date else None
                    e_date = r.get("expires_at")
                    e_day = e_date.date() if e_date else None
                    sku = str(r.get("pack_sku") or "").strip()
                    
                    # Group by purchase_id or by same purchase date, expiry date and pack sku
                    group_key = ("pid", p_id) if p_id else ("lot", p_day, e_day, sku.lower())
                    if group_key not in lots_map:
                        lots_map[group_key] = {
                            "purchased_at": p_date,
                            "expires_at": e_date,
                            "pack_sku": sku,
                            "remaining": 0
                        }
                    else:
                        if p_date and (lots_map[group_key]["purchased_at"] is None or p_date < lots_map[group_key]["purchased_at"]):
                            lots_map[group_key]["purchased_at"] = p_date
                        if e_date and (lots_map[group_key]["expires_at"] is None or e_date < lots_map[group_key]["expires_at"]):
                            lots_map[group_key]["expires_at"] = e_date
                    lots_map[group_key]["remaining"] += 1
                
                lots = []
                for lot in lots_map.values():
                    sku_clean = (lot["pack_sku"] or "").lower()
                    rem = lot["remaining"]
                    if sku_clean in ("license", "license bundle", "license_bundle") or not sku_clean:
                        pack_label = "License bundle"
                    elif sku_clean in ("1", "1 credit", "1_credit"):
                        pack_label = f"{rem} credit" if rem == 1 else f"{rem} credits"
                    elif sku_clean in ("3", "3 credits"):
                        pack_label = f"{rem} credits"
                    elif sku_clean in ("5", "5 credits"):
                        pack_label = f"{rem} credits"
                    elif sku_clean in ("10", "10 credits"):
                        pack_label = f"{rem} credits"
                    else:
                        pack_label = f"{rem} credit" if rem == 1 else f"{rem} credits"
                    
                    lots.append({
                        "purchased_str": _format_acct_date(lot["purchased_at"]),
                        "pack_name": pack_label,
                        "pack_qty": rem,
                        "expires_str": _format_acct_date(lot["expires_at"]),
                        "remaining": rem
                    })
                
                result["lots"] = lots
                if lots:
                    result["earliest_expiry_str"] = lots[0]["expires_str"]
                
                # 4. Recent purchases
                cur.execute("""
                    SELECT * FROM portal_creditpurchase 
                    WHERE LOWER(user_email) = %s 
                    ORDER BY id DESC
                """, (email,))
                raw_purchases = cur.fetchall()
                purchases = []
                for p in raw_purchases:
                    c_dt = p.get("created_at")
                    e_dt = p.get("expires_at")
                    purchases.append({
                        "purchased_str": _format_acct_datetime(c_dt),
                        "pack_sku": p.get("pack_sku") or "Credit Pack",
                        "credits_qty": p.get("credits_qty") or 1,
                        "paid_usd": f"{p.get('paid_usd', 0):.2f}",
                        "expires_str": _format_acct_date(e_dt)
                    })
                result["recent_purchases"] = purchases
                
                # 5. Used credits
                cur.execute("""
                    SELECT solution_slug, consumed_at, consumed_run_id 
                    FROM portal_solutioncredit 
                    WHERE LOWER(user_email) = %s AND status = 'consumed' 
                    ORDER BY consumed_at DESC NULLS LAST, id DESC
                """, (email,))
                raw_used = cur.fetchall()
                used = []
                for u in raw_used:
                    cons_dt = u.get("consumed_at")
                    used.append({
                        "solution_name": _format_solution_title(u.get("solution_slug")),
                        "consumed_str": _format_acct_datetime(cons_dt),
                        "consumed_date_str": _format_acct_date(cons_dt)
                    })
                result["used_credits"] = used

    except Exception as e:
        print(f"[database] get_account_data error: {e}")
        
    return result

# --- Cloud Neon PostgreSQL Session, Lead & EDA Functions ---

def save_user_lead(email: str, user_name: str = "", session_id: str = "") -> Dict[str, Any]:
    """
    Saves or updates user lead info (email, name, session_id) in Neon PostgreSQL Cloud Store
    so that reports and intelligence can be sent to this email in the future.
    """
    email = _clean_str(email).lower()
    user_name = _clean_str(user_name).strip()
    now_dt = datetime.utcnow()

    # 1. Update session in Neon if provided
    if session_id:
        update_session(session_id, user_name=user_name if user_name else None, email=email)

    # 2. Update/insert in Neon Postgres user_leads
    try:
        with get_neon_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_leads (email, user_name, last_session_id, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (email) DO UPDATE SET
                        user_name = CASE WHEN EXCLUDED.user_name != '' THEN EXCLUDED.user_name ELSE user_leads.user_name END,
                        last_session_id = EXCLUDED.last_session_id,
                        updated_at = EXCLUDED.updated_at
                """, (email, user_name, session_id, now_dt, now_dt))
                # Also ensure license record exists in Neon
                create_or_get_license(email)
    except Exception as e:
        print(f"[database] Neon save_user_lead error: {e}")

    return {"email": email, "user_name": user_name, "session_id": session_id}

def get_user_lead(email: str) -> Optional[Dict[str, Any]]:
    email = _clean_str(email).lower()
    try:
        with get_neon_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM user_leads WHERE LOWER(email) = %s", (email,))
                row = cur.fetchone()
                if row:
                    row_dict = dict(row)
                    if isinstance(row_dict.get("created_at"), datetime):
                        row_dict["created_at"] = row_dict["created_at"].isoformat()
                    if isinstance(row_dict.get("updated_at"), datetime):
                        row_dict["updated_at"] = row_dict["updated_at"].isoformat()
                    return row_dict
                return None
    except Exception as e:
        print(f"[database] get_user_lead error: {e}")
        return None

def get_or_create_session(session_id: str, user_name: str = "", stage: str = "awaiting_name") -> Dict[str, Any]:
    now_dt = datetime.utcnow()
    try:
        with get_neon_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM sessions WHERE session_id = %s", (session_id,))
                row = cur.fetchone()
                if row:
                    row_dict = dict(row)
                    if isinstance(row_dict.get("created_at"), datetime):
                        row_dict["created_at"] = row_dict["created_at"].isoformat()
                    if isinstance(row_dict.get("updated_at"), datetime):
                        row_dict["updated_at"] = row_dict["updated_at"].isoformat()
                    return row_dict

                cur.execute("""
                    INSERT INTO sessions (session_id, user_name, email, stage, created_at, updated_at)
                    VALUES (%s, %s, '', %s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE SET updated_at = EXCLUDED.updated_at
                    RETURNING *
                """, (session_id, user_name, stage, now_dt, now_dt))
                new_row = cur.fetchone()
                if new_row:
                    res = dict(new_row)
                    if isinstance(res.get("created_at"), datetime):
                        res["created_at"] = res["created_at"].isoformat()
                    if isinstance(res.get("updated_at"), datetime):
                        res["updated_at"] = res["updated_at"].isoformat()
                    return res
    except Exception as e:
        print(f"[database] get_or_create_session error: {e}")
    return {"session_id": session_id, "user_name": user_name, "email": "", "stage": stage, "created_at": now_dt.isoformat(), "updated_at": now_dt.isoformat()}

def update_session(session_id: str, user_name: Optional[str] = None, email: Optional[str] = None, stage: Optional[str] = None) -> Dict[str, Any]:
    now_dt = datetime.utcnow()
    try:
        with get_neon_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM sessions WHERE session_id = %s", (session_id,))
                existing = cur.fetchone()
                if not existing:
                    return get_or_create_session(session_id, user_name or "", stage or "awaiting_name")

                updates = ["updated_at = %s"]
                params = [now_dt]
                if user_name is not None:
                    updates.append("user_name = %s")
                    params.append(user_name)
                if email is not None:
                    updates.append("email = %s")
                    params.append(email)
                if stage is not None:
                    updates.append("stage = %s")
                    params.append(stage)
                params.append(session_id)
                query = f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = %s RETURNING *"
                cur.execute(query, params)
                updated = cur.fetchone()
                if updated:
                    res = dict(updated)
                    if isinstance(res.get("created_at"), datetime):
                        res["created_at"] = res["created_at"].isoformat()
                    if isinstance(res.get("updated_at"), datetime):
                        res["updated_at"] = res["updated_at"].isoformat()
                    return res
    except Exception as e:
        print(f"[database] update_session error: {e}")
    return get_or_create_session(session_id, user_name or "", stage or "awaiting_name")

def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    try:
        with get_neon_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM sessions WHERE session_id = %s", (session_id,))
                row = cur.fetchone()
                if row:
                    res = dict(row)
                    if isinstance(res.get("created_at"), datetime):
                        res["created_at"] = res["created_at"].isoformat()
                    if isinstance(res.get("updated_at"), datetime):
                        res["updated_at"] = res["updated_at"].isoformat()
                    return res
                return None
    except Exception as e:
        print(f"[database] get_session error: {e}")
        return None

def record_uploaded_file(session_id: str, filename: str, file_path: str, file_size_bytes: int, row_count: int, col_count: int) -> int:
    now_dt = datetime.utcnow()
    try:
        with get_neon_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO uploaded_files (session_id, filename, file_path, file_size_bytes, row_count, col_count, uploaded_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (session_id, filename, file_path, file_size_bytes, row_count, col_count, now_dt))
                row = cur.fetchone()
                return row[0] if row else 0
    except Exception as e:
        print(f"[database] record_uploaded_file error: {e}")
        return 0

def record_eda_report(session_id: str, file_id: int, summary_dict: Dict[str, Any], report_html_path: str) -> int:
    now_dt = datetime.utcnow()
    try:
        with get_neon_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO eda_reports (session_id, file_id, summary_json, report_html_path, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (session_id, file_id, json.dumps(summary_dict), report_html_path, now_dt))
                row = cur.fetchone()
                return row[0] if row else 0
    except Exception as e:
        print(f"[database] record_eda_report error: {e}")
        return 0

def get_latest_eda_report(session_id: str) -> Optional[Dict[str, Any]]:
    try:
        with get_neon_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT r.*, f.filename, f.file_path as source_file_path, f.file_size_bytes
                    FROM eda_reports r
                    LEFT JOIN uploaded_files f ON r.file_id = f.id
                    WHERE r.session_id = %s
                    ORDER BY r.id DESC LIMIT 1
                """, (session_id,))
                row = cur.fetchone()
                if row:
                    res = dict(row)
                    if isinstance(res.get("created_at"), datetime):
                        res["created_at"] = res["created_at"].isoformat()
                    if res.get("summary_json"):
                        try:
                            res["summary"] = json.loads(res["summary_json"])
                        except Exception:
                            res["summary"] = {}
                    return res
                return None
    except Exception as e:
        print(f"[database] get_latest_eda_report error: {e}")
        return None

def clear_session_data(session_id: str):
    try:
        with get_neon_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM eda_reports WHERE session_id = %s", (session_id,))
                cur.execute("DELETE FROM uploaded_files WHERE session_id = %s", (session_id,))
                cur.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
    except Exception as e:
        print(f"[database] clear_session_data error: {e}")


# --- Append-Only Decision Approval Audit Store ---

_LOCAL_APPROVALS_FALLBACK: List[Dict[str, Any]] = []
_APPROVALS_LOCK = threading.Lock()


def record_approval(
    session_id: str,
    user_name: str,
    dossier_version: str,
    action_id: str,
    decision: str,
    reason: str = "",
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Appends an immutable decision record to the agent_approvals DB table.
    Append-only: preserves complete audit history; later decisions supersede earlier ones.
    """
    from datetime import timezone
    now_iso = timestamp or datetime.now(timezone.utc).isoformat()
    record = {
        "session_id": session_id,
        "user_name": user_name,
        "dossier_version": dossier_version,
        "action_id": action_id,
        "decision": decision,
        "reason": reason,
        "created_at": now_iso,
    }
    recorded_db = False
    try:
        with get_neon_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO agent_approvals (session_id, user_name, dossier_version, action_id, decision, reason, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (session_id, user_name, dossier_version, action_id, decision, reason, now_iso))
                row = cur.fetchone()
                if row:
                    record["id"] = row[0]
                    recorded_db = True
    except Exception:
        pass

    with _APPROVALS_LOCK:
        if not recorded_db:
            record["id"] = len(_LOCAL_APPROVALS_FALLBACK) + 1
        _LOCAL_APPROVALS_FALLBACK.append(record)

    return record


def get_latest_approval(session_id: str, action_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves the latest superseding decision row for an action."""
    try:
        with get_neon_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM agent_approvals
                    WHERE session_id = %s AND action_id = %s
                    ORDER BY id DESC LIMIT 1
                """, (session_id, action_id))
                row = cur.fetchone()
                if row:
                    res = dict(row)
                    if isinstance(res.get("created_at"), datetime):
                        res["created_at"] = res["created_at"].isoformat()
                    return res
    except Exception:
        pass

    with _APPROVALS_LOCK:
        matches = [
            r for r in _LOCAL_APPROVALS_FALLBACK
            if r["session_id"] == session_id and r["action_id"] == action_id
        ]
        return matches[-1] if matches else None


def get_approvals_history(session_id: str, action_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves complete append-only audit trail for session actions."""
    try:
        with get_neon_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if action_id:
                    cur.execute("""
                        SELECT * FROM agent_approvals
                        WHERE session_id = %s AND action_id = %s
                        ORDER BY id ASC
                    """, (session_id, action_id))
                else:
                    cur.execute("""
                        SELECT * FROM agent_approvals
                        WHERE session_id = %s
                        ORDER BY id ASC
                    """, (session_id,))
                rows = cur.fetchall()
                if rows:
                    res_list = []
                    for row in rows:
                        r = dict(row)
                        if isinstance(r.get("created_at"), datetime):
                            r["created_at"] = r["created_at"].isoformat()
                        res_list.append(r)
                    return res_list
    except Exception:
        pass

    with _APPROVALS_LOCK:
        return [
            r for r in _LOCAL_APPROVALS_FALLBACK
            if r["session_id"] == session_id and (action_id is None or r["action_id"] == action_id)
        ]


init_db()

