import os
import json
import sqlite3
from datetime import datetime
from backend.config import settings

# Local Fallback Database
LOCAL_DB_PATH = os.path.join(settings.BASE_DIR, "Data", "patientpulse_local.db")

def init_local_db():
    """Initializes local fallback database"""
    os.makedirs(os.path.dirname(LOCAL_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(LOCAL_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_logs (
            id TEXT PRIMARY KEY,
            patient_name TEXT,
            report_type TEXT,
            analyzed_metrics TEXT,
            summary_text TEXT,
            created_at TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prescription_history (
            id TEXT PRIMARY KEY,
            patient_name TEXT,
            raw_ocr_text TEXT,
            deciphered_medicines TEXT,
            food_warnings TEXT,
            created_at TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS xray_triage_records (
            id TEXT PRIMARY KEY,
            patient_name TEXT,
            pathology_predictions TEXT,
            triage_status TEXT,
            guidance_notes TEXT,
            heatmap_path TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_local_db()

# Supabase Client Initialization
try:
    from supabase import create_client, Client
    if settings.SUPABASE_URL and settings.SUPABASE_KEY and "your-" not in settings.SUPABASE_URL:
        supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    else:
        supabase = None
except Exception as e:
    supabase = None
    print(f"[Database] Notice: Supabase client running in dual/local mode: {e}")

def _save_local(table: str, data: dict):
    """Saves record locally in SQLite fallback database"""
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        keys = list(data.keys())
        values = []
        for k in keys:
            val = data[k]
            if isinstance(val, (dict, list)):
                values.append(json.dumps(val))
            else:
                values.append(str(val) if val is not None else None)
                
        placeholders = ", ".join(["?"] * len(keys))
        columns = ", ".join(keys)
        cursor.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", values)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[Local DB Error] {e}")
        return False

def log_lab_report(patient_name: str, report_type: str, metrics: list, summary: str):
    """Logs analyzed lab report results into Supabase and local backup SQLite"""
    record_id = f"lab_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    timestamp = datetime.utcnow().isoformat()
    
    payload = {
        "id": record_id,
        "patient_name": patient_name,
        "report_type": report_type,
        "analyzed_metrics": metrics,
        "summary_text": summary,
        "created_at": timestamp
    }
    
    # Always save to local database
    _save_local("patient_logs", payload)
    
    # Sync to Supabase if reachable
    if supabase:
        try:
            res = supabase.table("patient_logs").insert({
                "patient_name": patient_name,
                "report_type": report_type,
                "analyzed_metrics": metrics,
                "summary_text": summary
            }).execute()
            return {"status": "synced_supabase", "data": res.data}
        except Exception as e:
            return {"status": "saved_locally", "error": str(e)}
            
    return {"status": "saved_locally", "id": record_id}

def log_prescription(patient_name: str, raw_text: str, medicines: list, warnings: list):
    """Logs deciphered prescription and FDA warnings into Supabase and local backup SQLite"""
    record_id = f"rx_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    timestamp = datetime.utcnow().isoformat()
    
    payload = {
        "id": record_id,
        "patient_name": patient_name,
        "raw_ocr_text": raw_text,
        "deciphered_medicines": medicines,
        "food_warnings": warnings,
        "created_at": timestamp
    }
    
    _save_local("prescription_history", payload)
    
    if supabase:
        try:
            res = supabase.table("prescription_history").insert({
                "patient_name": patient_name,
                "raw_ocr_text": raw_text,
                "deciphered_medicines": medicines,
                "food_warnings": warnings
            }).execute()
            return {"status": "synced_supabase", "data": res.data}
        except Exception as e:
            return {"status": "saved_locally", "error": str(e)}
            
    return {"status": "saved_locally", "id": record_id}

def log_xray_triage(patient_name: str, predictions: dict, status: str, notes: str, heatmap_path: str = ""):
    """Logs Chest X-Ray triage results into Supabase and local backup SQLite"""
    record_id = f"xray_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    timestamp = datetime.utcnow().isoformat()
    
    payload = {
        "id": record_id,
        "patient_name": patient_name,
        "pathology_predictions": predictions,
        "triage_status": status,
        "guidance_notes": notes,
        "heatmap_path": heatmap_path,
        "created_at": timestamp
    }
    
    _save_local("xray_triage_records", payload)
    
    if supabase:
        try:
            res = supabase.table("xray_triage_records").insert({
                "patient_name": patient_name,
                "pathology_predictions": predictions,
                "triage_status": status,
                "guidance_notes": notes,
                "heatmap_path": heatmap_path
            }).execute()
            return {"status": "synced_supabase", "data": res.data}
        except Exception as e:
            return {"status": "saved_locally", "error": str(e)}
            
    return {"status": "saved_locally", "id": record_id}

def get_recent_records(limit: int = 10):
    """Retrieves recent logs across all medical records from live Supabase Cloud or local SQLite fallback"""
    if supabase:
        try:
            res_labs = supabase.table("patient_logs").select("*").order("created_at", desc=True).limit(limit).execute()
            res_rxs = supabase.table("prescription_history").select("*").order("created_at", desc=True).limit(limit).execute()
            res_xrays = supabase.table("xray_triage_records").select("*").order("created_at", desc=True).limit(limit).execute()
            return {
                "source": "live_supabase_cloud",
                "lab_reports": res_labs.data or [],
                "prescriptions": res_rxs.data or [],
                "xray_records": res_xrays.data or []
            }
        except Exception as e:
            # Table not created yet or network issue -> gracefully fallback to local
            pass

    conn = sqlite3.connect(LOCAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM patient_logs ORDER BY created_at DESC LIMIT ?", (limit,))
    labs = [dict(row) for row in cursor.fetchall()]
    
    cursor.execute("SELECT * FROM prescription_history ORDER BY created_at DESC LIMIT ?", (limit,))
    rxs = [dict(row) for row in cursor.fetchall()]
    
    cursor.execute("SELECT * FROM xray_triage_records ORDER BY created_at DESC LIMIT ?", (limit,))
    xrays = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return {
        "source": "local_sqlite_backup",
        "lab_reports": labs,
        "prescriptions": rxs,
        "xray_records": xrays
    }

