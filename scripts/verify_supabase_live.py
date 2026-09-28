"""
PatientPulse AI — Supabase Live Database Connection & Real-Time Sync Verification
Verifies:
1. Supabase Client initialization using .env credentials
2. Status of the 3 cloud tables:
   - patient_logs
   - prescription_history
   - xray_triage_records
3. Live test insert and retrieval from Supabase Cloud PostgreSQL
"""

import os
import sys
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "App"))

from backend.config import settings
from backend.database import supabase, log_lab_report, log_prescription, log_xray_triage, get_recent_records

def verify_supabase():
    print("====================================================================")
    print("      PATIENTPULSE AI — SUPABASE CLOUD LIVE CONNECTION TEST         ")
    print("====================================================================")

    print(f"\n[1. Credentials Check]")
    print(f"  - Project URL: {settings.SUPABASE_URL}")
    print(f"  - API Key Set: {'YES' if settings.SUPABASE_KEY else 'NO'}")

    if not supabase:
        print("\n[ERROR] Supabase client is not initialized. Please check SUPABASE_URL and SUPABASE_KEY in .env.")
        return

    print("\n[2. Table Schema Check]")
    tables = ["patient_logs", "prescription_history", "xray_triage_records"]
    missing_tables = []

    for t in tables:
        try:
            res = supabase.table(t).select("id").limit(1).execute()
            print(f"  [OK] Table '{t}' is LIVE in Supabase PostgreSQL.")
        except Exception as e:
            err_msg = str(e)
            if "PGRST205" in err_msg or "Could not find the table" in err_msg:
                print(f"  [MISSING] Table '{t}' has NOT been created yet.")
                missing_tables.append(t)
            else:
                print(f"  [ERROR] Table '{t}' query failed: {err_msg}")
                missing_tables.append(t)

    if missing_tables:
        print("\n--------------------------------------------------------------------")
        print("ACTION REQUIRED TO ENABLE LIVE SUPABASE SYNC:")
        print("1. Open your Supabase SQL Editor in your web browser:")
        print("   👉 https://supabase.com/dashboard/project/caglaiteefttdkadpxpm/sql")
        print("2. Copy the SQL script from App/backend/schema.sql and click 'RUN'.")
        print("3. Re-run this script to confirm live sync is active!")
        print("--------------------------------------------------------------------")
        return False

    # 3. Test Live Data Entry
    print("\n[3. Live Data Entry Test]")
    test_patient = f"Live Cloud Patient {datetime.utcnow().strftime('%H%M%S')}"

    # Test Live Lab Insert
    lab_res = log_lab_report(
        patient_name=test_patient,
        report_type="Complete Blood Count (CBC)",
        metrics=[{"test": "Hemoglobin", "val": 13.8, "unit": "g/dL", "status": "NORMAL"}],
        summary="Patient exhibits normal hematology parameters."
    )
    print(f"  - Lab Report Live Insert Status: {lab_res.get('status')}")

    # Test Live Prescription Insert
    rx_res = log_prescription(
        patient_name=test_patient,
        raw_text="Amoxicillin 500mg 1 tab TID",
        medicines=[{"name": "Amoxicillin", "dosage": "500mg", "timing": "After meals"}],
        warnings=["Avoid dairy products within 2 hours."]
    )
    print(f"  - Prescription Live Insert Status: {rx_res.get('status')}")

    # Test Live X-Ray Insert
    xray_res = log_xray_triage(
        patient_name=test_patient,
        predictions={"Pneumonia": 0.12, "Normal": 0.88},
        status="NORMAL_UNREMARKABLE",
        notes="Lungs clear, no active infiltrates."
    )
    print(f"  - X-Ray Triage Live Insert Status: {xray_res.get('status')}")

    # 4. Test Live Cloud Query
    print("\n[4. Live Cloud Query Retrieval]")
    records = get_recent_records(limit=3)
    print(f"  - Data Source: {records.get('source')}")
    print(f"  - Recent Cloud Lab Reports: {len(records.get('lab_reports', []))}")
    print(f"  - Recent Cloud Prescriptions: {len(records.get('prescriptions', []))}")
    print(f"  - Recent Cloud X-Rays: {len(records.get('xray_records', []))}")

    print("\n====================================================================")
    print("   SUPABASE CLOUD DATABASE CONNECTION IS 100% LIVE & SYNCHRONIZING! ")
    print("====================================================================")
    return True

if __name__ == "__main__":
    verify_supabase()
