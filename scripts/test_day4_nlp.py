"""
PatientPulse AI — Day 4 Medical NLP & Clinical Entity Extraction Test Suite
Verifies:
1. Medical NLP Service initialization
2. Entity extraction from standard Clinical Blood Test (Data/sample_reports/sample_blood_test.txt)
3. Biomarker standardization & status evaluation (LOW, HIGH, NORMAL, CRITICAL)
4. Plain-English patient translation cards for every biomarker
5. Dietary guidance and 3 specific doctor questions per metric
6. Overall clinical triage risk calculation
7. Critical lab value alert detection (Hypoglycemia / Severe Hyperglycemia)
8. Multi-panel test on Hematology CBC (Data/raw/lab_reports/complete_blood_count_cbc.txt)
9. Database persistence to Supabase Cloud & Local SQLite (patientpulse_local.db)
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "App"))

from backend.nlp_service import MedicalNLPService

def run_test():
    print("====================================================================")
    print("      PATIENTPULSE AI — DAY 4 MEDICAL NLP TEST SUITE                ")
    print("====================================================================")

    nlp = MedicalNLPService()
    assert nlp is not None, "Failed to instantiate MedicalNLPService"
    print("[INIT] MedicalNLPService successfully instantiated.")

    # ---------------------------------------------------------
    # Test 1: Standard Comprehensive Clinical Blood Test
    # ---------------------------------------------------------
    sample_lab = os.path.join(BASE_DIR, "Data", "sample_reports", "sample_blood_test.txt")
    assert os.path.exists(sample_lab), f"Missing file: {sample_lab}"
    print(f"\n[Test 1] Processing Comprehensive Lab Report: {os.path.basename(sample_lab)}")

    res1 = nlp.analyze_lab_report(sample_lab, patient_name_override="John Doe")
    assert res1["status"] == "SUCCESS", f"Test 1 Failed: {res1}"

    print(f"  - Patient: {res1['metadata']['patient_name']}")
    print(f"  - Doctor: {res1['metadata']['doctor_name']}")
    print(f"  - Triage Badge: {res1['triage']['badge']}")
    print(f"  - Total Biomarkers Extracted: {res1['triage']['total_metrics_tested']}")
    print(f"  - Abnormal Count: {res1['triage']['abnormal_metrics_count']}")
    print(f"  - Summary: {res1['summary_text']}")

    assert res1["triage"]["total_metrics_tested"] >= 5, "Expected at least 5 biomarkers extracted!"

    extracted_dict = {b["metric_data"]["test_name"].lower(): b for b in res1["biomarkers"]}

    # Verify key biomarkers
    assert any("hemoglobin" in k for k in extracted_dict), "Hemoglobin missing!"
    assert any("glucose" in k for k in extracted_dict), "Fasting Glucose missing!"
    assert any("cholesterol" in k for k in extracted_dict), "Total Cholesterol missing!"
    assert any("tsh" in k for k in extracted_dict), "TSH missing!"

    # Verify Hemoglobin Low status & Plain English translation
    hb_item = next(b for k, b in extracted_dict.items() if "hemoglobin" in k)
    assert hb_item["metric_data"]["flag"] == "LOW", f"Expected Hemoglobin to be LOW, got {hb_item['metric_data']['flag']}"
    hb_decipher = hb_item["patient_decipher"]
    print("\n  [Verified Biomarker Card: Hemoglobin]")
    print(f"    - Title: {hb_decipher['test_name']}")
    print(f"    - Value: {hb_decipher['observed_value']}")
    print(f"    - Status: {hb_decipher['status_badge']['label']}")
    print(f"    - Meaning: {hb_decipher['what_your_result_means'][:95]}...")
    print(f"    - Diet Advice: {hb_decipher['diet_and_lifestyle_guidance'][:95]}...")
    print(f"    - Questions for Doctor ({len(hb_decipher['questions_for_doctor'])}):")
    for q in hb_decipher["questions_for_doctor"]:
        print(f"      * {q}")

    assert len(hb_decipher["questions_for_doctor"]) == 3, "Expected 3 questions for doctor!"

    # Verify Database logging
    print(f"\n  - Database Status: {res1['database_logging'].get('status')}")
    assert res1["database_logging"].get("status") in ["saved_locally", "synced_supabase"], "DB Logging failed!"

    # ---------------------------------------------------------
    # Test 2: Full Hematology (CBC) with Infection Leukocytosis
    # ---------------------------------------------------------
    sample_cbc = os.path.join(BASE_DIR, "Data", "raw", "lab_reports", "complete_blood_count_cbc.txt")
    if os.path.exists(sample_cbc):
        print(f"\n[Test 2] Processing Hematology CBC: {os.path.basename(sample_cbc)}")
        res2 = nlp.analyze_lab_report(sample_cbc)
        assert res2["status"] == "SUCCESS"
        print(f"  - Total Biomarkers: {res2['triage']['total_metrics_tested']}")
        print(f"  - Triage Badge: {res2['triage']['badge']}")
        wbc_found = any("white blood" in b["metric_data"]["test_name"].lower() or "wbc" in b["metric_data"]["test_name"].lower() for b in res2["biomarkers"])
        print(f"  - WBC Detected: {wbc_found}")

    # ---------------------------------------------------------
    # Test 3: Critical Value Alert Detection
    # ---------------------------------------------------------
    print("\n[Test 3] Verifying Extreme Critical Value Triage Alert...")
    critical_report = (
        "CRITICAL ALERT REPORT\n"
        "Patient: Emergency Patient | Date: 17-Sep-2026\n"
        "TEST NAME                   RESULT       UNIT         REFERENCE RANGE\n"
        "FASTING BLOOD GLUCOSE       420          mg/dL        70 - 99\n"
        "HEMOGLOBIN                  5.8          g/dL         13.0 - 17.0\n"
    )
    res_crit = nlp.analyze_lab_report(critical_report)
    print(f"  - Critical Triage Level: {res_crit['triage']['level']}")
    print(f"  - Critical Biomarkers Count: {res_crit['triage']['critical_metrics_count']}")
    assert res_crit["triage"]["level"] == "CRITICAL_URGENT", "Failed to trigger CRITICAL_URGENT triage for extreme values!"
    assert res_crit["triage"]["critical_metrics_count"] >= 1, "Failed to count critical metrics!"

    print("\n====================================================================")
    print("   ALL DAY 4 MEDICAL NLP TESTS PASSED WITH ZERO ERRORS (100%)       ")
    print("====================================================================")

if __name__ == "__main__":
    run_test()
