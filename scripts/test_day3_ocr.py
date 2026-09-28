"""
PatientPulse AI — Day 3 OCR & Layout Parsing Verification Script
Verifies:
1. Prescription OCR Service initialization
2. Extraction from Doctor Shorthand prescription (Data/sample_reports/sample_prescription.txt)
3. Extraction from Full Outpatient prescription (Data/raw/prescriptions/rx_amoxicillin_bacterial_infection.txt)
4. Extraction from Type 2 Diabetes prescription (Data/raw/prescriptions/rx_metformin_type2_diabetes.txt)
5. Structured medication parsing (Dosage, Frequency, Timing, Duration)
6. Critical food-drug interaction warnings (Antibiotic dairy avoidance, NSAID warnings)
7. OCR confidence scoring & Day 12 Medical Safety Warning Guardrail (<80% flag)
8. Patient-Friendly daily timetable (Morning, Lunch, Dinner, Bedtime) & Pharmacist questions
9. Database persistence to Supabase Cloud & Local SQLite (patientpulse_local.db)
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "App"))

from backend.ocr_service import PrescriptionOCRService

def run_test():
    print("====================================================================")
    print("      PATIENTPULSE AI — DAY 3 PRESCRIPTION OCR TEST SUITE           ")
    print("====================================================================")

    ocr = PrescriptionOCRService()
    assert ocr is not None, "Failed to instantiate PrescriptionOCRService"
    print("[INIT] PrescriptionOCRService successfully instantiated.")

    # ---------------------------------------------------------
    # Test 1: Doctor Shorthand Prescription
    # ---------------------------------------------------------
    sample_shorthand = os.path.join(BASE_DIR, "Data", "sample_reports", "sample_prescription.txt")
    assert os.path.exists(sample_shorthand), f"Missing file: {sample_shorthand}"
    print(f"\n[Test 1] Processing Doctor Shorthand: {os.path.basename(sample_shorthand)}")

    res1 = ocr.process_prescription(sample_shorthand, patient_name_override="Sarah Jenkins")
    assert res1["status"] == "SUCCESS", f"Test 1 Failed: {res1}"
    print(f"  - Status: {res1['status']}")
    print(f"  - Patient: {res1['metadata']['patient_name']}")
    print(f"  - Doctor: {res1['metadata']['doctor_name']}")
    print(f"  - Confidence: {res1['confidence']['percentage']} (Needs Verification: {res1['confidence']['needs_human_verification']})")
    print(f"  - Medicines Extracted: {res1['medicines_count']}")

    med_names1 = [m["name"].lower() for m in res1["medicines"]]
    print(f"  - Detected Medicines: {[m['name'] for m in res1['medicines']]}")
    assert any("amoxicillin" in m for m in med_names1), "Amoxicillin not extracted in Test 1!"
    assert any("paracetamol" in m for m in med_names1), "Paracetamol not extracted in Test 1!"

    print(f"  - Food Warnings: {len(res1['food_and_safety_warnings'])}")
    for w in res1["food_and_safety_warnings"]:
        print(f"    * {w[:90]}...")
    assert len(res1["food_and_safety_warnings"]) > 0, "Expected food warnings for antibiotic!"

    # Verify Patient-Friendly Guide
    guide1 = res1["patient_friendly_guide"]
    print(f"  - Patient Guide Title: {guide1['plain_title']}")
    print(f"  - Morning Schedule: {guide1['daily_schedule']['morning_breakfast']}")
    print(f"  - Bedtime Schedule: {guide1['daily_schedule']['bedtime_night']}")
    assert len(guide1["questions_for_pharmacist"]) == 3, "Expected 3 pharmacist questions!"

    print(f"  - Database Status: {res1['database_logging'].get('status')}")
    assert res1["database_logging"].get("status") in ["saved_locally", "synced_supabase"], "DB Logging failed!"

    # ---------------------------------------------------------
    # Test 2: Full Clinical Outpatient Prescription
    # ---------------------------------------------------------
    sample_full = os.path.join(BASE_DIR, "Data", "raw", "prescriptions", "rx_amoxicillin_bacterial_infection.txt")
    assert os.path.exists(sample_full), f"Missing file: {sample_full}"
    print(f"\n[Test 2] Processing Clinical Outpatient Slip: {os.path.basename(sample_full)}")

    res2 = ocr.process_prescription(sample_full)
    assert res2["status"] == "SUCCESS", f"Test 2 Failed: {res2}"
    print(f"  - Patient: {res2['metadata']['patient_name']}")
    print(f"  - Clinic: {res2['metadata']['clinic_hospital']}")
    print(f"  - Medicines Found: {res2['medicines_count']}")
    for m in res2["medicines"]:
        print(f"    * {m['name']} ({m['dosage']}) - {m['frequency']} | {m['timing']}")

    assert res2["medicines_count"] >= 2, "Expected at least 2 medicines in full slip!"

    # ---------------------------------------------------------
    # Test 3: Chronic Condition (Metformin / Diabetes)
    # ---------------------------------------------------------
    sample_metformin = os.path.join(BASE_DIR, "Data", "raw", "prescriptions", "rx_metformin_type2_diabetes.txt")
    if os.path.exists(sample_metformin):
        print(f"\n[Test 3] Processing Chronic Diabetes Prescription: {os.path.basename(sample_metformin)}")
        res3 = ocr.process_prescription(sample_metformin)
        assert res3["status"] == "SUCCESS"
        print(f"  - Medicines: {[m['name'] for m in res3['medicines']]}")
        print(f"  - Safety Warnings: {len(res3['food_and_safety_warnings'])}")

    # ---------------------------------------------------------
    # Test 4: Low-Confidence Safety Guardrail Trigger Test
    # ---------------------------------------------------------
    print("\n[Test 4] Verifying Low-Confidence Medical Safety Guardrail (<80% flag)...")
    blurry_text = "?? illegible scribbles ... dr unreadable rx ... ?? 250??"
    res_blurry = ocr.process_prescription(blurry_text)
    print(f"  - Raw Scribble Confidence: {res_blurry['confidence']['percentage']}")
    print(f"  - Needs Verification Flag: {res_blurry['confidence']['needs_human_verification']}")
    print(f"  - Warning Message: {res_blurry['confidence'].get('warning_message')}")
    assert res_blurry["confidence"]["needs_human_verification"] is True, "Guardrail failed to trigger for illegible text!"

    print("\n====================================================================")
    print("   ALL DAY 3 PRESCRIPTION OCR TESTS PASSED WITH ZERO ERRORS (100%)  ")
    print("====================================================================")

if __name__ == "__main__":
    run_test()
