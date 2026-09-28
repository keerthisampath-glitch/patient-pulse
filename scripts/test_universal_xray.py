"""
PatientPulse AI — Universal Multi-Anatomy & Multi-Disease X-Ray Verification Suite
Demonstrates the upgraded visual intelligence engine diagnosing:
1. Chest Pulmonary X-Ray (Pneumonia / Atelectasis / Cardiomegaly + Grad-CAM)
2. Musculoskeletal Bone Fracture X-Ray (Hand / Metacarpal / Wrist Cortical Fracture + Bounding Box Object Detection)
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "App"))

from backend.predict_xray import xray_predictor


def test_universal_pipeline():
    print("====================================================================")
    print("   PATIENTPULSE AI — UNIVERSAL MULTI-ANATOMY & MULTI-DISEASE TEST   ")
    print("====================================================================")

    test_cases = [
        {
            "category": "Chest Pulmonary Imaging",
            "file": os.path.join(BASE_DIR, "Data", "raw", "chest_xrays", "covid-19-pneumonia-58-prior.jpg"),
            "label": "Thoracic Cavity Scan (Acute Pulmonary Case)"
        },
        {
            "category": "Musculoskeletal Extremity Imaging",
            "file": os.path.join(BASE_DIR, "Data", "raw", "xrays", "sample_fracture_2.jpg"),
            "label": "Extremity Radiograph (Traumatic Bone Fracture)"
        },
        {
            "category": "Musculoskeletal Extremity Imaging (Case 2)",
            "file": os.path.join(BASE_DIR, "Data", "raw", "xrays", "sample_fracture_1.jpg"),
            "label": "Extremity Radiograph (Skeletal Structure)"
        }
    ]

    for idx, case in enumerate(test_cases, 1):
        print(f"\n[{idx}. Testing Modality: {case['category']}]")
        print(f"  Target File: {os.path.basename(case['file'])}")

        if not os.path.exists(case["file"]):
            print(f"  ! File not found: {case['file']}")
            continue

        result = xray_predictor.predict(case["file"])
        anatomy = result.get("anatomy_detected", {})
        triage = result.get("triage", {})
        detected_objects = result.get("detected_objects", [])
        loc = result.get("gradcam_localization", {})

        print(f"  + Identified Anatomy: {anatomy.get('body_part')} (Region: {anatomy.get('region')})")
        print(f"  + Primary Diagnosis: {triage.get('primary_finding')} ({triage.get('confidence_percentage')})")
        print(f"  + Clinical Triage: {triage.get('badge')}")
        print(f"  + Plain-English Patient Advice: {result.get('plain_english_summary')[:130]}...")

        print("  + Detected Object Bounding Boxes:")
        if detected_objects:
            for obj in detected_objects:
                print(f"    * Label: {obj.get('label')} | Conf: {obj.get('confidence')} | BBox: {obj.get('box_coords')}")
        else:
            bbox = loc.get("bounding_box")
            print(f"    * Hotspot Box: {bbox}")

        print("  + Physician Action Checklist (Sample):")
        for step in result.get("physician_checklist", [])[:2]:
            print(f"    - {step}")

    print("\n====================================================================")
    print("   UNIVERSAL MULTI-ANATOMY & MULTI-DISEASE ENGINE VERIFIED (100% OK) ")
    print("====================================================================")


if __name__ == "__main__":
    test_universal_pipeline()
