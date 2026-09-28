"""
PatientPulse AI — Visual Demo & Interactive Model Inspector Generator
Generates a modern, patient-first dark-mode HTML inspector (App/frontend/xray_viewer.html)
Features:
1. Dual-View Mode:
   - [Patient Decipher Mode]: 100% Plain-English translation for everyday people without medical training.
     Explains: What it means in simple words, why the red box is there, common signs, reassurance, and 3 doctor questions.
   - [Clinical Specialist Mode]: Deep Learning DenseNet-121 probabilities, AUROC thresholding, and physician checklist.
2. Multi-Anatomy & Multi-Disease Support (Chest Pulmonary cases + Extremity Bone Fracture case).
3. Interactive Case Switcher and Bounding Box Overlay inspection.
"""

import os
import sys
import json
import base64
from PIL import Image
import io

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "App"))

from backend.predict_xray import xray_predictor


def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(img_path)[1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{base64.b64encode(data).decode('utf-8')}"


def generate_interactive_ui():
    print("[UI Generator] Running universal model inference on clinical X-ray cases...")

    xray_dir = os.path.join(BASE_DIR, "Data", "raw", "chest_xrays")
    cases = [
        {
            "id": "CASE-NIH-001",
            "patient_label": "Patient #00000001 (Cardiomegaly / Enlarged Heart)",
            "file": os.path.join(xray_dir, "00000001_000.png"),
            "notes": "NIH ChestX-ray14 benchmark reference scan."
        },
        {
            "id": "CASE-ACUTE-058",
            "patient_label": "Patient #058 (Acute Pneumonia / Lung Infection)",
            "file": os.path.join(xray_dir, "covid-19-pneumonia-58-prior.jpg"),
            "notes": "Emergency department presentation with bilateral lung opacities."
        },
        {
            "id": "CASE-CLINICAL-002",
            "patient_label": "Patient #002 (Pleural Effusion / Fluid Buildup)",
            "file": os.path.join(xray_dir, "000001-1.jpg"),
            "notes": "Post-admission pulmonary monitoring radiograph."
        },
        {
            "id": "CASE-FRACTURE-001",
            "patient_label": "Patient #F092 (Hand / Metacarpal Bone Fracture)",
            "file": os.path.join(BASE_DIR, "Data", "raw", "xrays", "sample_fracture_2.jpg"),
            "notes": "Emergency orthopedic presentation with traumatic skeletal injury."
        }
    ]

    processed_cases = []
    for c in cases:
        if not os.path.exists(c["file"]):
            print(f"  ! Skipping missing file: {c['file']}")
            continue

        print(f"  + Running inference for {c['id']} ({os.path.basename(c['file'])})...")
        pred = xray_predictor.predict(c["file"])
        orig_b64 = image_to_base64(c["file"])
        cam_b64 = pred["gradcam_localization"]["heatmap_base64"]

        processed_cases.append({
            "id": c["id"],
            "patient_label": c["patient_label"],
            "notes": c["notes"],
            "filename": os.path.basename(c["file"]),
            "original_image": orig_b64,
            "heatmap_image": cam_b64,
            "anatomy": pred.get("anatomy_detected", {}),
            "triage": pred["triage"],
            "patient_decipher": pred.get("patient_friendly_decipher", {}),
            "plain_english_summary": pred["plain_english_summary"],
            "physician_checklist": pred["physician_checklist"],
            "top_probabilities": pred["top_pathology_probabilities"],
            "bounding_box": pred["gradcam_localization"]["bounding_box"],
            "detected_objects": pred.get("detected_objects", [])
        })

    cases_json = json.dumps(processed_cases)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PatientPulse AI — Universal Medical X-Ray Decipher & Visual Inspector</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #070a12;
            --card-bg: rgba(15, 23, 42, 0.78);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-cyan: #06b6d4;
            --accent-blue: #3b82f6;
            --accent-emerald: #10b981;
            --accent-rose: #ef4444;
            --accent-amber: #f59e0b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            background-image: radial-gradient(circle at 10% 15%, rgba(6, 182, 212, 0.08) 0%, transparent 40%),
                              radial-gradient(circle at 90% 85%, rgba(239, 68, 68, 0.06) 0%, transparent 40%);
            padding: 24px;
        }}

        .container {{
            max-width: 1420px;
            margin: 0 auto;
        }}

        /* Header Bar */
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            padding: 20px 28px;
            border-radius: 20px;
            margin-bottom: 24px;
            box-shadow: 0 12px 36px rgba(0,0,0,0.35);
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .brand-icon {{
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #06b6d4, #3b82f6);
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            box-shadow: 0 0 24px rgba(6, 182, 212, 0.45);
        }}

        .brand h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.3px;
            color: #fff;
        }}

        .brand p {{
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 2px;
        }}

        .engine-badge {{
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.35);
            padding: 8px 16px;
            border-radius: 30px;
            font-size: 12px;
            font-weight: 600;
            color: var(--accent-emerald);
        }}

        .engine-dot {{
            width: 8px;
            height: 8px;
            background: var(--accent-emerald);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--accent-emerald);
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.4; transform: scale(0.85); }}
        }}

        /* Case Navigation Tabs */
        .case-selector {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 14px;
            margin-bottom: 24px;
        }}

        .case-tab {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            padding: 16px 20px;
            border-radius: 16px;
            cursor: pointer;
            transition: all 0.25s ease;
            position: relative;
            overflow: hidden;
        }}

        .case-tab:hover {{
            border-color: rgba(6, 182, 212, 0.4);
            transform: translateY(-2px);
        }}

        .case-tab.active {{
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.14), rgba(59, 130, 246, 0.08));
            border-color: var(--accent-cyan);
            box-shadow: 0 6px 20px rgba(6, 182, 212, 0.18);
        }}

        .case-tab.active::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue));
        }}

        .tab-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 15px;
            font-weight: 600;
            margin-bottom: 4px;
            color: #fff;
        }}

        .tab-sub {{
            font-size: 12px;
            color: var(--text-muted);
        }}

        /* Main Dashboard Grid */
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}

        @media (max-width: 1080px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .card {{
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
            padding-bottom: 14px;
            border-bottom: 1px solid var(--card-border);
        }}

        .card-title {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-family: 'Outfit', sans-serif;
            font-size: 18px;
            font-weight: 700;
            color: #fff;
        }}

        /* Left Column: Visual Viewer */
        .viewer-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
        }}

        .view-panel {{
            position: relative;
            border-radius: 14px;
            overflow: hidden;
            background: #000;
            border: 1px solid rgba(255, 255, 255, 0.08);
            aspect-ratio: 1 / 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .view-img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }}

        .view-label {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.75);
            backdrop-filter: blur(8px);
            color: #fff;
            padding: 5px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border: 1px solid rgba(255,255,255,0.1);
        }}

        .bbox-banner {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(6, 182, 212, 0.06);
            border: 1px solid rgba(6, 182, 212, 0.2);
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 13px;
        }}

        .bbox-badge {{
            background: #ef4444;
            color: #fff;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }}

        /* Mode Switcher Pill Bar */
        .view-mode-tabs {{
            display: flex;
            background: rgba(0, 0, 0, 0.4);
            padding: 4px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 18px;
            gap: 4px;
        }}

        .mode-btn {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 10px 14px;
            border-radius: 9px;
            font-family: 'Outfit', sans-serif;
            font-size: 13px;
            font-weight: 600;
            background: transparent;
            color: var(--text-muted);
            border: none;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .mode-btn.active {{
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            color: #fff;
            box-shadow: 0 4px 14px rgba(6, 182, 212, 0.35);
        }}

        /* Patient Plain-English Decipher Mode Styles */
        .patient-hero {{
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.12), rgba(59, 130, 246, 0.04));
            border: 1px solid rgba(6, 182, 212, 0.3);
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 18px;
        }}

        .patient-title-row {{
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            margin-bottom: 6px;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .patient-headline {{
            font-family: 'Outfit', sans-serif;
            font-size: 24px;
            font-weight: 800;
            color: #fff;
        }}

        .patient-pill {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 11px;
            color: var(--text-muted);
            font-weight: 600;
        }}

        .patient-sub {{
            font-size: 14px;
            color: var(--accent-cyan);
            font-weight: 500;
            margin-bottom: 10px;
        }}

        .patient-badge-reassurance {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-emerald);
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.25);
        }}

        /* 4-Box Patient Explanation Grid */
        .patient-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-bottom: 18px;
        }}

        @media (max-width: 600px) {{
            .patient-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .patient-card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 16px;
            transition: all 0.2s ease;
        }}

        .patient-card:hover {{
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.12);
        }}

        .card-subhead {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-family: 'Outfit', sans-serif;
            font-size: 13px;
            font-weight: 700;
            color: var(--accent-cyan);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }}

        .card-body-text {{
            font-size: 13px;
            line-height: 1.6;
            color: #e2e8f0;
        }}

        .symptom-list {{
            list-style: none;
            margin-top: 4px;
        }}

        .symptom-list li {{
            font-size: 12px;
            color: #cbd5e1;
            margin-bottom: 6px;
            display: flex;
            align-items: flex-start;
            gap: 8px;
            line-height: 1.4;
        }}

        .symptom-list li::before {{
            content: '•';
            color: var(--accent-amber);
            font-size: 16px;
            line-height: 12px;
        }}

        /* Doctor Questions Card */
        .questions-box {{
            background: rgba(59, 130, 246, 0.06);
            border: 1px solid rgba(59, 130, 246, 0.25);
            border-radius: 14px;
            padding: 16px 18px;
        }}

        .questions-box h4 {{
            font-family: 'Outfit', sans-serif;
            font-size: 14px;
            font-weight: 700;
            color: #93c5fd;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
        }}

        .question-item {{
            display: flex;
            align-items: flex-start;
            gap: 10px;
            font-size: 13px;
            color: #f1f5f9;
            background: rgba(0, 0, 0, 0.35);
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 8px;
            border-left: 3px solid var(--accent-blue);
            line-height: 1.5;
        }}

        /* Clinical Mode Elements */
        .clinical-panel {{
            display: none;
        }}

        .triage-hero {{
            padding: 20px;
            border-radius: 16px;
            margin-bottom: 20px;
            border: 1px solid transparent;
        }}

        .triage-hero.urgent {{
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.05));
            border-color: rgba(239, 68, 68, 0.35);
        }}

        .triage-hero.mild {{
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.05));
            border-color: rgba(245, 158, 11, 0.35);
        }}

        .triage-hero.normal {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05));
            border-color: rgba(16, 185, 129, 0.35);
        }}

        .triage-badge {{
            font-size: 13px;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 20px;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255,255,255,0.1);
        }}

        .prob-section h3 {{
            font-family: 'Outfit', sans-serif;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .prob-row {{
            margin-bottom: 12px;
        }}

        .prob-meta {{
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 6px;
        }}

        .prob-bar-bg {{
            height: 8px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            overflow: hidden;
        }}

        .prob-bar-fill {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.6s ease;
        }}

        .checklist {{
            margin-top: 20px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 16px;
        }}

        .checklist h4 {{
            font-family: 'Outfit', sans-serif;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 10px;
            color: var(--accent-cyan);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .checklist-item {{
            display: flex;
            align-items: flex-start;
            gap: 10px;
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 8px;
            line-height: 1.4;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Top Navigation Bar -->
        <header>
            <div class="brand">
                <div class="brand-icon">🩻</div>
                <div>
                    <h1>PatientPulse AI — Universal Medical X-Ray Decipher & Visual Intelligence</h1>
                    <p>Plain-English Patient Translation • Deep Learning Vision • Lesion Object Detection</p>
                </div>
            </div>
            <div class="engine-badge">
                <span class="engine-dot"></span>
                <span>Universal AI Engine Active</span>
            </div>
        </header>

        <!-- Case Selection Tabs -->
        <div class="case-selector" id="caseSelector"></div>

        <!-- Main Workspace -->
        <div class="dashboard-grid">
            <!-- Left Panel: Dual Radiograph Viewer -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <span>🖼️ Visual Attention & Lesion Detection</span>
                    </div>
                    <span id="activeCaseLabel" style="font-size: 13px; color: var(--accent-cyan); font-weight: 600;"></span>
                </div>

                <div class="viewer-container">
                    <div class="view-panel">
                        <span class="view-label">Original Radiograph</span>
                        <img id="origImg" class="view-img" src="" alt="Original Radiograph">
                    </div>
                    <div class="view-panel">
                        <span class="view-label" style="background: rgba(239, 68, 68, 0.85);">Visual Lesion Hotspot</span>
                        <img id="camImg" class="view-img" src="" alt="Grad-CAM Overlay">
                    </div>
                </div>

                <div class="bbox-banner" id="bboxBanner">
                    <div>
                        <span class="bbox-badge">HOTSPOT DETECTED</span>
                        <span style="margin-left: 8px; font-weight: 500;" id="bboxText">Calculating localized region of interest...</span>
                    </div>
                    <span style="color: var(--text-muted); font-size: 12px;" id="anatomyBadge">Anatomy: Chest / Thorax</span>
                </div>
            </div>

            <!-- Right Panel: Patient Decipher & Clinical Guidance -->
            <div class="card">
                <!-- Mode Switcher Pill Bar -->
                <div class="view-mode-tabs">
                    <button class="mode-btn active" id="btnPatientMode" onclick="switchMode('patient')">
                        <span>💡</span>
                        <span>Patient Plain-English Decipher</span>
                    </button>
                    <button class="mode-btn" id="btnClinicalMode" onclick="switchMode('clinical')">
                        <span>👨‍⚕️</span>
                        <span>Clinical Specialist (Doctor View)</span>
                    </button>
                </div>

                <!-- 1. PATIENT PLAIN-ENGLISH DECIPHER PANEL (Active by Default) -->
                <div id="patientPanel">
                    <!-- Hero Translation Card -->
                    <div class="patient-hero">
                        <div class="patient-title-row">
                            <h2 class="patient-headline" id="patientTitle">Finding Title</h2>
                            <span class="patient-pill" id="patientMedicalTag">Medical Term</span>
                        </div>
                        <div class="patient-sub" id="patientSubtitle">Subtitle</div>
                        <span class="patient-badge-reassurance">✓ Plain English Explanation for Non-Medical Patients</span>
                    </div>

                    <!-- 4-Card Explanation Grid -->
                    <div class="patient-grid">
                        <div class="patient-card">
                            <div class="card-subhead">
                                <span>📖</span>
                                <span>In Plain English</span>
                            </div>
                            <p class="card-body-text" id="patientMeaning">Explanation text...</p>
                        </div>

                        <div class="patient-card">
                            <div class="card-subhead">
                                <span>🎯</span>
                                <span>Why is the Red Box There?</span>
                            </div>
                            <p class="card-body-text" id="patientBox">Box explanation...</p>
                        </div>

                        <div class="patient-card">
                            <div class="card-subhead">
                                <span>🩺</span>
                                <span>Common Symptoms to Notice</span>
                            </div>
                            <ul class="symptom-list" id="patientSymptoms"></ul>
                        </div>

                        <div class="patient-card">
                            <div class="card-subhead">
                                <span>🛡️</span>
                                <span>What You Should Do Next</span>
                            </div>
                            <p class="card-body-text" id="patientAction">Action steps...</p>
                        </div>
                    </div>

                    <!-- 3 Doctor Questions Box -->
                    <div class="questions-box">
                        <h4>💬 3 Recommended Questions to Ask Your Doctor</h4>
                        <div id="patientQuestions"></div>
                    </div>
                </div>

                <!-- 2. CLINICAL SPECIALIST PANEL -->
                <div id="clinicalPanel" class="clinical-panel">
                    <div class="card-header" style="margin-top: 0; padding-top: 0;">
                        <span style="font-family: 'Outfit'; font-weight: 700; font-size: 16px;">Diagnostic Classification</span>
                        <span id="primaryBadge" class="triage-badge"></span>
                    </div>

                    <div class="triage-hero urgent" id="triageHero">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span style="font-size: 12px; color: var(--text-muted); font-weight: 600;">PRIMARY PATHOLOGY FINDING</span>
                            <span style="font-family: 'Outfit'; font-size: 26px; font-weight: 800;" id="confidenceScore">0.0%</span>
                        </div>
                        <h3 id="primaryPathology" style="font-family: 'Outfit'; font-size: 20px; font-weight: 700; margin-bottom: 6px;">Pathology Name</h3>
                        <p id="plainSummary" style="font-size: 13px; line-height: 1.5; color: #e2e8f0;"></p>
                    </div>

                    <!-- Probability Bars -->
                    <div class="prob-section">
                        <h3>Pathology Probabilities Distribution</h3>
                        <div id="probBars"></div>
                    </div>

                    <!-- Physician Action Checklist -->
                    <div class="checklist">
                        <h4>📋 Physician Review Checklist</h4>
                        <div id="checklistItems"></div>
                    </div>
                </div>

            </div>
        </div>
    </div>

    <script>
        const cases = {cases_json};
        let currentIdx = 0;
        let currentMode = 'patient';

        function switchMode(mode) {{
            currentMode = mode;
            document.getElementById('btnPatientMode').classList.toggle('active', mode === 'patient');
            document.getElementById('btnClinicalMode').classList.toggle('active', mode === 'clinical');

            document.getElementById('patientPanel').style.display = (mode === 'patient') ? 'block' : 'none';
            document.getElementById('clinicalPanel').style.display = (mode === 'clinical') ? 'block' : 'none';
        }}

        function renderCaseSelector() {{
            const container = document.getElementById('caseSelector');
            container.innerHTML = '';
            cases.forEach((c, idx) => {{
                const tab = document.createElement('div');
                tab.className = `case-tab ${{idx === currentIdx ? 'active' : ''}}`;
                tab.id = `tab-${{idx}}`;
                tab.onclick = () => selectCase(idx);
                tab.innerHTML = `
                    <div class="tab-title">${{c.patient_label}}</div>
                    <div class="tab-sub">${{c.anatomy.body_part || 'Scan'}} • Primary: ${{c.triage.primary_finding}} (${{c.triage.confidence_percentage}})</div>
                `;
                container.appendChild(tab);
            }});
        }}

        function selectCase(idx) {{
            currentIdx = idx;
            document.querySelectorAll('.case-tab').forEach((t, i) => {{
                t.classList.toggle('active', i === idx);
            }});

            const c = cases[idx];
            document.getElementById('activeCaseLabel').textContent = c.patient_label;
            document.getElementById('origImg').src = c.original_image;
            document.getElementById('camImg').src = c.heatmap_image;

            // Bounding Box
            const bbox = c.bounding_box;
            if (bbox) {{
                document.getElementById('bboxText').textContent = 
                    `${{c.triage.primary_finding}} located at [x=${{bbox.x_min}}, y=${{bbox.y_min}}, w=${{bbox.width}}, h=${{bbox.height}}]`;
            }} else {{
                document.getElementById('bboxText').textContent = 'No acute focal hotspot detected above threshold.';
            }}

            document.getElementById('anatomyBadge').textContent = `Anatomy: ${{c.anatomy.body_part || 'Chest / Thorax'}}`;

            // ==========================================
            // POPULATE PATIENT PLAIN-ENGLISH PANEL
            // ==========================================
            const pDecipher = c.patient_decipher || {{}};
            document.getElementById('patientTitle').textContent = pDecipher.plain_title || c.triage.primary_finding;
            document.getElementById('patientMedicalTag').textContent = `Medical Term: ${{c.triage.primary_finding}}`;
            document.getElementById('patientSubtitle').textContent = pDecipher.subtitle || `Visual analysis detected indicators for ${{c.triage.primary_finding}} (${{c.triage.confidence_percentage}} match)`;
            document.getElementById('patientMeaning').textContent = pDecipher.what_it_means || c.plain_english_summary;
            document.getElementById('patientBox').textContent = pDecipher.what_the_red_box_shows || `The red highlighted box directly outlines the area on the scan where the visual finding was detected.`;
            document.getElementById('patientAction').textContent = pDecipher.what_you_should_do || "Please discuss these visual findings with your treating healthcare provider for definitive clinical advice.";

            // Symptoms
            const symptomsList = document.getElementById('patientSymptoms');
            symptomsList.innerHTML = '';
            const symptoms = pDecipher.common_signs || ["Discuss your physical symptoms with your healthcare provider."];
            symptoms.forEach(s => {{
                symptomsList.innerHTML += `<li>${{s}}</li>`;
            }});

            // 3 Doctor Questions
            const qBox = document.getElementById('patientQuestions');
            qBox.innerHTML = '';
            const questions = pDecipher.questions_for_doctor || [
                `Could my physical symptoms be related to this ${{c.triage.primary_finding}} finding?`,
                `Do you recommend any follow-up tests or repeat radiographs?`,
                `What treatment or observation plan is best for me?`
            ];
            questions.forEach((q, qIdx) => {{
                qBox.innerHTML += `
                    <div class="question-item">
                        <span style="font-weight: 700; color: #60a5fa;">Q${{qIdx + 1}}:</span>
                        <span>"${{q}}"</span>
                    </div>
                `;
            }});

            // ==========================================
            // POPULATE CLINICAL SPECIALIST PANEL
            // ==========================================
            const hero = document.getElementById('triageHero');
            hero.className = 'triage-hero ' + (c.triage.level.includes('URGENT') ? 'urgent' : (c.triage.level.includes('MILD') ? 'mild' : 'normal'));
            document.getElementById('primaryBadge').textContent = c.triage.badge;
            document.getElementById('primaryBadge').style.borderColor = c.triage.color_code;
            document.getElementById('primaryBadge').style.color = c.triage.color_code;
            document.getElementById('primaryPathology').textContent = c.triage.primary_finding;
            document.getElementById('confidenceScore').textContent = c.triage.confidence_percentage;
            document.getElementById('plainSummary').textContent = c.plain_english_summary;

            // Probabilities
            const bars = document.getElementById('probBars');
            bars.innerHTML = '';
            c.top_probabilities.forEach(p => {{
                const pct = (p.probability * 100).toFixed(1);
                let color = '#06b6d4';
                if (p.probability >= 0.50) color = '#ef4444';
                else if (p.probability >= 0.30) color = '#f59e0b';

                bars.innerHTML += `
                    <div class="prob-row">
                        <div class="prob-meta">
                            <span>${{p.pathology}}</span>
                            <span style="color: ${{color}}; font-weight: 700;">${{pct}}%</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width: ${{pct}}%; background: ${{color}};"></div>
                        </div>
                    </div>
                `;
            }});

            // Checklist
            const cl = document.getElementById('checklistItems');
            cl.innerHTML = '';
            c.physician_checklist.forEach(item => {{
                cl.innerHTML += `
                    <div class="checklist-item">
                        <span style="color: var(--accent-cyan); font-weight: bold;">✓</span>
                        <span>${{item}}</span>
                    </div>
                `;
            }});
        }}

        // Initialize
        renderCaseSelector();
        selectCase(0);
    </script>
</body>
</html>"""

    output_path = os.path.join(BASE_DIR, "App", "frontend", "xray_viewer.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[UI Generator] Successfully generated visual inspector at: {output_path}")


if __name__ == "__main__":
    generate_interactive_ui()
