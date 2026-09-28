"""
PatientPulse AI — Lab & Prescription Decipher Visual Viewer Generator
Generates App/frontend/lab_decipher_viewer.html:
Interactive visual demo for Medical Report Decipherer (nlp_service.py)
and Doctor Prescription Decipherer (ocr_service.py).
Includes:
- Dual-View Mode: [💡 Patient Plain-English Decipher] vs [👨‍⚕️ Clinical Specialist View]
- Biomarker Risk Gauges & Plain-English Explanation Cards
- 4-Slot Daily Medication Timetable (Morning, Lunch, Dinner, Bedtime)
- Critical Food-Drug Interaction Alerts
- 3 Questions to Ask Your Doctor / Pharmacist
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "App"))

from backend.nlp_service import nlp_service
from backend.ocr_service import ocr_service

HTML_OUTPUT = os.path.join(BASE_DIR, "App", "frontend", "lab_decipher_viewer.html")

def generate_viewer():
    print("[Viewer Generator] Running Medical Report & Prescription Decipherers...")

    # Case 1: Metabolic & CBC Blood Test
    lab_file = os.path.join(BASE_DIR, "Data", "sample_reports", "sample_blood_test.txt")
    lab_res = nlp_service.analyze_lab_report(lab_file)

    # Case 2: Outpatient CBC Hematology
    cbc_file = os.path.join(BASE_DIR, "Data", "raw", "lab_reports", "complete_blood_count_cbc.txt")
    cbc_res = nlp_service.analyze_lab_report(cbc_file) if os.path.exists(cbc_file) else lab_res

    # Case 3: Doctor Shorthand Prescription
    rx_file = os.path.join(BASE_DIR, "Data", "sample_reports", "sample_prescription.txt")
    rx_res = ocr_service.process_prescription(rx_file)

    # Case 4: Full Outpatient Prescription Slip
    rx_full_file = os.path.join(BASE_DIR, "Data", "raw", "prescriptions", "rx_amoxicillin_bacterial_infection.txt")
    rx_full_res = ocr_service.process_prescription(rx_full_file) if os.path.exists(rx_full_file) else rx_res

    payload = {
        "lab_cases": [
            {
                "id": "LAB-001",
                "title": "Comprehensive Blood & Metabolic Panel (John Doe)",
                "type": "LAB_REPORT",
                "data": lab_res
            },
            {
                "id": "LAB-002",
                "title": "Hematology CBC Panel with Leukocytosis (Michael Chang)",
                "type": "LAB_REPORT",
                "data": cbc_res
            }
        ],
        "rx_cases": [
            {
                "id": "RX-001",
                "title": "Doctor Shorthand Prescription (Sarah Jenkins)",
                "type": "PRESCRIPTION",
                "data": rx_res
            },
            {
                "id": "RX-002",
                "title": "Clinical Outpatient Antibiotic Slip (Emily Watson)",
                "type": "PRESCRIPTION",
                "data": rx_full_res
            }
        ]
    }

    payload_json = json.dumps(payload, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PatientPulse AI — Medical Report & Prescription Decipher Viewer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #07090e;
      --card-bg: #0e131f;
      --card-border: #1e293b;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.2);
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --purple: #8b5cf6;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: var(--bg);
      color: var(--text-main);
      min-height: 100vh;
      padding: 24px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 24px;
    }}
    .logo-area {{ display: flex; align-items: center; gap: 14px; }}
    .logo-badge {{
      background: linear-gradient(135deg, #0284c7, #8b5cf6);
      width: 44px; height: 44px; border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 22px; font-weight: 800; color: white;
    }}
    .title-box h1 {{ font-size: 22px; font-weight: 700; }}
    .title-box p {{ font-size: 13px; color: var(--text-muted); }}
    
    .nav-tabs {{
      display: flex; gap: 10px; background: #0b0f19; padding: 6px;
      border-radius: 12px; border: 1px solid var(--card-border);
    }}
    .nav-btn {{
      background: transparent; border: none; color: var(--text-muted);
      padding: 8px 18px; border-radius: 8px; cursor: pointer;
      font-weight: 600; font-size: 13px; transition: all 0.2s;
    }}
    .nav-btn.active {{
      background: var(--card-bg); color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.4);
      box-shadow: 0 0 12px var(--accent-glow);
    }}
    
    /* View Switcher */
    .view-switcher-bar {{
      background: linear-gradient(90deg, #111827, #0b1329);
      border: 1px solid #1e293b; border-radius: 16px;
      padding: 14px 20px; display: flex; justify-content: space-between;
      align-items: center; margin-bottom: 24px;
    }}
    .switcher-btns {{ display: flex; gap: 8px; }}
    .switch-btn {{
      padding: 10px 18px; border-radius: 10px; font-size: 14px;
      font-weight: 600; cursor: pointer; border: 1px solid transparent;
      display: flex; align-items: center; gap: 8px; transition: all 0.2s;
    }}
    .switch-btn.active {{
      background: #0284c7; color: white;
      border-color: #38bdf8; box-shadow: 0 0 16px rgba(56, 189, 248, 0.4);
    }}
    .switch-btn.inactive {{
      background: #1e293b; color: #94a3b8;
    }}

    .container {{
      display: grid; grid-template-columns: 320px 1fr; gap: 24px;
    }}
    .sidebar {{
      background: var(--card-bg); border: 1px solid var(--card-border);
      border-radius: 16px; padding: 18px; height: fit-content;
    }}
    .sidebar h3 {{ font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 12px; }}
    .case-item {{
      background: #090d16; border: 1px solid var(--card-border);
      border-radius: 10px; padding: 12px 14px; margin-bottom: 10px;
      cursor: pointer; transition: all 0.2s;
    }}
    .case-item:hover {{ border-color: var(--accent); }}
    .case-item.active {{
      border-color: var(--accent); background: rgba(56, 189, 248, 0.08);
    }}
    .case-item h4 {{ font-size: 14px; font-weight: 600; margin-bottom: 4px; }}
    .case-item p {{ font-size: 12px; color: var(--text-muted); }}

    .main-content {{
      background: var(--card-bg); border: 1px solid var(--card-border);
      border-radius: 16px; padding: 24px;
    }}
    .report-banner {{
      background: #090d16; border: 1px solid var(--card-border);
      border-radius: 12px; padding: 18px; margin-bottom: 24px;
      display: flex; justify-content: space-between; align-items: center;
    }}
    .banner-title h2 {{ font-size: 20px; font-weight: 700; }}
    .banner-title p {{ font-size: 13px; color: var(--text-muted); margin-top: 4px; }}
    .badge {{
      padding: 6px 14px; border-radius: 20px; font-size: 12px;
      font-weight: 700; text-transform: uppercase;
    }}
    .badge-critical {{ background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid #ef4444; }}
    .badge-warning {{ background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid #f59e0b; }}
    .badge-success {{ background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid #10b981; }}

    /* Cards Grid */
    .cards-grid {{
      display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 18px;
    }}
    .bio-card {{
      background: #090d16; border: 1px solid var(--card-border);
      border-radius: 14px; padding: 18px; transition: all 0.2s;
    }}
    .bio-card:hover {{ border-color: rgba(56, 189, 248, 0.4); }}
    .bio-header {{
      display: flex; justify-content: space-between; align-items: flex-start;
      margin-bottom: 12px;
    }}
    .bio-title {{ font-size: 16px; font-weight: 700; }}
    .bio-val {{ font-size: 18px; font-weight: 800; font-family: 'JetBrains Mono', monospace; }}
    .bio-section {{ margin-top: 12px; }}
    .bio-label {{ font-size: 12px; text-transform: uppercase; color: var(--text-muted); font-weight: 700; }}
    .bio-desc {{ font-size: 13px; color: #cbd5e1; margin-top: 4px; line-height: 1.5; }}
    
    .schedule-grid {{
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 16px;
    }}
    .schedule-slot {{
      background: #090d16; border: 1px solid var(--card-border);
      border-radius: 12px; padding: 14px;
    }}
    .schedule-slot h4 {{ font-size: 13px; color: var(--accent); margin-bottom: 8px; text-transform: uppercase; }}
    .schedule-slot ul {{ list-style: none; font-size: 13px; color: #cbd5e1; }}
    .schedule-slot li {{ margin-bottom: 6px; }}

    .warning-box {{
      background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3);
      border-radius: 12px; padding: 14px 18px; margin-bottom: 20px;
    }}
    .warning-box h4 {{ color: #ef4444; font-size: 14px; margin-bottom: 6px; display: flex; align-items: center; gap: 6px; }}
    .warning-box p {{ font-size: 13px; color: #fca5a5; line-height: 1.5; }}

    .questions-box {{
      background: #090d16; border: 1px solid #2e2640; border-radius: 14px;
      padding: 18px; margin-top: 24px;
    }}
    .questions-box h3 {{ color: var(--purple); font-size: 15px; margin-bottom: 12px; }}
    .questions-box ol {{ margin-left: 20px; font-size: 13px; color: #cbd5e1; line-height: 1.8; }}

    .table-view {{
      width: 100%; border-collapse: collapse; margin-top: 12px;
      font-size: 13px;
    }}
    .table-view th {{
      text-align: left; padding: 10px; border-bottom: 1px solid var(--card-border);
      color: var(--text-muted); font-size: 12px; text-transform: uppercase;
    }}
    .table-view td {{
      padding: 12px 10px; border-bottom: 1px solid #141c2e;
    }}
  </style>
</head>
<body>

  <div class="header">
    <div class="logo-area">
      <div class="logo-badge">P</div>
      <div class="title-box">
        <h1>PatientPulse AI — Medical Intelligence Platform</h1>
        <p>Zero-Jargon Lab Report & Doctor Prescription Plain-English Decipher Engine</p>
      </div>
    </div>
    <div class="nav-tabs">
      <a href="xray_viewer.html" style="text-decoration:none;"><button class="nav-btn">🩻 Universal X-Ray Engine</button></a>
      <button class="nav-btn active">🧪 Lab & Prescription Decipherer</button>
    </div>
  </div>

  <div class="view-switcher-bar">
    <div>
      <h3 style="font-size: 15px; font-weight: 700;">Active Perspective View</h3>
      <p style="font-size: 12px; color: var(--text-muted);">Toggle between zero-jargon patient explanations and specialist clinical data</p>
    </div>
    <div class="switcher-btns">
      <button class="switch-btn active" id="btn-patient" onclick="setViewMode('PATIENT')">
        💡 Patient Plain-English Decipher
      </button>
      <button class="switch-btn inactive" id="btn-doctor" onclick="setViewMode('DOCTOR')">
        👨‍⚕️ Clinical Specialist View
      </button>
    </div>
  </div>

  <div class="container">
    <div class="sidebar">
      <h3>Select Medical Document</h3>
      <div id="case-list"></div>
    </div>

    <div class="main-content" id="report-view">
      <!-- Dynamic render -->
    </div>
  </div>

  <script>
    const DATA = {payload_json};
    let currentMode = "PATIENT";
    let activeCase = DATA.lab_cases[0];

    function setViewMode(mode) {{
      currentMode = mode;
      document.getElementById("btn-patient").className = mode === "PATIENT" ? "switch-btn active" : "switch-btn inactive";
      document.getElementById("btn-doctor").className = mode === "DOCTOR" ? "switch-btn active" : "switch-btn inactive";
      renderActiveCase();
    }}

    function selectCase(type, id) {{
      if (type === "LAB_REPORT") {{
        activeCase = DATA.lab_cases.find(c => c.id === id);
      }} else {{
        activeCase = DATA.rx_cases.find(c => c.id === id);
      }}
      renderSidebar();
      renderActiveCase();
    }}

    function renderSidebar() {{
      const list = document.getElementById("case-list");
      let html = `<div style="margin-bottom: 12px; font-size: 12px; font-weight:700; color: #38bdf8;">LAB & BLOOD TESTS (DAY 4 NLP)</div>`;
      
      DATA.lab_cases.forEach(c => {{
        const isAct = activeCase.id === c.id;
        html += `
          <div class="case-item ${{isAct ? 'active' : ''}}" onclick="selectCase('LAB_REPORT', '${{c.id}}')">
            <h4>${{c.title}}</h4>
            <p>${{c.data.triage.badge}}</p>
          </div>
        `;
      }});

      html += `<div style="margin: 16px 0 8px 0; font-size: 12px; font-weight:700; color: #8b5cf6;">DOCTOR PRESCRIPTIONS (DAY 3 OCR)</div>`;
      DATA.rx_cases.forEach(c => {{
        const isAct = activeCase.id === c.id;
        html += `
          <div class="case-item ${{isAct ? 'active' : ''}}" onclick="selectCase('PRESCRIPTION', '${{c.id}}')">
            <h4>${{c.title}}</h4>
            <p>${{c.data.medicines_count}} Medicines • ${{c.data.confidence.percentage}} Conf</p>
          </div>
        `;
      }});

      list.innerHTML = html;
    }}

    function renderActiveCase() {{
      const container = document.getElementById("report-view");
      if (activeCase.type === "LAB_REPORT") {{
        renderLabReport(activeCase.data, container);
      }} else {{
        renderPrescription(activeCase.data, container);
      }}
    }}

    function renderLabReport(data, el) {{
      const badgeCls = data.triage.level === "CRITICAL_URGENT" ? "badge-critical" : (data.triage.level === "MILD_OBSERVATION" ? "badge-warning" : "badge-success");

      let html = `
        <div class="report-banner">
          <div class="banner-title">
            <h2>${{data.metadata.report_title}}</h2>
            <p>Patient: <strong>${{data.metadata.patient_name}}</strong> | Ordering Physician: <strong>${{data.metadata.doctor_name}}</strong> | Date: ${{data.metadata.date}}</p>
          </div>
          <span class="badge ${{badgeCls}}">${{data.triage.badge}}</span>
        </div>
      `;

      if (currentMode === "PATIENT") {{
        html += `
          <div class="warning-box" style="background: rgba(56, 189, 248, 0.08); border-color: rgba(56, 189, 248, 0.3);">
            <h4 style="color: #38bdf8;">💡 Plain-English Executive Summary</h4>
            <p style="color: #e2e8f0;">${{data.summary_text}}</p>
          </div>

          <div class="cards-grid">
        `;

        data.biomarkers.forEach(b => {{
          const m = b.metric_data;
          const d = b.patient_decipher;
          const flagColor = m.flag === "HIGH" ? "#f59e0b" : (m.flag === "LOW" ? "#3b82f6" : "#10b981");

          html += `
            <div class="bio-card" style="border-left: 4px solid ${{flagColor}};">
              <div class="bio-header">
                <div>
                  <div class="bio-title">${{d.test_name}}</div>
                  <div style="font-size: 12px; color: var(--text-muted); margin-top:2px;">Target: ${{d.reference_range}}</div>
                </div>
                <div class="bio-val" style="color: ${{flagColor}};">${{d.observed_value}}</div>
              </div>
              <span class="badge" style="background: rgba(255,255,255,0.06); color: ${{flagColor}};">${{d.status_badge.label}}</span>
              
              <div class="bio-section">
                <div class="bio-label">What This Test Does</div>
                <div class="bio-desc">${{d.what_it_does}}</div>
              </div>

              <div class="bio-section">
                <div class="bio-label">What Your Result Means</div>
                <div class="bio-desc">${{d.what_your_result_means}}</div>
              </div>

              <div class="bio-section">
                <div class="bio-label">Diet & Lifestyle Tip</div>
                <div class="bio-desc">${{d.diet_and_lifestyle_guidance}}</div>
              </div>
            </div>
          `;
        }});

        html += `</div>`;

        // 3 Questions box from first abnormal biomarker
        const firstAbnormal = data.biomarkers.find(b => b.metric_data.flag !== "NORMAL") || data.biomarkers[0];
        html += `
          <div class="questions-box">
            <h3>💬 3 Smart Questions To Ask Your Doctor at Your Next Visit</h3>
            <ol>
              ${{firstAbnormal.patient_decipher.questions_for_doctor.map(q => `<li>${{q}}</li>`).join("")}}
            </ol>
          </div>
        `;

      }} else {{
        // Doctor Specialist Table View
        html += `
          <h3 style="font-size: 16px; margin-bottom: 12px;">Clinical Diagnostic Table (Extracted by Bio_ClinicalBERT)</h3>
          <table class="table-view">
            <thead>
              <tr>
                <th>Investigation</th>
                <th>Observed Value</th>
                <th>Unit</th>
                <th>Reference Range</th>
                <th>Diagnostic Flag</th>
              </tr>
            </thead>
            <tbody>
              ${{data.biomarkers.map(b => `
                <tr>
                  <td><strong>${{b.metric_data.test_name}}</strong></td>
                  <td style="font-family:'JetBrains Mono'; font-weight:700;">${{b.metric_data.value}}</td>
                  <td>${{b.metric_data.unit}}</td>
                  <td style="color: var(--text-muted);">${{b.metric_data.reference_range}}</td>
                  <td><span class="badge" style="color:${{b.metric_data.flag==='NORMAL'?'#10b981':'#ef4444'}}">${{b.metric_data.flag}}</span></td>
                </tr>
              `).join("")}}
            </tbody>
          </table>
          <div style="margin-top: 20px; font-size: 13px; color: var(--text-muted);">
            <strong>Clinical Impression:</strong> ${{data.metadata.clinical_impression || "Standard outpatient screening."}}
          </div>
        `;
      }}

      el.innerHTML = html;
    }}

    function renderPrescription(data, el) {{
      const guide = data.patient_friendly_guide;
      let html = `
        <div class="report-banner">
          <div class="banner-title">
            <h2>${{data.document_type.replace(/_/g, ' ')}}</h2>
            <p>Prescribing Physician: <strong>${{data.metadata.doctor_name}}</strong> | Clinic: <strong>${{data.metadata.clinic_hospital}}</strong></p>
          </div>
          <span class="badge badge-success">OCR Confidence: ${{data.confidence.percentage}}</span>
        </div>
      `;

      if (data.food_and_safety_warnings.length > 0) {{
        html += `
          <div class="warning-box">
            <h4>⚠️ Critical Food & Drug Safety Alerts</h4>
            ${{data.food_and_safety_warnings.map(w => `<p style="margin-bottom: 6px;">• ${{w}}</p>`).join("")}}
          </div>
        `;
      }}

      if (currentMode === "PATIENT") {{
        html += `
          <h3 style="font-size: 16px; margin-bottom: 8px;">${{guide.plain_title}}</h3>
          <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 16px;">${{guide.treatment_overview}}</p>

          <h4 style="font-size: 14px; text-transform: uppercase; color: var(--accent); margin-top: 20px;">📅 Daily Medication Timetable</h4>
          <div class="schedule-grid">
            <div class="schedule-slot">
              <h4>🌅 Morning (Breakfast)</h4>
              <ul>${{guide.daily_schedule.morning_breakfast.map(m => `<li>• ${{m}}</li>`).join("")}}</ul>
            </div>
            <div class="schedule-slot">
              <h4>☀️ Midday (Lunch)</h4>
              <ul>${{guide.daily_schedule.afternoon_lunch.map(m => `<li>• ${{m}}</li>`).join("")}}</ul>
            </div>
            <div class="schedule-slot">
              <h4>🌆 Evening (Dinner)</h4>
              <ul>${{guide.daily_schedule.evening_dinner.map(m => `<li>• ${{m}}</li>`).join("")}}</ul>
            </div>
            <div class="schedule-slot">
              <h4>🌙 Bedtime (Night)</h4>
              <ul>${{guide.daily_schedule.bedtime_night.map(m => `<li>• ${{m}}</li>`).join("")}}</ul>
            </div>
          </div>

          <div class="questions-box">
            <h3>💬 3 Questions to Ask Your Pharmacist</h3>
            <ol>
              ${{guide.questions_for_pharmacist.map(q => `<li>${{q}}</li>`).join("")}}
            </ol>
          </div>
        `;
      }} else {{
        // Doctor / Pharmacist View
        html += `
          <h3 style="font-size: 16px; margin-bottom: 12px;">Deciphered Medication List (TrOCR / Layout Parsing)</h3>
          <table class="table-view">
            <thead>
              <tr>
                <th>Medicine</th>
                <th>Dosage</th>
                <th>Form</th>
                <th>Frequency</th>
                <th>Timing</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              ${{data.medicines.map(m => `
                <tr>
                  <td><strong>${{m.name}}</strong></td>
                  <td style="font-family:'JetBrains Mono'; font-weight:700;">${{m.dosage}}</td>
                  <td>${{m.dosage_form}}</td>
                  <td>${{m.frequency}} (${{m.abbreviation}})</td>
                  <td>${{m.timing}}</td>
                  <td>${{m.duration}}</td>
                </tr>
              `).join("")}}
            </tbody>
          </table>
        `;
      }}

      el.innerHTML = html;
    }}

    // Initial render
    renderSidebar();
    renderActiveCase();
  </script>
</body>
</html>
"""

    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[Viewer Generator] Successfully created interactive visual viewer: {HTML_OUTPUT}")

if __name__ == "__main__":
    generate_viewer()
