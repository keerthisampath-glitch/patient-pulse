# Prompt for Claude / AI Studio to Generate the Exact PatientPulse AI Frontend

> **How to use this:**
> 1. Copy everything inside the prompt box below.
> 2. Open [Claude](https://claude.ai) (or Google AI Studio / ChatGPT).
> 3. Paste the entire prompt and press Enter.
> 4. Claude will generate a single-file `index.html` (with modern CSS & JavaScript) that connects directly to your live FastAPI backend (`http://127.0.0.1:8000`) and Supabase cloud database!

---

```markdown
You are an expert Principal Frontend Engineer and UI/UX Designer specialized in modern clinical healthcare web applications.

Your task is to build a complete, single-file, production-ready frontend web application (`index.html` containing all HTML, CSS in `<style>`, and JavaScript in `<script>`) for our medical multimodal AI platform called **PatientPulse AI**.

The backend is ALREADY built in FastAPI and running on `http://127.0.0.1:8000`. Your frontend will connect to this backend via standard JavaScript `fetch()` calls.

---

### 🎨 DESIGN & AESTHETIC REQUIREMENTS (Must Look Like a $100M HealthTech App):
1. **Theme**: Futuristic yet trustworthy Clinical Dark Mode:
   - Background: Deep navy/slate (`#07090e`, `#0e131f`, `#1e293b`)
   - Accents: Electric Cyan (`#38bdf8`), Medical Teal (`#0d9488`), Violet Glow (`#8b5cf6`), Amber Warning (`#f59e0b`), Emerald Normal (`#10b981`), Rose Critical (`#f43f5e`).
   - Typography: Google Font `'Plus Jakarta Sans'` or `'Inter'`, with `'JetBrains Mono'` for numbers/dosages.
   - Style: Glassmorphism (`backdrop-filter: blur(12px)`), subtle glowing borders, modern rounded cards (16px), smooth micro-animations on hover and tab transitions.
2. **Layout Structure**:
   - **Header**: App Logo ("PatientPulse AI"), subtitle ("Multimodal Clinical Intelligence System"), Live Status Badge ("🟢 FastAPI Backend Connected: http://127.0.0.1:8000"), and active Supabase Cloud Sync pill.
   - **Top Navigation Bar**: 5 interactive tabs with icons:
     1. 🩻 **Radiograph & Bone Vision** (X-Ray triage, fracture detection, Grad-CAM heatmap)
     2. 📝 **Doctor Handwriting OCR** (Prescription deciphering, timetable, FDA warnings)
     3. 🔬 **Biomarker Lab Explainer** (Blood tests, normal/high/low flags, plain English)
     4. 💬 **24/7 AI Health Companion** (Conversational chatbot with NIH/FDA citations)
     5. 📊 **Patient Cloud Records** (Live audit history from Supabase)
   - **Sample Data Buttons**: Every tab MUST include a "⚡ Try Sample Demo" button so the user can test the UI in 1 click even without uploading a file.

---

### 🔌 EXACT FASTAPI BACKEND API CONTRACT:

Base URL variable in JavaScript:
```javascript
const API_BASE = "http://127.0.0.1:8000";
```

#### TAB 1: Radiograph & Bone X-Ray Vision Engine
- **Endpoint**: `POST ${API_BASE}/api/predict-xray`
- **Request Format**: `multipart/form-data`
  - Form field `file`: File upload (PNG/JPEG/DICOM)
  - Form field `patient_name`: Optional string (defaults to "Anonymous Patient")
- **Response JSON**:
  ```json
  {
    "patient_name": "John Doe",
    "primary_finding": "Atelectasis",
    "confidence": 0.644,
    "severity_level": "MODERATE",
    "urgency_badge": "TRIAGE_LEVEL_2_URGENT",
    "plain_english_summary": "Small Collapsed Air Sacs (Partial Lung Deflation)",
    "patient_guidance": "Deep breathing exercises recommended. Consult a physician.",
    "pathology_probabilities": {
      "Atelectasis": 0.644,
      "Effusion": 0.182,
      "Infiltration": 0.125,
      "Pneumonia": 0.089,
      "Normal": 0.041
    },
    "heatmap_image_base64": "data:image/png;base64,...",
    "detections": [
      { "box": [120, 80, 240, 200], "label": "Atelectasis", "score": 0.64 }
    ]
  }
  ```
- **UI Elements Needed**:
  - Drag-and-drop file upload area + image preview.
  - Interactive toggles: "Show Grad-CAM Heatmap Overlay" and "Show Bounding Boxes".
  - Finding pill with severity color (Green for Normal, Yellow for Moderate, Red for Critical).
  - Progress bars for the top 5 predicted pathology probabilities.
  - "Plain-English Deciphered for Patient" explanation card.

---

#### TAB 2: Doctor Handwriting OCR & Daily Timetable
- **Endpoint**: `POST ${API_BASE}/api/decipher-prescription`
- **Request Format**: `multipart/form-data`
  - Form field `file`: Uploaded prescription image/PDF, OR
  - Form field `raw_text`: Text area if pasting handwriting notes
  - Form field `patient_name`: Optional string
- **Response JSON**:
  ```json
  {
    "patient_name": "Jane Smith",
    "doctor_info": { "name": "Dr. R. V. Patel", "specialty": "Internal Medicine", "confidence": 0.92 },
    "handwriting_confidence": 0.92,
    "confidence_warning": false,
    "deciphered_medicines": [
      { "name": "Amoxicillin", "dosage": "500mg", "frequency": "TID", "instructions": "Take after meals" },
      { "name": "Paracetamol", "dosage": "650mg", "frequency": "SOS", "instructions": "For fever/pain" }
    ],
    "daily_timetable": {
      "morning": ["Amoxicillin 500mg (After breakfast)"],
      "afternoon": ["Amoxicillin 500mg (After lunch)"],
      "evening": ["Paracetamol 650mg (If needed)"],
      "night": ["Amoxicillin 500mg (Before bed)"]
    },
    "fda_food_drug_warnings": [
      { "drug": "Amoxicillin", "warning": "Avoid dairy products within 2 hours of ingestion." }
    ]
  }
  ```
- **UI Elements Needed**:
  - Image uploader or text area tab switch.
  - Doctor badge + Handwriting Confidence Meter (Warning pill if <80%).
  - Clean cards for each deciphered medicine with dosage, frequency, and purpose.
  - Visual 4-Column Daily Timetable: 🌅 Morning | ☀️ Afternoon | 🌇 Evening | 🌙 Night.
  - High-priority FDA Food-Drug Interaction Caution Box (glowing amber/red alert).

---

#### TAB 3: Medical Lab Report Explainer
- **Endpoint**: `POST ${API_BASE}/api/analyze-lab-report`
- **Request Format**: `multipart/form-data`
  - Form field `file`: Uploaded lab test PDF/Image, OR
  - Form field `raw_text`: Pasted text
  - Form field `patient_name`: Optional string
- **Response JSON**:
  ```json
  {
    "patient_name": "Robert Taylor",
    "report_type": "Complete Blood Count (CBC) & Lipid Profile",
    "triage_badge": "[CAUTION] Follow-Up Advised",
    "overall_summary": "Several biomarkers are outside normal reference limits, notably fasting blood glucose.",
    "biomarkers": [
      {
        "name": "Hemoglobin",
        "value": 11.2,
        "unit": "g/dL",
        "reference_range": "13.0 - 17.0",
        "status": "LOW",
        "plain_english": "The protein inside red blood cells carrying oxygen throughout your body.",
        "interpretation": "Mildly low; may cause mild fatigue. Consult your doctor regarding iron intake."
      },
      {
        "name": "Fasting Glucose",
        "value": 142.0,
        "unit": "mg/dL",
        "reference_range": "70.0 - 99.0",
        "status": "HIGH",
        "plain_english": "Amount of sugar in your blood after an overnight fast.",
        "interpretation": "Elevated level indicative of pre-diabetes or diabetes requiring medical review."
      }
    ],
    "doctor_discussion_questions": [
      "Could my elevated glucose be influenced by recent medications or diet?",
      "Would you recommend repeating this test with an HbA1c screening?"
    ]
  }
  ```
- **UI Elements Needed**:
  - Triage Alert Banner at the top (Green: "All Within Normal Range", Amber: "Follow-Up Advised", Red: "Urgent Review").
  - Biomarker Cards with Status Badges (🟢 NORMAL, 🟡 HIGH, 🔴 CRITICAL, 🔵 LOW).
  - "What does this mean?" Plain-English explanation per biomarker.
  - "Personalized Questions for Your Doctor" accordion or checklist.

---

#### TAB 4: 24/7 AI Health Companion Chat
- **Endpoint**: `POST ${API_BASE}/api/patient-chat`
- **Request Format**: `application/json`
  - Body: `{ "message": "What should I eat while taking Amoxicillin?" }`
- **Response JSON**:
  ```json
  {
    "response": "When taking Amoxicillin, it is best to take it with a light meal or water to avoid stomach irritation. However, avoid calcium-rich dairy products or antacids within 2 hours...",
    "citations": [
      "https://medlineplus.gov/druginfo/meds/a685001.html",
      "https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=BasicSearch.process&SearchTerm=amoxicillin"
    ],
    "tools_used": ["check_drug_safety", "explain_biomarker"]
  }
  ```
- **UI Elements Needed**:
  - Full-height conversational messenger with chat bubbles (patient avatar vs AI companion avatar).
  - Quick Suggestion Prompts ("What are the side effects of Metformin?", "Explain my high glucose", "Can I take Paracetamol with food?").
  - Clickable Government Citation Pills linking directly to NIH MedlinePlus & openFDA.
  - Tools Used badge showing autonomous agent execution.

---

#### TAB 5: Live Supabase Patient History & Audit Log
- **Endpoint**: `GET ${API_BASE}/api/patient-history?limit=10`
- **Response JSON**:
  ```json
  {
    "source": "live_supabase_cloud",
    "lab_reports": [ ... ],
    "prescriptions": [ ... ],
    "xray_records": [ ... ]
  }
  ```
- **UI Elements Needed**:
  - Live data status badge showing "☁️ Synced with Supabase PostgreSQL".
  - Timeline of recent patient analyses across all 3 modalities.
  - Refresh button to reload latest entries.

---

### ⚙️ TECHNICAL IMPLEMENTATION RULES:
1. **Single File Only**: Output all HTML, CSS inside `<style>`, and JavaScript inside `<script>` in one complete `index.html` file so it can be opened directly in any browser.
2. **Zero External CSS Frameworks**: Use Vanilla CSS with Flexbox/Grid for maximum reliability and styling precision (no broken CDN links).
3. **Robust Error Handling**: If the backend is loading or an upload fails, show an elegant inline toast notification or fallback demo card, NEVER crash or leave the screen blank.
4. **Interactive Mock Fallbacks**: If the user clicks "⚡ Try Sample Demo" on any tab, populate the UI immediately with rich, realistic clinical sample data so they can see the full visual experience without waiting.
5. Provide the complete code from `<!DOCTYPE html>` to `</html>` without truncation.
```
