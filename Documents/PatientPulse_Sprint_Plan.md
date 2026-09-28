# PatientPulse AI — 3-Week Day-by-Day Sprint Plan
## 15-Day Engineering & Deployment Roadmap (100% Free Stack)

---

## 📅 WEEKS AT A GLANCE

| Week | Phase Focus | Key Deliverable |
| :--- | :--- | :--- |
| **Week 1** | **Backend AI Engines & Models** | X-Ray DL, Handwriting OCR, BioBERT NLP & NIH/FDA RAG |
| **Week 2** | **FastAPI Server & Web UI** | Fully connected HTML5/CSS3/JS Web App & AI Agent |
| **Week 3** | **Testing, Cloud Deploy & Review Prep** | Free Cloud Hosting (Render + Vercel) & Manager Review Prep |

---

## 🚀 WEEK 1: Core Foundation, AI Models & RAG Engine

### 🔹 Day 1: Project Setup & Supabase Database Configuration
* **Tasks**:
  1. Initialize project folder structure (`app/backend/`, `app/frontend/`, `data/sample_reports/`).
  2. Setup `.env` configuration file for API keys (Groq/Gemini API Free Tier, Supabase URL & Service Key).
  3. Create Supabase Cloud PostgreSQL database tables (`patient_logs`, `prescription_history`, `xray_triage_records`).
* **Deliverable**: Working environment & connected Supabase database.

### 🔹 Day 2: Chest X-Ray Visual AI Model (`TorchXRayVision`)
* **Tasks**:
  1. Create `backend/predict_xray.py` script.
  2. Load PyTorch `TorchXRayVision` DenseNet-121 pre-trained weights (`densenet121-res224-all`).
  3. Implement pathology probability calculator (Pneumonia, Effusion, Infiltration, Normal).
  4. Implement Grad-CAM heatmap bounding-box generator.
* **Deliverable**: Functional PyTorch X-Ray analysis module.

### 🔹 Day 3: Document Layout Parsing & Handwriting OCR
* **Tasks**:
  1. Create `backend/ocr_service.py` script.
  2. Integrate `PyMuPDF` / `pdfplumber` to extract tables from Blood Test PDFs.
  3. Load HuggingFace `microsoft/trocr-base-handwritten` model to convert handwritten doctor prescriptions into digital text.
* **Deliverable**: PDF table parser & handwriting OCR module.

### 🔹 Day 4: Medical NLP & Entity Extraction (`BioBERT`)
* **Tasks**:
  1. Create `backend/nlp_service.py` script.
  2. Integrate `BioBERT` / Clinical NER (`emilyalsentzer/Bio_ClinicalBERT`) to extract metric names (*Hemoglobin*, *TSH*, *Glucose*), values, and reference ranges.
  3. Implement high/low status categorizer (Normal, High, Low).
* **Deliverable**: Medical entity & metric status extraction engine.

### 🔹 Day 5: Grounded NIH MedlinePlus & openFDA RAG Integration
* **Tasks**:
  1. Create `backend/rag_engine.py` script.
  2. Build live REST API fetcher for **NIH MedlinePlus** (lab term definitions).
  3. Build live REST API fetcher for **openFDA Drug API** (drug side-effects & food interaction warnings).
  4. Implement local ChromaDB vector store for instant document retrieval.
* **Deliverable**: Zero-hallucination RAG engine with exact source citations.

---

## ⚡ WEEK 2: AI Agent, FastAPI Routers & Web Interface

### 🔹 Day 6: AI Agent & Tool-Calling Integration
* **Tasks**:
  1. Create `backend/agent.py` script.
  2. Integrate Groq Llama 3 8B / Gemini Flash function-calling LLM.
  3. Create Agent tools: `check_drug_safety()`, `generate_doctor_checklist()`, and `save_to_supabase()`.
* **Deliverable**: Function-calling AI Agent connected to tools and database.

### 🔹 Day 7: FastAPI Backend Server & API Endpoints
* **Tasks**:
  1. Create `backend/main.py` FastAPI application.
  2. Build API endpoints:
     * `POST /api/analyze-lab-report`
     * `POST /api/decipher-prescription`
     * `POST /api/analyze-xray`
     * `POST /api/patient-chat`
  3. Enable CORS middleware for seamless frontend requests.
* **Deliverable**: Fully functional FastAPI backend server.

### 🔹 Day 8: Modern Custom Web Frontend UI (HTML5/CSS3)
* **Tasks**:
  1. Create `frontend/index.html` structure.
  2. Create `frontend/styles.css` custom design system (Dark mode, glassmorphism UI, vibrant status badges, responsive grids).
  3. Build 3 core tool tabs: *Lab Explainer*, *Prescription Reader*, *Chest X-Ray Triage*.
* **Deliverable**: Pixel-perfect responsive web UI template.

### 🔹 Day 9: Frontend JavaScript & Drag-and-Drop Dropzones
* **Tasks**:
  1. Create `frontend/app.js` logic script.
  2. Build interactive file upload dropzones with real-time upload progress bars.
  3. Connect dropzone buttons to FastAPI endpoints using `fetch()` API.
  4. Render color-coded biomarker risk gauges and dosage rule cards.
* **Deliverable**: Fully interactive Web UI sending requests to FastAPI backend.

### 🔹 Day 10: 24/7 AI Health Companion Slide-Out Chatbot UI
* **Tasks**:
  1. Build floating slide-out chat window component in HTML/CSS.
  2. Connect chatbot input to `/api/patient-chat` endpoint.
  3. Implement real-time streaming text responses with NIH/FDA citation source links.
* **Deliverable**: Live interactive AI companion chatbot inside the web app.

---

## 🌐 WEEK 3: Pre-Packaged Demo Data, Free Cloud Deploy & Defense

### 🔹 Day 11: Pre-Packaged "1-Click Try Sample Data" Integration
* **Tasks**:
  1. Populate `data/sample_reports/` with 3 sample Blood Test PDFs, 3 Prescription photos, and 3 Chest X-Ray images.
  2. Add **"Try Sample Data"** buttons on the website so reviewers can test all features in 1-click.
* **Deliverable**: 1-click instant live testing mode on the website.

### 🔹 Day 12: Medical Safety Guardrails & Fallback Testing
* **Tasks**:
  1. Implement OCR confidence score warning banners ($<80\%$ confidence warning).
  2. Add medical disclaimer banners across all pages (*"Educational guidance only — Consult your physician"*).
  3. Test edge cases (unclear images, missing API keys, rate limits) and verify graceful fallback responses.
* **Deliverable**: 100% stable, error-handled application.

### 🔹 Day 13: Free Cloud Deployment (Vercel + Render + Supabase)
* **Tasks**:
  1. Deploy FastAPI backend server for free on **Render.com** (or Koyeb).
  2. Deploy custom Web Frontend for free on **Vercel** (or Netlify).
  3. Verify live cross-origin API connection (`https://patientpulse.vercel.app` $\rightarrow$ `https://patientpulse-api.onrender.com`).
* **Deliverable**: Live, public, 100% free HTTPS website link.

### 🔹 Day 14: Project Documentation & GitHub Repository Setup
* **Tasks**:
  1. Update repository `README.md` with architectural diagrams, dataset links, and setup commands.
  2. Include `PatientPulse_Project_Overview.docx` in project documentation folder.
  3. Commit and push clean codebase to GitHub repository.
* **Deliverable**: Professional, recruiter-ready GitHub repository.

### 🔹 Day 15: Manager Review Presentation & Q&A Defense Practice
* **Tasks**:
  1. Create 10-slide presentation deck covering Industry, Domain, Architecture, AI Stack, and Live Demo.
  2. Practice 5-point response to manager pushback questions.
  3. Conduct live end-to-end demo walkthrough.
* **Deliverable**: Successful internship project completion & final review defense!
