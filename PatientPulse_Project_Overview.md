## Multimodal Medical Report & Diagnostic Intelligence Platform
### Project Overview & Pre-Trained Model Specifications

---

## 1. Industry & Domain

* **Industry**: Healthcare & Digital Health (HealthTech)
* **Domain**: Patient Health Intelligence & Clinical Decision Support Systems (CDSS)
* **Primary Target Users**: Patients, Family Caregivers, Outpatient Visitors, and Clinic Administrative Staff

---

## 2. Project Overview

This Project is a multimodal healthcare web application designed to bridge the communication gap between complex clinical data and non-technical patients. Medical test reports, doctor handwritten prescriptions, and radiology X-ray scans contain highly specialized medical terminology that patients often find confusing and anxiety-inducing. 

This Project allows users to upload medical documents, prescriptions, and radiology images, automatically extracting key clinical information and translating it into plain-English, grounded, and actionable guidance.

The application is built on a **100% free-of-cost technical architecture (₹0 budget)**, integrating pre-trained Deep Learning computer vision models, Optical Character Recognition (OCR), Natural Language Processing (NLP), Retrieval-Augmented Generation (RAG), and an AI Agent backed by a cloud database.

---

## 3. How It Works (System Workflow & Architecture)

The platform operates via an end-to-end multimodal pipeline structured into four primary technical layers:

### A. Document & Image Input Layer
* Users upload Blood Test PDFs, scanned Lab Reports, handwritten Doctor Prescriptions, or Chest X-Ray images via a clean drag-and-drop web interface.
* Includes a **"Try Sample Data"** feature with pre-loaded medical files for instant 1-click testing.

### B. Computer Vision & OCR Processing Layer
* **Chest X-Ray Analysis (Deep Learning Vision)**: Uses `TorchXRayVision` DenseNet-121 to scan Chest X-Ray images and evaluate visual findings for pathologies such as Pneumonia, Effusion, Infiltration, or Normal lung structures.
* **Handwriting & Report OCR (PyMuPDF & TrOCR)**: Reads scanned lab test tables and deciphers handwritten doctor prescription slips into digital text.

### C. Natural Language Processing & RAG Engine
* **Entity Recognition (BioBERT NLP)**: Identifies medical entities, categorizes lab metric statuses (Normal, High, Low), and extracts medication names.
* **Grounded Knowledge RAG**: Queries live **NIH MedlinePlus** and **openFDA** REST APIs to fetch official, grounded definitions, food-drug interaction warnings, and precautions with exact source citations.

### D. AI Agent & Database Layer
* **Patient Companion**: Powers a 24/7 interactive patient Q&A chatbot using LLM function-calling grounded in NIH/FDA documentation.
* **Cloud Storage**: Automatically logs patient report history, extracted medication schedules, and clinical timeline records into a **Supabase Cloud PostgreSQL** database.

---

## 4. What The System Outputs

This Project provides four primary outputs across its dedicated application modules:

1. **Lab Report Analysis Output**: Extracts test metrics, compares values against reference ranges, and displays color-coded status badges (Normal / High / Low) alongside plain-English explanations.
2. **Prescription Deciphering Output**: Displays transcribed medicine names, dosage instructions, timing (before/after meals), and flags critical food-drug interaction warnings (e.g., *"Take with meals"*, *"Avoid dairy"*).
3. **Chest X-Ray Triage Output**: Renders visual finding probabilities (Pneumonia vs Normal) and provides clear precautionary advice and warning signs.
4. **AI Companion & Patient Log Output**: Answers patient health queries in real-time with source-cited facts and logs patient history into Supabase Cloud Database.

---

## 5. Pre-Trained Models & Data Sources (By Feature)

To eliminate the need for manual dataset downloads or heavy local training, This Project utilizes verified open-source pre-trained model weights imported directly via PyTorch and HuggingFace Hub, paired with official government REST APIs:

### Feature 1: Chest X-Ray & Radiology Assistant
* **Pre-Trained Visual Models**: `TorchXRayVision` DenseNet-121 (`xrv.models.get_model('densenet121-res224-all')`) & `ResNet-50` (`torchvision.models.resnet50`). Pre-trained to classify 18 medical pathologies.
* **Visual Localization Technique**: Grad-CAM (Gradient-Weighted Class Activation Mapping) Heatmap Bounding-Box Generator.
* **Pre-Training Datasets**: NIH ChestX-ray14 (112,000+ open radiology images) & CheXpert Medical Imaging Benchmarks.

### Feature 2: Doctor Handwritten Prescription Deciphering
* **Pre-Trained OCR Model**: Microsoft TrOCR (`microsoft/trocr-base-handwritten` from HuggingFace Hub). Converts handwritten cursive strokes into digital text.
* **Document Layout Parsers**: PyMuPDF (`fitz`), `pdfplumber`, and LayoutLMv3 for text line bounding-box extraction.
* **Pre-Training Datasets**: IAM Handwriting Database & Kaggle Medical Prescription Handwriting Dataset.

### Feature 3: Medical Report Metric Extraction & Status Categorization
* **Pre-Trained NLP Models**: BioBERT (`emilyalsentzer/Bio_ClinicalBERT`) & `dslim/bert-base-NER` (HuggingFace Hub). Extracts metric names, patient values, and reference ranges.
* **Pre-Training Datasets**: PubMed Central Open Access Corpus & MIMIC Clinical Text Annotations.

### Feature 4: Grounded Medical Knowledge Base & Drug Safety (GenAI / RAG)
* **Medical Dictionary API**: NIH MedlinePlus REST API (`medlineplus.gov`) providing official definitions for 5,000+ lab metrics.
* **Drug Safety API**: openFDA Drug Labeling REST API (`api.fda.gov`) providing official safety, side-effect, and food warning data for 100,000+ approved drugs.
* **RAG Vector Store**: ChromaDB local vector store for fast embedding retrieval.

### Feature 5: Interactive Patient Companion Chatbot & AI Agent
* **Pre-Trained LLM Engine**: Groq Llama 3.1 8B Instant (`llama-3.1-8b-instant`) / Google Gemini 1.5 Flash API (Free Tier).
* **AI Agent Tools & Database**: Autonomous tools: `check_drug_safety()`, `save_to_supabase()`. Logs patient timelines into Supabase Cloud PostgreSQL.

---

## 6. Technology Stack

* **Frontend Web UI**: HTML5, CSS3, Vanilla JavaScript (Modern, Dark-Mode Web App)
* **Backend Server**: Python FastAPI (`uvicorn` server)
* **Machine Learning / DL**: PyTorch, TorchXRayVision, HuggingFace Transformers, BioBERT, Microsoft TrOCR
* **GenAI / RAG Engine**: ChromaDB vector store + NIH MedlinePlus & openFDA REST APIs
* **AI Agent LLM**: Groq API / Gemini API (Free Tier LLM)
* **Cloud Database**: Supabase Cloud PostgreSQL (Free Tier)
* **Project Budget**: 100% Free (₹0 spent)
