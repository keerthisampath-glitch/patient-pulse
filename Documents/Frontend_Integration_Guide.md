# PatientPulse AI — Frontend & Google AI Studio Integration Guide

This guide provides the exact JavaScript `fetch()` integration code to connect the frontend web application (generated in **Google AI Studio**) to the **FastAPI Backend Server** (deployed on **Render** or running locally on `http://127.0.0.1:8000`).

---

## 🌐 API Base URL Configuration

In your frontend script (`app.js` or script tag):

```javascript
// Toggle between Localhost and Live Render Deployment
const API_BASE_URL = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? "http://127.0.0.1:8000"
  : "https://patientpulse-api.onrender.com"; // Replace with your Render URL
```

---

## 1. Feature 1: Universal Medical X-Ray Diagnostic Engine

**Endpoint**: `POST /api/analyze-xray`  
**Payload**: `multipart/form-data` with `file` (Image: JPG, PNG)

```javascript
async function uploadAndAnalyzeXRay(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  try {
    const response = await fetch(`${API_BASE_URL}/api/analyze-xray`, {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    console.log("X-Ray Diagnosis:", data);

    // Key output fields for UI:
    // 1. Anatomy: data.anatomy_detected.body_part
    // 2. Triage Badge: data.triage.badge (Color: data.triage.color_code)
    // 3. Primary Finding: data.triage.primary_finding (Confidence: data.triage.confidence_percentage)
    // 4. Plain-English Title: data.patient_friendly_decipher.plain_title
    // 5. What it Means: data.patient_friendly_decipher.what_it_means
    // 6. Why Red Box is there: data.patient_friendly_decipher.what_the_red_box_shows
    // 7. Symptoms: data.patient_friendly_decipher.common_signs
    // 8. 3 Questions for Doctor: data.patient_friendly_decipher.questions_for_doctor
    // 9. Visual Overlay: data.gradcam_localization.heatmap_base64

    return data;
  } catch (error) {
    console.error("Error analyzing X-ray:", error);
    alert("Failed to analyze X-ray scan. Please check connection.");
  }
}
```

---

## 2. Feature 2: Doctor Handwritten Prescription Decipherer

**Endpoint**: `POST /api/decipher-prescription`  
**Payload**: `multipart/form-data` with `file` (Image/PDF) OR `raw_text` (String)

```javascript
async function decipherPrescription(prescriptionFile, patientName = "Patient") {
  const formData = new FormData();
  formData.append("file", prescriptionFile);
  formData.append("patient_name", patientName);

  try {
    const response = await fetch(`${API_BASE_URL}/api/decipher-prescription`, {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    console.log("Prescription Output:", data);

    // Key output fields for UI:
    // 1. Doctor Name: data.metadata.doctor_name
    // 2. OCR Confidence: data.confidence.percentage
    // 3. Safety Guardrail Warning: data.confidence.warning_message (if confidence < 80%)
    // 4. Medicines List: data.medicines (array of {name, dosage, form, frequency, timing})
    // 5. Food & Safety Warnings: data.food_and_safety_warnings (e.g. avoid dairy, take with food)
    // 6. 4-Slot Daily Schedule: data.patient_friendly_guide.daily_schedule (morning, lunch, dinner, bedtime)
    // 7. Questions for Pharmacist: data.patient_friendly_guide.questions_for_pharmacist

    return data;
  } catch (error) {
    console.error("Error deciphering prescription:", error);
  }
}
```

---

## 3. Feature 3: Medical Lab Report & Biomarker Explainer

**Endpoint**: `POST /api/analyze-lab-report`  
**Payload**: `multipart/form-data` with `file` (PDF) OR `raw_text` (String)

```javascript
async function analyzeLabReport(labReportFile, patientName = "Patient") {
  const formData = new FormData();
  formData.append("file", labReportFile);
  formData.append("patient_name", patientName);

  try {
    const response = await fetch(`${API_BASE_URL}/api/analyze-lab-report`, {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    console.log("Lab Analysis Output:", data);

    // Key output fields for UI:
    // 1. Report Title: data.metadata.report_title
    // 2. Triage Level: data.triage.badge (Color: data.triage.color_code)
    // 3. Biomarkers Extracted: data.biomarkers (array of {metric_data, patient_decipher})
    //    Each biomarker has:
    //    - Test Name: item.patient_decipher.test_name
    //    - Observed Value: item.patient_decipher.observed_value
    //    - Reference Range: item.patient_decipher.reference_range
    //    - Status Badge: item.patient_decipher.status_badge.label (Color: item.patient_decipher.status_badge.color)
    //    - What it Means: item.patient_decipher.what_your_result_means
    //    - Diet & Lifestyle Tip: item.patient_decipher.diet_and_lifestyle_guidance
    //    - 3 Doctor Questions: item.patient_decipher.questions_for_doctor

    return data;
  } catch (error) {
    console.error("Error analyzing lab report:", error);
  }
}
```

---

## 4. Feature 4: 24/7 Grounded AI Health Companion Chatbot

**Endpoint**: `POST /api/patient-chat`  
**Payload**: JSON `{"message": string, "patient_name": string}`

```javascript
async function sendChatMessage(userMessage) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/patient-chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userMessage, patient_name: "Patient" })
    });

    const data = await response.json();
    console.log("AI Companion Response:", data);

    // Key output fields for UI:
    // 1. Reassuring Answer: data.response
    // 2. Citations & Official Sources: data.citations (links to NIH MedlinePlus & US FDA)
    // 3. Medical Safety Disclaimer: data.disclaimer

    return data;
  } catch (error) {
    console.error("Chatbot request failed:", error);
  }
}
```

---

## 5. Feature 5: Medical Timeline & Patient History

**Endpoint**: `GET /api/patient-history?limit=10`

```javascript
async function fetchPatientHistory() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/patient-history?limit=10`);
    const data = await response.json();
    
    // Returns:
    // data.lab_reports (Recent blood test logs)
    // data.prescriptions (Recent prescription logs)
    // data.xray_records (Recent X-ray triage logs)
    return data;
  } catch (error) {
    console.error("Failed to load patient history:", error);
  }
}
```
