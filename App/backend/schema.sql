-- ====================================================================
-- PATIENTPULSE AI — SUPABASE CLOUD DATABASE SCHEMA DEFINITION
-- ====================================================================

-- 1. Patient Lab Reports Table (Biomarker Triaging & Plain-English Summaries)
CREATE TABLE IF NOT EXISTS public.patient_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_name TEXT DEFAULT 'Anonymous Patient',
    report_type TEXT NOT NULL,
    analyzed_metrics JSONB NOT NULL,
    summary_text TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Doctor Prescription History Table (OCR & FDA Food/Drug Warnings)
CREATE TABLE IF NOT EXISTS public.prescription_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_name TEXT DEFAULT 'Anonymous Patient',
    raw_ocr_text TEXT NOT NULL,
    deciphered_medicines JSONB NOT NULL,
    food_warnings JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Radiograph X-Ray Triage Records Table (Pathologies & Urgency Levels)
CREATE TABLE IF NOT EXISTS public.xray_triage_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_name TEXT DEFAULT 'Anonymous Patient',
    pathology_predictions JSONB NOT NULL,
    triage_status TEXT NOT NULL,
    guidance_notes TEXT NOT NULL,
    heatmap_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ====================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES FOR LIVE FRONTEND & BACKEND ACCESS
-- ====================================================================

ALTER TABLE public.patient_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.prescription_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.xray_triage_records ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow anon select on patient_logs" ON public.patient_logs FOR SELECT TO anon USING (true);
CREATE POLICY "Allow anon insert on patient_logs" ON public.patient_logs FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "Allow anon select on prescription_history" ON public.prescription_history FOR SELECT TO anon USING (true);
CREATE POLICY "Allow anon insert on prescription_history" ON public.prescription_history FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "Allow anon select on xray_triage_records" ON public.xray_triage_records FOR SELECT TO anon USING (true);
CREATE POLICY "Allow anon insert on xray_triage_records" ON public.xray_triage_records FOR INSERT TO anon WITH CHECK (true);
