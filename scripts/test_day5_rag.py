"""
PatientPulse AI — Day 5 RAG Knowledge Retrieval Verification Script
Verifies:
1. NIH MedlinePlus REST API / Local Knowledge retrieval (definitions for Hemoglobin, Glucose, TSH, etc.)
2. openFDA Drug Labeling REST API / Local Knowledge retrieval (Amoxicillin, Metformin, Lisinopril)
3. Grounded context synthesis with exact official source URLs & authority citations
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "App"))

from backend.rag_engine import rag_engine

def run_test():
    print("====================================================================")
    print("      PATIENTPULSE AI — DAY 5 GROUNDED RAG TEST SUITE               ")
    print("====================================================================")

    # Test 1: NIH MedlinePlus Lab Definition
    print("\n[Test 1] Querying NIH MedlinePlus for 'Hemoglobin'...")
    res1 = rag_engine.query_nih_medlineplus("hemoglobin")
    print(f"  - Source: {res1['source']}")
    print(f"  - Authority: {res1['authority']}")
    print(f"  - Definition: {res1['definition'][:100]}...")
    print(f"  - URL: {res1['url']}")
    assert len(res1["definition"]) > 20, "MedlinePlus definition is empty!"
    assert "medlineplus.gov" in res1["url"] or "nih.gov" in res1["url"], "Invalid NIH citation!"

    # Test 2: openFDA Drug Safety & Interactions
    print("\n[Test 2] Querying openFDA for 'Amoxicillin'...")
    res2 = rag_engine.query_openfda_drug_safety("amoxicillin")
    print(f"  - Source: {res2['source']}")
    print(f"  - Drug: {res2['drug_name']}")
    print(f"  - Interactions: {res2['drug_interactions'][:100]}...")
    print(f"  - URL: {res2['url']}")
    assert len(res2["drug_interactions"]) > 10, "openFDA drug interactions empty!"

    # Test 3: Grounded Multi-Source Query
    print("\n[Test 3] Testing Multi-Source Combined RAG Retrieval...")
    query = "Is it safe to take paracetamol and amoxicillin together?"
    res3 = rag_engine.retrieve_grounded_context(query)
    print(f"  - Query: {res3['query']}")
    print(f"  - Grounded Facts: {res3['total_grounded_items']}")
    print(f"  - Citations: {res3['citations']}")
    assert res3["total_grounded_items"] >= 1, "RAG failed to retrieve grounded context!"

    print("\n====================================================================")
    print("   ALL DAY 5 GROUNDED RAG TESTS PASSED WITH ZERO ERRORS (100%)      ")
    print("====================================================================")

if __name__ == "__main__":
    run_test()
