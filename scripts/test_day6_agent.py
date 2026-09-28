"""
PatientPulse AI — Day 6 AI Companion Agent & Tool-Calling Verification Script
Verifies:
1. AI Companion Agent initialization
2. Live tool execution:
   - check_drug_safety
   - explain_lab_biomarker
   - generate_doctor_checklist
3. Grounded patient chat responses with official government citations
4. Medical safety disclaimer embedding
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "App"))

from backend.agent import patient_agent

def run_test():
    print("====================================================================")
    print("      PATIENTPULSE AI — DAY 6 AI COMPANION AGENT TEST SUITE         ")
    print("====================================================================")

    # Test 1: Autonomous Tool Execution
    print("\n[Test 1] Testing Agent Tool Execution...")
    tool1_res = patient_agent.execute_tool("check_drug_safety", {"drug_name": "metformin"})
    print(f"  - Tool: check_drug_safety('metformin')")
    print(f"  - Output Drug: {tool1_res.get('drug_name')}")
    print(f"  - Source: {tool1_res.get('source')}")
    assert tool1_res.get("drug_name") == "metformin", "Tool execution failed for check_drug_safety!"

    tool2_res = patient_agent.execute_tool("explain_lab_biomarker", {"biomarker_name": "glucose"})
    print(f"  - Tool: explain_lab_biomarker('glucose')")
    print(f"  - Output Definition: {tool2_res.get('definition')[:90]}...")
    assert "glucose" in tool2_res.get("biomarker", "").lower(), "Tool execution failed for explain_lab_biomarker!"

    tool3_res = patient_agent.execute_tool("generate_doctor_checklist", {"symptoms": "persistent dry cough"})
    print(f"  - Tool: generate_doctor_checklist('persistent dry cough')")
    print(f"  - Questions Generated: {len(tool3_res.get('checklist', []))}")
    assert len(tool3_res.get("checklist", [])) >= 3, "Checklist generator failed!"

    # Test 2: Grounded Patient Chat
    print("\n[Test 2] Testing Grounded Patient Conversation...")
    user_query = "Why is it important to take Metformin with food?"
    chat_res = patient_agent.chat(user_query)
    print(f"  - Query: {chat_res['patient_query']}")
    print(f"  - Grounded Facts Count: {chat_res['grounded_sources_count']}")
    print(f"  - AI Response Preview: {chat_res['response'][:250]}...")
    print(f"  - Citations: {chat_res['citations']}")
    print(f"  - Disclaimer: {chat_res['disclaimer']}")

    assert len(chat_res["response"]) > 30, "AI response is empty!"
    assert len(chat_res["citations"]) > 0, "No citations generated in AI response!"
    assert "Disclaimer" in chat_res["disclaimer"], "Medical safety disclaimer missing!"

    print("\n====================================================================")
    print("   ALL DAY 6 AI AGENT TESTS PASSED WITH ZERO ERRORS (100%)          ")
    print("====================================================================")

if __name__ == "__main__":
    run_test()
