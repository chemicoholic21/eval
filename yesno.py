import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

INPUT_FILE = "sarvamai.csv"          # Change to your input filename
OUTPUT_FILE = "sarvam.csv"       # Change to your output filename
RESULT_COL = "Result[LLM]"
NEW_VERDICT_COL = "Manual Intervention verdict?"
JUSTIFICATION_COL = "Justification" # New column for justification text

# --- Main Script ---

def get_gemini_verdict(jd_text, resume_text, candidate_profile):
    """
    Calls the Gemini API to get a final verdict and justification for a candidate.
    Returns a tuple: (verdict, justification)
    """
    print("[Gemini] Preparing prompt for Gemini API call...")
    prompt = f"""
You are an expert HR reviewer. You are given a candidate profile, Recruiter requirements (RR), and a job description (JD). This candidate was previously marked as "Maybe" by an automated system. Your task is to thoroughly review the candidate’s profile and the JD, and then make a clear, final decision: either "Advanced" (the candidate is a strong fit and should move forward) or "No" (the candidate is not a fit and should not move forward).

Instructions:

1.  **Review the Job Description (JD):** Carefully read the requirements, responsibilities, and qualifications.
2.  **Review the Recruiter Requirements (RR):** Carefully read the "must-have" and "should-have" sections. [Ignore if absent]
3.  **Review the Candidate Profile:** Examine the candidate’s experience, skills, and education. 
IMPORTANT: DO NOT REJECT CANDIDATES BASED ON YEARS OF EXPERIENCE, YOUR REASON MUST BE BASED ON SKILLS AND QUALIFICATIONS. IF THE JD MATCH IS STRONG, YOU CAN CONSIDER IT A POSITIVE FACTOR, IRRESPECTIVE OF YOE.
5.  **Be Decisive:** Avoid ambiguity. You must choose either "Advanced" or "No".
6.  **Justify Your Decision:** Provide a brief, clear explanation (2-3 sentences) for your decision.

Output Format:

Final Decision: [Advanced/No]
Justification: [Your explanation]

Example Output:

Final Decision: Advanced
Justification: The candidate has 5+ years of relevant experience, meets all must-have skills, and has led similar projects as described in the JD.

---
Job Description:
{jd_text}

Candidate Profile:
{candidate_profile}

Resume Text:
{resume_text}
"""
    try:
        print("[Gemini] Calling Gemini API...")
        model = genai.GenerativeModel("gemini-1.5-flash") # Using 1.5-flash for potentially better instruction following
        response = model.generate_content(prompt)
        print("[Gemini] API call complete.")

        # More robust parsing for both verdict and justification.
        response_text = response.text.strip()
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]

        verdict = "Parsing Error"
        justification = "Could not parse justification."

        # Parse verdict from the first line
        if lines:
            first_line = lines[0].lower()
            if "advanced" in first_line:
                verdict = "Advanced"
            elif "no" in first_line:
                verdict = "No"

        # Parse justification from the line starting with "Justification:"
        justification_line = next((line for line in lines if line.lower().startswith("justification:")), None)
        if justification_line:
            # Split once on the first colon and take everything after it.
            justification = justification_line.split(":", 1)[-1].strip()

        print(f"[Gemini] Parsed Verdict: '{verdict}'")
        return verdict, justification

    except Exception as e:
        print(f"[Gemini] API error: {e}")
        return "API Error", str(e)

def main():
    """
    Main function to run the script.
    """
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    # Ensure the result column exists before trying to insert after it
    if RESULT_COL not in df.columns:
        print(f"Error: Column '{RESULT_COL}' not found in the input file.")
        return

    # Insert the new verdict column if it doesn't already exist
    if NEW_VERDICT_COL not in df.columns:
        insert_at_verdict = df.columns.get_loc(RESULT_COL) + 1
        df.insert(insert_at_verdict, NEW_VERDICT_COL, "")

    # Insert the new justification column if it doesn't already exist
    if JUSTIFICATION_COL not in df.columns:
        insert_at_justification = df.columns.get_loc(NEW_VERDICT_COL) + 1
        df.insert(insert_at_justification, JUSTIFICATION_COL, "")


    # Identify rows that need processing
    manual_intervention_rows = df[df[RESULT_COL].str.strip().str.lower() == "manual intervention"]
    total_rows_to_process = len(manual_intervention_rows)
    print(f"[Main] Found {total_rows_to_process} rows for manual intervention.")

    # Process only the identified rows
    for i, (row_idx, row) in enumerate(manual_intervention_rows.iterrows()):
        print(f"\n[Main] Processing row {i+1}/{total_rows_to_process} (Original Index: {row_idx})...")
        jd_text = row.get("Grapevine Job - Job → Description", "")
        resume_text = row.get("Grapevine Userresume - Resume → Metadata → Resume Text", "")
        candidate_profile = row.get("Candidate Profile", "")

        verdict, justification = get_gemini_verdict(jd_text, resume_text, candidate_profile)

        # Use .at for efficient and guaranteed cell update
        df.at[row_idx, NEW_VERDICT_COL] = verdict
        df.at[row_idx, JUSTIFICATION_COL] = justification
        print(f"[Main] Row {i+1} verdict: {verdict}")

        # Add a small delay to avoid hitting API rate limits
        time.sleep(1)

    # Save the result
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone! Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
