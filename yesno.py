import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load Gemini API key
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

INPUT_FILE = "supahealth (01 sept) - supahealth posteval.csv"  # Change to your input filename
OUTPUT_FILE = "output.csv"
RESULT_COL = "Result[LLM]"
RESUME_TEXT_COL = "Grapevine Userresume - Resume → Metadata → Resume Text"

# Load the sheet
df = pd.read_csv(INPUT_FILE)

# Insert the new column after "Result[LLM]"
insert_at = df.columns.get_loc(RESULT_COL) + 1
new_col = "Manual Intervention verdict?"
df.insert(insert_at, new_col, "")

# Gemini prompt function
def get_gemini_flag(jd_text, resume_text, candidate_profile):
    print("[Gemini] Preparing prompt for Gemini API call...")
    prompt = f"""
You are an expert HR reviewer. You are given a candidate profile, Recruiter requirements (RR), and a job description (JD). This candidate was previously marked as "Maybe" by an automated system. Your task is to thoroughly review the candidate’s profile and the JD, and then make a clear, final decision: either Advanced (the candidate is a strong fit and should move forward) or No (the candidate is not a fit and should not move forward).

Instructions:

Review the Job Description (JD): Carefully read the requirements, responsibilities, and qualifications.
Review the Recruiter Requirements (RR): Carefully read the requirements, must have and should have. [optional]
Review the Candidate Profile: Examine the candidate’s experience, skills, education, and any other relevant information.
Compare and Analyze: Assess how well the candidate matches the must-have and nice-to-have requirements in the JD.
Be Decisive: Avoid ambiguity. Do not select "Maybe." You must choose either Advanced or No.
Justify Your Decision: Provide a brief, clear explanation (2-3 sentences) for your decision, referencing specific requirements or gaps.

Output Format:

Final Decision: [Yes/No]
Justification: [Your explanation]

Example Output:

Final Decision: Yes
Justification: The candidate has 5+ years of relevant experience, meets all must-have skills, and has led similar projects as described in the JD.

Job Description:
{jd_text}

Candidate Profile:
{candidate_profile}

Resume Text:
{resume_text}
"""
    try:
        print("[Gemini] Calling Gemini API...")
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        print("[Gemini] API call complete.")
        result = response.text.strip().split("\n")[0]
        print(f"[Gemini] API result: {result}")
        if result.lower().startswith("yes"):
            return "Yes"
        if result.lower().startswith("no"):
            return "No"
        return ""
    except Exception as e:
        print(f"Gemini API error: {e}")
        return ""

# Apply logic for Manual Intervention rows
for idx, row in df.iterrows():
    if str(row[RESULT_COL]).strip().lower() == "manual intervention":
        print(f"[Main] Processing row {idx+1}/{len(df)} (Manual Intervention)...")
        jd_text = row.get("Grapevine Job - Job → Description", "")
        resume_text = row.get(RESUME_TEXT_COL, "")
        candidate_profile = row.get("Candidate Profile", "")
        verdict = get_gemini_flag(jd_text, resume_text, candidate_profile)
        df.at[idx, new_col] = verdict
        print(f"[Main] Row {idx+1} verdict: {verdict}")

# Save the result
df.to_csv(OUTPUT_FILE, index=False)
print(f"Done! Output saved to {OUTPUT_FILE}")
