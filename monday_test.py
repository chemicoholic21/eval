import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import threading
from functools import lru_cache
from typing import List, Dict, Tuple
from tqdm import tqdm

# --- GLOBAL CONFIGURATION & SETUP ---
print("🚀 Starting the UNIFIED Candidate Evaluation Pipeline...")
PIPELINE_START_TIME = time.time()
load_dotenv()

# Configure Google Generative AI
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Google Generative AI configured successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not configure Google Generative AI. Reason: {e}")
    exit(1)

# --- COLUMN NAME CONSTANTS ---
# Define all column names here for easy management
# Input Columns
COL_JOB_DESC = "Grapevine Job - Job → Description"
COL_INTERVIEW = "Grapevine Aiinterviewinstance → Transcript → Conversation"
COL_RESUME = "Grapevine Userresume - Resume → Metadata → Resume Text"
COL_CRITERIA = "Recruiter GPT Response "
COL_CANDIDATE_NAME = "Grapevine Userresume - Resume → Metadata → User Real Name"
COL_COMPANY_NAME = "Title"
COL_RESUME_URL = "Grapevine Userresume - Resume → Resume URL"
COL_PHONE = "Grapevine Userresume - Resume → Metadata → Phone Number"
COL_EMAIL = "Grapevine Userresume - Resume → Metadata → Email"
COL_CTC = "Grapevine Userresume - Resume → Metadata → Current Salary"
COL_EXPERIENCE = "Grapevine Userresume - Resume → Metadata → Experience"

# Stage 1 Output Columns
COL_INTERVIEW_EVAL = "Interview Evaluator Agent (RAG-LLM)"
COL_RESUME_EVAL = "Resume Evaluator Agent (RAG-LLM)"
COL_SUMMARIZER = "Resume + Interview Summarizer Agent"
COL_RESULT = "Result[LLM]" # This column determines filtering

# Stage 2 Output Columns
COL_GOOD_FIT = "Good Fit"
COL_PROFILE = "Candidate Profile"


# --- ULTRA-FAST GEMINI PROCESSOR CLASS ---
_response_cache = {}
_cache_lock = threading.Lock()

class UltraFastGeminiProcessor:
    """A highly concurrent, batch-processing class for the Gemini API."""
    def __init__(self, max_workers=50, cache_enabled=True, concurrent_batches=8):
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled
        self.concurrent_batches = concurrent_batches
        self.api_call_count = 0
        self.cache_hits = 0

    def _generate_cache_key(self, model_name: str, prompt: str) -> str:
        content = f"{model_name}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> str:
        with _cache_lock:
            if cache_key in _response_cache:
                self.cache_hits += 1
                return _response_cache[cache_key]
            return None

    def _set_cached_response(self, cache_key: str, response: str):
        with _cache_lock:
            _response_cache[cache_key] = response

    def _gemini_generate_single(self, model_name: str, prompt: str) -> str:
        """Optimized single API call with caching."""
        if not prompt or not prompt.strip():
            return "Error: Empty prompt provided."

        cache_key = self._generate_cache_key(model_name, prompt)
        if self.cache_enabled:
            cached_result = self._get_cached_response(cache_key)
            if cached_result:
                return cached_result

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    time.sleep(random.uniform(0.5, 1.5))
                
                model = genai.GenerativeModel(model_name)
                safety_settings = [
                    {"category": c, "threshold": "BLOCK_NONE"}
                    for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
                ]
                response = model.generate_content(prompt, safety_settings=safety_settings)
                
                result = response.text.strip() if hasattr(response, 'text') and response.text else "Error: No response text"
                
                if self.cache_enabled:
                    self._set_cached_response(cache_key, result)
                
                self.api_call_count += 1
                return result
            except Exception as e:
                if attempt == max_retries:
                    return f"Error: API call failed after {max_retries} retries. Reason: {e}"
        return "Error: Max retries exceeded"

    def process_prompts_in_parallel(self, model_name: str, prompts: List[str], task_description: str) -> List[str]:
        """Processes a list of prompts using a thread pool for high concurrency."""
        if not prompts:
            return []

        print(f"🔥 Starting parallel processing for '{task_description}' ({len(prompts)} prompts)...")
        results = [""] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {executor.submit(self._gemini_generate_single, model_name, prompt): i for i, prompt in enumerate(prompts)}
            
            with tqdm(as_completed(future_to_index), total=len(prompts), desc=f"⚡️ {task_description}") as progress_bar:
                for future in progress_bar:
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        results[index] = f"Error: Task failed with exception: {e}"
        
        print(f"✅ Completed '{task_description}'.")
        return results

processor = UltraFastGeminiProcessor(max_workers=32, cache_enabled=True)


# --- STAGE 1: INITIAL EVALUATION PROMPTS ---

@lru_cache(maxsize=1000)
def create_resume_prompt(job_desc, resume, criteria):
    # This prompt is from your first script
    return f"""Analyze the provided resume against the job role and criteria. Output a JSON object evaluating 'Education and Company Pedigree', 'Skills & Specialties', 'Work Experience', 'Basic Contact Details', and 'Educational Background Details' on specified scales, including a justification for each. Also, provide a summary for 'Input on Job Specific Criteria'. You are an AI hiring manager. Follow the instructions exactly. Only use the information provided in the input. Never invent or assume details. If information is missing, state this in your justification. Always output in the specified JSON format. 


*Input:*
- Job Role: {job_desc}
- Resume: {resume}
- Job-Specific Criteria: {criteria}



*Output your assessment as a JSON object only.*
"""

@lru_cache(maxsize=1000)
def create_interview_prompt(job_desc, transcript, criteria):
    # This prompt is from your first script
    return f"""Analyze the interview transcript to assess the candidate's 'Demonstrated Technical/Role Knowledge/Skills'.You are an AI interviewer. Only use the information provided in the transcript and job role. Never invent, assume, or add details. If information is missing, state this in your justification. Always output in the specified JSON format. 



*Input:*
- Job Role: {job_desc}
- Interview Transcript: {transcript}
- Job-Specific Criteria: {criteria}


*Output your assessment as a JSON object with a 'value' (0-10) and 'justification'.*
"""



@lru_cache(maxsize=1000)
def create_summarizer_prompt(job_desc, resume_eval, interview_eval, criteria):
    # This prompt is from your first script
    return f"""As an AI hiring coordinator, summarize the candidate's evaluation. Provide bullet points for key strengths, weaknesses, and a recommendation (advance/reject/manual intervention). Also, provide an 'Overall Recommendation Score' as a JSON object with a 'value' and 'justification'.

*Input:*
- Job Role: {job_desc}
- Resume Evaluation: {resume_eval}
- Interview Evaluation: {interview_eval}
- Job-Specific Criteria: {criteria}
    
    
    **Decision Guidelines:**
- "Advanced": Strong candidate with clear alignment to job requirements and good evaluation scores
- "Reject": Clear mismatch with job requirements, poor evaluation scores, or significant concerns
- "Manual Intervention": ONLY when there are conflicting signals, borderline scores, or insufficient data/ interview transcript to make a clear decision. 

    Consider matching keywords from the JD with the resume text in terms of Tech Stack, and role responsibilities to make a decision. Higher matches must be considered for "Advanced" and lower matches for "Reject". If the interview script is not enough, you can mark as "Manual Intervention".
   CRITICAL: DO NOT HALLUCINATE OR MAKE UP DATA. 
*Output both the bulleted summary and the final JSON score block.*
"""

@lru_cache(maxsize=1000)
def create_verdict_prompt(text):
    # This prompt is from your first script
    return f"""From the text below, extract only one of these exact words: "Advanced", "Reject", or "Manual Intervention".You are an extraction assistant. Only output one of the following exact words based on the decision in the provided text: "Advanced", "Reject", or "Manual Intervention". Do not output anything else. Do not add explanations, formatting, or extra text. Do not output the word "Advance".



Text: {text}
"""


# --- STAGE 2: DETAILED PROFILING PROMPTS ---

def create_good_fit_prompt(row_data: dict) -> str:
    """Creates the prompt for the 'Good Fit' summary, adapted from your second script."""
    candidate_name = row_data.get(COL_CANDIDATE_NAME, 'The Candidate')
    company_name = row_data.get(COL_COMPANY_NAME, 'the Company')
    job_description = row_data.get(COL_JOB_DESC, '')
    resume_text = row_data.get(COL_RESUME, '')
    
    # Handle potential NaN values
    if pd.isna(job_description) or pd.isna(resume_text):
        return "Error: Cannot generate summary because Job Description or Resume Text is missing."
    
    # Convert to string if not already
    job_description = str(job_description) if job_description else ''
    resume_text = str(resume_text) if resume_text else ''

    if not job_description or not resume_text:
        return "Error: Cannot generate summary because Job Description or Resume Text is missing."
    
    # A simple way to get job title, can be improved
    job_title = "the Role"
    if 'title' in job_description.lower():
        try:
            job_title = re.search(r'title\s*:\s*(.*)', job_description, re.IGNORECASE).group(1).strip()
        except AttributeError:
            pass # Keep default

    first_name = str(candidate_name).split(' ')[0]

    return f"""
You are a professional recruitment analyst. Analyze the provided information and generate a concise, 4-point bulleted summary explaining why the candidate is a strong fit for the role.

**Context:**
- Candidate: {candidate_name}
- Company: {company_name}
- Role: {job_title}
- Job Description: {job_description}
- Resume: {resume_text}

**Task:**
Generate a 4-point summary. Each point must be a clear, role-specific reason supported by evidence from the resume.

**Strict Output Format Required:**
**Why {first_name} for {job_title} at {company_name}:**
1. **[Core Strength 1]**: [Brief explanation linking candidate's experience to a job requirement].
2. **[Core Strength 2]**: [Brief explanation linking candidate's experience to a job requirement].
3. **[Core Strength 3]**: [Brief explanation linking candidate's experience to a job requirement].
4. **[Core Strength 4]**: [Brief explanation linking candidate's experience to a job requirement].
"""

def create_candidate_profile_prompt(row_data: dict, good_fit_summary: str) -> str:
    """Creates the prompt for the detailed 'Candidate Profile'."""
    if "Error:" in good_fit_summary:
        return good_fit_summary

    candidate_name = row_data.get(COL_CANDIDATE_NAME, "Not Provided")
    company_name = row_data.get(COL_COMPANY_NAME, "Not Provided")
    job_description = row_data.get(COL_JOB_DESC, "")
    resume_text = row_data.get(COL_RESUME, "")
    resume_url = row_data.get(COL_RESUME_URL, "")
    phone = row_data.get(COL_PHONE, "Not Provided")
    email = row_data.get(COL_EMAIL, "Not Provided")
    ctc = row_data.get(COL_CTC, "Not Provided")
    experience = row_data.get(COL_EXPERIENCE, "Not Provided")

    # Extract LinkedIn URL from resume text if present
    linkedin_url = "Not Present"
    if isinstance(resume_text, str) and pd.notna(resume_text):
        match = re.search(r'(https?://(?:www\.)?linkedin\.com/in/[\w\-/]+)', resume_text)
        if match:
            linkedin_url = match.group(0)

    resume_hyperlink = f"[Resume]({resume_url})" if resume_url else "[Resume Not Provided]"
    linkedin_hyperlink = f"[LinkedIn]({linkedin_url})" if linkedin_url != "Not Present" else "[LinkedIn Not Present]"

    first_name = str(candidate_name).split(" ")[0] if candidate_name and candidate_name != "Not Provided" else "Candidate"

    # Try to infer a job title from the job description (fallback to 'the Role')
    job_title = "the Role"
    if isinstance(job_description, str) and job_description:
        m = re.search(r'(?:title\s*[:\-]\s*)([^\n\r]+)', job_description, re.IGNORECASE)
        if m:
            job_title = m.group(1).strip()

    prompt = f"""You are an expert Talent Agency Analyst. Produce a clean, Notion-friendly candidate profile using only the provided information. If a field is missing, write 'Not Provided'. Do not invent numbers, institutions, company names, or degrees.

Context:
- Candidate: {candidate_name}
- Company: {company_name}
- Role: {job_title}
- Job Description: {job_description}
- Resume excerpt: {resume_text}

Good Fit Summary:
{good_fit_summary}

Output requirements (plain text, ready to paste into Notion):
1) Full Name (Current Role at Current Company)
   (Degree, Institution | Degree, Institution) (Last Company)
   Phone: {phone} | Email: {email} | Current CTC: {ctc}
   Experience: {experience}

2) Why {first_name} for {job_title} at {company_name}:
   1. [Clear, role-specific reason supported by evidence from resume or provided data]
   2. [Clear, role-specific reason]
   3. [Clear, role-specific reason]
   4. [Clear, role-specific reason]

3) Technical Alignment:
   - [Short bullets of relevant technical and soft skills]

4) What Stood Out in the Interview:
   - [Brief narrative or bullets; if transcript not available, base this on resume and JD]

5) Links:
   Resume: {resume_hyperlink}
   LinkedIn: {linkedin_hyperlink}

6) One-sentence personalized summary of candidate fit.

Formatting rules:
- Output plain text only (no markdown code fences).
- Use clear line breaks and bullets as above.
- Do NOT hallucinate or invent any numbers, colleges, companies, or names. If information is missing, state 'Not Provided'.
"""

    return prompt
def run_initial_evaluations(df: pd.DataFrame) -> pd.DataFrame:
    """Runs Stage 1: Resume, Interview, Summary, and Verdict evaluations."""
    print("\n" + "="*25 + " STAGE 1: INITIAL EVALUATION " + "="*25)

    # --- Create Prompts ---
    resume_prompts = [create_resume_prompt(r.get(COL_JOB_DESC, ""), r.get(COL_RESUME, ""), r.get(COL_CRITERIA, "")) for _, r in df.iterrows()]
    interview_prompts = [create_interview_prompt(r.get(COL_JOB_DESC, ""), r.get(COL_INTERVIEW, ""), r.get(COL_CRITERIA, "")) if pd.notna(r.get(COL_INTERVIEW)) and r.get(COL_INTERVIEW, "").strip() else "" for _, r in df.iterrows()]

    # --- Run Evaluations in Parallel ---
    df[COL_RESUME_EVAL] = processor.process_prompts_in_parallel("gemini-2.5-pro", resume_prompts, "Resume Evaluations")

    valid_interview_prompts = [p for p in interview_prompts if p]
    interview_results_partial = processor.process_prompts_in_parallel("gemini-2.5-pro", valid_interview_prompts, "Interview Evaluations")

    # Map interview results back
    full_interview_results = [""] * len(df)
    interview_result_idx = 0
    for i in range(len(df)):
        if interview_prompts[i]:
            full_interview_results[i] = interview_results_partial[interview_result_idx]
            interview_result_idx += 1
        else:
            full_interview_results[i] = "Interview not conducted or transcript unavailable."
    df[COL_INTERVIEW_EVAL] = full_interview_results

    # --- Create and Run Summarizer and Verdict ---
    summarizer_prompts = [create_summarizer_prompt(row.get(COL_JOB_DESC, ""), row.get(COL_RESUME_EVAL, ""), row.get(COL_INTERVIEW_EVAL, ""), row.get(COL_CRITERIA, "")) for _, row in df.iterrows()]
    df[COL_SUMMARIZER] = processor.process_prompts_in_parallel("gemini-2.5-pro", summarizer_prompts, "Summaries")

    verdict_prompts = [create_verdict_prompt(summary) for summary in df[COL_SUMMARIZER]]
    df[COL_RESULT] = processor.process_prompts_in_parallel("gemini-2.5-flash", verdict_prompts, "Verdicts")

    print("✅ STAGE 1: Initial Evaluation Complete.")
    return df
def run_detailed_profiling(df: pd.DataFrame) -> pd.DataFrame:
    """Runs Stage 2: 'Good Fit' and 'Candidate Profile' generation."""
    print("\n" + "="*25 + " STAGE 2: DETAILED PROFILING " + "="*25)

    if df.empty:
        print("ℹ️ No candidates to process for detailed profiling. Skipping Stage 2.")
        return pd.DataFrame(columns=[COL_GOOD_FIT, COL_PROFILE])

    print(f"Found {len(df)} candidates for detailed profiling.")

    # --- Create and Run "Good Fit" ---
    row_data_list = [row.to_dict() for _, row in df.iterrows()]
    good_fit_prompts = [create_good_fit_prompt(data) for data in row_data_list]
    good_fit_results = processor.process_prompts_in_parallel("gemini-2.5-pro", good_fit_prompts, "Good Fit Summaries")

    # --- Create and Run "Candidate Profile" ---
    profile_prompts = [create_candidate_profile_prompt(row_data_list[i], good_fit_results[i]) for i in range(len(df))]
    profile_results = processor.process_prompts_in_parallel("gemini-2.5-pro", profile_prompts, "Candidate Profiles")

    # Clean up markdown code blocks and special characters from results
    cleaned_profiles = []
    for p in profile_results:
        # Remove markdown code blocks
        cleaned = re.sub(r"```markdown\n?|```", "", p)
        # Remove flower brackets and other special characters
        cleaned = re.sub(r'[{}]', '', cleaned)  # Remove curly braces
        cleaned = re.sub(r'[^\w\s.,;:!?()-\[\]#*]', '', cleaned)  # Remove other special characters except basic punctuation and markdown
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned_profiles.append(cleaned)

    # Create a results DataFrame to merge back
    results_df = pd.DataFrame({
        COL_GOOD_FIT: good_fit_results,
        COL_PROFILE: cleaned_profiles
    }, index=df.index)

    print("✅ STAGE 2: Detailed Profiling Complete.")
    return results_df
# --- MAIN ORCHESTRATOR ---

def main():
    # --- Load Data ---
    try:
        # Prioritize CSV, then look for Excel
        if os.path.exists("Supahealth (01 sept) - Supahealth.csv"):
            df = pd.read_csv("Supahealth (01 sept) - Supahealth.csv")
            print(f"📄 Successfully loaded 'Supahealth (01 sept) - Supahealth.csv' with {len(df)} rows.")
        else:
            excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
            if not excel_files:
                raise FileNotFoundError("No 'correct - Sheet1.csv' or .xlsx file found.")
            df = pd.read_excel(excel_files[0])
            print(f"📄 Successfully loaded '{excel_files[0]}' with {len(df)} rows.")
    except Exception as e:
        print(f"❌ FATAL ERROR loading data: {e}")
        return

    # --- Title Case for Candidate Name ---
    if COL_CANDIDATE_NAME in df.columns:
        df[COL_CANDIDATE_NAME] = df[COL_CANDIDATE_NAME].astype(str).apply(lambda x: x.title() if pd.notna(x) else x)
    # --- Title Case Resume Text ---
    if COL_RESUME in df.columns:
        df[COL_RESUME] = df[COL_RESUME].astype(str).apply(lambda x: x.title() if pd.notna(x) else x)
    # --- Clean Phone Numbers: Remove +91/91 and keep last 10 digits ---
    if COL_PHONE in df.columns:
        def clean_phone(phone):
            if pd.isna(phone):
                return phone
            phone_str = str(phone)
            # Remove spaces, dashes, parentheses
            phone_str = re.sub(r'[\s\-()]+', '', phone_str)
            # Remove leading +91 or 91
            phone_str = re.sub(r'^\+?91', '', phone_str)
            # Keep only digits
            phone_str = re.sub(r'\D', '', phone_str)
            # Return last 10 digits if available
            return phone_str[-10:] if len(phone_str) >= 10 else phone_str
        df[COL_PHONE] = df[COL_PHONE].apply(clean_phone)
    """Main function to run the entire pipeline."""
    # ...existing code...

    # --- Deduplicate by Resume Link (immediately after loading data) ---
    initial_rows = len(df)
    df.drop_duplicates(subset=[COL_RESUME_URL], keep='first', inplace=True)
    if (removed_count := initial_rows - len(df)) > 0:
        print(f"🔍 Deduplication removed {removed_count} duplicate entries based on resume link.")

    # --- Run Stage 1 ---
    df = run_initial_evaluations(df)

    # --- Filter Rejected Candidates ---
    print("\n" + "="*25 + " FILTERING REJECTED CANDIDATES " + "="*25)
    initial_count = len(df)
    # Normalize verdict text for reliable filtering
    df[COL_RESULT] = df[COL_RESULT].str.strip().str.title()

    non_rejected_df = df[df[COL_RESULT] != 'Reject'].copy()

    print(f"Initial candidates: {initial_count}")
    print(f"Candidates marked 'Rejected': {initial_count - len(non_rejected_df)}")
    print(f"Candidates remaining for profiling ('Advanced' or 'Manual Intervention'): {len(non_rejected_df)}")

    # --- Run Stage 2 ---
    if not non_rejected_df.empty:
        detailed_profiles_df = run_detailed_profiling(non_rejected_df)
        
        # --- Merge Results ---
        print("\n" + "="*25 + " MERGING RESULTS " + "="*25)
        # Initialize columns if they don't exist
        if COL_GOOD_FIT not in df.columns:
            df[COL_GOOD_FIT] = ""
            df[COL_PROFILE] = ""
        
        # Update the main dataframe with the results from the detailed profiling stage
        df.update(detailed_profiles_df)
        print("✅ Detailed profiles merged into the main dataset.")
    else:
        print("ℹ️ No candidates to profile, skipping merge step.")

    # --- Save Output ---
    output_filename = "supahealth (01 sept) - supahealth posteval.csv"
    try:
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n🎉 Pipeline Complete! All results saved to '{output_filename}'.")
    except Exception as e:
        print(f"❌ ERROR saving results: {e}")

    total_time = time.time() - PIPELINE_START_TIME
    print(f"⏱️ Total pipeline execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes).")

if __name__ == "__main__":
    from tqdm import tqdm
    main()