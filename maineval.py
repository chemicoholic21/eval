import os
import sys
import time
import random
import re
import json
import hashlib
import threading
import pandas as pd
from functools import lru_cache
from gemini_rest import gemini_generate_content
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from tqdm import tqdm

# --- RECRUITER GPT EVALUATION PROMPT ---
@lru_cache(maxsize=1000)
def create_recruiter_gpt_eval_prompt(recruiter_request, resume_text):
    return f"""You are an expert AI recruiting assistant. Your task is to evaluate a candidate's resume based **only** on a single, high-priority request from a recruiter.\n\nAnalyze the resume provided below and determine if the candidate's resume text directly matches the recruiter's request.\n\nRecruiter Request: {recruiter_request}\nResume: {resume_text}\n\nOutput a short evaluation and a clear yes/no match statement."""

## (rest of the script remains unchanged)

# --- GLOBAL CONFIGURATION & SETUP ---
print("Starting the UNIFIED Candidate Evaluation Pipeline...")
PIPELINE_START_TIME = time.time()
load_dotenv()

# Configure Google Generative AI
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    print("Google Generative AI API key loaded.")
except Exception as e:
    print(f"FATAL ERROR: Could not configure Google Generative AI. Reason: {e}")
    exit(1)



# --- COLUMN NAME CONSTANTS (MATCHING devops_plum - Sheet1.csv) ---
# Input Columns
COL_ID = "Grapevine Aiinterviewinstance → ID"
COL_RESUME = "Grapevine Userresume - Resume → Metadata → Resume Text"
COL_RESUME_URL = "Grapevine Userresume - Resume → Resume URL"
COL_CANDIDATE_NAME = "Grapevine Userresume - Resume → Metadata → User Real Name"
COL_DURATION = "Grapevine Aiinterviewinstance → Duration"
COL_EXPERIENCE = "Grapevine Userresume - Resume → Metadata → Phone Number"
COL_PHONE = "Grapevine Userresume - Resume → Metadata → Phone Number"
COL_CTC = "Grapevine Userresume - Resume → Metadata → Current Salary"
COL_EMAIL = "Grapevine Userresume - Resume → Metadata → Email"
COL_RESUME_EVAL = "Resume Evaluator Agent (RAG-LLM)"
# Read this column from input CSV, but rename it in output
COL_SUMMARIZER_INPUT = "Resume + Interview Summarizer Agent"
COL_SUMMARIZER = "Resume + RecruiterGPT Summarizer Agent"
COL_RESULT = "Result[LLM]"
COL_RECRUITER_GPT = "Recruiter GPT"
COL_USER_ID = "Grapevine Aiinterviewinstance → User ID"
COL_FINAL_DECISION = "Final Decision"
COL_GOOD_FIT = "Good Fit"
COL_PROFILE = "Candidate Profile"
COL_NOTICE_PERIOD = "Grapevine Userresume - Resume → Metadata → Notice Period"
COL_NOTICE_PERIOD_PREF = "User → User Settings → Round1 Preference → Notice Period"

# Columns not present in this CSV are omitted.
COL_GOOD_FIT = "Good Fit"
COL_PROFILE = "Candidate Profile"


# For compatibility in code logic (fallbacks for missing columns)
COL_COMPANY_NAME = COL_CANDIDATE_NAME  # No company name column, fallback to candidate name
COL_CRITERIA = COL_RECRUITER_GPT  # Use Recruiter GPT as criteria


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
                print(f"[CACHE HIT] Key: {cache_key}")
                return _response_cache[cache_key]
            return ""

    def _set_cached_response(self, cache_key: str, response: str):
        with _cache_lock:
            _response_cache[cache_key] = response
            print(f"[CACHE STORE] Key: {cache_key}")

    def _gemini_generate_single(self, model_name: str, prompt: str) -> str:
        """Optimized single API call with caching."""
        if not prompt or not prompt.strip():
            print("[ERROR] Empty prompt provided.")
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
                    print(f"[RETRY] API call attempt {attempt+1} for key: {cache_key}")
                    time.sleep(random.uniform(0.5, 1.5))
                print(f"[API CALL] Model: {model_name}, Attempt: {attempt+1}, Key: {cache_key}")
                # Use Gemini REST API helper
                result = gemini_generate_content(GEMINI_API_KEY, prompt)
                if result:
                    result = result.strip()
                else:
                    result = "Error: No response text"
                if self.cache_enabled:
                    self._set_cached_response(cache_key, result)
                self.api_call_count += 1
                print(f"[API RESULT] Key: {cache_key}, Result: {str(result)[:80]}...")
                return result
            except Exception as e:
                print(f"[API ERROR] Attempt {attempt+1} failed for key: {cache_key}. Reason: {e}")
                if attempt == max_retries:
                    return f"Error: API call failed after {max_retries} retries. Reason: {e}"
        print(f"[ERROR] Max retries exceeded for key: {cache_key}")
        return "Error: Max retries exceeded"

    def process_prompts_in_parallel(self, model_name: str, prompts: List[str], task_description: str) -> List[str]:
        """Processes a list of prompts using a thread pool for high concurrency."""
        if not prompts:
            print(f"[WARN] No prompts provided for {task_description}.")
            return []
        print(f"[START] Parallel processing for '{task_description}' ({len(prompts)} prompts)...")
        results = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {executor.submit(self._gemini_generate_single, model_name, prompt): i for i, prompt in enumerate(prompts)}
            with tqdm(as_completed(future_to_index), total=len(prompts), desc=f"{task_description}") as progress_bar:
                for future in progress_bar:
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                        print(f"[PROGRESS] {task_description}: Completed {index+1}/{len(prompts)}")
                    except Exception as e:
                        print(f"[ERROR] {task_description}: Task {index+1} failed with exception: {e}")
                        results[index] = f"Error: Task failed with exception: {e}"
        print(f"[COMPLETE] '{task_description}'.")
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
def create_recruitergpt_prompt(resume_text, criteria):
    """
    Generates a prompt for RecruiterGPT to evaluate a candidate's resume based only on a single, high-priority recruiter request.
    """
    return f"""You are an expert AI recruiting assistant. Your task is to evaluate a candidate's resume based **only** on a single, high-priority request from a recruiter.

Analyze the resume provided below and determine if the candidate's resume text directly matches the recruiter's request. Only use the information provided in the resume. Never invent, assume, or add details. If information is missing, state this in your justification. Always output in the specified JSON format.


*Input:*
- Resume: {resume_text}
- Job-Specific Criteria: {criteria}

*Output your assessment as a JSON object with a 'value' (0-10) and 'justification'.*
"""



@lru_cache(maxsize=1000)
def create_summarizer_prompt(job_desc, resume_eval, criteria):
    # This prompt is from your first script
    return f"""As an AI hiring coordinator, summarize the candidate's evaluation. Provide bullet points for key strengths, weaknesses, and a recommendation (advance/reject/manual intervention). Also, provide an 'Overall Recommendation Score' as a JSON object with a 'value' and 'justification'.

*Input:*
- Job Role: {job_desc}
- Resume Evaluation: {resume_eval}

- Job-Specific Criteria: {criteria}
    
    
    **Decision Guidelines:**
- "Advanced": Strong candidate with clear alignment to job requirements and good evaluation scores
- "Reject": Clear mismatch with job requirements, poor evaluation scores, or significant concerns
- "Manual Intervention": ONLY when there are conflicting signals, borderline scores, or insufficient data to make a clear decision. But they are from top product companies or startups or top universities/institutes.

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
    # Use the job description column from the DataFrame if present
    job_description = row_data.get('Grapevine Job - Job → Description', '')
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
        m = re.search(r'title\s*:\s*(.*)', job_description, re.IGNORECASE)
        if m:
            job_title = m.group(1).strip()

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
    job_description = ''  # No job description column in new CSV
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
    print(f"[INFO] Running initial evaluation for {len(df)} candidates.")
    # --- Create Prompts ---
    resume_prompts = [create_resume_prompt('', r.get(COL_RESUME, ''), r.get(COL_CRITERIA, '')) for _, r in df.iterrows()]
    print(f"[INFO] Created {len(resume_prompts)} resume prompts.")
    # No interview prompts or columns in this CSV
    df[COL_RESUME_EVAL] = processor.process_prompts_in_parallel('gemini-2.5-pro', resume_prompts, 'Resume Evaluations')
    print(f"[INFO] Resume evaluations complete.")
    # No interview evaluation
    summary_prompts = [create_summarizer_prompt('', row.get(COL_RESUME_EVAL, ''), row.get(COL_CRITERIA, '')) for _, row in df.iterrows()]
    print(f"[INFO] Created {len(summary_prompts)} summary prompts.")
    df[COL_SUMMARIZER] = processor.process_prompts_in_parallel('gemini-2.5-pro', summary_prompts, 'Summaries')
    print(f"[INFO] Summaries complete.")
    verdict_prompts = [create_verdict_prompt(summary) for summary in df[COL_SUMMARIZER]]
    print(f"[INFO] Created {len(verdict_prompts)} verdict prompts.")
    df[COL_RESULT] = processor.process_prompts_in_parallel('gemini-2.5-flash', verdict_prompts, 'Verdicts')
    print('STAGE 1: Initial Evaluation Complete.')
    return df
def run_detailed_profiling(df: pd.DataFrame) -> pd.DataFrame:
    """Runs Stage 2: 'Good Fit' and 'Candidate Profile' generation."""
    print("\n" + "="*25 + " STAGE 2: DETAILED PROFILING " + "="*25)
    if df.empty:
        print("[INFO] No candidates to process for detailed profiling. Skipping Stage 2.")
        return pd.DataFrame(columns=[COL_GOOD_FIT, COL_PROFILE])

    print(f"[INFO] Found {len(df)} candidates for detailed profiling.")

    # --- Create and Run "Good Fit" ---
    row_data_list = [row.to_dict() for _, row in df.iterrows()]
    good_fit_prompts = [create_good_fit_prompt(data) for data in row_data_list]
    print(f"[INFO] Created {len(good_fit_prompts)} good fit prompts.")
    good_fit_results = processor.process_prompts_in_parallel("gemini-2.5-pro", good_fit_prompts, "Good Fit Summaries")
    print(f"[INFO] Good fit summaries complete.")

    # --- Create and Run "Candidate Profile" ---
    profile_prompts = [create_candidate_profile_prompt(row_data_list[i], good_fit_results[i]) for i in range(len(df))]
    print(f"[INFO] Created {len(profile_prompts)} candidate profile prompts.")
    profile_results = processor.process_prompts_in_parallel("gemini-2.5-pro", profile_prompts, "Candidate Profiles")
    print(f"[INFO] Candidate profiles complete.")

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

    print("STAGE 2: Detailed Profiling Complete.")
    return results_df
# --- MAIN ORCHESTRATOR ---

def main():
    # ...existing code...
    print("\n" + "="*25 + " MAIN PIPELINE START " + "="*25)
    print("[DEBUG] Entered main() function.")
    # --- Recruiter Gpt evaluation column ---
    COL_RECRUITER_GPT_INPUT = 'Recruiter GPT'
    COL_RECRUITER_GPT_EVAL = 'Recruiter Gpt evaluation'
    global df
    df = None
    # --- Load Data ---
    input_path = None
    output_path = None
    if len(sys.argv) >= 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    print(f"[DEBUG] input_path: {input_path}, output_path: {output_path}")
    try:
<<<<<<< HEAD:monday_test.py
        # Prioritize CSV, then look for Excel
        if os.path.exists("Supahealth (01 sept) - Supahealth.csv"):
            df = pd.read_csv("Supahealth (01 sept) - Supahealth.csv")
            print(f"📄 Successfully loaded 'Supahealth (01 sept) - Supahealth.csv' with {len(df)} rows.")
=======
        if input_path:
            print(f"[INFO] Loading input file: {input_path}")
            df = pd.read_csv(input_path)
            print(f"[INFO] Successfully loaded '{input_path}' with {len(df)} rows.")
            # --- Always overwrite 'Recruiter GPT' column with Gemini 2.5 Flash evaluation ---
            COL_RECRUITER_GPT = 'Recruiter GPT'
            COL_RECRUITER_GPT_RESPONSE = 'Recruiter GPT Response'
            if COL_RECRUITER_GPT_RESPONSE in df.columns and COL_RESUME in df.columns:
                print(f"[INFO] Overwriting 'Recruiter GPT' column with Gemini 2.5 Flash evaluation...")
                recruiter_gpt_fit_prompts = [
                    f"""You are an expert AI recruiting assistant. Based on the recruiter's request and the candidate's resume, state if the candidate is fit for the role (yes/no) and provide a two-line explanation.\n\nRecruiter Request: {row[COL_RECRUITER_GPT_RESPONSE]}\nResume: {row[COL_RESUME]}\n\nOutput: 1. Fit for role: Yes/No 2. Two-line explanation."""
                    for _, row in df.iterrows()
                ]
                recruiter_gpt_fit_results = processor.process_prompts_in_parallel(
                    'gemini-2.5-flash', recruiter_gpt_fit_prompts, 'Recruiter GPT Fit Evaluation'
                )
                df[COL_RECRUITER_GPT] = recruiter_gpt_fit_results
                print(f"[INFO] 'Recruiter GPT' column overwritten with fit evaluation.")
                print("[DEBUG] Sample of 'Recruiter GPT' column after overwrite:")
                print(df[['Recruiter GPT']].head(5))
>>>>>>> b2c64d6fa1ab50c0e6bf4bf901e6c9da6179f24a:maineval.py
        else:
            print("[FATAL ERROR] No input file provided. Please specify an input CSV file as a command line argument.")
            return
    except Exception as e:
        print(f"[FATAL ERROR] loading data: {e}")
        return
    # Print columns with non-ASCII characters replaced to avoid UnicodeEncodeError
    safe_columns = [col.encode('ascii', 'replace').decode() for col in df.columns.tolist()]
    print("[DEBUG] DataFrame loaded. Columns: ", safe_columns)

    # --- Title Case for Candidate Name ---
    if COL_CANDIDATE_NAME in df.columns:
        print(f"[INFO] Title-casing candidate names.")
        df[COL_CANDIDATE_NAME] = df[COL_CANDIDATE_NAME].astype(str).apply(lambda x: x.title() if pd.notna(x) else x)
    else:
        print(f"[DEBUG] Column {COL_CANDIDATE_NAME.encode('ascii', 'replace').decode()} not found in DataFrame.")
    # --- Title Case Resume Text ---
    if COL_RESUME in df.columns:
        print(f"[INFO] Title-casing resume text.")
        df[COL_RESUME] = df[COL_RESUME].astype(str).apply(lambda x: x.title() if pd.notna(x) else x)
    else:
        print(f"[DEBUG] Column {COL_RESUME.encode('ascii', 'replace').decode()} not found in DataFrame.")
    # --- Clean Phone Numbers: Remove +91/91 and keep last 10 digits ---
    if COL_PHONE in df.columns:
        print(f"[INFO] Cleaning phone numbers.")
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
    else:
        print(f"[DEBUG] Column {COL_PHONE.encode('ascii', 'replace').decode()} not found in DataFrame.")

    # --- Add 'title' column from UI input (if provided as env or argument) ---
    title_value = os.environ.get('TITLE_INPUT', None)
    if len(sys.argv) > 3:
        title_value = sys.argv[3]
    if title_value and df is not None:
        print(f"[INFO] Adding 'title' column to DataFrame with value: {title_value}")
        df['title'] = title_value
    else:
        print(f"[DEBUG] Title value not set or DataFrame is None.")

    # --- Deduplicate by Resume Link (immediately after loading data) ---
    initial_rows = len(df)
    print(f"[DEBUG] Deduplicating by {COL_RESUME_URL.encode('ascii', 'replace').decode()}...")
    df.drop_duplicates(subset=[COL_RESUME_URL], keep='first', inplace=True)
    removed_count = initial_rows - len(df)
    if removed_count > 0:
        print(f"[INFO] Deduplication removed {removed_count} duplicate entries based on resume link.")
    else:
        print(f"[DEBUG] No duplicates found for {COL_RESUME_URL.encode('ascii', 'replace').decode()}.")

    # --- Recruiter Gpt evaluation column ---
    COL_RECRUITER_GPT_RESPONSE = 'Recruiter GPT Response'
    COL_RECRUITER_GPT_EVAL = 'Recruiter Gpt evaluation'
    if COL_RECRUITER_GPT_RESPONSE in df.columns and COL_RESUME in df.columns:
        print(f"[INFO] Evaluating resumes for recruiter GPT response using Gemini 2.5 Flash...")
        recruiter_gpt_prompts = [
            f"""You are an expert AI recruiting assistant. Your task is to evaluate a candidate's resume based **only** on a single, high-priority request from a recruiter.\n\nAnalyze the resume provided below and determine if the candidate's resume text directly matches the recruiter's request.\n\nRecruiter Request: {row[COL_RECRUITER_GPT_RESPONSE]}\nResume: {row[COL_RESUME]}\n\nOutput a short evaluation (max 2 lines) and a clear yes/no match statement."""
            for _, row in df.iterrows()
        ]
        print(f"[INFO] Created {len(recruiter_gpt_prompts)} recruiter GPT response prompts.")
        recruiter_gpt_results = processor.process_prompts_in_parallel(
            'gemini-2.5-flash', recruiter_gpt_prompts, 'Recruiter GPT Evaluation'
        )
        df[COL_RECRUITER_GPT_EVAL] = recruiter_gpt_results
        print(f"[INFO] Recruiter Gpt evaluation (from response) complete.")
    else:
        print(f"[DEBUG] Skipping recruiter GPT evaluation: columns missing.")

    # --- Run Stage 1 ---
    print("[DEBUG] Running initial evaluations...")
    df = run_initial_evaluations(df)
    print("[DEBUG] Initial evaluations complete.")

    # --- Filter Rejected Candidates ---
    print("\n" + "="*25 + " FILTERING REJECTED CANDIDATES " + "="*25)
    initial_count = len(df)
    # Normalize verdict text for reliable filtering
    df[COL_RESULT] = df[COL_RESULT].str.strip().str.title()

    print("[DEBUG] Filtering rejected candidates...")
    non_rejected_df = df[df[COL_RESULT] != 'Reject'].copy()

    print(f"[INFO] Initial candidates: {initial_count}")
    print(f"[INFO] Candidates marked 'Rejected': {initial_count - len(non_rejected_df)}")
    print(f"[INFO] Candidates remaining for profiling ('Advanced' or 'Manual Intervention'): {len(non_rejected_df)}")

    # --- Run Stage 2 ---
    if not non_rejected_df.empty:
        print("[DEBUG] Running detailed profiling...")
        detailed_profiles_df = run_detailed_profiling(non_rejected_df)
        # --- Merge Results ---
        print("\n" + "="*25 + " MERGING RESULTS " + "="*25)
        # Initialize columns if they don't exist
        if COL_GOOD_FIT not in df.columns:
            df[COL_GOOD_FIT] = ""
            df[COL_PROFILE] = ""
        # Update the main dataframe with the results from the detailed profiling stage
        df.update(detailed_profiles_df)
        print("[INFO] Detailed profiles merged into the main dataset.")
    else:
        print("[DEBUG] No candidates to profile, skipping merge step.")

    # --- Save Output ---
<<<<<<< HEAD:monday_test.py
    output_filename = "supahealth (01 sept) - supahealth posteval.csv"
=======
>>>>>>> b2c64d6fa1ab50c0e6bf4bf901e6c9da6179f24a:maineval.py
    try:
        print(f"[DEBUG] Attempting to write output file...")
        if output_path:
            print(f"[INFO] Attempting to write output to: {output_path}")
            print("[DEBUG] Sample of 'Recruiter GPT' column just before saving output file:")
            print(df[['Recruiter GPT']].head(5))
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            if os.path.exists(output_path):
                print(f"\n[INFO] Pipeline Complete! All results saved to '{output_path}'.")
                # Play a loud sound when processing is complete (Windows only)
                try:
                    import winsound
                    duration = 1000  # milliseconds
                    freq = 1200      # Hz
                    winsound.Beep(freq, duration)
                except Exception as e:
                    print(f"[WARN] Could not play sound: {e}")
            else:
                print(f"[ERROR] Output file was not created at '{output_path}'. Check for earlier errors.")
        else:
            output_filename = "platform_posteval_ads.csv"
            print(f"[INFO] Attempting to write output to: {output_filename}")
            df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            if os.path.exists(output_filename):
                print(f"\n[INFO] Pipeline Complete! All results saved to '{output_filename}'.")
                # Play a loud sound when processing is complete (Windows only)
                try:
                    import winsound
                    duration = 1000  # milliseconds
                    freq = 1200      # Hz
                    winsound.Beep(freq, duration)
                except Exception as e:
                    print(f"[WARN] Could not play sound: {e}")
            else:
                print(f"[ERROR] Output file was not created at '{output_filename}'. Check for earlier errors.")
    except Exception as e:
        print(f"[FATAL ERROR] Exception while saving results: {e}")
    total_time = time.time() - PIPELINE_START_TIME
    print(f"[INFO] Total pipeline execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes).")

if __name__ == "__main__":
    from tqdm import tqdm
    main()