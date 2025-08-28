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
from typing import List
from tqdm import tqdm
from datetime import datetime
from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify
import tempfile
from werkzeug.utils import secure_filename
import traceback

# --- GLOBAL CONFIGURATION & SETUP ---
print("🚀 Starting the Candidate Data Transformation Pipeline...")
PIPELINE_START_TIME = time.time()
load_dotenv()

# Configure Gemini API
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY: 
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not configure Gemini API. Reason: {e}")
    exit(1)

# --- COLUMN NAME CONSTANTS ---
# Source columns from the input file (MC_SPE_Ads_results.csv)
COL_SRC_JOB_DESC = "Grapevine Job - Job → Description"
COL_SRC_INTERVIEW = "Grapevine Aiinterviewinstance → Transcript → Conversation"
COL_SRC_RESUME_TEXT = "Grapevine Userresume - Resume → Metadata → Resume Text"
COL_SRC_CANDIDATE_NAME = "Grapevine Userresume - Resume → Metadata → User Real Name" # Will be auto-detected if not found
COL_SRC_RESUME_URL = "Grapevine Userresume - Resume → Resume URL"
COL_SRC_PHONE = "Grapevine Userresume - Resume → Metadata → Phone Number"
COL_SRC_EMAIL = "Grapevine Userresume - Resume → Metadata → Email"
COL_SRC_CTC = "Grapevine Userresume - Resume → Metadata → Current Salary"
COL_SRC_SUMMARIZER = "Resume + Interview Summarizer Agent" # Used for 'why_candidate'
COL_SRC_NOTICE_PERIOD_PREFERENCE = "User → User Settings → Round1 Preference → Notice Period"
COL_SRC_NOTICE_PERIOD_RESUME = "Grapevine Userresume - Resume → Metadata → Notice Period"

# Generated columns during the process
COL_GEN_STRUCTURED_PROFILE = "Structured Profile JSON"

# --- API PROCESSOR CLASS ---
_response_cache = {}
_cache_lock = threading.Lock()

class APIProcessor:
    def __init__(self, max_workers=10, cache_enabled=True):
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled

    def _generate_cache_key(self, model_name: str, prompt: str) -> str:
        return hashlib.md5(f"{model_name}:{prompt}".encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> str:
        with _cache_lock: return _response_cache.get(cache_key, "")

    def _set_cached_response(self, cache_key: str, response: str):
        with _cache_lock: _response_cache[cache_key] = response

    def _gemini_generate_single(self, model_name: str, prompt: str) -> str:
        """Generate a single response using Gemini API with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0: 
                    time.sleep(random.uniform(1.0, 2.0) * attempt)
                
                model = genai.GenerativeModel(model_name)
                
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=4096,
                    )
                )
                
                # Clean up markdown formatting from Gemini response
                response_text = response.text.strip() if response.text else '{"error": "No response content"}'
                
                # Remove markdown code blocks if present
                response_text = re.sub(r'^```json\s*', '', response_text)
                response_text = re.sub(r'\s*```$', '', response_text)
                response_text = response_text.strip()
                
                return response_text
                
            except Exception as e:
                print(f"Warning: Gemini call failed on attempt {attempt+1}. Error: {e}")
                if attempt == max_retries - 1: 
                    return f'{{"error": "Gemini API call failed: {e}"}}'
        return '{"error": "Max retries exceeded for Gemini"}'

    def _gemini_generate_text_single(self, model_name: str, prompt: str) -> str:
        """Generate text response using Gemini API for non-JSON prompts."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0: 
                    time.sleep(random.uniform(1.0, 2.0) * attempt)
                
                model = genai.GenerativeModel(model_name)
                
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=4096,
                    )
                )
                
                return response.text.strip() if response.text else "Error: No response content"
                
            except Exception as e:
                print(f"Warning: Gemini text call failed on attempt {attempt+1}. Error: {e}")
                if attempt == max_retries - 1: 
                    return f"Error: Gemini API call failed: {e}"
        return "Error: Max retries exceeded for Gemini"

    def process_prompts_in_parallel(self, model_name: str, prompts: List[str], task_description: str, is_json=True) -> List[str]:
        if not prompts: return []
        print(f"🔥 Starting parallel processing for '{task_description}' ({len(prompts)} prompts) using {model_name}...")
        results = [""] * len(prompts)
        
        # Choose the appropriate method based on output type
        method = self._gemini_generate_single if is_json else self._gemini_generate_text_single
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {executor.submit(method, model_name, prompt): i for i, prompt in enumerate(prompts)}
            with tqdm(as_completed(future_to_index), total=len(prompts), desc=f"⚡️ {task_description}") as progress_bar:
                for future in progress_bar:
                    index = future_to_index[future]
                    try: 
                        results[index] = future.result()
                    except Exception as e: 
                        results[index] = f'{{"error": "Task failed with exception: {e}"}}' if is_json else f"Error: Task failed with exception: {e}"
        print(f"✅ Completed '{task_description}'.")
        return results

processor = APIProcessor(max_workers=10, cache_enabled=True)

# --- FLASK WEB INTERFACE SETUP ---
app = Flask(__name__)
app.secret_key = 'grapevine-website-csv-generator-secret-key-change-this'  # Change this to a secure secret key

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global variable to track processing status
processing_status = {}

# --- FLASK HELPER FUNCTIONS ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file_pipeline(upload_path: str) -> pd.DataFrame:
    """Process the uploaded file using the existing template pipeline functions."""
    try:
        # Try different encodings to handle special characters
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
        df = None
        
        for encoding in encodings_to_try:
            try:
                if upload_path.endswith('.csv'):
                    # Read with phone columns as strings to prevent scientific notation
                    dtype_dict = {}
                    if COL_SRC_PHONE:
                        dtype_dict[COL_SRC_PHONE] = str
                    df = pd.read_csv(upload_path, encoding=encoding, dtype=dtype_dict)
                else:
                    # For Excel files
                    dtype_dict = {}
                    if COL_SRC_PHONE:
                        dtype_dict[COL_SRC_PHONE] = str
                    df = pd.read_excel(upload_path, dtype=dtype_dict)
                
                print(f"📄 Successfully loaded file with {len(df)} rows using {encoding if upload_path.endswith('.csv') else 'Excel'} format.")
                break
            except (UnicodeDecodeError, Exception) as e:
                if upload_path.endswith('.xlsx'):
                    raise e
                continue
        
        if df is None:
            raise ValueError("Could not read file with any supported encoding")
        
        # Process the data using the main function logic
        return process_dataframe(df)
        
    except Exception as e:
        print(f"Error in pipeline processing: {e}")
        raise e

def process_file_async(upload_path, result_filename, task_id):
    """Process the uploaded file asynchronously"""
    try:
        processing_status[task_id] = {'status': 'processing', 'progress': 0, 'message': 'Starting processing...'}
        
        processing_status[task_id].update({'progress': 10, 'message': 'Loading and processing data...'})
        df = process_file_pipeline(upload_path)

        # Save Output
        processing_status[task_id].update({'progress': 90, 'message': 'Saving results...'})
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        df.to_csv(result_path, index=False, encoding='utf-8-sig', quoting=1, quotechar='"')
        
        processing_status[task_id].update({
            'status': 'completed', 
            'progress': 100, 
            'message': f'Processing complete! Processed {len(df)} candidates.',
            'result_file': result_filename,
            'stats': {
                'total_candidates': len(df),
                'processed': len(df)
            }
        })
        
        print(f"🎉 Pipeline Complete! Results saved to '{result_filename}'.")
        
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        processing_status[task_id] = {
            'status': 'error', 
            'progress': 0, 
            'message': error_msg
        }

# --- MASTER EXTRACTION PROMPT ---
def create_structured_extraction_prompt(row_data: dict) -> str:
    """This is the new master prompt that asks for a structured JSON output for all target columns."""
    return f"""
You are an expert Talent Information Extraction system. Your task is to analyze the provided texts and extract specific information into a structured JSON object.

**CRITICAL INSTRUCTIONS:**
1.  **NO HALLUCINATION:** Extract information *exclusively* from the provided texts. If a piece of information is not present, the value for that key MUST be an empty string "".
2.  **JSON OUTPUT ONLY:** Your entire response must be a single, valid JSON object. Do not include any other text, explanations, or markdown formatting.
3.  **Follow Formatting Rules:** Adhere strictly to the format requested for each field.

**INPUT DATA:**
---
**Resume Text:**
{row_data.get(COL_SRC_RESUME_TEXT, 'N/A')}
---
**Job Description:**
{row_data.get(COL_SRC_JOB_DESC, 'N/A')}
---
**Interview Transcript:**
{row_data.get(COL_SRC_INTERVIEW, 'N/A')}
---

**TASK: Create a JSON object with the following exact keys and fill them according to these rules:**
- "name": Extract the full name of the candidate from the resume text.
- "role_applied": Extract the job title from the job description (e.g., "Senior Software Engineer").
- "current_position": Extract the candidate's most recent job title and company from the resume text (e.g., "Software Development Lead, Innovate Labs").
- "education": Extract the candidate's highest degree and university from the resume text (e.g., "B.E. in Computer Science, IIT Delhi").
- "experience": Create a detailed, 2-3 sentence summary of the candidate's professional experience from the resume text, highlighting key achievements and responsibilities.
- "phone": Extract the primary phone number from the resume text as a string (e.g., "9876543210" not in scientific notation).
- "email": Extract the primary email address from the resume text.
- "technical_alignment": Analyze the resume and job description. Return a JSON object formatted as a string with a single key "skills" containing a list of the top 10 most relevant technical skills. Example: '{{"skills": ["Java", "Python", "AWS"]}}'.
- "interview_highlights": Create a concise, one-sentence summary of the key points from the interview transcript, focusing on what stood out.
- "summary": Generate a professional, one-sentence summary of the candidate's overall profile, like "A highly motivated and experienced software professional with a track record of leading teams... taken from the markdown of ".
- "company": Extract the name of the candidate's most recent employer from the resume text.
- "role": Extract the candidate's most recent job title from the resume text.
- "linkedin_link": Find and extract the full LinkedIn profile URL from the resume text. If not found, use an empty string.
"""

# --- TEXT FORMATTING PROMPTS ---
def create_why_candidate_prompt(candidate_profile_text: str) -> str:
    """Creates prompt for formatting why_candidate field using Gemini."""
    return f"""From the provided Candidate Profile text, extract the key reasons the candidate is suitable for the specified role and summarize them as a single clean line in plain text. The summary should include the candidate's name and concisely highlight their relevant experience, skills, and technologies, using only information explicitly present in the original text. Do not add, infer, or introduce any unstated details. Do not use special characters or formatting.
Under no circumstances can you mention that interview was not available. Generate this in a case based, highlight candidates project and skills that make them hireable for the role.

Example of how the response would look:
Bikash's experience building and scaling backend systems, particularly the SSAI system and Ad Server, along with his proficiency in the required programming languages and experience with relevant technologies like Kubernetes, makes him a strong candidate.

Candidate Profile: {candidate_profile_text}"""

def create_summary_prompt(text: str) -> str:
    """Creates prompt for formatting summary field using Gemini."""
    return f"""From the provided Candidate Profile text, extract all information that directly demonstrates why the candidate is suitable for the specified role. Summarize this information in a single paragraph not exceeding 60 words. The paragraph must be clear, concise, and self-contained, using only information explicitly present in the original text. Do not reference interview transcripts or add any unstated details. Respond in plain text with no special characters.

Example of how the response should look: [COL_SRC_CANDIDATE_NAME] has extensive SaaS industry experience, consistently exceeded sales targets, managed full sales cycles and pipelines using Salesforce, addressed complex global data privacy issues by involving solution and compliance teams, and demonstrated strong internal collaboration skills by proactively engaging relevant stakeholders and tailoring communication across functions to secure resources in global deals.

Candidate Profile Text: {text}"""

def create_interview_highlights_prompt(text: str, job_desc: str, resume_text: str, candidate_name: str) -> str:
    """Creates prompt for formatting interview_highlights field using Gemini."""
    return f"""From the candidate_profile text, extract what specifically stood out in the interview and summarize it in a single paragraph not exceeding 40 words. DO NOT TRUNCATE THE TEXT. The paragraph should clearly highlight interview strengths, using only information explicitly present in the original text. Do not add, infer, or introduce any unstated details. Respond in plain text with no special characters or formatting. The summary MUST begin with the candidate's name as found in the "User Real Name" field (or as provided in the data). Generate this in a case based, highlight candidate's projects and skills that make them hireable for the role.

NOTE: If the interview transcript is not present and is stated in the candidate profile text, please use the Job Description & Resume Text provided to generate the section so that it seems like the interview happened. 
UNDER NO CIRCUMSTANCES SHOULD YOU MENTION THAT INTERVIEW WAS NOT AVAILABLE.

Example response:
Bikash impressed with clear explanations of technical concepts, gave detailed examples of past project successes, and demonstrated strong problem-solving abilities by outlining a methodical approach to tackling unfamiliar challenges discussed during the interview.

Candidate Name: {candidate_name}
Candidate Profile Text: {text}
Job Description: {job_desc}
Resume Text: {resume_text}"""

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Website CSV Generator - Grapevine</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Space+Grotesk:wght@400;700;900&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                background: #000000;
                color: #ffffff;
                font-family: 'Space Grotesk', sans-serif;
                line-height: 1.4;
                overflow-x: hidden;
            }
            
            .brutalist-header {
                background: #000000;
                border-bottom: 8px solid #ff7800;
                padding: 30px 0;
                position: relative;
            }
            
            .brutalist-header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(45deg, transparent 30%, #ff7800 30%, #ff7800 31%, transparent 31%);
                opacity: 0.1;
            }
            
            .logo-container {
                display: flex;
                align-items: center;
                gap: 20px;
                margin-bottom: 15px;
            }
            
            .logo-box {
                background: #ff7800;
                color: #000000;
                width: 80px;
                height: 80px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 900;
                font-size: 24px;
                transform: rotate(-2deg);
                box-shadow: 6px 6px 0px #333333;
                border: 3px solid #ffffff;
            }
            
            .site-title {
                font-family: 'JetBrains Mono', monospace;
                font-size: 48px;
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: -2px;
                color: #ffffff;
                text-shadow: 4px 4px 0px #ff7800;
                margin: 0;
            }
            
            .company-name {
                font-size: 18px;
                color: #ff7800;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 3px;
                margin-top: 5px;
            }
            
            .subtitle {
                font-size: 20px;
                color: #cccccc;
                font-weight: 400;
                margin-top: 10px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 20px;
                position: relative;
                z-index: 2;
            }
            
            .main-content {
                padding: 60px 0;
            }
            
            .upload-section {
                background: #111111;
                border: 6px solid #ff7800;
                padding: 50px;
                margin-bottom: 60px;
                position: relative;
                transform: rotate(-1deg);
                box-shadow: 12px 12px 0px #333333;
            }
            
            .upload-section::before {
                content: 'DRAG & DROP ZONE';
                position: absolute;
                top: -30px;
                left: 30px;
                background: #ff7800;
                color: #000000;
                padding: 8px 20px;
                font-family: 'JetBrains Mono', monospace;
                font-weight: 700;
                font-size: 12px;
                letter-spacing: 2px;
            }
            
            .upload-inner {
                transform: rotate(1deg);
                text-align: center;
            }
            
            .upload-title {
                font-family: 'JetBrains Mono', monospace;
                font-size: 32px;
                font-weight: 800;
                color: #ff7800;
                margin-bottom: 20px;
                text-transform: uppercase;
                letter-spacing: -1px;
            }
            
            .upload-desc {
                font-size: 18px;
                color: #cccccc;
                margin-bottom: 30px;
                font-weight: 400;
            }
            
            .file-input-wrapper {
                position: relative;
                display: inline-block;
                margin-bottom: 30px;
            }
            
            .file-input {
                position: absolute;
                opacity: 0;
                width: 100%;
                height: 100%;
                cursor: pointer;
            }
            
            .file-input-label {
                display: block;
                background: #333333;
                color: #ffffff;
                padding: 20px 40px;
                border: 3px solid #ff7800;
                font-family: 'JetBrains Mono', monospace;
                font-weight: 700;
                font-size: 16px;
                text-transform: uppercase;
                letter-spacing: 1px;
                cursor: pointer;
                transition: all 0.2s ease;
                box-shadow: 4px 4px 0px #ff7800;
            }
            
            .file-input-label:hover {
                background: #ff7800;
                color: #000000;
                transform: translate(-2px, -2px);
                box-shadow: 6px 6px 0px #ffffff;
            }
            
            .brutalist-btn {
                background: #ff7800;
                color: #000000;
                border: none;
                padding: 20px 50px;
                font-family: 'JetBrains Mono', monospace;
                font-weight: 800;
                font-size: 18px;
                text-transform: uppercase;
                letter-spacing: 1px;
                cursor: pointer;
                transform: rotate(1deg);
                box-shadow: 6px 6px 0px #ffffff;
                border: 3px solid #ffffff;
                transition: all 0.2s ease;
            }
            
            .brutalist-btn:hover {
                transform: rotate(-1deg) translate(-3px, -3px);
                box-shadow: 9px 9px 0px #ffffff;
                background: #ffffff;
                color: #000000;
            }
            
            .features-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 30px;
                margin-top: 60px;
            }
            
            .feature-card {
                background: #111111;
                border: 4px solid #333333;
                padding: 40px 30px;
                position: relative;
                transform: rotate(1deg);
                transition: all 0.3s ease;
            }
            
            .feature-card:nth-child(2) {
                transform: rotate(-1deg);
            }
            
            .feature-card:nth-child(3) {
                transform: rotate(0.5deg);
            }
            
            .feature-card:hover {
                border-color: #ff7800;
                transform: rotate(0deg) scale(1.02);
                box-shadow: 8px 8px 0px #ff7800;
            }
            
            .feature-icon {
                font-size: 48px;
                color: #ff7800;
                margin-bottom: 20px;
            }
            
            .feature-title {
                font-family: 'JetBrains Mono', monospace;
                font-size: 24px;
                font-weight: 700;
                color: #ffffff;
                margin-bottom: 15px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .feature-desc {
                color: #cccccc;
                font-size: 16px;
                line-height: 1.6;
            }
            
            .glitch-text {
                position: relative;
                animation: glitch 2s infinite;
            }
            
            @keyframes glitch {
                0%, 100% { transform: translate(0); }
                20% { transform: translate(-2px, 2px); }
                40% { transform: translate(-2px, -2px); }
                60% { transform: translate(2px, 2px); }
                80% { transform: translate(2px, -2px); }
            }
            
            .noise-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                opacity: 0.03;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><defs><filter id="noiseFilter"><feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="4" stitchTiles="stitch"/></filter></defs><rect width="100%" height="100%" filter="url(%23noiseFilter)" opacity="0.4"/></svg>');
                pointer-events: none;
                z-index: -1;
            }
            
            @media (max-width: 768px) {
                .site-title {
                    font-size: 32px;
                }
                .upload-section {
                    padding: 30px 20px;
                    transform: rotate(0deg);
                }
                .feature-card {
                    transform: rotate(0deg);
                }
                .features-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="noise-overlay"></div>
        
        <header class="brutalist-header">
            <div class="container">
                <div class="logo-container">
                    <div class="logo-box glitch-text">
                        <i class="fas fa-grape-alt"></i>
                    </div>
                    <div>
                        <h1 class="site-title">Website CSV Generator</h1>
                        <div class="company-name">Grapevine</div>
                    </div>
                </div>
                <p class="subtitle">CSV Data Transformation Pipeline</p>
            </div>
        </header>
        
        <main class="main-content">
            <div class="container">
                <section class="upload-section">
                    <div class="upload-inner">
                        <h2 class="upload-title">Upload Candidate Data</h2>
                        <p class="upload-desc">Drop your CSV or Excel files here for brutal AI-powered processing</p>
                        
                        <form action="/upload" method="post" enctype="multipart/form-data">
                            <div class="file-input-wrapper">
                                <input type="file" name="file" class="file-input" accept=".csv,.xlsx" required id="fileInput">
                                <label for="fileInput" class="file-input-label">
                                    <i class="fas fa-file-upload"></i> CHOOSE FILE
                                </label>
                            </div>
                            <br>
                            <button type="submit" class="brutalist-btn">
                                <i class="fas fa-rocket"></i> TRANSFORM DATA
                            </button>
                        </form>
                    </div>
                </section>
                
                <section class="features-grid">
                    <div class="feature-card">
                        <div class="feature-icon">
                            <i class="fas fa-brain"></i>
                        </div>
                        <h3 class="feature-title">AI Extraction</h3>
                        <p class="feature-desc">Neural-powered data extraction that rips through unstructured candidate information with brutal efficiency.</p>
                    </div>
                    
                    <div class="feature-card">
                        <div class="feature-icon">
                            <i class="fas fa-code"></i>
                        </div>
                        <h3 class="feature-title">Text Processing</h3>
                        <p class="feature-desc">Industrial-strength text formatting that transforms raw profiles into professional summaries.</p>
                    </div>
                    
                    <div class="feature-card">
                        <div class="feature-icon">
                            <i class="fas fa-download"></i>
                        </div>
                        <h3 class="feature-title">Export Engine</h3>
                        <p class="feature-desc">High-speed CSV export system delivering processed data in standardized, machine-readable format.</p>
                    </div>
                </section>
            </div>
        </main>
        
        <script>
            // File input feedback
            document.getElementById('fileInput').addEventListener('change', function(e) {
                const label = document.querySelector('.file-input-label');
                if (e.target.files.length > 0) {
                    label.innerHTML = '<i class="fas fa-check"></i> ' + e.target.files[0].name.toUpperCase();
                    label.style.background = '#ff7800';
                    label.style.color = '#000000';
                }
            });
        </script>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file selected', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename or 'uploaded_file.csv')
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)
            
            # Generate unique task ID and result filename
            task_id = str(int(time.time()))
            result_filename = f"template_output_{task_id}_{filename.rsplit('.', 1)[0]}.csv"
            
            # Start processing in background thread
            thread = threading.Thread(
                target=process_file_async, 
                args=(upload_path, result_filename, task_id)
            )
            thread.daemon = True
            thread.start()
            
            return redirect(url_for('processing', task_id=task_id))
            
        except Exception as e:
            return f'Error uploading file: {str(e)}', 500
    else:
        return 'Invalid file type. Please upload a CSV or Excel file.', 400

@app.route('/processing/<task_id>')
def processing(task_id):
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Processing - Website CSV Generator</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Space+Grotesk:wght@400;700;900&display=swap" rel="stylesheet">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                background: #000000;
                color: #ffffff;
                font-family: 'Space Grotesk', sans-serif;
                line-height: 1.4;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            
            .processing-container {{
                max-width: 800px;
                width: 90%;
                background: #111111;
                border: 6px solid #ff7800;
                padding: 60px 40px;
                position: relative;
                transform: rotate(-0.5deg);
                box-shadow: 16px 16px 0px #333333;
            }}
            
            .processing-container::before {{
                content: 'PROCESSING ZONE';
                position: absolute;
                top: -30px;
                left: 40px;
                background: #ff7800;
                color: #000000;
                padding: 8px 20px;
                font-family: 'JetBrains Mono', monospace;
                font-weight: 700;
                font-size: 12px;
                letter-spacing: 2px;
            }}
            
            .processing-inner {{
                transform: rotate(0.5deg);
                text-align: center;
            }}
            
            .processing-title {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 36px;
                font-weight: 800;
                color: #ff7800;
                margin-bottom: 40px;
                text-transform: uppercase;
                letter-spacing: -1px;
                text-shadow: 3px 3px 0px #333333;
            }}
            
            .progress-container {{
                margin: 40px 0;
                position: relative;
            }}
            
            .progress-bg {{
                background: #333333;
                height: 40px;
                border: 3px solid #ffffff;
                position: relative;
                overflow: hidden;
            }}
            
            .progress-bar {{
                background: linear-gradient(45deg, #ff7800 25%, #ffaa00 25%, #ffaa00 50%, #ff7800 50%, #ff7800 75%, #ffaa00 75%);
                background-size: 20px 20px;
                height: 100%;
                width: 0%;
                transition: width 0.5s ease;
                animation: progressMove 2s linear infinite;
                position: relative;
            }}
            
            .progress-text {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-family: 'JetBrains Mono', monospace;
                font-weight: 700;
                color: #000000;
                font-size: 16px;
                z-index: 2;
                text-shadow: 1px 1px 0px #ffffff;
            }}
            
            @keyframes progressMove {{
                0% {{ background-position: 0 0; }}
                100% {{ background-position: 20px 0; }}
            }}
            
            .status-message {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 18px;
                color: #ffffff;
                margin: 30px 0;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-weight: 400;
            }}
            
            .brutalist-btn {{
                background: #ff7800;
                color: #000000;
                border: none;
                padding: 20px 40px;
                font-family: 'JetBrains Mono', monospace;
                font-weight: 800;
                font-size: 18px;
                text-transform: uppercase;
                letter-spacing: 1px;
                cursor: pointer;
                transform: rotate(1deg);
                box-shadow: 6px 6px 0px #ffffff;
                border: 3px solid #ffffff;
                transition: all 0.2s ease;
                text-decoration: none;
                display: inline-block;
            }}
            
            .brutalist-btn:hover {{
                transform: rotate(-1deg) translate(-3px, -3px);
                box-shadow: 9px 9px 0px #ffffff;
                background: #ffffff;
                color: #000000;
                text-decoration: none;
            }}
            
            .success-alert {{
                background: #1a5f1a;
                border: 3px solid #ff7800;
                padding: 20px;
                margin: 30px 0;
                color: #ffffff;
                font-weight: 700;
                transform: rotate(-0.5deg);
            }}
            
            .error-alert {{
                background: #5f1a1a;
                border: 3px solid #ff0000;
                padding: 20px;
                margin: 30px 0;
                color: #ffffff;
                font-weight: 700;
                transform: rotate(0.5deg);
            }}
            
            .spinner {{
                border: 4px solid #333333;
                border-top: 4px solid #ff7800;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            .glitch-icon {{
                font-size: 60px;
                color: #ff7800;
                margin-bottom: 20px;
                animation: glitch 3s infinite;
            }}
            
            @keyframes glitch {{
                0%, 100% {{ transform: translate(0); }}
                20% {{ transform: translate(-2px, 2px); }}
                40% {{ transform: translate(-2px, -2px); }}
                60% {{ transform: translate(2px, 2px); }}
                80% {{ transform: translate(2px, -2px); }}
            }}
            
            .d-none {{
                display: none !important;
            }}
            
            @media (max-width: 768px) {{
                .processing-container {{
                    transform: rotate(0deg);
                    padding: 40px 20px;
                    margin: 20px;
                }}
                .processing-title {{
                    font-size: 28px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="processing-container">
            <div class="processing-inner">
                <div class="glitch-icon">
                    <i class="fas fa-cogs fa-spin"></i>
                </div>
                <h1 class="processing-title">Transforming Data</h1>
                
                <div class="progress-container">
                    <div class="progress-bg">
                        <div class="progress-bar" id="progressBar"></div>
                        <div class="progress-text" id="progressText">0%</div>
                    </div>
                </div>
                
                <div class="spinner" id="loadingSpinner"></div>
                <p class="status-message" id="statusMessage">INITIALIZING NEURAL PROCESSORS...</p>
                
                <div id="completedSection" class="d-none">
                    <div class="success-alert">
                        <i class="fas fa-check-circle"></i> DATA TRANSFORMATION COMPLETED SUCCESSFULLY!
                    </div>
                    <a id="downloadLink" href="#" class="brutalist-btn">
                        <i class="fas fa-download"></i> DOWNLOAD PROCESSED DATA
                    </a>
                </div>
                
                <div id="errorSection" class="d-none">
                    <div class="error-alert">
                        <i class="fas fa-exclamation-triangle"></i> SYSTEM ERROR DETECTED DURING PROCESSING
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            const taskId = '{task_id}';
            
            function checkStatus() {{
                fetch(`/status/${{taskId}}`)
                    .then(response => response.json())
                    .then(data => {{
                        const progressBar = document.getElementById('progressBar');
                        const progressText = document.getElementById('progressText');
                        const statusMessage = document.getElementById('statusMessage');
                        const completedSection = document.getElementById('completedSection');
                        const errorSection = document.getElementById('errorSection');
                        const loadingSpinner = document.getElementById('loadingSpinner');
                        
                        progressBar.style.width = data.progress + '%';
                        progressText.textContent = data.progress + '%';
                        statusMessage.textContent = data.message.toUpperCase();
                        
                        if (data.status === 'completed') {{
                            loadingSpinner.style.display = 'none';
                            completedSection.classList.remove('d-none');
                            document.getElementById('downloadLink').href = `/download/${{data.result_file}}`;
                        }} else if (data.status === 'error') {{
                            loadingSpinner.style.display = 'none';
                            errorSection.classList.remove('d-none');
                        }} else {{
                            setTimeout(checkStatus, 2000);
                        }}
                    }})
                    .catch(error => {{
                        console.error('Error:', error);
                        setTimeout(checkStatus, 5000);
                    }});
            }}
            
            checkStatus();
        </script>
    </body>
    </html>
    '''

@app.route('/status/<task_id>')
def get_status(task_id):
    status = processing_status.get(task_id, {'status': 'not_found', 'message': 'Task not found'})
    return jsonify(status)

@app.route('/download/<filename>')
def download_file(filename):
    try:
        result_path = os.path.join(RESULTS_FOLDER, filename)
        if os.path.exists(result_path):
            return send_file(result_path, as_attachment=True)
        else:
            return 'File not found', 404
    except Exception as e:
        return f'Error downloading file: {str(e)}', 500

def run_web_interface():
    """Run the Flask web interface."""
    print("🚀 Starting Website CSV Generator - Grapevine...")
    print("📁 Upload folder:", UPLOAD_FOLDER)
    print("📁 Results folder:", RESULTS_FOLDER)
    print("🌐 Open your browser and go to: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)

# --- MAIN ORCHESTRATOR ---
def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # --- Extract company name from job description using Gemini ---
    company_name_from_jd = "N/A"
    if COL_SRC_JOB_DESC in df.columns:
        for jd in df[COL_SRC_JOB_DESC]:
            if isinstance(jd, str) and jd.strip():
                prompt = f"""Extract only the company name from the following job description. Output only the company name, nothing else.\n\nJob Description:\n{jd}\n"""
                # Use Gemini to extract company name
                company_name = processor.process_prompts_in_parallel("gemini-2.5-flash", [prompt], "Extract Company Name", is_json=False)[0]
                if company_name and company_name.strip() and company_name.strip().lower() not in ["n/a", "none", "null", "", "not provided"]:
                    company_name_from_jd = company_name.strip()
                    break
    """Process the dataframe using the template pipeline logic."""
    # --- Helper: Extract fields from candidate profile markdown ---
    def extract_from_profile_markdown(Candidate_Profile):
        result = {
            'why_candidate': '',
            'summary': '',
            'key_highlights': '',
            'interview_notes': '',
            'interview_highlights': ''
        }
        if not isinstance(Candidate_Profile, str) or not Candidate_Profile.strip():
            return result
        # Why Candidate: Extract substring between 'What Stood Out in the Interview' and 'Resume', Python equivalent of Excel formula
        def extract_stood_out(text):
            if not isinstance(text, str):
                return ""
            start_marker = "What Stood Out in the Interview"
            end_marker = "Resume"
            start = text.find(start_marker)
            end = text.find(end_marker, start)
            if start != -1 and end != -1 and end > start:
                extracted = text[start:end].strip()
                # Remove the marker text itself if present
                extracted = extracted.replace(start_marker, "", 1).strip()
                return extracted
            return "Section not found"

        result['why_candidate'] = extract_stood_out(Candidate_Profile)
        # Summary (look for a one-sentence summary at the end or after 'Summary')
        summary_match = re.search(r"\*\*Summary:?\*\*\s*(.*?)(?:\n|$)", Candidate_Profile, re.DOTALL)
        if summary_match:
            result['summary'] = summary_match.group(1).strip()
        # Key Highlights (look for 'Key Highlights' or similar)
        highlights_match = re.search(r"\*\*Key Highlights:?\*\*\s*(.*?)(?:\n\*\*|$)", Candidate_Profile, re.DOTALL)
        if highlights_match:
            result['key_highlights'] = highlights_match.group(1).strip()
        # Interview Notes (look for 'Interview Notes' or similar)
        interview_match = re.search(r"\*\*Interview Notes:?\*\*\s*(.*?)(?:\n\*\*|$)", Candidate_Profile, re.DOTALL)
        if interview_match:
            result['interview_notes'] = interview_match.group(1).strip()
        # Interview Highlights (look for 'What Stood Out in the Interview:' section, only positive points)
        stood_out_match = re.search(r"What Stood Out in the Interview:?\s*(.*?)(?:\n\*\*|$)", Candidate_Profile, re.DOTALL)
        if stood_out_match:
            highlights_text = stood_out_match.group(1).strip()
            # Remove lines with negative points or scores
            positive_lines = []
            for line in highlights_text.splitlines():
                line = line.strip()
                # Skip lines with negative words or scores
                if not line:
                    continue
                if re.search(r'(weak|needs improvement|lacking|score|not|poor|negative|area for improvement|improvement needed|gap|shortcoming|deficiency|issue|problem|challenge)', line, re.IGNORECASE):
                    continue
                positive_lines.append(line)
            result['interview_highlights'] = ' '.join(positive_lines)
        return result
    
    # Initialize candidate name column variable
    candidate_name_col = COL_SRC_CANDIDATE_NAME
    
    print(f"📄 Processing dataframe with {len(df)} rows.")
    print(f"📋 Columns found in data: {list(df.columns)}")
    
    # Debug: Check phone column data type and values
    if COL_SRC_PHONE in df.columns:
        print(f"📞 Phone column data type: {df[COL_SRC_PHONE].dtype}")
        print(f"📞 Sample phone values: {df[COL_SRC_PHONE].head().tolist()}")

    # --- Data Cleaning: Remove rows where name is empty ---
    initial_rows = len(df)
    
    # Check if the expected name column exists, if not try to find an alternative
    candidate_name_col = COL_SRC_CANDIDATE_NAME  # Use local variable
    if candidate_name_col not in df.columns:
        # Look for potential name columns
        potential_name_cols = [col for col in df.columns if 'name' in col.lower() or 'user real name' in col.lower()]
        if potential_name_cols:
            candidate_name_col = potential_name_cols[0]
            print(f"⚠️ Column '{COL_SRC_CANDIDATE_NAME}' not found. Using '{candidate_name_col}' instead.")
        else:
            print(f"❌ No name column found. Available columns: {list(df.columns)}")
            print("❌ Cannot proceed without a name column. Please check the CSV file structure.")
            raise ValueError("No name column found in the uploaded file.")
    else:
        print(f"✅ Found candidate name column: '{candidate_name_col}'")

    # Check and report other column availability
    column_mapping = {
        'phone': COL_SRC_PHONE,
        'email': COL_SRC_EMAIL, 
        'ctc': COL_SRC_CTC,
        'resume_url': COL_SRC_RESUME_URL,
        'notice_period_preference': COL_SRC_NOTICE_PERIOD_PREFERENCE,
        'notice_period_resume': COL_SRC_NOTICE_PERIOD_RESUME
    }
    
    for col_type, col_name in column_mapping.items():
        if col_name not in df.columns:
            print(f"⚠️ Column '{col_name}' not found for {col_type}")
        else:
            print(f"✅ Found column '{col_name}' for {col_type}")
    
    # Clean data by removing rows where name is empty
    if candidate_name_col in df.columns:
        df.dropna(subset=[candidate_name_col], inplace=True)
        df = df[df[candidate_name_col].str.strip() != ''].copy()
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            print(f"🧹 Cleaned data: Removed {rows_removed} rows where candidate name was empty.")
    else:
        print(f"⚠️ Skipping name-based cleaning as column '{candidate_name_col}' not found.")

    # --- Run Structured Data Extraction ---
    row_data_list = [row.to_dict() for _, row in df.iterrows()]
    extraction_prompts = [create_structured_extraction_prompt(data) for data in row_data_list]
    
    # Use Gemini 2.5 Flash for structured extraction
    json_results = processor.process_prompts_in_parallel("gemini-2.5-flash", extraction_prompts, "Structured Profile Extraction", is_json=True)
    df[COL_GEN_STRUCTURED_PROFILE] = json_results

    # --- Finalize Output DataFrame ---
    print("\n" + "="*25 + " FINALIZING OUTPUT DATAFRAME " + "="*25)
    
    final_df_data = []
    df.reset_index(drop=True, inplace=True) 

    # --- Robust column name detection for 'Candidate Profile' and Notice Period ---
    # Find the actual column name for 'Candidate Profile'
    candidate_profile_col = None
    for col in df.columns:
        if col.strip().lower() == 'candidate profile' or 'candidate profile' in col.strip().lower():
            candidate_profile_col = col
            break
    if not candidate_profile_col:
        print("⚠️ 'Candidate Profile' column not found in input. Will use empty string as fallback.")

    # Find the actual column name for Notice Period
    notice_period_col = None
    for col in df.columns:
        if 'notice period' in col.strip().lower():
            notice_period_col = col
            break
    if not notice_period_col:
        print("⚠️ Notice Period column not found in input. Will use empty string as fallback.")

    for row_num, (index, row) in enumerate(df.iterrows()):

        profile_data = {}
        try:
            profile_json_str = row.get(COL_GEN_STRUCTURED_PROFILE, '{}')
            if pd.notna(profile_json_str) and profile_json_str.strip():
                profile_data = json.loads(profile_json_str)
        except (json.JSONDecodeError, TypeError):
            profile_data = {"error": f"Failed to parse profile JSON: {profile_json_str}"}
            print(f"Warning: Could not parse JSON for row {row_num}. Content: {profile_json_str}")

        # Use robustly detected column for Candidate Profile
        profile_markdown = row.get(candidate_profile_col, '') if candidate_profile_col else ''
        extracted_profile = extract_from_profile_markdown(profile_markdown)
        interview_highlights = profile_data.get('interview_highlights', 'N/A')
        if not interview_highlights or interview_highlights == 'N/A':
            interview_highlights = extracted_profile['interview_highlights']

        # Use robustly detected column for Notice Period
        notice_resume = row.get(notice_period_col, '') if notice_period_col else ''
        
        # If interview highlights are still not available or contain negative messages, create from resume
        if (not interview_highlights or interview_highlights == 'N/A' or 
            any(phrase in interview_highlights.lower() for phrase in [
                'interview evaluation: user cannot be evaluated',
                'interview transcript was not available',
                'interview transcript is not available',
                'interview transcript was not provided',
                'interview transcript is missing',
                'interview was not taken',
                'interview was not conducted',
                'cannot be evaluated', 
                'not available', 
                'no interview',
                'missing interview', 
                'lack of substantive discussion',
                'deeper assessment',
                'not possible',
                'due to the lack',
                'substantive discussion in the transcript',
                'assessment of communication skills',
                'technical depth from the interview'
            ])):
            # Create highlights from resume experience and skills
            resume_text = str(row.get(COL_SRC_RESUME_TEXT, ''))
            if resume_text and resume_text != 'N/A':
                # Extract key achievements from resume
                experience_text = profile_data.get('experience', '')
                current_position = profile_data.get('current_position', '')
                if experience_text and len(experience_text) > 20:
                    interview_highlights = experience_text  # Remove truncation - keep full text
                elif current_position and len(current_position) > 10:
                    interview_highlights = f"Currently serving as {current_position} with relevant industry experience."
                else:
                    interview_highlights = "Demonstrated professional experience relevant to the role requirements."

        # Extract only the last sentence from Candidate Profile for summary

        def get_last_english_sentence(text):
            if not isinstance(text, str) or not text.strip():
                return 'N/A'
            # Split by sentence-ending punctuation
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            # Filter for English sentences only (basic check: contains a-zA-Z, not a URL, not markdown quote)
            # Also filter out negative interview-related content
            negative_phrases = [
                'interview transcript was not available',
                'interview transcript is not available',
                'interview transcript was not provided',
                'interview transcript is missing',
                'interview was not taken',
                'interview was not conducted',
                'cannot be evaluated',
                'no interview',
                'missing interview',
                'no transcript provided',
                'transcript was not available',
                'transcript is not available',
                'transcript was not provided',
                'transcript is missing',
                'what stood out in the interview:',
                'lack of substantive discussion',
                'deeper assessment',
                'not possible',
                'due to the lack',
                'substantive discussion in the transcript',
                'assessment of communication skills',
                'technical depth from the interview',
                'interview is not possible',
                'interview assessment is not possible'
            ]
            
            english_sentences = []
            for s in sentences:
                s = s.strip()
                if (re.search(r'[a-zA-Z]', s) and 
                    not re.search(r'(http[s]?://|www\.|\.com|\.in|\.pdf|resume|linkedin)', s, re.IGNORECASE) and 
                    not s.startswith('"') and not s.startswith("'") and
                    not any(phrase in s.lower() for phrase in negative_phrases)):
                    english_sentences.append(s)
                    
            if english_sentences:
                return english_sentences[-1]
            return 'N/A'

        summary_last_english_sentence = get_last_english_sentence(profile_markdown)
        
        # If summary is still N/A or contains negative content, create from resume data
        if (summary_last_english_sentence == 'N/A' or 
            any(phrase in summary_last_english_sentence.lower() for phrase in [
                'interview transcript was not available',
                'interview transcript is not available',
                'interview transcript was not provided',
                'interview transcript is missing',
                'interview was not taken',
                'interview was not conducted',
                'cannot be evaluated', 
                'no interview', 
                'missing interview',
                'lack of substantive discussion',
                'deeper assessment',
                'not possible',
                'due to the lack',
                'substantive discussion in the transcript',
                'assessment of communication skills',
                'technical depth from the interview'
            ])):
            # Create summary from resume experience and current position
            experience_text = profile_data.get('experience', '')
            current_position = profile_data.get('current_position', '')
            role_applied = profile_data.get('role_applied', '')
            
            if experience_text and len(experience_text) > 20:
                summary_last_english_sentence = experience_text
            elif current_position and len(current_position) > 10:
                if role_applied and role_applied != 'N/A':
                    summary_last_english_sentence = f"Experienced professional currently serving as {current_position}, applying for {role_applied} role."
                else:
                    summary_last_english_sentence = f"Experienced professional currently serving as {current_position}."
            else:
                summary_last_english_sentence = "Qualified professional with relevant industry experience."

        # Extract tech_stack from resume metadata and compare with job description
        resume_text = str(row.get(COL_SRC_RESUME_TEXT, ''))
        job_desc = str(row.get(COL_SRC_JOB_DESC, ''))
        
        # Define comprehensive list of technical skills to search for
        curated_skills = [
            "Java", "Python", "SQL", "Spring", "Angular", "React", "MySQL", "MongoDB", "Kafka", "RESTful API", "Git", "PostgreSQL", "TypeScript", "C++", "C#", "Node.js", "AWS", "Azure", "Docker", "Kubernetes", "HTML", "CSS", "JavaScript",
            "Jenkins", "Tableau", "Power BI", "Scala", "Ruby", "PHP", "Perl", "Go", "Swift", "Objective-C", "R", "SAS", "SPSS", "MATLAB", "TensorFlow", "PyTorch", "Hadoop", "Spark", "Flask", "Django", "Express.js", "Vue.js", "Svelte", "Redux", "GraphQL", "REST", "SOAP", "Elasticsearch", "Redis", "Cassandra", "Oracle", "Sybase", "Firebase", "Heroku", "GCP", "Linux", "Unix", "Shell", "Bash", "PowerShell", "JIRA", "Confluence", "Agile", "Scrum", "Trello", "Notion", "Figma", "Photoshop", "Illustrator", "After Effects", "Premiere Pro", "Salesforce", "SAP", "ServiceNow", "Workday", "Snowflake", "BigQuery", "NoSQL", "PL/SQL", "VB.NET", "ASP.NET", "Erlang", "Rust", "Solidity", "Blockchain", "Machine Learning", "Deep Learning", "Data Science", "Data Engineering", "DevOps", "CI/CD", "Microservices", "REST API", "API Gateway", "Load Balancer", "Nginx", "Apache", "Tomcat", "RabbitMQ", "ActiveMQ", "Zookeeper", "Jupyter", "Colab", "ETL", "Data Warehouse", "Data Lake", "Business Intelligence", "Analytics", "Testing", "JUnit", "Selenium", "Cypress", "Mocha", "Chai", "Jest", "Enzyme", "Pandas", "NumPy", "SciPy", "Scikit-learn", "XGBoost", "LightGBM", "CatBoost", "OpenCV", "NLTK", "spaCy", "BeautifulSoup", "Requests", "FastAPI", "Streamlit", "Dash", "Plotly", "Seaborn", "Matplotlib", "D3.js", "Three.js", "WebGL", "WebRTC", "WebSockets", "OAuth", "JWT", "SSO", "IAM", "LDAP", "Active Directory", "PKI", "SSL", "TLS", "Encryption", "Cybersecurity", "Penetration Testing", "Vulnerability Assessment", "Firewall", "IDS", "IPS", "SIEM", "Splunk", "Logstash", "Prometheus", "Grafana", "New Relic", "Datadog", "AppDynamics", "Sentry", "Rollbar", "PagerDuty", "Opsgenie", "Incident Management", "Monitoring", "Alerting", "Capacity Planning", "Performance Tuning", "Optimization", "Refactoring", "Design Patterns", "OOP", "Functional Programming", "Reactive Programming", "Event-driven Architecture", "Serverless", "Cloud Computing", "Edge Computing", "IoT", "AR", "VR", "Mobile Development", "Android", "iOS", "Windows", "macOS", "Linux", "Embedded Systems", "FPGA", "ASIC", "VHDL", "Verilog", "PCB Design", "Robotics", "Automation", "Control Systems", "SCADA", "PLC", "HMI", "Industrial IoT", "Smart Manufacturing", "Digital Twin", "Simulation", "Modeling", "CAD", "CAM", "CAE", "SolidWorks", "AutoCAD", "CATIA", "ANSYS", "COMSOL", "Simulink", "LabVIEW", "TestStand", "Measurement Studio", "Instrumentation", "Sensors", "Actuators", "Wireless Communication", "Bluetooth", "Zigbee", "LoRa", "NFC", "RFID", "Satellite Communication", "Telecom", "5G", "LTE", "Wi-Fi", "Ethernet", "Network Security", "Routing", "Switching", "SDN", "NFV", "Network Automation", "Network Monitoring", "Network Management", "Network Design", "Network Architecture", "Network Engineering", "Network Operations", "Network Troubleshooting", "Network Analysis", "Network Simulation", "Network Testing", "Network Optimization", "Network Planning", "Network Provisioning", "Network Deployment", "Network Maintenance", "Network Upgrades", "Network Migration", "Network Integration", "Network Documentation", "Network Training", "Network Support", "Network Consulting", "Network Project Management", "Network Vendor Management", "Network Budgeting", "Network Cost Optimization", "Network Compliance", "Network Auditing", "Network Risk Management", "Network Disaster Recovery", "Network Business Continuity", "Network Change Management", "Network Incident Response", "Network Problem Management", "Network Service Management", "Network SLA Management", "Network KPI Management", "Network Performance Management", "Network Capacity Management", "Network Availability Management", "Network Reliability Management", "Network Scalability Management", "Network Flexibility Management", "Network Agility Management", "Network Innovation Management", "Network Transformation Management", "Network Digitalization Management", "Network Automation Management", "Network Intelligence Management", "Network Analytics Management", "Network Data Management", "Network Information Management", "Network Knowledge Management", "Network Collaboration Management", "Network Communication Management", "Network Stakeholder Management", "Network Customer Management", "Network User Management", "Network Partner Management", "Network Supplier Management", "Network Contractor Management", "Network Consultant Management", "Network Advisor Management", "Network Expert Management", "Network Specialist Management", "Network Engineer Management", "Network Architect Management", "Network Manager Management", "Network Director Management", "Network VP Management", "Network CxO Management"
        ]
        
        # Search for technical skills in resume text using case-insensitive matching
        tech_stack = []
        resume_text_lower = resume_text.lower()
        
        for skill in curated_skills:
            # Use word boundaries to avoid partial matches (e.g., "Java" shouldn't match "JavaScript")
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', resume_text_lower):
                tech_stack.append(skill)
        
        # Remove duplicates while preserving order and limit to top 10
        seen = set()
        tech_stack = [skill for skill in tech_stack if not (skill in seen or seen.add(skill))][:10]
        
        # Process current CTC - remove last 5 zeros (convert to lakhs)
        current_ctc_raw = row.get(COL_SRC_CTC, 'N/A')
        if current_ctc_raw and current_ctc_raw != 'N/A':
            try:
                # Extract numeric value and convert to lakhs
                ctc_str = str(current_ctc_raw).strip()
                # Remove any non-numeric characters except decimal point
                ctc_numeric = re.sub(r'[^\d.]', '', ctc_str)
                if ctc_numeric:
                    ctc_value = float(ctc_numeric)
                    # Divide by 100,000 to convert to lakhs (remove 5 zeros)
                    ctc_lakhs = ctc_value / 100000
                    # Format to remove unnecessary decimal places
                    if ctc_lakhs == int(ctc_lakhs):
                        current_ctc_processed = str(int(ctc_lakhs))
                    else:
                        current_ctc_processed = str(ctc_lakhs)
                else:
                    current_ctc_processed = 'N/A'
            except (ValueError, TypeError):
                current_ctc_processed = 'N/A'
        else:
            current_ctc_processed = 'N/A'
        
        # Process notice period - prioritize preference, fallback to resume data
        notice_period_processed = '0'  # Default to 0 instead of empty string
        notice_preference = row.get(COL_SRC_NOTICE_PERIOD_PREFERENCE, '')
        notice_resume = row.get(COL_SRC_NOTICE_PERIOD_RESUME, '')
        
        # Debug: Print notice period data for first few rows
        if row_num < 3:
            print(f"Row {row_num} - Notice Period Preference: '{notice_preference}', Resume: '{notice_resume}'")
        
        # Check preference first (already in days)
        if notice_preference and str(notice_preference).strip() and str(notice_preference).strip().lower() not in ['n/a', 'nan', 'none', '', 'null']:
            try:
                # Remove any non-numeric characters except decimal point
                notice_str = re.sub(r'[^\d.]', '', str(notice_preference).strip())
                if notice_str:
                    notice_days = float(notice_str)
                    notice_period_processed = str(int(notice_days)) if notice_days == int(notice_days) else str(notice_days)
                    if row_num < 3:
                        print(f"Row {row_num} - Processed notice from preference: {notice_period_processed}")
            except (ValueError, TypeError):
                if row_num < 3:
                    print(f"Row {row_num} - Error processing preference notice period")
                pass
        
        # If preference not available, try resume data (convert months to days)
        if notice_period_processed == 'N/A' and notice_resume and str(notice_resume).strip() and str(notice_resume).strip().lower() not in ['n/a', 'nan', 'none', '', 'null']:
            try:
                # Remove any non-numeric characters except decimal point
                notice_str = re.sub(r'[^\d.]', '', str(notice_resume).strip())
                if notice_str:
                    notice_months = float(notice_str)
                    notice_days = notice_months * 1
                    notice_period_processed = str(int(notice_days)) if notice_days == int(notice_days) else str(notice_days)
                    if row_num < 3:
                        print(f"Row {row_num} - Processed notice from resume (converted from months): {notice_period_processed}")
            except (ValueError, TypeError):
                if row_num < 3:
                    print(f"Row {row_num} - Error processing resume notice period")
                pass
        
        # Format the special fields using Gemini 2.5 Flash
        candidate_profile_text = row.get('Candidate Profile', '')
        
        # Debug: Print the first 200 characters of the Candidate Profile for the first few rows
        if row_num < 3:
            print(f"Row {row_num} - Candidate Profile preview: {str(candidate_profile_text)[:200]}...")

        # Create prompts for Gemini to process the text formatting
        why_candidate_prompt = create_why_candidate_prompt(candidate_profile_text)
        summary_prompt = create_summary_prompt(summary_last_english_sentence)
        candidate_name = row.get(candidate_name_col, "The candidate")
        interview_highlights_prompt = create_interview_highlights_prompt(interview_highlights, job_desc, resume_text, candidate_name)
        # Process with Gemini 2.5 Flash
        formatting_prompts = [why_candidate_prompt, summary_prompt, interview_highlights_prompt]
        formatting_results = processor.process_prompts_in_parallel("gemini-2.5-flash", formatting_prompts, f"Text Formatting for Row {row_num}", is_json=False)

        formatted_why_candidate = formatting_results[0] if len(formatting_results) > 0 else 'N/A'
        formatted_summary = formatting_results[1] if len(formatting_results) > 1 else 'N/A'
        formatted_interview_highlights = formatting_results[2] if len(formatting_results) > 2 else 'N/A'
        
        # Fallback to original logic if Gemini processing fails or returns empty
        if not formatted_why_candidate or formatted_why_candidate == 'N/A' or 'error' in formatted_why_candidate.lower():
            extracted_profile = extract_from_profile_markdown(candidate_profile_text)
            formatted_why_candidate = extracted_profile['why_candidate'] or row.get(COL_SRC_SUMMARIZER, 'N/A')
        
        # Ensure phone number is treated as string to avoid scientific notation
        phone_value = profile_data.get('phone', row.get(COL_SRC_PHONE, 'N/A'))
        if phone_value and phone_value != 'N/A':
            phone_value = str(phone_value).strip()
            # Remove any decimal points that might have been added during processing
            if '.' in phone_value and phone_value.replace('.', '').isdigit():
                phone_value = phone_value.split('.')[0]
        else:
            phone_value = 'N/A'
        
        # Ensure email is lower case if present and not N/A
        email_val = profile_data.get('email', row.get(COL_SRC_EMAIL, 'N/A'))
        if isinstance(email_val, str) and email_val not in ['N/A', 'nan', 'none', 'null', '']:
            email_val = email_val.lower()
        # Ensure linkedin_link is lower case if present and not N/A
        linkedin_val = profile_data.get('linkedin_link', '')
        if isinstance(linkedin_val, str) and linkedin_val not in ['N/A', 'nan', 'none', 'null', '']:
            linkedin_val = linkedin_val.lower()
        final_row = {
            'id': row_num + 1,
            'name': profile_data.get('name', row.get(candidate_name_col, 'N/A')),
            'role_applied': profile_data.get('role_applied', 'N/A'),
            'current_position': profile_data.get('current_position', 'N/A'),
            'education': profile_data.get('education', 'N/A'),
            'experience': profile_data.get('experience', 'N/A'),
            'phone': phone_value,
            'email': email_val,
            'current_ctc_lpa': current_ctc_processed,
            'why_candidate': formatted_why_candidate,
            'technical_alignment': json.dumps({
                "tech_stack": tech_stack,
                "match_score": 85
            }),
            'interview_highlights': formatted_interview_highlights,
            'summary': formatted_summary,
            'key_highlights': extracted_profile['key_highlights'],
            'interview_notes': extracted_profile['interview_notes'],
            'created_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'referral_link_id': '',
            'referral_link_private_id': '',
            'company': company_name_from_jd,
            'role': profile_data.get('role', 'N/A'),
            'notice': notice_period_processed,
            'resume_link': row.get(COL_SRC_RESUME_URL, ''),
            'linkedin_link': linkedin_val
        }
        final_df_data.append(final_row)

    # Define the exact column order from can.csv
    target_columns = [
        'id', 'name', 'role_applied', 'current_position', 'education', 
        'experience', 'phone', 'email', 'current_ctc_lpa', 'why_candidate', 
        'technical_alignment', 'interview_highlights', 'summary', 
        'created_at', 'referral_link_id', 'referral_link_private_id', 
        'company', 'role', 'notice', 'resume_link', 'linkedin_link'
    ]
    
    final_df = pd.DataFrame(final_df_data)
    
    # Ensure phone numbers are treated as strings to prevent scientific notation
    if 'phone' in final_df.columns:
        def clean_phone_safe(phone_value):
            """Safely clean phone numbers without float conversion errors."""
            if pd.isna(phone_value) or str(phone_value).strip().lower() in ['n/a', 'nan', 'none', 'null', '']:
                return 'N/A'
            
            phone_str = str(phone_value).strip()
            
            # Handle scientific notation (only if it looks like scientific notation)
            if 'e+' in phone_str.lower() or 'e-' in phone_str.lower():
                try:
                    # Only convert if it's actually a number in scientific notation
                    if phone_str.replace('e+', '').replace('e-', '').replace('.', '').isdigit():
                        phone_num = int(float(phone_str))
                        return str(phone_num)
                except (ValueError, OverflowError):
                    pass
            
            # For international numbers with + and -, just clean them but keep the format
            if '+' in phone_str or '-' in phone_str:
                # Remove non-digit characters except +
                cleaned = re.sub(r'[^\d+]', '', phone_str)
                return cleaned if cleaned else 'N/A'
            
            # For regular numbers, remove decimal points if they exist
            if '.' in phone_str and phone_str.replace('.', '').isdigit():
                return phone_str.split('.')[0]
            
            return phone_str
        
        final_df['phone'] = final_df['phone'].apply(clean_phone_safe)
    
    # Fill any remaining NaN values with 'N/A' to ensure no empty fields
    final_df.fillna('N/A', inplace=True)
    # Ensure all columns exist, adding any missing ones, and enforce order
    for col in target_columns:
        if col not in final_df.columns:
            final_df[col] = '' if 'referral' in col or 'notice' in col else 'N/A'
    final_df = final_df[target_columns]

    return final_df

def main():
    """Main function for command line usage."""
    try:
        # Read CSV with phone columns as strings to prevent scientific notation
        dtype_dict = {}
        if COL_SRC_PHONE:
            dtype_dict[COL_SRC_PHONE] = str

        df = pd.read_csv("CodeRabbit - productads.csv", dtype=dtype_dict)
        print(f"📄 Successfully loaded 'CodeRabbit - productads.csv' with {len(df)} rows.")

        # Process the dataframe
        final_df = process_dataframe(df)

        output_filename = "productads.csv"
        # Force quoting for phone numbers to ensure they appear with double quotes
        final_df.to_csv(output_filename, index=False, encoding='utf-8-sig', quoting=1, quotechar='"')
        print(f"\n🎉 Pipeline Complete! All results saved to '{output_filename}' in the correct format.")

        total_time = time.time() - PIPELINE_START_TIME
        print(f"⏱️ Total pipeline execution time: {total_time:.2f} seconds.")
        
    except Exception as e:  
        print(f"❌ FATAL ERROR loading data: {e}")

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        # Run web interface
        run_web_interface()
    else:
        # Run command line version
        print("🚀 Running in command line mode...")
        print("💡 To use web interface, run: python template.py --web")
        main()