import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import time
import re
from io import BytesIO
import base64
import textwrap
import os
import json
import pickle
from pathlib import Path
import zipfile
from datetime import datetime

# App title and configuration
st.set_page_config(
    page_title="LLM for Medical Notes Simplification Tutorial",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #0e4c92;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #0e4c92;
        margin: 1rem 0;
    }
    .disclaimer {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #ffa000;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .success-box {
        background-color: #e6f4ea;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #34a853;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fce8e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #ea4335;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #0e4c92;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #0b3b73;
    }
    .styled-table {
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 0.9em;
        font-family: sans-serif;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        width: 100%;
    }
    .styled-table thead tr {
        background-color: #0e4c92;
        color: #ffffff;
        text-align: left;
    }
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #0e4c92;
    }
    .code-box {
        background-color: #f6f8fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
        white-space: pre-wrap;
    }
    .two-column {
        display: flex;
        gap: 20px;
    }
    .column {
        flex: 1;
    }
    .method-box {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
    }
    .method-title {
        font-weight: bold;
        color: #0e4c92;
        margin-bottom: 10px;
    }
    .comparison-header {
        text-align: center;
        padding: 10px;
        background-color: #e8f0fe;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .tab-content {
        padding: 20px 0;
    }
    .prompt-title {
        font-weight: bold;
        color: #333;
        margin-bottom: 5px;
    }
    .prompt-text {
        background-color: #f6f8fa;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #0e4c92;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        margin-bottom: 20px;
    }
    .metric-box {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
    }
    .metric-title {
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        color: #0e4c92;
    }
    .metric-description {
        text-align: center;
        font-size: 12px;
        color: #666;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False

if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []
    
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "tutorial"
    
if 'cache_dir' not in st.session_state:
    # Create a cache directory if it doesn't exist
    cache_dir = Path("streamlit_cache")
    cache_dir.mkdir(exist_ok=True)
    st.session_state.cache_dir = cache_dir

# Sidebar for API key and navigation
with st.sidebar:
    st.image("https://www.creativefabrica.com/wp-content/uploads/2021/03/20/Medical-Logo-Graphics-9786532-1-580x386.jpg", width=100)
    st.title("LLM for Healthcare")
    
    # Navigation
    st.subheader("Navigation")
    nav_selection = st.radio(
        "Go to:",
        ["📚 Tutorial", "🔬 Live Demo", "📊 Results Explorer", "📝 Code Examples", "ℹ️ About"]
    )
    
    if nav_selection == "📚 Tutorial":
        st.session_state.current_tab = "tutorial"
    elif nav_selection == "🔬 Live Demo":
        st.session_state.current_tab = "demo"
    elif nav_selection == "📊 Results Explorer":
        st.session_state.current_tab = "results"
    elif nav_selection == "📝 Code Examples":
        st.session_state.current_tab = "code"
    elif nav_selection == "ℹ️ About":
        st.session_state.current_tab = "about"
    
    # API Key Configuration
    st.markdown("---")
    st.subheader("API Configuration")
    
    # Using Streamlit secrets if available, otherwise ask for API key
    if "openai" in st.secrets:
        openai.api_key = st.secrets["openai"]["api_key"]
        st.success("API Key configured from Streamlit secrets!")
        st.session_state.api_key_configured = True
    else:
        api_key = st.text_input("Enter your OpenAI API Key", type="password")
        if api_key:
            openai.api_key = api_key
            st.session_state.api_key_configured = True
            st.success("API Key configured!")
        else:
            st.session_state.api_key_configured = False
            st.warning("Please enter your API key to use the demo features.")
    
    # Settings (only shown when relevant)
    if st.session_state.current_tab == "demo" and st.session_state.api_key_configured:
        st.markdown("---")
        st.subheader("Demo Settings")
        
        model_choice = st.selectbox(
            "Choose LLM Model",
            options=["gpt-3.5-turbo", "gpt-4-turbo"],
            index=0
        )
        
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3, 
            step=0.1, 
            help="Lower values make output more deterministic, higher values make it more random"
        )
        
        target_group = st.radio(
            "Target patient group",
            options=["General", "Elderly", "Low Literacy", "ESL (English as Second Language)"]
        )
    
    st.markdown("---")
    st.caption("Created for Healthcare LLM Application Tutorial")

# Sample medical notes from Synthea
default_medical_note = """PATIENT MEDICAL NOTE
Patient ID: 1a2b3c4d-5e6f
Demographics: 67 year old male, White, Non-Hispanic

MEDICAL HISTORY:
Conditions: Essential hypertension, Hyperlipidemia, Type 2 diabetes mellitus, Chronic kidney disease, stage 2 (mild)

MEDICATIONS:
Lisinopril 10mg daily, Atorvastatin 20mg daily, Metformin 1000mg BID

ENCOUNTERS:
- 2024-03-10: Outpatient visit - Reason: Follow-up
- 2024-01-15: Laboratory encounter - Reason: Routine labs

LABORATORY RESULTS:
- 2024-01-15: Hemoglobin A1c - 7.2 %
- 2024-01-15: Creatinine - 1.3 mg/dL
- 2024-01-15: LDL Cholesterol - 110 mg/dL"""

# Load sample medical notes
sample_notes = {
    "Sample 1: Diabetes and Hypertension": default_medical_note,
    "Sample 2: Cardiac Condition": """PATIENT MEDICAL NOTE
Patient ID: 7g8h9i0j-1k2l
Demographics: 72 year old female, Asian, Non-Hispanic

MEDICAL HISTORY:
Conditions: Coronary atherosclerosis, Atrial fibrillation, Congestive heart failure, Osteoarthritis

MEDICATIONS:
Metoprolol 25mg BID, Warfarin 5mg daily, Furosemide 40mg daily, Lisinopril 20mg daily, Atorvastatin 40mg daily

ENCOUNTERS:
- 2024-02-05: Emergency visit - Reason: Chest pain
- 2024-02-06: Inpatient admission - Reason: Non-ST elevation myocardial infarction
- 2024-02-10: Discharge - Disposition: Home

LABORATORY RESULTS:
- 2024-02-05: Troponin I - 0.2 ng/mL
- 2024-02-05: BNP - 450 pg/mL
- 2024-02-05: Creatinine - 1.1 mg/dL""",
    "Sample 3: Respiratory Condition": """PATIENT MEDICAL NOTE
Patient ID: 3m4n5o6p-7q8r
Demographics: 58 year old female, Black, Non-Hispanic

MEDICAL HISTORY:
Conditions: Chronic obstructive pulmonary disease (COPD), Gastroesophageal reflux disease (GERD), Anxiety disorder

MEDICATIONS:
Albuterol inhaler PRN, Fluticasone/Salmeterol inhaler BID, Omeprazole 20mg daily, Sertraline 50mg daily

ENCOUNTERS:
- 2024-04-02: Outpatient visit - Reason: COPD exacerbation
- 2024-04-02: Pulmonary function test

LABORATORY RESULTS:
- 2024-04-02: SpO2 - 94 %
- 2024-04-02: FEV1 - 65 % predicted
- 2024-04-02: FEV1/FVC ratio - 0.65""",
    "Custom Note (Enter your own)": ""
}

# Functions for different prompting methods
def zero_shot_simplification(medical_note, target_group="General", model="gpt-3.5-turbo", temp=0.3):
    """Simplifies a medical note using zero-shot prompting approach."""
    
    # Customize for target patient group
    if target_group == "Elderly":
        audience = "elderly patients (70+ years) who may have some vision or hearing difficulties"
        specific_instructions = "Use larger conceptual chunks, clear organization with headings, and avoid information overload."
    elif target_group == "Low Literacy":
        audience = "patients with low health literacy (reading at a 4th-5th grade level)"
        specific_instructions = "Use very simple words (1-2 syllables when possible), short sentences, and concrete examples."
    elif target_group == "ESL":
        audience = "patients who speak English as a second language"
        specific_instructions = "Use common everyday vocabulary, avoid idioms and cultural references, and use consistent terminology."
    else:  # General
        audience = "patients with limited health literacy"
        specific_instructions = "Use plain language at approximately an 8th grade reading level."
    
    prompt = f"""
Please simplify the following medical note to make it more understandable for {audience}:

{medical_note}

The simplified note should:
- Use plain language instead of medical jargon
- Maintain all important medical information
- Be organized in a clear structure
- Explain medical terms when necessary
- {specific_instructions}
"""
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that specializes in making medical information accessible to patients."},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=1000
        )
        
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error: {str(e)}"

def few_shot_simplification(medical_note, target_group="General", model="gpt-3.5-turbo", temp=0.3):
    """Simplifies a medical note using few-shot (in-context learning) approach."""
    
    # Customize examples for target patient group
    if target_group == "Elderly":
        instruction = "elderly patients (70+ years) who may have some vision or hearing difficulties"
        examples = """
EXAMPLE 1:
ORIGINAL: 
Patient is a 75-year-old female with hypertension, hyperlipidemia, and osteoarthritis. Patient reports increasing joint pain and difficulty with mobility. Physical examination reveals decreased range of motion in bilateral knees.

SIMPLIFIED:
YOUR HEALTH SUMMARY

You are a 75-year-old woman with high blood pressure, high cholesterol, and arthritis in your joints.

YOUR CURRENT SYMPTOMS:
You mentioned that your joint pain is getting worse and you're having more trouble moving around. When we examined you, we noticed you can't bend your knees as fully as normal.

WHAT THIS MEANS:
Your arthritis may be progressing. This is causing the increased pain and making it harder for you to walk and move.

NEXT STEPS:
We should discuss pain management options and possibly physical therapy to help maintain your mobility and independence.

EXAMPLE 2:
ORIGINAL:
Patient presents with exacerbation of COPD. Pulmonary function tests show FEV1 of 45% predicted and SpO2 of 92% on room air. Started on prednisone 40mg daily for 5 days and increased albuterol inhaler frequency.

SIMPLIFIED:
YOUR HEALTH UPDATE

Your lung condition (COPD) is having a flare-up right now.

YOUR TEST RESULTS:
• Breathing test: Shows your lungs are working at about 45% of normal capacity
• Oxygen level: 92% (normal is 95-100%)

YOUR TREATMENT PLAN:
• New medication: Prednisone pills (40mg) once daily for 5 days
  This helps reduce inflammation in your lungs
• Increase your rescue inhaler (albuterol) as needed
  Use it more often until your breathing improves

IMPORTANT REMINDER:
• Take all medications as directed
• Call us if your breathing gets worse or doesn't improve
"""
    elif target_group == "Low Literacy":
        instruction = "patients with low health literacy (reading at a 4th-5th grade level)"
        examples = """
EXAMPLE 1:
ORIGINAL: 
Patient is a 67-year-old male with hypertension, hyperlipidemia, and type 2 diabetes mellitus. Patient reports dyspnea on exertion and occasional orthopnea. Physical examination reveals bilateral lower extremity edema.

SIMPLIFIED:
YOUR HEALTH

You are a 67-year-old man with:
• High blood pressure
• High fat in your blood
• Sugar disease (diabetes)

You told us:
• You get short of breath when you move around
• Sometimes it's hard to breathe when you lie down

We found:
• Your legs are swollen on both sides

What this means:
Your heart may be working too hard. The swelling in your legs happens when fluid builds up.

Next steps:
We need to check your heart. Take your pills every day.

EXAMPLE 2:
ORIGINAL:
Patient presents with complaints of dyspepsia and epigastric pain for 2 weeks, worse after meals. Endoscopy revealed gastric erosions consistent with NSAID gastropathy. H. pylori testing negative.

SIMPLIFIED:
YOUR HEALTH PROBLEM

What you told us:
• Your stomach hurts
• The pain has lasted 2 weeks
• Pain gets worse after you eat

What we found:
• Your stomach has some raw, sore areas inside
• These sores likely came from pain pills you take
• You do not have the stomach germ called H. pylori

What to do now:
• Stop taking ibuprofen, naproxen, or aspirin
• Take the new stomach medicine every day
• Eat smaller meals
• Call us if you see blood in your throw-up or poop
"""
    elif target_group == "ESL":
        instruction = "patients who speak English as a second language"
        examples = """
EXAMPLE 1:
ORIGINAL: 
Patient is a 58-year-old female who presents with acute onset of severe headache, photophobia, and nuchal rigidity. CT scan negative for hemorrhage. Lumbar puncture performed, results pending. Started on empiric antibiotics for presumed meningitis.

SIMPLIFIED:
YOUR MEDICAL SITUATION

Your symptoms:
• You have a sudden, very bad headache
• Bright light hurts your eyes
• Your neck feels stiff and painful

Tests we did:
• Head scan (CT): No bleeding was found in your brain
• Spinal fluid test: We took some fluid from your spine to test it. We are waiting for results.

Current treatment:
• We started you on strong antibiotics through your IV
• These medications fight infection

What we think might be happening:
We are concerned you might have an infection around your brain and spinal cord. This is called "meningitis."

Next steps:
• You need to stay in the hospital
• We will check your test results when they are ready
• We will watch you closely for any changes

EXAMPLE 2:
ORIGINAL:
Patient with history of CHF presents with increased dyspnea, orthopnea, and peripheral edema. BNP elevated at 850 pg/mL. CXR shows pulmonary edema and cardiomegaly. Started on IV furosemide and increased ACE inhibitor dosage.

SIMPLIFIED:
YOUR HEART CONDITION

Your symptoms now:
• You are having trouble breathing
• You cannot breathe well when lying flat
• Your legs and ankles are swollen

Your test results:
• Blood test: Shows your heart is under stress
• Chest X-ray: Shows fluid in your lungs and your heart is enlarged

Your treatment plan:
• Water pill through IV: This helps remove extra fluid from your body
• Increased dose of your heart medicine: This helps your heart work better

What is happening:
Your heart failure is getting worse right now. This means your heart is not pumping blood well enough. This causes fluid to build up in your lungs and legs.

Important information:
• You need to limit salt in your food
• You need to limit how much liquid you drink
• You should weigh yourself every day
• Call us if you gain more than 2 kg (4 pounds) in one day
"""
    else:  # General
        instruction = "patients with limited health literacy"
        examples = """
EXAMPLE 1:
ORIGINAL: 
Patient is a 67-year-old male with hypertension, hyperlipidemia, and type 2 diabetes mellitus. Patient reports dyspnea on exertion and orthopnea. Physical examination reveals bilateral lower extremity edema.

SIMPLIFIED:
You are a 67-year-old man with high blood pressure, high cholesterol, and type 2 diabetes. You mentioned feeling short of breath during activity and when lying flat. During the exam, we noticed swelling in both of your legs.

EXAMPLE 2:
ORIGINAL:
Patient presents with persistent cough for 2 weeks, associated with low-grade fever and myalgia. Chest auscultation reveals rhonchi in the right lower lobe. WBC count elevated at 11,000.

SIMPLIFIED:
You came in with a cough that has lasted for 2 weeks, along with a mild fever and muscle aches. When listening to your lungs, we heard abnormal breathing sounds in the lower right part of your lungs. Your white blood cell count is high at 11,000, which might indicate an infection.
"""
    
    prompt = f"""
I'll show you how to simplify medical notes for {instruction}. Here are some examples:

{examples}

Now, please simplify the following medical note in a similar way:

{medical_note}
"""
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that specializes in making medical information accessible to patients."},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=1000
        )
        
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error: {str(e)}"

def chain_of_thought_simplification(medical_note, target_group="General", model="gpt-3.5-turbo", temp=0.3):
    """Simplifies a medical note using chain of thought prompting approach."""
    
    # Customize for target patient group
    if target_group == "Elderly":
        audience = "elderly patients (70+ years) who may have some vision or hearing difficulties"
        specific_instructions = "Use larger conceptual chunks, clear organization with headings, and avoid information overload."
    elif target_group == "Low Literacy":
        audience = "patients with low health literacy (reading at a 4th-5th grade level)"
        specific_instructions = "Use very simple words (1-2 syllables when possible), short sentences, and concrete examples."
    elif target_group == "ESL":
        audience = "patients who speak English as a second language"
        specific_instructions = "Use common everyday vocabulary, avoid idioms and cultural references, and use consistent terminology."
    else:  # General
        audience = "patients with limited health literacy"
        specific_instructions = "Use plain language at approximately an 8th grade reading level."
    
    prompt = f"""
Please simplify the following medical note for {audience}. Think step by step:

1. First, identify all medical terms and jargon that need simplification
2. Determine the core medical information that must be preserved
3. Reorganize the information in a more logical flow for the patient
4. Rewrite each section using plain language appropriate for the patient
5. Add brief explanations for medical terms and values when needed
6. Ensure all important information is included and accurate
7. Format the information in a patient-friendly way with clear headings
8. Check that the simplification addresses these specific needs: {specific_instructions}

Medical Note:
{medical_note}

Now, first identify the medical terms that need simplification:
"""
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that specializes in making medical information accessible to patients."},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=1500
        )
        
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error: {str(e)}"

def tree_of_thoughts_simplification(medical_note, target_group="General", model="gpt-3.5-turbo", temp=0.3):
    """Simplifies a medical note using tree of thoughts approach."""
    
    # Customize for target patient group
    if target_group == "Elderly":
        audience = "elderly patients (70+ years) who may have some vision or hearing difficulties"
        specific_instructions = "larger conceptual chunks, clear organization with headings, and avoid information overload"
    elif target_group == "Low Literacy":
        audience = "patients with low health literacy (reading at a 4th-5th grade level)"
        specific_instructions = "very simple words (1-2 syllables when possible), short sentences, and concrete examples"
    elif target_group == "ESL":
        audience = "patients who speak English as a second language"
        specific_instructions = "common everyday vocabulary, no idioms or cultural references, and consistent terminology"
    else:  # General
        audience = "patients with limited health literacy"
        specific_instructions = "plain language at approximately an 8th grade reading level"
    
    prompt = f"""
I will simplify this medical note for {audience} by exploring different approaches and selecting the best one.

Medical Note:
{medical_note}

Approach 1: Focus on simplifying vocabulary while maintaining the structure
- Identify all medical terms
- Replace with simpler alternatives or brief explanations
- Keep the original structure of the note
- Use {specific_instructions}

Approach 2: Restructure the note to be more narrative and conversational
- Convert the note into a summary of what happened and what it means
- Use second-person perspective ("you have..." instead of "patient has...")
- Group related information together regardless of original structure
- Use {specific_instructions}

Approach 3: Create a hybrid approach with simplified sections and explanations
- Keep key sections (history, medications, etc.) but rename them to be more patient-friendly
- Simplify the language within each section
- Add brief explanations of what each section means for the patient's health
- Use {specific_instructions}

Let me evaluate each approach for this specific note:

Approach 1 Evaluation:

Approach 2 Evaluation:

Approach 3 Evaluation:

Based on my evaluation, the most effective approach for this specific case is:

Here's the simplified note using the best approach:
"""
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that specializes in making medical information accessible to patients."},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=1500
        )
        
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Function to calculate readability score (Flesch Reading Ease)
def calculate_readability(text):
    sentences = len(re.split(r'[.!?]+', text))
    words = len(re.findall(r'\b\w+\b', text))
    syllables = 0
    for word in re.findall(r'\b\w+\b', text.lower()):
        syllables += max(1, len(re.findall(r'[aeiouy]+', word)))
    
    if sentences == 0 or words == 0:
        return 0
    
    flesch_score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    return max(0, min(100, flesch_score))  # Clip between 0 and 100

# Function to calculate medical term density
def calculate_medical_term_density(text):
    # Sample list of medical terms (would be expanded in a real application)
    medical_terms = [
        'hypertension', 'diabetes', 'mellitus', 'dyspnea', 'orthopnea', 'edema',
        'hyperlipidemia', 'myocardial', 'infarction', 'stroke', 'arrhythmia',
        'tachycardia', 'bradycardia', 'fibrillation', 'cholesterol', 'triglycerides',
        'glucose', 'insulin', 'hyperglycemia', 'hypoglycemia', 'neuropathy',
        'retinopathy', 'nephropathy', 'cardiomyopathy', 'angina', 'stent',
        'bypass', 'angioplasty', 'catheterization', 'echocardiogram', 'electrocardiogram',
        'coronary', 'atherosclerosis', 'atrial', 'congestive', 'creatinine',
        'hemoglobin', 'ldl', 'troponin', 'bnp', 'copd', 'gerd', 'prn', 'bid',
        'chronic', 'obstructive', 'pulmonary', 'gastroesophageal', 'reflux',
        'albuterol', 'fluticasone', 'salmeterol', 'omeprazole', 'sertraline',
        'metoprolol', 'warfarin', 'furosemide', 'lisinopril', 'atorvastatin',
        'metformin', 'SpO2', 'FEV1', 'FVC'
    ]
    
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    
    if word_count == 0:
        return 0
    
    medical_term_count = sum(1 for word in words if word in medical_terms)
    
    return (medical_term_count / word_count) * 100  # Return as a percentage

# Function to download results as text file
def get_download_link(text, filename="simplified_medical_note.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download as Text File</a>'
    return href

# Function to save results to session_state history
def save_to_history(original_note, simplified_note, method, target_group, metrics):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    history_item = {
        "timestamp": timestamp,
        "method": method,
        "target_group": target_group,
        "original_note": original_note,
        "simplified_note": simplified_note,
        "metrics": metrics
    }
    
    st.session_state.processing_history.append(history_item)
    
    # Save to disk as well (for persistence between sessions)
    try:
        history_file = st.session_state.cache_dir / "processing_history.pkl"
        with open(history_file, 'wb') as f:
            pickle.dump(st.session_state.processing_history, f)
    except Exception as e:
        st.warning(f"Could not save history to disk: {str(e)}")
  # Function to save results to session_state history
def save_to_history(original_note, simplified_note, method, target_group, metrics):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    history_item = {
        "timestamp": timestamp,
        "method": method,
        "target_group": target_group,
        "original_note": original_note,
        "simplified_note": simplified_note,
        "metrics": metrics
    }
    
    st.session_state.processing_history.append(history_item)
    
    # Save to disk as well (for persistence between sessions)
    try:
        history_file = st.session_state.cache_dir / "processing_history.pkl"
        with open(history_file, 'wb') as f:
            pickle.dump(st.session_state.processing_history, f)
    except Exception as e:
        st.warning(f"Could not save history to disk: {str(e)}")

# Function to load processing history from disk
def load_history():
    try:
        history_file = st.session_state.cache_dir / "processing_history.pkl"
        if history_file.exists():
            with open(history_file, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load history from disk: {str(e)}")
    
    return []

# Try to load history at startup
if len(st.session_state.processing_history) == 0:
    st.session_state.processing_history = load_history()

# Function to generate a report as a PDF
def generate_report(history_items):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import MaxNLocator
    from datetime import datetime
    
    pdf_buffer = BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        # Title page
        plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.9, "Medical Note Simplification Report", ha='center', fontsize=24)
        plt.text(0.5, 0.85, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ha='center', fontsize=14)
        plt.text(0.5, 0.8, f"Contains analysis of {len(history_items)} simplified notes", ha='center', fontsize=14)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Summary statistics
        plt.figure(figsize=(8.5, 11))
        plt.subplot(2, 1, 1)
        
        # Group by method
        methods = [item['method'] for item in history_items]
        unique_methods = list(set(methods))
        method_counts = [methods.count(m) for m in unique_methods]
        
        plt.bar(unique_methods, method_counts)
        plt.title('Notes Simplified by Method')
        plt.ylabel('Count')
        plt.subplot(2, 1, 2)
        
        # Average readability by method
        readability_by_method = {}
        for method in unique_methods:
            scores = [item['metrics']['readability_score'] for item in history_items if item['method'] == method]
            readability_by_method[method] = sum(scores) / len(scores) if scores else 0
        
        methods = list(readability_by_method.keys())
        scores = list(readability_by_method.values())
        
        plt.bar(methods, scores)
        plt.title('Average Readability Score by Method')
        plt.ylabel('Flesch Reading Ease Score')
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Include sample notes (first 3 at most)
        for i, item in enumerate(history_items[:3]):
            plt.figure(figsize=(8.5, 11))
            plt.text(0.5, 0.95, f"Sample Simplification #{i+1}", ha='center', fontsize=16)
            plt.text(0.5, 0.9, f"Method: {item['method']} | Target: {item['target_group']}", ha='center', fontsize=12)
            
            # Metrics
            metrics = item['metrics']
            plt.text(0.5, 0.85, f"Readability: {metrics['readability_score']:.1f}/100 | Term Density: {metrics['term_density']:.1f}% | Length Ratio: {metrics['length_ratio']:.2f}", ha='center', fontsize=10)
            
            # Original note (truncated if very long)
            original = item['original_note']
            if len(original) > 500:
                original = original[:500] + "..."
            
            wrapped_original = textwrap.fill(original, width=80)
            plt.text(0.1, 0.8, "ORIGINAL NOTE:", fontsize=10, fontweight='bold')
            plt.text(0.1, 0.78, wrapped_original, fontsize=8, va='top', linespacing=1.5)
            
            # Simplified note (truncated if very long)
            simplified = item['simplified_note']
            if len(simplified) > 500:
                simplified = simplified[:500] + "..."
                
            wrapped_simplified = textwrap.fill(simplified, width=80)
            plt.text(0.1, 0.45, "SIMPLIFIED NOTE:", fontsize=10, fontweight='bold')
            plt.text(0.1, 0.43, wrapped_simplified, fontsize=8, va='top', linespacing=1.5)
            
            plt.axis('off')
            pdf.savefig()
            plt.close()
    
    pdf_buffer.seek(0)
    return pdf_buffer

# Prepare materials for the Tutorial tab
def load_tutorial_content():
    # Introduction Section
    introduction = """
    # Using LLMs for Medical Note Simplification
    
    This tutorial demonstrates how to apply Large Language Models (LLMs) to simplify medical notes and make healthcare information more accessible to patients with different levels of health literacy.
    
    ## The Healthcare Challenge
    
    Medical notes are written for healthcare professionals, not patients. They contain complex terminology, abbreviations, and clinical language that can be difficult for patients to understand. This communication barrier can lead to:
    
    - Confusion about diagnoses and treatment plans
    - Poor medication adherence
    - Increased anxiety about health conditions
    - Reduced patient engagement in care
    
    According to research, approximately 36% of US adults have limited health literacy, making medical documents particularly challenging to understand.
    
    ## The Solution: LLM-Assisted Simplification
    
    Large Language Models like GPT can be used to translate complex medical information into more accessible language while preserving critical clinical details. This tutorial demonstrates how to use different prompting techniques to achieve optimal results.
    """
    
    # Dataset Information
    dataset_info = """
    ## Dataset: Synthea™ Synthetic Patient Data
    
    This tutorial uses **Synthea™** - an open-source synthetic patient generator that creates realistic but not real patient data. This allows us to work with medical data without privacy concerns.
    
    ### About Synthea Data:
    
    - 100% synthetic - contains no real protected health information (PHI)
    - Follows realistic clinical patterns and disease progression
    - Uses standard medical coding systems (ICD-10, LOINC, etc.)
    - Available in multiple formats (CSV, FHIR, etc.)
    
    The examples in this tutorial use a dataset containing 100 synthetic patients with various medical conditions, medications, and laboratory results.
    """
    
    # Prompting Methods
    prompting_methods = """
    ## LLM Prompting Methods
    
    This tutorial explores four different prompting techniques for medical note simplification:
    
    ### 1. Zero-Shot Prompting
    
    Direct instructions without examples - relies on the model's pre-trained knowledge of medical terminology and plain language.
    
    ### 2. Few-Shot (In-Context Learning)
    
    Provides examples of medical notes paired with their simplified versions before asking the model to simplify a new note.
    
    ### 3. Chain of Thought (CoT)
    
    Guides the model through a step-by-step reasoning process for simplification, breaking down the task into explicit steps.
    
    ### 4. Tree of Thoughts (ToT)
    
    Explores multiple approaches to simplification, evaluates each, and selects the most effective one for the specific note.
    """
    
    # Patient-Specific Approaches
    patient_specific = """
    ## Patient-Specific Approaches
    
    Different patient populations have different communication needs. This tutorial explores tailoring simplified notes for:
    
    ### Elderly Patients (70+ years)
    - Larger conceptual chunks
    - Clear organization with headings
    - More context for medical terms
    - Avoiding information overload
    
    ### Low Health Literacy Patients
    - Very simple vocabulary (4th-5th grade level)
    - Short sentences (8-10 words)
    - Concrete examples and analogies
    - Avoiding abbreviations
    
    ### ESL Patients (English as Second Language)
    - Common vocabulary
    - Avoiding idioms and cultural references
    - Explicit connections between ideas
    - Consistent terminology
    
    ### General Patients
    - 8th grade reading level (standard health communication)
    - Balance between simplicity and completeness
    """
    
    # Evaluation Methods
    evaluation_methods = """
    ## Evaluation Methods
    
    How do we measure the success of our simplification efforts? This tutorial uses both quantitative and qualitative metrics:
    
    ### Quantitative Metrics
    
    1. **Flesch Reading Ease Score**
       - Measures text readability from 0-100
       - Higher scores indicate easier reading
       - Target: 70-80 for general audiences (7th-8th grade level)
    
    2. **Medical Term Density**
       - Percentage of specialized medical terms in the text
       - Lower percentages indicate better simplification
       - Calculated by comparing against a medical terminology dictionary
    
    3. **Length Ratio**
       - Ratio of simplified text length to original note length
       - Values near 1.0 indicate preserving information while reducing complexity
       - Too low may indicate lost information; too high may indicate verbosity
    
    ### Qualitative Analysis
    
    1. **Information Accuracy**
       - Are all essential diagnoses preserved?
       - Are medication details accurately translated?
       - Are test results presented with proper context?
    
    2. **Clarity of Explanations**
       - How well are medical terms explained?
       - Is context provided for values and results?
       - Is there a logical flow to the information?
    
    3. **Organization and Structure**
       - Is information grouped logically?
       - Are headings clear and informative?
       - Is the most important information prioritized?
    """
    
    return {
        "introduction": introduction,
        "dataset_info": dataset_info,
        "prompting_methods": prompting_methods,
        "patient_specific": patient_specific,
        "evaluation_methods": evaluation_methods
    }

# Code examples for the Code tab
def load_code_examples():
    zero_shot_code = """
# Zero-Shot Prompting
def zero_shot_simplification(medical_note, target_group="General"):
    # Customize for target patient group
    if target_group == "Elderly":
        audience = "elderly patients (70+ years)"
        specific_instructions = "Use clear organization with headings."
    elif target_group == "Low Literacy":
        audience = "patients with low health literacy (4th-5th grade level)"
        specific_instructions = "Use simple words and short sentences."
    else:  # General or ESL
        audience = "patients with limited health literacy"
        specific_instructions = "Use plain language at 8th grade level."
    
    prompt = f'''
Please simplify the following medical note for {audience}:

{medical_note}

The simplified note should:
- Use plain language instead of medical jargon
- Maintain all important medical information
- Be organized in a clear structure
- Explain medical terms when necessary
- {specific_instructions}
'''
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You specialize in making medical information accessible."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message["content"].strip()
"""

    few_shot_code = """
# Few-Shot (In-Context Learning) Prompting
def few_shot_simplification(medical_note, target_group="General"):
    # Define example pairs (original → simplified)
    examples = '''
EXAMPLE 1:
ORIGINAL: 
Patient is a 67-year-old male with hypertension, hyperlipidemia, and type 2 diabetes mellitus.

SIMPLIFIED:
You are a 67-year-old man with high blood pressure, high cholesterol, and type 2 diabetes.

EXAMPLE 2:
ORIGINAL:
Patient presents with persistent cough for 2 weeks, associated with low-grade fever and myalgia.

SIMPLIFIED:
You came in with a cough that has lasted for 2 weeks, along with a mild fever and muscle aches.
'''
    
    prompt = f'''
I'll show you how to simplify medical notes for patients with limited health literacy.
Here are some examples:

{examples}

Now, please simplify the following medical note in a similar way:

{medical_note}
'''
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You specialize in making medical information accessible."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message["content"].strip()
"""

    chain_of_thought_code = """
# Chain of Thought Prompting
def chain_of_thought_simplification(medical_note, target_group="General"):
    prompt = f'''
Please simplify the following medical note for patients with limited health literacy. Think step by step:

1. First, identify all medical terms and jargon that need simplification
2. Determine the core medical information that must be preserved
3. Rewrite each section using plain language
4. Ensure all important information is included and accurate
5. Organize the information in a patient-friendly format

Medical Note:
{medical_note}

Now, first identify the medical terms that need simplification:
'''
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You specialize in making medical information accessible."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    
    return response.choices[0].message["content"].strip()
"""

    tree_of_thoughts_code = """
# Tree of Thoughts Prompting
def tree_of_thoughts_simplification(medical_note, target_group="General"):
    prompt = f'''
I will simplify this medical note by exploring different approaches and selecting the best one.

Medical Note:
{medical_note}

Approach 1: Focus on simplifying vocabulary while maintaining the structure
- Identify all medical terms
- Replace with simpler alternatives or brief explanations
- Keep the original structure of the note

Approach 2: Restructure the note to be more narrative and conversational
- Convert the note into a summary of what happened and what it means
- Use second-person perspective ("you have..." instead of "patient has...")
- Group related information together regardless of original structure

Approach 3: Create a hybrid approach with simplified sections and explanations
- Keep key sections (history, medications, etc.) but rename them to be patient-friendly
- Simplify the language within each section
- Add brief explanations of what each section means for the patient's health

Let me evaluate each approach for this specific note:

Approach 1 Evaluation:

Approach 2 Evaluation:

Approach 3 Evaluation:

Based on my evaluation, the most effective approach for this specific case is:

Here's the simplified note using the best approach:
'''
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You specialize in making medical information accessible."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    
    return response.choices[0].message["content"].strip()
"""

    evaluation_code = """
# Evaluation Functions

# Calculate readability score (Flesch Reading Ease)
def calculate_readability(text):
    sentences = len(re.split(r'[.!?]+', text))
    words = len(re.findall(r'\\b\\w+\\b', text))
    syllables = 0
    for word in re.findall(r'\\b\\w+\\b', text.lower()):
        syllables += max(1, len(re.findall(r'[aeiouy]+', word)))
    
    if sentences == 0 or words == 0:
        return 0
    
    flesch_score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    return max(0, min(100, flesch_score))  # Clip between 0 and 100

# Calculate medical term density
def calculate_medical_term_density(text):
    # Sample medical term list (would be expanded in a real application)
    medical_terms = [
        'hypertension', 'diabetes', 'mellitus', 'dyspnea', 'orthopnea', 
        'hyperlipidemia', 'myocardial', 'infarction', 'arrhythmia',
        'tachycardia', 'bradycardia', 'fibrillation', 'cholesterol', 
        'glucose', 'insulin', 'hyperglycemia', 'hypoglycemia'
        # Add many more terms in a real application
    ]
    
    words = re.findall(r'\\b\\w+\\b', text.lower())
    word_count = len(words)
    
    if word_count == 0:
        return 0
    
    medical_term_count = sum(1 for word in words if word in medical_terms)
    
    return (medical_term_count / word_count) * 100  # Return as a percentage
"""

    return {
        "zero_shot": zero_shot_code,
        "few_shot": few_shot_code,
        "chain_of_thought": chain_of_thought_code,
        "tree_of_thoughts": tree_of_thoughts_code,
        "evaluation": evaluation_code
    }

# Main app layout based on selected tab
if st.session_state.current_tab == "tutorial":
    # TUTORIAL TAB
    tutorial_content = load_tutorial_content()
    
    st.title("📚 LLM for Medical Notes Simplification Tutorial")
    
    # Tutorial sections as tabs
    tutorial_tabs = st.tabs(["Introduction", "Dataset", "Prompting Methods", "Patient-Specific", "Evaluation", "Step-by-Step Guide"])
    
    with tutorial_tabs[0]:
        st.markdown(tutorial_content["introduction"])
        
        st.markdown("### Try it now!")
        st.markdown("Click on the **Live Demo** tab in the sidebar to try simplifying medical notes with different methods.")
    
    with tutorial_tabs[1]:
        st.markdown(tutorial_content["dataset_info"])
        
        # Display sample Synthea data
        st.subheader("Sample Data Snippets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Patient Demographics (CSV)**")
            st.markdown("""
            ```
            Id,BIRTHDATE,GENDER,RACE,ETHNICITY
            1a2b3c4d-5e6f,1952-07-15,M,white,non-hispanic
            7g8h9i0j-1k2l,1979-02-23,F,asian,non-hispanic
            ```
            """)
        
        with col2:
            st.markdown("**Conditions (CSV)**")
            st.markdown("""
            ```
            START,STOP,PATIENT,DESCRIPTION,CODE
            2018-05-12,,1a2b3c4d-5e6f,Essential hypertension,I10
            2020-11-30,,1a2b3c4d-5e6f,Type 2 diabetes mellitus,E11.9
            ```
            """)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Medications (CSV)**")
            st.markdown("""
            ```
            START,STOP,PATIENT,DESCRIPTION,DISPENSES
            2018-05-15,,1a2b3c4d-5e6f,Lisinopril 10 MG,4
            2020-12-02,,1a2b3c4d-5e6f,Metformin 1000 MG,3
            ```
            """)
        
        with col4:
            st.markdown("**Observations (CSV)**")
            st.markdown("""
            ```
            DATE,PATIENT,DESCRIPTION,VALUE,UNITS
            2022-01-15,1a2b3c4d-5e6f,Hemoglobin A1c,7.2,%
            2022-01-15,1a2b3c4d-5e6f,Systolic BP,142,mm[Hg]
            ```
            """)
        
        st.markdown("### Creating Structured Medical Notes")
        st.markdown("""
        For this tutorial, we process the raw Synthea CSV data to create realistic medical notes. Each note includes:
        
        - Patient demographics
        - Medical history (conditions)
        - Current medications
        - Recent encounters
        - Laboratory results
        
        These structured notes serve as input for our LLM simplification experiments.
        """)
    
    with tutorial_tabs[2]:
        st.markdown(tutorial_content["prompting_methods"])
        
        # Add illustrations for each method
        st.subheader("Method Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Zero-Shot Prompting")
            st.markdown("""
            ```
            Please simplify the following medical note for patients with limited health literacy:
            
            [MEDICAL NOTE]
            
            The simplified note should:
            - Use plain language instead of medical jargon
            - Maintain all important medical information
            - Be organized in a clear structure
            - Explain medical terms when necessary
            ```
            """)
            
            st.markdown("#### Chain of Thought Prompting")
            st.markdown("""
            ```
            Please simplify the following medical note. Think step by step:
            
            1. First, identify all medical terms that need simplification
            2. Determine core medical information to preserve
            3. Rewrite each section using plain language
            4. Ensure all information is included and accurate
            5. Organize the information in a patient-friendly format
            
            [MEDICAL NOTE]
            ```
            """)
        
        with col2:
            st.markdown("#### Few-Shot Prompting")
            st.markdown("""
            ```
            I'll show you how to simplify medical notes for patients with limited health literacy. Here are some examples:
            
            EXAMPLE 1:
            ORIGINAL: 
            Patient is a 67-year-old male with hypertension.
            SIMPLIFIED:
            You are a 67-year-old man with high blood pressure.
            
            [Additional example]
            
            Now, please simplify the following medical note:
            [MEDICAL NOTE]
            ```
            """)
            
            st.markdown("#### Tree of Thoughts Prompting")
            st.markdown("""
            ```
            I will simplify this medical note by exploring different approaches:
            
            Approach 1: Focus on simplifying vocabulary
            Approach 2: Restructure to be more conversational
            Approach 3: Create a hybrid with sections and explanations
            
            Let me evaluate each approach for this note...
            
            [After evaluation]
            The simplified note based on the most effective approach:
            ```
            """)
        
        st.markdown("### Comparative Effectiveness of Methods")
        st.markdown("""
        Different prompting methods have different strengths:
        
        | Method | Readability | Information Retention | Computation Required |
        |--------|-------------|------------------------|----------------------|
        | Zero-Shot | ⭐⭐ | ⭐⭐⭐ | ⭐ (Lowest) |
        | Few-Shot | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
        | Chain of Thought | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
        | Tree of Thoughts | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ (Highest) |
        
        The most effective method depends on your specific requirements and constraints.
        """)
    
    with tutorial_tabs[3]:
        st.markdown(tutorial_content["patient_specific"])
        
        st.subheader("Example Adaptations")
        
        st.markdown("#### Original Medical Note")
        st.markdown("""
        ```
        Patient is a 67-year-old male with hypertension, hyperlipidemia, and type 2 diabetes mellitus. 
        Patient reports dyspnea on exertion and orthopnea. Hemoglobin A1c 7.2%.
        ```
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### For Elderly Patients")
            st.markdown("""
            ```
            YOUR HEALTH SUMMARY
            
            You are a 67-year-old man with:
              • High blood pressure
              • High cholesterol 
              • Type 2 diabetes
            
            Your symptoms:
              • You feel short of breath during activity
              • You have trouble breathing when lying flat
            
            Your recent blood sugar test (Hemoglobin A1c) 
            was 7.2%, which is slightly above the target 
            of less than 7%.
            ```
            """)
            
            st.markdown("#### For ESL Patients")
            st.markdown("""
            ```
            YOUR HEALTH INFORMATION
            
            You are a 67-year-old man.
            
            You have these health conditions:
              • High blood pressure (hypertension)
              • High levels of fat in blood (hyperlipidemia)
              • Diabetes type 2
            
            Your symptoms now:
              • You cannot breathe easily when you exercise
              • You cannot breathe easily when you lie down
            
            Your blood sugar test result:
              • HbA1c = 7.2% (normal is less than 7.0%)
            ```
            """)
        
        with col2:
            st.markdown("#### For Low Literacy Patients")
            st.markdown("""
            ```
            YOUR HEALTH
            
            You are a 67-year-old man with:
              • High blood pressure
              • Too much fat in your blood
              • Sugar problem (diabetes)
            
            You told us:
              • You can't breathe well when you move
              • You can't breathe well when you lie down
            
            Your blood test:
              • Your sugar level is a bit high (7.2)
            ```
            """)
            
            st.markdown("#### For General Patients")
            st.markdown("""
            ```
            YOUR HEALTH SUMMARY
            
            You are a 67-year-old man with high blood pressure, 
            high cholesterol, and type 2 diabetes. You mentioned 
            feeling short of breath during activity and when 
            lying flat. Your A1c (3-month average blood sugar) 
            is 7.2%, which is slightly above the target of 
            less than 7%.
            ```
            """)
    
    with tutorial_tabs[4]:
        st.markdown(tutorial_content["evaluation_methods"])
        
        # Add visualization of evaluation metrics
        st.subheader("Visualizing Evaluation Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Flesch Reading Ease Score")
            
            # Simple chart showing score ranges
            chart_data = pd.DataFrame({
                'Category': ['Original', 'Zero-Shot', 'Few-Shot', 'Chain of Thought', 'Tree of Thoughts'],
                'Score': [35, 50, 65, 75, 72]
            })
            
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(chart_data['Category'], chart_data['Score'], color=['#d32f2f', '#8884d8', '#82ca9d', '#ffc658', '#ff8042'])
            ax.axhline(y=60, color='r', linestyle='--', alpha=0.7)
            ax.axhline(y=80, color='g', linestyle='--', alpha=0.7)
            ax.set_ylim(0, 100)
            ax.set_ylabel('Flesch Reading Ease Score')
            ax.text(4.2, 62, 'Minimum Target (60)', color='r', ha='right')
            ax.text(4.2, 82, 'Ideal Target (80)', color='g', ha='right')
            
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            st.pyplot(fig)
            
            st.markdown("""
            **Interpreting the Scores:**
            - 0-30: Very difficult (College graduate level)
            - 30-50: Difficult (College level)
            - 50-60: Fairly difficult (10th-12th grade)
            - 60-70: Standard (8th-9th grade)
            - 70-80: Fairly easy (7th grade)
            - 80-90: Easy (6th grade)
            - 90-100: Very easy (5th grade)
            """)
        
        with col2:
            st.markdown("#### Medical Term Density")
            
            # Medical Term Density chart
            term_data = pd.DataFrame({
                'Category': ['Original', 'Zero-Shot', 'Few-Shot', 'Chain of Thought', 'Tree of Thoughts'],
                'Density': [12.5, 8.3, 6.2, 3.5, 4.1]
            })
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            bars2 = ax2.bar(term_data['Category'], term_data['Density'], color=['#d32f2f', '#8884d8', '#82ca9d', '#ffc658', '#ff8042'])
            ax2.set_ylim(0, 15)
            ax2.set_ylabel('Medical Term Density (%)')
            
            for bar in bars2:
                height = bar.get_
            for bar in bars2:
                height = bar.get_height()
                ax2.annotate(f'{height}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            st.pyplot(fig2)
            
            st.markdown("""
            **Medical Term Density:**
            
            This metric measures the percentage of specialized medical terminology in the text. Lower values indicate better simplification of technical language.
            
            **Target Values:**
            - Original medical notes: 10-15%
            - Simplified for professionals: 5-10%
            - Simplified for patients: Below 5%
            """)
    with tutorial_tabs[5]:
        st.header("Step-by-Step Implementation Guide")
    
        st.markdown("""
        Follow these steps to implement medical note simplification with LLMs in your own projects:
    
        ### 1. Setting Up Your Environment
    
        ```python
        # Install required packages
        pip install openai pandas matplotlib scikit-learn nltk streamlit
    
        # Import necessary libraries
        import openai
        import pandas as pd
        import re
        import time
    
        # Set up OpenAI API
        openai.api_key = "your-api-key-here"
        ```
    
        ### 2. Preparing Your Data
    
        ```python
        # Load your medical notes
        # For Synthea data, you might process CSVs to create notes
        def create_medical_note(patient_data, conditions, medications, observations):
        note = f"""
        PATIENT MEDICAL NOTE
        Patient ID: {patient_data['Id']}
        Demographics: {patient_data['age']} year old {patient_data['GENDER']}, {patient_data['RACE']}, {patient_data['ETHNICITY']}
        
        MEDICAL HISTORY:
        Conditions: {', '.join(conditions['DESCRIPTION'].tolist())}
        
        MEDICATIONS:
        {', '.join(medications['DESCRIPTION'].tolist())}
        
        LABORATORY RESULTS:
        """
        
        for _, obs in observations.iterrows():
            note += f"- {obs['DATE']}: {obs['DESCRIPTION']} - {obs['VALUE']} {obs['UNITS']}\\n"
            
        return note
    ```
    
    ### 3. Implementing Simplification Functions
    
    Choose the appropriate method for your needs:
    
    ```python
    # Example: Chain of Thought method
    def chain_of_thought_simplification(medical_note, target_group="General"):
        # See the Code Examples tab for the full implementation
        # ...
    ```
    
    ### 4. Evaluating Results
    
    ```python
    # Calculate readability score
    def calculate_readability(text):
        # Flesch Reading Ease implementation
        # ...
        
    # Calculate medical term density
    def calculate_medical_term_density(text):
        # Term density implementation
        # ...
        
    # Evaluate simplification
    def evaluate_simplification(original, simplified):
        metrics = {
            "readability_score": calculate_readability(simplified),
            "original_readability": calculate_readability(original),
            "term_density": calculate_medical_term_density(simplified),
            "original_term_density": calculate_medical_term_density(original),
            "length_ratio": len(simplified.split()) / len(original.split())
        }
        
        return metrics
    ```
    
    ### 5. Building a User Interface
    
    ```python
    # Example Streamlit app (simplified)
    import streamlit as st
    
    st.title("Medical Note Simplification")
    
    medical_note = st.text_area("Enter medical note:")
    method = st.selectbox("Select method", 
                         ["Zero-Shot", "Few-Shot", "Chain of Thought", "Tree of Thoughts"])
    
    if st.button("Simplify") and medical_note:
        # Call appropriate function based on method
        # ...
        
        # Display results
        st.subheader("Simplified Note")
        st.write(simplified_note)
        
        # Show metrics
        metrics = evaluate_simplification(medical_note, simplified_note)
        st.metric("Readability Score", f"{metrics['readability_score']:.1f}/100")
    ```
    
    ### Try it yourself!
    
    Explore the **Live Demo** tab to see these steps in action and experiment with different methods and parameters.
    """)
        
elif st.session_state.current_tab == "demo":
    # LIVE DEMO TAB
    st.title("🔬 Live LLM Medical Note Simplification Demo")
    
    st.markdown("""
    <div class="highlight">
    This interactive demo allows you to test different LLM prompting methods for simplifying medical notes. Select a sample note or enter your own, choose a simplification method, and see the results.
    </div>
    """, unsafe_allow_html=True)
    
    # Check if API key is configured
    if not st.session_state.api_key_configured:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use this demo.")
        st.markdown("""
        <div class="disclaimer">
        You'll need an OpenAI API key to use this demo. If you don't have one, you can sign up at 
        <a href="https://platform.openai.com/signup" target="_blank">OpenAI</a> to get started.
        </div>
        """, unsafe_allow_html=True)
    
    # Medical note input
    st.subheader("Step 1: Select or Enter a Medical Note")
    
    note_selection = st.selectbox("Choose a sample medical note or enter your own:", list(sample_notes.keys()))
    
    if note_selection == "Custom Note (Enter your own)":
        medical_note = st.text_area("Enter medical note:", height=300)
    else:
        medical_note = sample_notes[note_selection]
        st.text_area("Medical Note:", value=medical_note, height=300, disabled=True)
    
    # Prompting method selection
    st.subheader("Step 2: Choose Simplification Method")
    
    prompting_method = st.radio(
        "Select a prompting technique:",
        options=["Zero-Shot", "Few-Shot (In-Context Learning)", "Chain of Thought", "Tree of Thoughts"],
        help="Different prompting techniques instruct the LLM in different ways"
    )
    
    # Method explanation based on selection
    if prompting_method == "Zero-Shot":
        st.markdown("""
        <div class="highlight">
        <b>Zero-Shot Prompting</b> gives direct instructions without examples. The LLM must rely on its pre-trained knowledge to simplify the medical note.
        
        <b>Best for:</b> Simple medical notes with few technical terms, quick implementations, minimal token usage
        </div>
        """, unsafe_allow_html=True)
    
    elif prompting_method == "Few-Shot (In-Context Learning)":
        st.markdown("""
        <div class="highlight">
        <b>Few-Shot Prompting</b> provides examples of original and simplified medical notes. This helps the LLM understand the pattern and style of simplification needed.
        
        <b>Best for:</b> Consistent simplification style, better handling of specific terminology, maintaining uniform output format
        </div>
        """, unsafe_allow_html=True)
    
    elif prompting_method == "Chain of Thought":
        st.markdown("""
        <div class="highlight">
        <b>Chain of Thought</b> guides the LLM through a step-by-step reasoning process. It breaks down the simplification task into explicit steps to follow.
        
        <b>Best for:</b> Complex medical notes, thorough simplifications, complete explanations of medical terms
        </div>
        """, unsafe_allow_html=True)
    
    else:  # Tree of Thoughts
        st.markdown("""
        <div class="highlight">
        <b>Tree of Thoughts</b> explores multiple approaches to simplification, evaluates each, and selects the most effective one for the specific note.
        
        <b>Best for:</b> Variable or complex patient scenarios, balancing completeness with simplicity, adapting to different note structures
        </div>
        """, unsafe_allow_html=True)
    
    # Target patient group selection (if API key is configured)
    if st.session_state.api_key_configured:
        st.subheader("Step 3: Select Target Patient Group")
        
        target_group = st.radio(
            "Who is this simplified note for?",
            options=["General", "Elderly", "Low Literacy", "ESL (English as Second Language)"],
            help="Different patient populations have different communication needs"
        )
        
        # Target group explanation
        if target_group == "Elderly":
            st.markdown("""
            <div class="disclaimer">
            <b>Elderly Patient Focus:</b> Larger conceptual chunks, clear organization with headings, and avoiding information overload.
            </div>
            """, unsafe_allow_html=True)
        elif target_group == "Low Literacy":
            st.markdown("""
            <div class="disclaimer">
            <b>Low Literacy Focus:</b> Very simple words (1-2 syllables), short sentences (8-10 words), and concrete examples.
            </div>
            """, unsafe_allow_html=True)
        elif target_group == "ESL":
            st.markdown("""
            <div class="disclaimer">
            <b>ESL Patient Focus:</b> Common everyday vocabulary, no idioms or cultural references, and consistent terminology.
            </div>
            """, unsafe_allow_html=True)
        else:  # General
            st.markdown("""
            <div class="disclaimer">
            <b>General Patient Focus:</b> Plain language at approximately an 8th grade reading level (standard for health communication).
            </div>
            """, unsafe_allow_html=True)
        
        # Model settings (if applicable)
        with st.expander("Advanced Settings"):
            model_choice = st.selectbox(
                "LLM Model",
                options=["gpt-3.5-turbo", "gpt-4-turbo"],
                index=0
            )
            
            temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.3, 
                step=0.1, 
                help="Lower values make output more deterministic, higher values make it more random"
            )
    
    # Process button
    process_clicked = st.button(
        "Simplify Medical Note", 
        disabled=(not st.session_state.api_key_configured or not medical_note)
    )
    
    # Results section
    if process_clicked and st.session_state.api_key_configured and medical_note:
        st.subheader("Step 4: Review Results")
        
        with st.spinner("Simplifying medical note... Please wait."):
            start_time = time.time()
            
            if prompting_method == "Zero-Shot":
                simplified_note = zero_shot_simplification(
                    medical_note, 
                    target_group=target_group,
                    model=model_choice,
                    temp=temperature
                )
            elif prompting_method == "Few-Shot (In-Context Learning)":
                simplified_note = few_shot_simplification(
                    medical_note, 
                    target_group=target_group,
                    model=model_choice,
                    temp=temperature
                )
            elif prompting_method == "Chain of Thought":
                simplified_note = chain_of_thought_simplification(
                    medical_note, 
                    target_group=target_group,
                    model=model_choice,
                    temp=temperature
                )
            else:  # Tree of Thoughts
                simplified_note = tree_of_thoughts_simplification(
                    medical_note, 
                    target_group=target_group,
                    model=model_choice,
                    temp=temperature
                )
            
            processing_time = time.time() - start_time
        
        if "Error:" in simplified_note:
            st.error(simplified_note)
        else:
            # Display simplified note
            st.markdown("### Simplified Medical Note")
            st.markdown(f"<div class='highlight'>{simplified_note.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
            
            # Calculate metrics
            readability_score = calculate_readability(simplified_note)
            original_readability = calculate_readability(medical_note)
            term_density = calculate_medical_term_density(simplified_note)
            original_term_density = calculate_medical_term_density(medical_note)
            simplified_words = len(re.findall(r'\b\w+\b', simplified_note))
            original_words = len(re.findall(r'\b\w+\b', medical_note))
            length_ratio = simplified_words / original_words if original_words > 0 else 0
            
            metrics = {
                "readability_score": readability_score,
                "original_readability": original_readability,
                "term_density": term_density,
                "original_term_density": original_term_density,
                "length_ratio": length_ratio,
                "processing_time": processing_time
            }
            
            # Save to history
            save_to_history(medical_note, simplified_note, prompting_method, target_group, metrics)
            
            # Display metrics
            st.markdown("### Evaluation Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Readability Score", 
                    f"{readability_score:.1f}/100", 
                    delta=f"{readability_score - original_readability:.1f}",
                    delta_color="normal"
                )
                st.caption("Higher is more readable. 70-80 is good for general audiences.")
            
            with col2:
                st.metric(
                    "Medical Term Density", 
                    f"{term_density:.1f}%", 
                    delta=f"{term_density - original_term_density:.1f}%", 
                    delta_color="inverse"
                )
                st.caption("Lower percentages indicate better simplification of terminology.")
            
            with col3:
                st.metric(
                    "Length Ratio", 
                    f"{length_ratio:.2f}", 
                    delta=None
                )
                st.caption("Ratio of simplified to original length. Target: 0.8-1.2")
            
            # Processing information
            st.markdown(f"Processing time: {processing_time:.2f} seconds")
            
            # Download option
            st.markdown(get_download_link(simplified_note), unsafe_allow_html=True)
            
            # Suggestion for improvement
            st.markdown("### Potential Improvements")
            
            if readability_score < 60:
                st.warning("The simplified note could be more readable. Consider using shorter sentences and simpler vocabulary.")
            elif term_density > 5:
                st.warning("The medical term density is still high. Further simplification of technical terms might be helpful.")
            else:
                st.success("The simplification looks good! The readability is improved, and medical terminology is well-simplified.")

elif st.session_state.current_tab == "results":
    # RESULTS EXPLORER TAB
    st.title("📊 Results Explorer")
    
    st.markdown("""
    <div class="highlight">
    Explore and compare the results of different simplification methods across your processing history.
    This analysis helps understand which approaches work best for different types of medical notes and patient groups.
    </div>
    """, unsafe_allow_html=True)
    
    # Check if we have any history
    if len(st.session_state.processing_history) == 0:
        st.warning("No processing history found. Try simplifying some medical notes in the Live Demo tab first.")
    else:
        # Summary statistics
        st.subheader("Summary Statistics")
        
        # Count by method
        methods = [item['method'] for item in st.session_state.processing_history]
        unique_methods = list(set(methods))
        method_counts = [methods.count(m) for m in unique_methods]
        
        # Average readability by method
        readability_by_method = {}
        term_density_by_method = {}
        processing_time_by_method = {}
        
        for method in unique_methods:
            scores = [item['metrics']['readability_score'] for item in st.session_state.processing_history if item['method'] == method]
            readability_by_method[method] = sum(scores) / len(scores) if scores else 0
            
            densities = [item['metrics']['term_density'] for item in st.session_state.processing_history if item['method'] == method]
            term_density_by_method[method] = sum(densities) / len(densities) if densities else 0
            
            times = [item['metrics']['processing_time'] for item in st.session_state.processing_history if item['method'] == method]
            processing_time_by_method[method] = sum(times) / len(times) if times else 0
        
        # Create DataFrame for easier visualization
        summary_df = pd.DataFrame({
            'Method': unique_methods,
            'Count': method_counts,
            'Avg. Readability': [readability_by_method[m] for m in unique_methods],
            'Avg. Term Density': [term_density_by_method[m] for m in unique_methods],
            'Avg. Processing Time': [processing_time_by_method[m] for m in unique_methods]
        })
        
        # Display summary table
        st.dataframe(summary_df.round(2))
        
        # Visualizations
        st.subheader("Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Readability comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(
                summary_df['Method'], 
                summary_df['Avg. Readability'],
                color=['#8884d8', '#82ca9d', '#ffc658', '#ff8042'][:len(unique_methods)]
            )
            ax.set_ylim(0, 100)
            ax.set_ylabel('Flesch Reading Ease Score')
            ax.set_title('Average Readability by Method')
            
            # Add target line
            ax.axhline(y=70, color='g', linestyle='--', alpha=0.7)
            ax.text(len(unique_methods)-0.5, 72, 'Target Readability (70)', color='g', ha='right')
            
            # Add values on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            st.pyplot(fig)
        
        with col2:
            # Term density comparison
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            bars2 = ax2.bar(
                summary_df['Method'], 
                summary_df['Avg. Term Density'],
                color=['#8884d8', '#82ca9d', '#ffc658', '#ff8042'][:len(unique_methods)]
            )
            ax2.set_ylim(0, max(summary_df['Avg. Term Density'])*1.2)
            ax2.set_ylabel('Medical Term Density (%)')
            ax2.set_title('Average Medical Term Density by Method')
            
            # Add target line
            ax2.axhline(y=5, color='g', linestyle='--', alpha=0.7)
            ax2.text(len(unique_methods)-0.5, 5.2, 'Target Density (5%)', color='g', ha='right')
            
            # Add values on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            st.pyplot(fig2)
        
        # Processing time comparison
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        bars3 = ax3.bar(
            summary_df['Method'], 
            summary_df['Avg. Processing Time'],
            color=['#8884d8', '#82ca9d', '#ffc658', '#ff8042'][:len(unique_methods)]
        )
        ax3.set_ylabel('Processing Time (seconds)')
        ax3.set_title('Average Processing Time by Method')
        
        # Add values on bars
        for bar in bars3:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}s',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        st.pyplot(fig3)
        
        # Detailed history table
        st.subheader("Processing History")
        
        # Prepare data for table
        history_table = []
        for i, item in enumerate(st.session_state.processing_history):
            history_table.append({
                'ID': i+1,
                'Timestamp': item['timestamp'],
                'Method': item['method'],
                'Target Group': item['target_group'],
                'Readability': f"{item['metrics']['readability_score']:.1f}",
                'Term Density': f"{item['metrics']['term_density']:.1f}%",
                'Length Ratio': f"{item['metrics']['length_ratio']:.2f}"
            })
        
        history_df = pd.DataFrame(history_table)
        st.dataframe(history_df)
        
        # Export options
        st.subheader("Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate PDF Report"):
                pdf_buffer = generate_report(st.session_state.processing_history)
                b64_pdf = base64.b64encode(pdf_buffer.read()).decode()
                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="simplification_report.pdf">Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            if st.button("Export Data (CSV)"):
                csv = history_df.to_csv(index=False)
                b64_csv = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64_csv}" download="simplification_history.csv">Download CSV Data</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        # Note example detail viewer
        st.subheader("View Example Details")
        
        selected_example = st.selectbox(
            "Select an example to view:", 
            [f"Example {i+1}: {item['method']} ({item['timestamp']})" for i, item in enumerate(st.session_state.processing_history)]
        )
        
        if selected_example:
            example_index = int(selected_example.split(':')[0].replace('Example ', '')) - 1
            example = st.session_state.processing_history[example_index]
            
            st.markdown(f"**Method:** {example['method']} | **Target Group:** {example['target_group']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Medical Note**")
                st.text_area("", value=example['original_note'], height=300, disabled=True)
            
            with col2:
                st.markdown("**Simplified Medical Note**")
                st.text_area("", value=example['simplified_note'], height=300, disabled=True)
            
            st.markdown("**Metrics:**")
            metrics = example['metrics']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Readability Improvement", 
                    f"{metrics['readability_score']:.1f}/100", 
                    delta=f"+{metrics['readability_score'] - metrics['original_readability']:.1f}"
                )
            
            with col2:
                st.metric(
                    "Term Density Reduction", 
                    f"{metrics['term_density']:.1f}%", 
                    delta=f"-{metrics['original_term_density'] - metrics['term_density']:.1f}%", 
                    delta_color="inverse"
                )
            
            with col3:
                st.metric("Length Ratio", f"{metrics['length_ratio']:.2f}")

elif st.session_state.current_tab == "code":
    # CODE EXAMPLES TAB
    st.title("📝 Code Examples")
    
    st.markdown("""
    <div class="highlight">
    This section provides practical code examples for implementing medical note simplification using various LLM prompting techniques.
    Use these examples as a starting point for your own implementations.
    </div>
    """, unsafe_allow_html=True)
    
    code_examples = load_code_examples()
    
    # Code example tabs
    code_tabs = st.tabs([
        "Zero-Shot Prompting", 
        "Few-Shot Prompting", 
        "Chain of Thought", 
        "Tree of Thoughts", 
        "Evaluation Methods"
    ])
    
    with code_tabs[0]:
        st.subheader("Zero-Shot Prompting Implementation")
        st.markdown("""
        The zero-shot approach gives the LLM direct instructions without examples. It relies on the model's pre-trained knowledge to simplify medical terminology.
        
        **Key Features:**
        - Simple to implement
        - Minimal prompt engineering
        - Works well for straightforward medical notes
        - Lowest token usage
        """)
        
        st.code(code_examples["zero_shot"], language="python")
        
        st.markdown("""
        **Best Practices:**
        
        1. Be very explicit about the target audience and reading level
        2. Include specific instructions about maintaining all important medical information
        3. Request explanations for necessary medical terms
        4. Specify the desired output format and structure
        """)
    
    with code_tabs[1]:
        st.subheader("Few-Shot (In-Context Learning) Implementation")
        st.markdown("""
        The few-shot approach provides examples of the transformation you want. This helps the LLM understand the pattern and style of simplification needed.
        
        **Key Features:**
        - Examples guide the model's output style
        - More consistent results than zero-shot
        - Better handling of specific medical terminology
        - Higher token usage due to examples
        """)
        
        st.code(code_examples["few_shot"], language="python")
        
        st.markdown("""
        **Best Practices:**
        
        1. Choose diverse but representative examples
        2. Include examples that cover various types of medical terms
        3. Keep example pairs brief but illustrative
        4. Make sure examples demonstrate the appropriate reading level for your target audience
        5. Consider creating specialized example sets for different patient populations
        """)
    
    with code_tabs[2]:
        st.subheader("Chain of Thought Implementation")
        st.markdown("""
        The Chain of Thought approach breaks down the simplification task into explicit steps, guiding the LLM through a systematic reasoning process.
        
        **Key Features:**
        - Step-by-step reasoning
        - More thorough identification of medical terms
        - Systematic approach to simplification
        - Higher token usage and processing time
        """)
        
        st.code(code_examples["chain_of_thought"], language="python")
        
        st.markdown("""
        **Best Practices:**
        
        1. Break down the task into logical, sequential steps
        2. Make the first step explicit identification of medical terms
        3. Include a step for determining what information must be preserved
        4. Add a final verification step to ensure all information is accurate
        5. For longer notes, consider processing sections separately
        """)
    
    with code_tabs[3]:
        st.subheader("Tree of Thoughts Implementation")
        st.markdown("""
        The Tree of Thoughts approach explores multiple simplification strategies, evaluates each, and selects the most effective one for the specific note.
        
        **Key Features:**
        - Multiple approaches evaluated
        - Adaptable to different note structures
        - Optimal balance of simplicity and completeness
        - Highest token usage and processing time
        """)
        
        st.code(code_examples["tree_of_thoughts"], language="python")
        
        st.markdown("""
        **Best Practices:**
        
        1. Define distinctly different approaches for the model to explore
        2. Include specific evaluation criteria for each approach
        3. Request an explicit selection of the best approach before generating output
        4. Consider the note's complexity when deciding if this method is worth the additional computation
        5. Use specific patient population needs as evaluation criteria
        """)
    
    with code_tabs[4]:
        st.subheader("Evaluation Methods")
        st.markdown("""
        These functions calculate key metrics for evaluating the quality of simplified medical notes.
        
        **Key Metrics:**
        - Flesch Reading Ease Score (readability)
        - Medical Term Density (terminology simplification)
        - Length Ratio (information preservation)
        """)
        
        st.code(code_examples["evaluation"], language="python")
        
        st.markdown("""
        **Implementation Notes:**
        
        1. For a production system, use a comprehensive medical terminology dictionary
        2. Consider implementing multiple readability formulas (Flesch-Kincaid, SMOG, etc.)
        3. Add domain-specific metrics like medical concept preservation
        4. For thorough evaluation, include human reviews by healthcare professionals and patient representatives
        """)
    
    # Complete application example
    st.subheader("Complete Streamlit Application Example")
    
    st.markdown("""
    The complete code for this tutorial application is available below. This includes all functionality demonstrated here.
    
    **Key Components:**
    
    1. Multiple prompting methods implementation
    2. Evaluation metrics calculation
    3. Interactive UI with Streamlit
    4. Result visualization and analysis
    5. Export functionality
    """)
    
    with st.expander("View Complete Application Code"):
        st.markdown("""
        The complete application code is quite extensive. To access it, you can:
        
        1. [View the GitHub repository](https://github.com/yourusername/medical-note-simplification)
        2. Download the source code using the button below
        """)
        
        if st.button("Download Complete Source Code"):
            # Create a zip file with the source code
            try:
                with open(__file__, 'r') as f:
                    source_code = f.read()
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zf:
                    zf.writestr('app.py', source_code)
                    zf.writestr('README.md', """
  zf.writestr('README.md', """
# Medical Note Simplification with LLMs

This Streamlit application demonstrates how to use LLMs to simplify medical notes for various patient populations.

## Features

- Multiple LLM prompting techniques (Zero-Shot, Few-Shot, Chain of Thought, Tree of Thoughts)
- Patient-specific simplification (General, Elderly, Low Literacy, ESL)
- Quantitative evaluation metrics
- Result visualization and analysis
- Export functionality

## Installation

```
pip install -r requirements.txt
```

## Usage

```
streamlit run app.py
```

## Requirements

- Python 3.7+
- OpenAI API key
- Streamlit
- Pandas
- Matplotlib
- NLTK
```
""")
                    zf.writestr('requirements.txt', """
streamlit>=1.22.0
openai>=0.27.0
pandas>=1.3.5
matplotlib>=3.5.1
seaborn>=0.11.2
nltk>=3.7
""")
                
                zip_buffer.seek(0)
                b64_zip = base64.b64encode(zip_buffer.getvalue()).decode()
                href = f'<a href="data:application/zip;base64,{b64_zip}" download="medical_note_simplification.zip">Download Source Code ZIP</a>'
                st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to create download: {str(e)}")

elif st.session_state.current_tab == "about":
    # ABOUT TAB
    st.title("ℹ️ About This Tutorial")
    
    st.markdown("""
    # LLM for Medical Notes Simplification Tutorial
    
    This interactive tutorial demonstrates how to leverage Large Language Models (LLMs) for healthcare communication, specifically focusing on simplifying medical notes for improved patient understanding.
    
    ## Purpose
    
    Medical notes are written for healthcare professionals, not patients. They often contain complex terminology, abbreviations, and clinical language that can be difficult for patients to understand. This communication barrier can lead to confusion, poor medication adherence, and reduced patient engagement.
    
    This tutorial shows how LLMs can help bridge this gap by transforming complex medical language into accessible, patient-friendly text while preserving critical clinical information.
    
    ## Features
    
    - **Interactive Demonstrations**: Test different LLM prompting methods in real-time
    - **Multiple Simplification Approaches**: Zero-shot, Few-shot, Chain of Thought, and Tree of Thoughts
    - **Patient-Specific Targeting**: Tailor simplifications for different patient populations
    - **Comprehensive Evaluation**: Quantitative metrics and qualitative analysis
    - **Practical Code Examples**: Ready-to-use implementations
    - **Result Analysis**: Visualization and comparison tools
    
    ## Data
    
    This tutorial uses Synthea - an open-source synthetic patient generator that creates realistic but not real patient data. This allows for demonstration with medical-like data without privacy concerns.
    
    ## Technologies
    
    - **LLM**: OpenAI GPT models (via API)
    - **Frontend**: Streamlit
    - **Data Processing**: Pandas, NLTK
    - **Visualization**: Matplotlib, Seaborn
    
    ## Educational Goals
    
    This tutorial aims to help you:
    
    1. Understand different LLM prompting techniques and their applications in healthcare
    2. Learn how to adapt language for different patient populations
    3. Develop methods to evaluate the effectiveness of medical language simplification
    4. Implement a practical application of LLMs in healthcare communication
    
    ## Citation and References
    
    If you use this tutorial in your research or applications, please cite:
    
    ```
    Joe (2025). LLM for Medical Notes Simplification Tutorial.
    ```
    
    ## Acknowledgments
    
    - Synthea for providing open-source synthetic patient data
    - OpenAI for LLM API access
    - Streamlit for the interactive web application framework
    
    ## Disclaimer
    
    This tutorial is for educational purposes only. While the simplified notes aim to be accurate, any real medical application should involve healthcare professionals in the review process. The simplifications generated should complement, not replace, direct communication between healthcare providers and patients.
    )
    
    # Contact information
    st.markdown("---")
    st.markdown("""
    ## Contact Information
    
    For questions, feedback, or collaboration opportunities, please contact:
    
    - **Email**: your.email@example.com
    - **GitHub**: [Your GitHub Profile](https://github.com/yourusername)
    
    We welcome contributions and suggestions for improving this tutorial.
    """)

# Run the app
if __name__ == '__main__':
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 Joe")
