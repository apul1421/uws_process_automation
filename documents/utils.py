import fitz  # PyMuPDF
import os
import cv2
import numpy as np
# import pytesseract
import re
import requests
import json
import easyocr
from datetime import datetime
import google.generativeai as genai
from pdf2image import convert_from_path
from django.conf import settings
from rapidfuzz import fuzz
# from paddleocr import PaddleOCR
from PIL import Image, ImageFile
from datetime import datetime


# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- API Tokens ---
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

# Gemini API Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# --- OCR Engines ---
#ocr_engine = PaddleOCR(
#    use_angle_cls=True,
#    lang='en',
#    det_db_box_thresh=0.5,
#    use_gpu=False
#)
easyocr_reader = easyocr.Reader(['en'])

# --- Utility Functions ---

def split_pdf_into_pages(original_pdf_path, output_folder):
    doc = fitz.open(original_pdf_path)
    page_paths = []
    for page_number in range(doc.page_count):
        page = doc.load_page(page_number)
        single_page_doc = fitz.open()
        single_page_doc.insert_pdf(doc, from_page=page_number, to_page=page_number)
        output_path = os.path.join(output_folder, f"page_{page_number + 1}.pdf")
        single_page_doc.save(output_path)
        page_paths.append(output_path)
    return page_paths

def easyocr_text_from_pdf(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300, thread_count=1)
    full_text = ""
    for idx, page in enumerate(pages):
        temp_path = f"temp_page_{idx}.jpg"
        page.save(temp_path)

        try:
            img = Image.open(temp_path)
            img.verify()
            img = Image.open(temp_path)
            results = easyocr_reader.readtext(temp_path, detail=0)
            for line in results:
                full_text += line + "\n"
        except Exception as e:
            print(f"Skipping unreadable page {temp_path}: {e}")
            continue

    return full_text

#def paddle_ocr_text_from_pdf(pdf_path):
#    pages = convert_from_path(pdf_path, dpi=300)
#   full_text = ""
#    for idx, page in enumerate(pages):
#        temp_path = f"temp_page_{idx}.jpg"
#        page.save(temp_path)
#        result = ocr_engine.ocr(temp_path, cls=True)
#        for line in result[0]:
#            text_line = line[1][0]
#            full_text += text_line + "\n"
#    return full_text

def merge_pdfs(page_paths, output_path):
    merged_doc = fitz.open()
    for path in page_paths:
        single_doc = fitz.open(path)
        merged_doc.insert_pdf(single_doc)
    merged_doc.save(output_path)
    return output_path

def classify_text_with_llm(text, retries=3, wait_time=5):
    text_lower = text.lower()
    keyword_map = {
        "payslip": ["payslip", "net pay", "gross pay", "salary", "income tax"],
        "contract": ["employment contract", "job title", "employer", "position"],
        "bank_statement": ["account number", "sort code", "bank statement", "transaction"],
        "id proof": ["passport", "driving license", "identity card"],
        "p60": ["p60", "tax year ending", "hmrc"],
    }
    scores = {}
    for doc_type, keywords in keyword_map.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        scores[doc_type] = score

    best_match = max(scores, key=scores.get)
    highest_score = scores[best_match]

    if best_match == 'id proof' and highest_score >= 1:
        return best_match
    if highest_score >= 2:
        return best_match

    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": ["Payslip", "Contract", "Bank Statement", "ID Proof", "P60 Form"]
        }
    }
    for attempt in range(retries):
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        response_data = response.json()
        if 'labels' in response_data:
            return response_data['labels'][0].lower()
    return "unknown"

def check_page_quality(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300)
    results = []
    for idx, page in enumerate(pages):
        open_cv_image = np.array(page.convert('RGB'))
        image = open_cv_image[:, :, ::-1].copy()
        blurry, blur_score = is_blurry(image)
        blank, blank_score = is_blank(image)
        page_result = {
            "page": idx + 1,
            "blurry": blurry,
            "blur_score": blur_score,
            "blank": blank,
            "blank_score": blank_score,
        }
        results.append(page_result)
    return results

def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

def is_blank(image, threshold=0.99):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white_pixels = np.sum(gray > 245)
    total_pixels = gray.shape[0] * gray.shape[1]
    blank_ratio = white_pixels / total_pixels
    return blank_ratio > threshold, blank_ratio

mandatory_fields = {
    "payslip": {
        "net pay": ["net pay", "net salary", "net amount"],
        "gross pay": ["gross pay", "gross salary", "gross amount"],
        "employer": ["employer", "company name", "organization"],
    },
    "contract": {
        "job title": ["job title", "designation", "role"],
        "employer": ["employer", "company name", "organization"],
        "joining": ["joining date", "commencement of employment", "start date", "effective date"],
    },
    "id proof": {
        "name": ["name", "full name"],
        "date of birth": ["date of birth", "dob"],
        "id number": ["passport number", "driving license number", "id number"],
    },
    "bank_statement": {
        "account number": ["account number", "acc no", "acct no"],
        "sort code": ["sort code", "routing number"],
        "transaction": ["transaction", "payment", "debit", "credit"],
    },
    "p60": {
        "tax year": ["tax year", "year ending"],
        "national insurance": ["national insurance", "ni number"],
        "total pay": ["total pay", "total earnings"],
    }
}

def check_ocr_completeness(text, document_type):
    text_lower = text.lower()
    missing_fields = []
    expected_fields = mandatory_fields.get(document_type, {})
    for field, synonyms in expected_fields.items():
        found = False
        for synonym in synonyms:
            for line in text_lower.split("\n"):
                similarity = fuzz.partial_ratio(synonym, line)
                if similarity > 80:
                    found = True
                    break
            if found:
                break
        if not found:
            missing_fields.append(field)
    return missing_fields

def llm_extract_fields_with_gemini(document_text):
    prompt = f"""
Extract the following fields strictly as JSON:
- Full Name
- Employer Name
- Salary Amount
- Address
- Date of Birth

Document Text:
\"\"\"{document_text}\"\"\"
"""

    model = genai.GenerativeModel('gemini-1.5-flash')

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json"
            }
        )
        if response and response.text:
            parsed_output = json.loads(response.text)
            return parsed_output
        else:
            return {}
    except Exception as e:
        print(f"Error in Gemini LLM extraction: {e}")
        return {}

# ---------- Anomaly Detection ----------

def detect_cross_document_anomalies(extracted_data):
    anomalies = []

    names = []
    employers = []
    addresses = []
    net_pays = []
    gross_pays = []
    bank_salary_credits = []
    contract_salaries = []
    p60_incomes = []
    p60_dates = []

    name_sources = []
    employer_sources = []
    address_sources = []
    net_pay_sources = []
    salary_sources = []
    gross_pay_sources = []

    payslip_tax_present = False

    print("\n\n====== STARTING ANOMALY CHECK ======\n")

    for doc_type, fields in extracted_data.items():
        if not fields:
            continue

        print(f"--- Extracting Fields from {doc_type} ---")
        print(json.dumps(fields, indent=2))

        # Normalize extracted field names
        standardized_fields = {}
        for key, value in fields.items():
            key_lower = key.strip().lower()

            if "name" in key_lower and "employer" in key_lower:
                standardized_fields["employer"] = value
            elif "name" in key_lower and "employee" in key_lower:
                standardized_fields["employee"] = value
            elif "name" in key_lower and "account" in key_lower:
                standardized_fields["employee"] = value
            elif "full name" in key_lower:
                standardized_fields["employee"] = value
            elif "address" in key_lower:
                standardized_fields["address"] = value
            elif "net monthly income" in key_lower or "net pay" in key_lower:
                standardized_fields["net_pay"] = value
            elif "gross monthly income" in key_lower or "gross pay" in key_lower:
                standardized_fields["gross_pay"] = value
            elif "monthly deposits" in key_lower or "salary" in key_lower:
                standardized_fields["bank_salary_credit"] = value
            elif "annual salary" in key_lower:
                standardized_fields["contract_annual_salary"] = value
            elif "annual gross income" in key_lower:
                standardized_fields["p60_income"] = value
            elif "tax/ni deductions" in key_lower or "tax deductions" in key_lower:
                standardized_fields["tax_deductions"] = value
            elif "date of birth" in key_lower:
                standardized_fields["dob"] = value

        # Now extract standardized fields
        name = standardized_fields.get("employee")
        employer = standardized_fields.get("employer")
        address = standardized_fields.get("address")
        net_pay = standardized_fields.get("net_pay")
        gross_pay = standardized_fields.get("gross_pay")
        salary_credit = standardized_fields.get("bank_salary_credit")
        annual_salary = standardized_fields.get("contract_annual_salary")
        total_pay = standardized_fields.get("p60_income")
        tax_deductions = standardized_fields.get("tax_deductions")
        dob = standardized_fields.get("dob")

        if name:
            names.append(name.lower())
            name_sources.append(doc_type)
        if employer:
            employers.append(employer.lower())
            employer_sources.append(doc_type)
        if address:
            addresses.append(address.lower())
            address_sources.append(doc_type)
        if net_pay:
            cleaned_net = clean_salary_value(net_pay)
            if cleaned_net is not None:
                net_pays.append(cleaned_net)
                net_pay_sources.append(doc_type)
        if gross_pay:
            cleaned_gross = clean_salary_value(gross_pay)
            if cleaned_gross is not None:
                gross_pays.append(cleaned_gross)
                gross_pay_sources.append(doc_type)
        if salary_credit:
            cleaned_credit = clean_salary_value(salary_credit)
            if cleaned_credit is not None:
                bank_salary_credits.append(cleaned_credit)
                salary_sources.append(doc_type)
        if total_pay:
            cleaned_total = clean_salary_value(total_pay)
            if cleaned_total is not None:
                p60_incomes.append(cleaned_total)
        if annual_salary:
            salary_value = clean_salary_value(annual_salary)
            if salary_value is not None:
                contract_salaries.append(salary_value)

        if tax_deductions and tax_deductions != "Not found":
            payslip_tax_present = True

        if doc_type.lower() == 'p60' and dob:
            try:
                parsed_date = datetime.strptime(dob, "%d.%m.%Y")
                p60_dates.append(parsed_date)
            except Exception as e:
                pass

    print("\n====== Aggregated Data ======")
    print(f"Names: {names}")
    print(f"Employers: {employers}")
    print(f"Addresses: {addresses}")
    print(f"Net Pays: {net_pays}")
    print(f"Gross Pays: {gross_pays}")
    print(f"Bank Credits: {bank_salary_credits}")
    print(f"Contract Salaries: {contract_salaries}")
    print(f"P60 Incomes: {p60_incomes}")
    print(f"P60 Dates: {p60_dates}")
    print("====== End Aggregated Data ======\n")

    # --- Anomaly Checks ---

    # Name Consistency
    if names:
        for idx, other in enumerate(names[1:], start=1):
            similarity = fuzz.partial_ratio(names[0], other)
            print(f"Name similarity: {names[0]} <-> {other} = {similarity}")
            if similarity < 90:
                anomalies.append(f"Mismatch in Customer Name between {name_sources[0]} and {name_sources[idx]}.")

    # Employer Consistency
    if employers:
        for idx, other in enumerate(employers[1:], start=1):
            similarity = fuzz.partial_ratio(employers[0], other)
            print(f"Employer similarity: {employers[0]} <-> {other} = {similarity}")
            if similarity < 90:
                anomalies.append(f"Mismatch in Employer Name between {employer_sources[0]} and {employer_sources[idx]}.")

    # Address Consistency
    if addresses:
        for idx, other in enumerate(addresses[1:], start=1):
            similarity = fuzz.partial_ratio(addresses[0], other)
            print(f"Address similarity: {addresses[0]} <-> {other} = {similarity}")
            if similarity < 80:
                anomalies.append(f"Mismatch in Address between {address_sources[0]} and {address_sources[idx]}.")

    # Net Pay vs Bank Salary Credit
    if net_pays and bank_salary_credits:
        for idx, net in enumerate(net_pays):
            matched = False
            for jdx, credit in enumerate(bank_salary_credits):
                # Only consider reasonable salary credits
                if 500 <= credit <= 10000:
                    if abs(net - credit) <= 10:  # tolerance 10 GBP
                        matched = True
                    else:
                        anomalies.append(
                            f"Mismatch: Net Pay {net} from {net_pay_sources[idx]} vs Bank Credit {credit} from {salary_sources[jdx]}."
                        )
            if not matched:
                anomalies.append(f"No matching salary credit found in Bank Statement for Net Pay {net} from {net_pay_sources[idx]}.")

    # Gross Pay vs Contract Salary
    if gross_pays and contract_salaries:
        for idx, gross in enumerate(gross_pays):
            for salary in contract_salaries:
                diff = abs(gross * 12 - salary)
                print(f"Gross * 12: {gross*12} vs Contract Salary: {salary}, Diff={diff}")
                if diff > 200:
                    anomalies.append(f"Gross *12 from {gross_pay_sources[idx]} does not match Contract Annual Salary.")

    # Gross Pay vs P60 Income
    if gross_pays and p60_incomes:
        for idx, gross in enumerate(gross_pays):
            for p60_income in p60_incomes:
                diff = abs(gross * 12 - p60_income)
                print(f"Gross * 12: {gross*12} vs P60 Income: {p60_income}, Diff={diff}")
                if diff > 200:
                    anomalies.append(f"Gross *12 from {gross_pay_sources[idx]} does not match P60 Annual Income.")

    # Tax Deductions
    print(f"Payscale Tax Present: {payslip_tax_present}")
    if not payslip_tax_present:
        anomalies.append("Tax/NI Deductions not found on Payslip.")

    # P60 recency
    if p60_dates:
        latest_p60 = max(p60_dates)
        diff_days = (datetime.now() - latest_p60).days
        print(f"P60 Latest Date: {latest_p60}, Age (days): {diff_days}")
        if diff_days > 400:
            anomalies.append("P60 document is older than 12 months.")

    # Large suspicious salary credits
    for idx, credit in enumerate(bank_salary_credits):
        if credit > 10000:  # annual income wrongly tagged as salary
            anomalies.append(f"Unusually large salary credit ({credit}) detected from {salary_sources[idx]}. Please review manually.")

    print("\n====== FINAL ANOMALIES DETECTED ======")
    print(json.dumps(anomalies, indent=2))
    print("====== END ======\n")

    return anomalies

def llm_full_page_analysis(document_text):
    """
    Use Gemini to classify, extract fields, check missing fields, and confidence scores in one go.
    """
    prompt = f"""
You are an intelligent document analyst part of the underwriting team in a bank. Based on the provided document text:

Tasks:
1. Identify document type (Payslip, Contract of Employment, Bank Statement, ID Proof(Passport, Driving License), P60, Unknown).
2. Extract relevant fields according to document type.
2.1 Passport/ Driving License:
a) Full name
b) Date of birth
c) Passport/Driving License number
d) Expiry date
e) Address
2.2 Payslip
a) Employer name
b) Employee name
c) Gross monthly income
d) Net monthly income
e) Tax/NI deductions
f) Address
2.3  P60:
a) Annual gross income
b) Total tax paid
c) Employee name
d) Employer name
e) Address
2.4 Bank Statements
a) Account holder name
b) Monthly deposits (income/Salary)
c) Monthly expenses (Summation of expenses for every month)
d) Overdraft usage
e) Address
2.5 Contract of Employement
a) Employee name
b) Employer name
c) Type of contract (Permanent if nothing is mentioned the contract is permanent] or Fixed term[Fixed term contract will mention the temporary terms of contract])
d) Job Start Date
e) Annual Salary
f) Address
3. Identify missing fields from the relevent field list.
4. Provide confidence scores (0 to 1) for each extracted field.

Return strictly as JSON in this format:
{{
  "document_type": "payslip",
  "extracted_fields": {{
    "Employer Name": "...",
    "Employee Name": "..."
  }},
  "missing_fields": ["Net Pay", "Tax Deductions"],
  "confidence_scores": {{
    "Employer Name": 0.95,
    "Employee Name": 0.90
  }}
}}

Document Text:
\"\"\"
{document_text}
\"\"\"
"""

    model = genai.GenerativeModel('gemini-2.0-flash-lite')

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json"
            }
        )
        if response and response.text:
            parsed_output = json.loads(response.text)
            return parsed_output
        else:
            return {}
    except Exception as e:
        print(f"Error in Full Page Analysis Gemini call: {e}")
        return {}
    
    # ---------- Salary and Amount Cleaning ----------

def clean_salary_value(value):
    """
    Clean and extract a float salary from noisy OCR extracted text.
    Handles £, E, commas, spaces, 'per month' like suffixes etc.
    """
    if not value:
        return None

    value_str = str(value).lower()

    # Replace OCR mistakes
    value_str = value_str.replace('e', '').replace('£', '').replace(',', '').strip()

    # Remove any non-digit characters after number
    value_str = re.split(r'\s+', value_str)[0]

    try:
        return float(value_str)
    except Exception as e:
        print(f"Error parsing salary: {value} -> {e}")
        return None