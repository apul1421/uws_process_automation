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


# ---------- Anomaly Detection (updated) ----------
def detect_cross_document_anomalies(extracted_data):
    anomalies = []

    # For Inter Document Checks (ticks and crosses)
    inter_document_checks_result = {
        "Employee Name (At least 2 words)": True,
        "Employer Name Present": True,
        "Gross Pay Present": True,
        "Address Present": True,
        "Tax/NI Deductions Present": True,
        "Minimum Payslips Uploaded": True,
    }

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
    payslip_found = False

    print("\n\n====== STARTING ANOMALY CHECK ======\n")

    for doc_type, fields in extracted_data.items():
        if not fields:
            continue

        print(f"--- Extracting Fields from {doc_type} ---")
        print(json.dumps(fields, indent=2))

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
            elif "monthly deposits" in key_lower:
                standardized_fields["bank_salary_credit"] = value
            elif "annual salary" in key_lower:
                standardized_fields["contract_annual_salary"] = value
            elif "annual gross income" in key_lower:
                standardized_fields["p60_income"] = value
            elif "tax/ni deductions" in key_lower or "tax deductions" in key_lower:
                standardized_fields["tax_deductions"] = value
            elif "date of birth" in key_lower:
                standardized_fields["dob"] = value

        # 🔥 If it's a Payslip, perform Inter Document Checks
        if 'payslip' in doc_type.lower():
            payslip_found = True

            # Inter document rules
            emp_name = standardized_fields.get("employee", "")
            if not emp_name or len(emp_name.split()) < 2:
                inter_document_checks_result["Employee Name (At least 2 words)"] = False

            emp_company = standardized_fields.get("employer", "")
            if not emp_company:
                inter_document_checks_result["Employer Name Present"] = False

            gross_pay = standardized_fields.get("gross_pay", "")
            if not gross_pay:
                inter_document_checks_result["Gross Pay Present"] = False

            address = standardized_fields.get("address", "")
            if not address or address.lower() == "null":
                inter_document_checks_result["Address Present"] = False

            tax_deductions = standardized_fields.get("tax_deductions", "")
            if not tax_deductions or tax_deductions.lower() == "not found":
                inter_document_checks_result["Tax/NI Deductions Present"] = False

        # Standardized field extraction
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
        if address and address.lower() != "null":
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
            cleaned_credit = extract_latest_salary(salary_credit)
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

        if tax_deductions and tax_deductions.lower() != "not found":
            payslip_tax_present = True

        if 'p60' in doc_type.lower() and dob:
            try:
                parsed_date = datetime.strptime(dob, "%d.%m.%Y")
                p60_dates.append(parsed_date)
            except Exception:
                pass

    # === Minimum Payslip Check ===
    if not payslip_found:
        inter_document_checks_result["Minimum Payslips Uploaded"] = False

    print("\n====== Aggregated Data ======")
    print(f"Names: {names}")
    print(f"Employers: {employers}")
    print(f"Addresses: {addresses}")
    print(f"Net Pays: {net_pays}")
    print(f"Gross Pays: {gross_pays}")
    print(f"Bank Credits: {bank_salary_credits}")
    print(f"Contract Salaries: {contract_salaries}")
    print(f"P60 Incomes: {p60_incomes}")
    print("====== End Aggregated Data ======\n")

    # === INTRA DOCUMENT (Cross document comparisons) ===

    # Name Consistency
    if names:
        base = names[0]
        for idx, other in enumerate(names[1:], start=1):
            similarity = fuzz.partial_ratio(base, other)
            print(f"Name similarity: {base} <-> {other} = {similarity}")
            if similarity < 90:
                anomalies.append({
                    "severity": "High",
                    "type": "Name Mismatch",
                    "details": f"{name_sources[0]} vs {name_sources[idx]}: {base} <-> {other}"
                })

    # Employer Consistency
    if employers:
        base = employers[0]
        for idx, other in enumerate(employers[1:], start=1):
            similarity = fuzz.partial_ratio(base, other)
            print(f"Employer similarity: {base} <-> {other} = {similarity}")
            if similarity < 90:
                anomalies.append({
                    "severity": "Medium",
                    "type": "Employer Mismatch",
                    "details": f"{employer_sources[0]} vs {employer_sources[idx]}: {base} <-> {other}"
                })

    # Address Consistency
    if addresses:
        base = addresses[0]
        for idx, other in enumerate(addresses[1:], start=1):
            if not base or base == "null" or not other or other == "null":
                continue
            similarity = fuzz.partial_ratio(base, other)
            print(f"Address similarity: {base} <-> {other} = {similarity}")
            if similarity < 80:
                anomalies.append({
                    "severity": "Medium",
                    "type": "Address Mismatch",
                    "details": f"{address_sources[0]} vs {address_sources[idx]}: {base} <-> {other}"
                })

    # Net Pay vs Bank Salary Credit
    if net_pays and bank_salary_credits:
        for net in net_pays:
            matches = [credit for credit in bank_salary_credits if abs(net - credit) <= 10]
            if not matches:
                anomalies.append({
                    "severity": "Medium",
                    "type": "Salary Mismatch",
                    "details": f"Net Pay {net} (Payslip) vs Bank Credits {','.join(map(str, bank_salary_credits))}"
                })

    # Gross Pay vs Contract Salary
    if gross_pays and contract_salaries:
        for gross in gross_pays:
            for salary in contract_salaries:
                diff = abs(gross * 12 - salary)
                print(f"Gross *12: {gross*12} vs Contract Salary: {salary}, Diff={diff}")
                if diff > 200:
                    anomalies.append({
                        "severity": "Medium",
                        "type": "Gross Salary Mismatch",
                        "details": "Gross *12 from Payslip does not match Contract Salary"
                    })

    # Gross Pay vs P60 Income
    if gross_pays and p60_incomes:
        for gross in gross_pays:
            for income in p60_incomes:
                diff = abs(gross * 12 - income)
                print(f"Gross *12: {gross*12} vs P60 Income: {income}, Diff={diff}")
                if diff > 200:
                    anomalies.append({
                        "severity": "Medium",
                        "type": "P60 Salary Mismatch",
                        "details": "Gross *12 from Payslip does not match P60 Income"
                    })

    # P60 Age Check
    if p60_dates:
        latest_p60 = max(p60_dates)
        diff_days = (datetime.now() - latest_p60).days
        print(f"P60 Latest Date: {latest_p60}, Age (days): {diff_days}")
        if diff_days > 400:
            anomalies.append({
                "severity": "Medium",
                "type": "P60 Expired",
                "details": "P60 document is older than 12 months"
            })

    print("\n====== FINAL STRUCTURED ANOMALIES ======")
    print(json.dumps(anomalies, indent=2))
    print("====== END ======\n")

    return inter_document_checks_result, anomalies

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
e) Address (Residential Address only, if available; do not extract Place of Birth))
2.2 Payslip
a) Employer name
b) Employee name
c) Gross monthly income
d) Net monthly income
e) Tax/NI deductions
f) Address (Residential Address of employee only)
2.3  P60:
a) Annual gross income
b) Total tax paid
c) Employee name
d) Employer name
e) Address (Residential Address of employee only)
2.4 Bank Statements
a) Account holder name
b) Monthly deposits (income/Salary)
c) Monthly expenses (Summation of expenses for every month like Jan:45 Feb:55 )
d) Overdraft usage
e) Address (Residential Address of employee and not Employer's office address)
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
    
def extract_latest_salary(deposits_str):
    """
    Extract latest month salary from monthly deposit string like 'Jan:2630.00, Feb:2630.00, Mar:2630.00'.
    """
    if not deposits_str:
        return None
    try:
        deposits = deposits_str.lower().split(',')
        if not deposits:
            return None
        last_entry = deposits[-1].strip()
        amount = last_entry.split(':')[-1].strip()
        return clean_salary_value(amount)
    except Exception as e:
        print(f"Error parsing monthly deposits: {e}")
        return None
    
def check_payslip_rules(payslip_fields, extracted_data):
    anomalies = []
    
    employee_name = payslip_fields.get("Employee Name") or payslip_fields.get("Employee name")
    employer_name = payslip_fields.get("Employer Name") or payslip_fields.get("Employer name")
    gross_income = payslip_fields.get("Gross monthly income") or payslip_fields.get("Gross Pay")
    net_income = payslip_fields.get("Net monthly income") or payslip_fields.get("Net Pay")
    tax_deductions = payslip_fields.get("Tax/NI deductions") or payslip_fields.get("Tax deductions")
    address = payslip_fields.get("Address")
    ytd_income = payslip_fields.get("YTD Income")  # Optional for future
    pay_date = payslip_fields.get("Pay Date")  # Optional for future

    # 1. Employee Name checks
    if not employee_name or len(employee_name.split()) < 2:
        anomalies.append({
            "severity": "High",
            "type": "Employee Name Missing or Invalid",
            "details": "Employee name must contain at least two words (first and last name)."
        })
    
    # Cross-verify name with ID Proof
    id_name = None
    for doc_type, fields in extracted_data.items():
        if 'id proof' in doc_type.lower() or 'passport' in doc_type.lower():
            id_name = fields.get("Full name")
            break
    if employee_name and id_name:
        similarity = fuzz.partial_ratio(employee_name.lower(), id_name.lower())
        if similarity < 90:
            anomalies.append({
                "severity": "High",
                "type": "Name Mismatch",
                "details": f"Employee name on Payslip ({employee_name}) does not match ID Proof ({id_name})."
            })

    # 2. Employer Name checks
    if not employer_name:
        anomalies.append({
            "severity": "High",
            "type": "Employer Name Missing",
            "details": "Employer name not found on payslip."
        })

    else:
        employer_match_found = False
        for doc_type, fields in extracted_data.items():
            if 'contract' in doc_type.lower() or 'p60' in doc_type.lower():
                doc_employer = fields.get("Employer Name") or fields.get("Employer name")
                if doc_employer:
                    similarity = fuzz.partial_ratio(employer_name.lower(), doc_employer.lower())
                    if similarity >= 90:
                        employer_match_found = True
                        break
        if not employer_match_found:
            anomalies.append({
                "severity": "High",
                "type": "Employer Mismatch",
                "details": "Employer name on Payslip does not match with other documents."
            })

    # 3. Gross Pay must be present
    if not gross_income:
        anomalies.append({
            "severity": "High",
            "type": "Gross Income Missing",
            "details": "Gross monthly income not found on payslip."
        })

    # 4. Address presence
    if not address or address.lower() == "null":
        anomalies.append({
            "severity": "Medium",
            "type": "Address Missing",
            "details": "Address not found on Payslip."
        })

    # 5. Tax/NI deductions must be present
    if not tax_deductions or tax_deductions.lower() == "null":
        anomalies.append({
            "severity": "High",
            "type": "Tax Deductions Missing",
            "details": "Tax/NI deductions not found on Payslip."
        })

    return anomalies


def generate_memo_from_fields(extracted_data):
    print("\n======= DEBUG: Memo Field Extraction Start =======")

    # Static fields
    buyer_type = "First Time Buyer"
    amount_to_be_borrowed = "£247,000"
    mortgage_term = "35 Years 0 Months"
    deposit_source = "Savings"
    deposit_amount = "£13,000"
    savings_and_investments = "£300.00"

    # Dynamic fields
    basic_pay = "Not Available"
    net_income = "Not Available"
    salary_credited = "Not Available"
    monthly_expenditure = "Not Available"

    for doc_type, fields in extracted_data.items():
        print(f"\n--- Extracting Fields from {doc_type} ---")
        print(json.dumps(fields, indent=2))

        doc_type_lower = doc_type.lower()

        if "payslip" in doc_type_lower:
            basic_pay = fields.get("Gross monthly income", basic_pay)
            net_income = fields.get("Net monthly income", net_income)

        elif "bank statement" in doc_type_lower:
            for k, v in fields.items():
                k_lower = k.strip().lower()
                if fuzz.partial_ratio(k_lower, "monthly deposits") > 85:
                    salary_credited = v
                elif fuzz.partial_ratio(k_lower, "monthly expenses") > 85:
                    monthly_expenditure = v

    memo_text = f"""
Lending Details:
a) Buyer type/Remortgage reason: {buyer_type}
b) Total amount to be borrowed: {amount_to_be_borrowed}
c) Mortgage term: {mortgage_term}

Deposit/HOL (Application summary > DepositDetail):
a) Source of deposit for mortgage: {deposit_source}
b) Deposit amount: {deposit_amount}

FOR STANDARD
- Was Income verified as declared: {'Y' if basic_pay != 'Not Available' else 'N'} – Basic Pay: {basic_pay}
- Outgoing verified as declared: {'Y' if monthly_expenditure != 'Not Available' else 'N'} – Monthly Expenditure: {monthly_expenditure}

Can you evidence customer's salary credit:
a) Total Personal Expenditure: {monthly_expenditure}
b) Savings and Investment: {savings_and_investments}
c) Salary Credited: {salary_credited}
    """.strip()

    print("\n======= DEBUG: Final Memo Text =======")
    print(memo_text)

    return memo_text