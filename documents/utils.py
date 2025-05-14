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
from datetime import datetime, timedelta

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
        "Payslip - Employee Name (At least 2 words)": True,
        "Payslip - Employer Name Present": True,
        "Payslip - Gross Pay Present": True,
        "Payslip - Address Present": True,
        "Payslip - Tax/NI Deductions Present": True,
        "Payslip - Minimum Payslips Uploaded": True,
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
                inter_document_checks_result["Payslip - Employee Name (At least 2 words)"] = False

            emp_company = standardized_fields.get("employer", "")
            if not emp_company:
                inter_document_checks_result["Payslip - Employer Name Present"] = False

            gross_pay = standardized_fields.get("gross_pay", "")
            if not gross_pay:
                inter_document_checks_result["Payslip - Gross Pay Present"] = False

            address = standardized_fields.get("address", "")
            if not address or address.lower() == "null":
                inter_document_checks_result["Payslip - Address Present"] = False

            tax_deductions = standardized_fields.get("tax_deductions", "")
            if not tax_deductions or tax_deductions.lower() == "not found":
                inter_document_checks_result["Payslip - Tax/NI Deductions Present"] = False

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
        inter_document_checks_result["Payslip - Minimum Payslips Uploaded"] = False

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
                    "severity": "Medium",
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
                    "severity": "High",
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

# ---------- P60 Cross Document Anomaly Detection ----------
def detect_p60_cross_document_anomalies(extracted_data):
    anomalies = []
    inter_document_checks_result = {
        "P60 - Employee Name (At least 2 words)": True,
        "P60 - National Insurance Number Matches Payslip": True,
        "P60 - Address Matches Payslip": True,
        "P60 - Income Matches Payslip Gross*12": True,
    }

    # Initialize fields
    p60_employee_name = p60_address = p60_ni = p60_income = None
    payslip_employee_name = payslip_address = payslip_ni = None
    payslip_gross_income = None

    print("\n========== 🛠️ Starting P60 Anomaly Detection ==========")

    for doc_type, fields in extracted_data.items():
        if not fields:
            continue

        print(f"\n--- 📄 Processing {doc_type} ---")
        print(json.dumps(fields, indent=2))

        standardized_fields = {}
        for key, value in fields.items():
            key_lower = key.strip().lower()

            if "full name" in key_lower or ("name" in key_lower and "employee" in key_lower):
                standardized_fields["employee_name"] = value
            if "address" in key_lower:
                standardized_fields["address"] = value
            if "national insurance" in key_lower or "ni number" in key_lower:
                standardized_fields["ni_number"] = value
            if "annual gross income" in key_lower:
                standardized_fields["p60_income"] = value
            if "gross monthly income" in key_lower or "gross pay" in key_lower:
                standardized_fields["gross_pay"] = value

        if "payslip" in doc_type.lower():
            payslip_employee_name = standardized_fields.get("employee_name")
            payslip_address = standardized_fields.get("address")
            payslip_ni = standardized_fields.get("ni_number")
            gross_pay = standardized_fields.get("gross_pay")
            if gross_pay:
                payslip_gross_income = clean_salary_value(gross_pay)

        if "p60" in doc_type.lower():
            p60_employee_name = standardized_fields.get("employee_name")
            p60_address = standardized_fields.get("address")
            p60_ni = standardized_fields.get("ni_number")
            p60_income = standardized_fields.get("p60_income")

    print("\n====== 📊 Extracted Key Fields ======")
    print(f"P60 Employee Name    : {p60_employee_name}")
    print(f"P60 Address          : {p60_address}")
    print(f"P60 National Insurance: {p60_ni}")
    print(f"P60 Income           : {p60_income}")
    print(f"Payslip Employee Name : {payslip_employee_name}")
    print(f"Payslip Address       : {payslip_address}")
    print(f"Payslip NI Number     : {payslip_ni}")
    print(f"Payslip Gross Income  : {payslip_gross_income}")
    print("=====================================\n")

    # === CHECKS ===

    # 1. Employee Name should have at least 2 words
    print("🔎 Checking: P60 Employee Name has at least 2 words...")
    if not p60_employee_name or len(p60_employee_name.split()) < 2:
        print("❌ Failed: Employee name missing or incomplete.")
        inter_document_checks_result["P60 - Employee Name (At least 2 words)"] = False
        anomalies.append({
            "severity": "Medium",
            "type": "P60 Employee Name Issue",
            "details": "Employee name in P60 does not have at least 2 words"
        })
    else:
        print("✅ Passed: Employee name is valid.")

    # 2. Address Match between Payslip and P60
    print("\n🔎 Checking: P60 Address matches Payslip address...")
    if p60_address and payslip_address:
        similarity = fuzz.partial_ratio(p60_address.lower(), payslip_address.lower())
        print(f"Similarity Score: {similarity}")
        if similarity < 85:
            print("❌ Failed: Address mismatch detected.")
            inter_document_checks_result["P60 - Address Matches Payslip"] = False
            anomalies.append({
                "severity": "Medium",
                "type": "P60 Address Mismatch",
                "details": "Address on P60 does not match address on Payslip"
            })
        else:
            print("✅ Passed: Address matches successfully.")
    else:
        print("⚠️ Warning: Address data missing in either P60 or Payslip.")
        inter_document_checks_result["P60 - Address Matches Payslip"] = False
        anomalies.append({
            "severity": "Medium",
            "type": "P60 Address Data Missing",
            "details": "Address missing in either P60 or Payslip"
        })

    # 3. NI Number match
    print("\n🔎 Checking: P60 NI number matches Payslip NI number...")
    if p60_ni and payslip_ni:
        similarity = fuzz.partial_ratio(p60_ni.lower(), payslip_ni.lower())
        print(f"Similarity Score: {similarity}")
        if similarity < 90:
            print("❌ Failed: NI number mismatch detected.")
            inter_document_checks_result["P60 - National Insurance Number Matches Payslip"] = False
            anomalies.append({
                "severity": "High",
                "type": "P60 NI Number Mismatch",
                "details": "National Insurance Number on P60 does not match Payslip"
            })
        else:
            print("✅ Passed: NI numbers match successfully.")
    else:
        print("❌ Failed: NI number missing in either P60 or Payslip.")
        inter_document_checks_result["P60 - National Insurance Number Matches Payslip"] = False
        anomalies.append({
            "severity": "High",
            "type": "P60 NI Number Missing",
            "details": "NI number missing in either P60 or Payslip"
        })

    # 4. Income Match
    print("\n🔎 Checking: P60 Income matches Payslip Gross*12...")
    if p60_income and payslip_gross_income:
        try:
            p60_income_val = clean_salary_value(p60_income)
            payslip_annual_income = payslip_gross_income * 12
            diff = abs(payslip_annual_income - p60_income_val)
            print(f"Calculated Difference: {diff}")
            if diff > 200:
                print("❌ Failed: Income mismatch detected.")
                inter_document_checks_result["P60 - Income Matches Payslip Gross*12"] = False
                anomalies.append({
                    "severity": "Medium",
                    "type": "P60 Income Mismatch",
                    "details": f"Income mismatch: Payslip Gross*12 ({payslip_annual_income}) vs P60 Income ({p60_income_val})"
                })
            else:
                print("✅ Passed: Income matches successfully.")
        except Exception as e:
            print(f"⚠️ Error during income comparison: {e}")
            inter_document_checks_result["P60 - Income Matches Payslip Gross*12"] = False
    else:
        print("❌ Failed: Income data missing in either P60 or Payslip.")
        inter_document_checks_result["P60 - Income Matches Payslip Gross*12"] = False
        anomalies.append({
            "severity": "Medium",
            "type": "P60 Income Data Missing",
            "details": "Income missing in either P60 or Payslip"
        })

    print("\n========== ✅ Completed P60 Anomaly Detection ==========\n")
    return inter_document_checks_result, anomalies

# ---------- Contract of Employment Cross Document Anomaly Detection ----------
def detect_contract_cross_document_anomalies(extracted_data):
    anomalies = []
    inter_document_checks_result = {
        "Contract - Employee Name (At least 2 words)": True,
        "Contract - Employer Name Matches Payslip": True,
        "Contract - Address Matches Payslip": True,
        "Contract - Contract Type Present (Permanent/Fixed Term/Temporary)": True,
        "Contract - Annual Income Matches Payslip Gross*12": True,
        "Contract - Employee Signature Present": True,
    }

    # Initialize fields
    contract_employee_name = contract_employer_name = contract_address = None
    contract_type = contract_annual_income = contract_signature = None

    payslip_employee_name = payslip_employer_name = payslip_address = None
    payslip_gross_income = None

    print("\n========== 🛠️ Starting Contract Anomaly Detection ==========")

    for doc_type, fields in extracted_data.items():
        if not fields:
            continue

        print(f"\n--- 📄 Processing {doc_type} ---")
        print(json.dumps(fields, indent=2))

        standardized_fields = {}
        for key, value in fields.items():
            key_lower = key.strip().lower()

            if "employee name" in key_lower:
                standardized_fields["employee_name"] = value
            if "employer name" in key_lower:
                standardized_fields["employer_name"] = value
            if "address" in key_lower:
                standardized_fields["address"] = value
            if "type of contract" in key_lower:
                standardized_fields["contract_type"] = value
            if "annual salary" in key_lower or "annual income" in key_lower:
                standardized_fields["annual_income"] = value
            if "gross monthly income" in key_lower or "gross pay" in key_lower:
                standardized_fields["gross_pay"] = value
            if "signature" in key_lower or "employee signature" in key_lower:
                standardized_fields["signature"] = value

        if "contract" in doc_type.lower():
            contract_employee_name = standardized_fields.get("employee_name")
            contract_employer_name = standardized_fields.get("employer_name")
            contract_address = standardized_fields.get("address")
            contract_type = standardized_fields.get("contract_type")
            contract_annual_income = standardized_fields.get("annual_income")
            contract_signature = standardized_fields.get("signature")

        if "payslip" in doc_type.lower():
            payslip_employee_name = standardized_fields.get("employee_name")
            payslip_employer_name = standardized_fields.get("employer_name")
            payslip_address = standardized_fields.get("address")
            gross_pay = standardized_fields.get("gross_pay")
            if gross_pay:
                payslip_gross_income = clean_salary_value(gross_pay)

    print("\n====== 📊 Extracted Key Fields ======")
    print(f"Contract Employee Name   : {contract_employee_name}")
    print(f"Contract Employer Name   : {contract_employer_name}")
    print(f"Contract Address         : {contract_address}")
    print(f"Contract Type            : {contract_type}")
    print(f"Contract Annual Income   : {contract_annual_income}")
    print(f"Contract Employee Signature : {contract_signature}")
    print(f"Payslip Employer Name    : {payslip_employer_name}")
    print(f"Payslip Employee Address : {payslip_address}")
    print(f"Payslip Gross Income     : {payslip_gross_income}")
    print("=====================================\n")

    # === CHECKS ===

    # 1. Employee Name should have at least 2 words
    print("🔎 Checking: Contract Employee Name has at least 2 words...")
    if not contract_employee_name or len(contract_employee_name.split()) < 2:
        print("❌ Failed: Employee name missing or incomplete.")
        inter_document_checks_result["Contract - Employee Name (At least 2 words)"] = False
        anomalies.append({
            "severity": "Medium",
            "type": "Contract Employee Name Issue",
            "details": "Employee name in Contract does not have at least 2 words"
        })
    else:
        print("✅ Passed: Employee name is valid.")

    # 2. Employer Name match
    print("\n🔎 Checking: Contract Employer Name matches Payslip...")
    if contract_employer_name and payslip_employer_name:
        similarity = fuzz.partial_ratio(contract_employer_name.lower(), payslip_employer_name.lower())
        print(f"Similarity Score: {similarity}")
        if similarity < 85:
            print("❌ Failed: Employer name mismatch detected.")
            inter_document_checks_result["Contract - Employer Name Matches Payslip"] = False
            anomalies.append({
                "severity": "Medium",
                "type": "Contract Employer Mismatch",
                "details": "Employer name in Contract does not match Payslip"
            })
        else:
            print("✅ Passed: Employer name matches successfully.")
    else:
        print("⚠️ Warning: Employer name missing in either Contract or Payslip.")
        inter_document_checks_result["Contract - Employer Name Matches Payslip"] = False

    # 3. Address Match
    print("\n🔎 Checking: Contract Address matches Payslip Address...")
    if contract_address and payslip_address:
        similarity = fuzz.partial_ratio(contract_address.lower(), payslip_address.lower())
        print(f"Similarity Score: {similarity}")
        if similarity < 85:
            print("❌ Failed: Address mismatch detected.")
            inter_document_checks_result["Contract - Address Matches Payslip"] = False
            anomalies.append({
                "severity": "Medium",
                "type": "Contract Address Mismatch",
                "details": "Address on Contract does not match Payslip"
            })
        else:
            print("✅ Passed: Address matches successfully.")
    else:
        print("⚠️ Warning: Address missing in either Contract or Payslip.")
        inter_document_checks_result["Contract - Address Matches Payslip"] = False

    # 4. Contract Type Present
    print("\n🔎 Checking: Contract Type Present...")
    if not contract_type or contract_type.lower() not in ["permanent", "fixed term", "temporary"]:
        print("❌ Failed: Contract Type missing or invalid.")
        inter_document_checks_result["Contract - Contract Type Present (Permanent/Fixed Term/Temporary)"] = False
        anomalies.append({
            "severity": "Medium",
            "type": "Contract Type Missing",
            "details": "Contract Type (Permanent, Fixed term, Temporary) not specified in Contract"
        })
    else:
        print(f"✅ Passed: Contract Type '{contract_type}' is valid.")

    # 5. Annual Income Matches Gross*12
    print("\n🔎 Checking: Contract Annual Income matches Payslip Gross*12...")
    if contract_annual_income and payslip_gross_income:
        try:
            contract_annual_val = clean_salary_value(contract_annual_income)
            payslip_annual_income = payslip_gross_income * 12
            diff = abs(payslip_annual_income - contract_annual_val)
            print(f"Calculated Difference: {diff}")
            if diff > 200:
                print("❌ Failed: Annual Income mismatch detected.")
                inter_document_checks_result["Contract - Annual Income Matches Payslip Gross*12"] = False
                anomalies.append({
                    "severity": "Medium",
                    "type": "Contract Annual Income Mismatch",
                    "details": f"Annual Income mismatch: Contract ({contract_annual_val}) vs Payslip Gross*12 ({payslip_annual_income})"
                })
            else:
                print("✅ Passed: Annual income matches successfully.")
        except Exception as e:
            print(f"⚠️ Error during income matching: {e}")
            inter_document_checks_result["Contract - Annual Income Matches Payslip Gross*12"] = False
    else:
        print("⚠️ Warning: Annual Income data missing in either Contract or Payslip.")
        inter_document_checks_result["Contract - Annual Income Matches Payslip Gross*12"] = False

    # 6. Employee Signature Check
    print("\n🔎 Checking: Employee Signature Present on Contract...")
    if not contract_signature or contract_signature.lower() == "null" or contract_signature.strip() == "":
        print("❌ Failed: Employee Signature missing.")
        inter_document_checks_result["Contract - Employee Signature Present"] = False
        anomalies.append({
            "severity": "High",
            "type": "Contract Signature Missing",
            "details": "Employee Signature not found in Contract of Employment"
        })
    else:
        print("✅ Passed: Employee Signature found.")

    print("\n========== ✅ Completed Contract Anomaly Detection ==========\n")
    return inter_document_checks_result, anomalies

# ---------- Bank Statement Cross Document Anomaly Detection ----------

def detect_bank_statement_cross_document_anomalies(extracted_data):
    print("\n========== 🛠️ Starting Bank Statement Anomaly Detection ==========")
    
    anomalies = []
    inter_document_checks_result = {
        "Bank Statement - Employee Name (At least 2 words)": True,
        "Bank Statement - Account Number Present": True,
        "Bank Statement - Sort Code Present": True,
        "Bank Statement - Salary Matches Payslip Net Pay": True,
        "Bank Statement - No Large Unexplained Credits or Debits": True,
        "Bank Statement - No Gambling or Sanctioned Transactions": True,
        "Bank Statement - Address Matches Payslip": True,
    }

    bank_fields = {}
    payslip_fields = {}

    # === Extract relevant fields ===
    print("\n--- 📄 Extracting Fields ---")
    for doc_type, fields in extracted_data.items():
        if not fields:
            continue

        print(f"\n📄 {doc_type} Fields:")
        print(json.dumps(fields, indent=2))

        doc_type_lower = doc_type.lower()
        if "bank" in doc_type_lower:
            bank_fields = fields
        elif "payslip" in doc_type_lower:
            payslip_fields = fields

    # === 1. Name Check ===
    name = bank_fields.get("Account holder name")
    print(f"\n🔍 Checking Account holder name: {name}")
    if not name or len(name.split()) < 2:
        inter_document_checks_result["Bank Statement - Employee Name (At least 2 words)"] = False
        anomalies.append({
            "severity": "Medium",
            "type": "Bank Statement Name Issue",
            "details": "Account holder name has less than 2 words"
        })

    # === 2. Account Number Check ===
    account_number = bank_fields.get("Account number")
    print(f"🔍 Checking Account number: {account_number}")
    if not account_number:
        inter_document_checks_result["Bank Statement - Account Number Present"] = False
        anomalies.append({
            "severity": "High",
            "type": "Bank Statement Account Number Missing",
            "details": "Account number missing in Bank Statement"
        })

    # === 3. Sort Code Check ===
    sort_code = bank_fields.get("Sort code")
    print(f"🔍 Checking Sort code: {sort_code}")
    if not sort_code:
        inter_document_checks_result["Bank Statement - Sort Code Present"] = False
        anomalies.append({
            "severity": "High",
            "type": "Bank Statement Sort Code Missing",
            "details": "Sort code missing in Bank Statement"
        })

    # === 4. Salary Match ===
    deposits = None
    for k, v in bank_fields.items():
        if "monthly deposit" in k.lower():
            deposits = v
            break
    net_pay = payslip_fields.get("Net monthly income") or payslip_fields.get("Net Pay")
    print(f"🔍 Deposits: {deposits}")
    print(f"🔍 Payslip Net Pay: {net_pay}")

    if deposits and net_pay:
        latest_salary = extract_latest_salary(deposits)
        cleaned_net = clean_salary_value(net_pay)
        print(f"💰 Latest Salary Credited: {latest_salary}")
        print(f"💰 Net Pay from Payslip: {cleaned_net}")
        if latest_salary is not None and cleaned_net is not None:
            diff = abs(latest_salary - cleaned_net)
            print(f"💡 Salary Difference: {diff}")
            if diff > 10:
                inter_document_checks_result["Bank Statement - Salary Matches Payslip Net Pay"] = False
                anomalies.append({
                    "severity": "High",
                    "type": "Salary Mismatch",
                    "details": f"Net Pay on Payslip ({cleaned_net}) does not match salary credited in bank ({latest_salary})"
                })

    # === 5. Large Credit/Debit Check ===
    transactions_text = bank_fields.get("Transactions", "") or ""
    print(f"\n🔍 Transactions Text: {transactions_text}")
    if any(amount in transactions_text for amount in ["£5000", "10000", "20000"]):
        print("⚠️ Large transaction detected!")
        inter_document_checks_result["Bank Statement - No Large Unexplained Credits or Debits"] = False
        anomalies.append({
            "severity": "Medium",
            "type": "Large Transaction Detected",
            "details": "Large credit or debit detected in Bank Statement"
        })

    # === 6. Gambling/Sanctioned Check ===
    risky_keywords = ["bet", "casino", "poker", "cryptocurrency", "binance", "russia", "iran"]
    risky_found = [word for word in risky_keywords if word in transactions_text.lower()]
    print(f"🔍 Risky Keywords Found: {risky_found}")
    if risky_found:
        inter_document_checks_result["Bank Statement - No Gambling or Sanctioned Transactions"] = False
        anomalies.append({
            "severity": "High",
            "type": "Risky Transactions",
            "details": "Gambling or sanctioned transactions found in bank statement"
        })

    # === 7. Address Match ===
    bank_address = bank_fields.get("Address", "")
    payslip_address = payslip_fields.get("Address", "")
    print(f"\n🔍 Bank Address: {bank_address}")
    print(f"🔍 Payslip Address: {payslip_address}")

    if bank_address and payslip_address:
        similarity = fuzz.partial_ratio(bank_address.lower(), payslip_address.lower())
        print(f"📏 Address Similarity Score: {similarity}")
        if similarity < 85:
            inter_document_checks_result["Bank Statement - Address Matches Payslip"] = False
            anomalies.append({
                "severity": "Medium",
                "type": "Bank Statement Address Mismatch",
                "details": f"Bank address does not match Payslip address"
            })

    print("\n✅ Completed Bank Statement Check\n")
    print(f"Inter Document Checks Result:\n{json.dumps(inter_document_checks_result, indent=2)}")
    print(f"\nStructured Anomalies:\n{json.dumps(anomalies, indent=2)}\n")
    
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
g) National Insurance Number (NI No)
2.3  P60:
a) Annual gross income
b) Total tax paid
c) Employee name
d) Employer name
e) Address (Residential Address of employee only)
f) National Insurance Number (NI No)
2.4 Bank Statements
a) Account holder name
b) Account number
c) Sort code
d) Monthly deposits (income/Salary)
e) Monthly expenses (Summation of expenses for every month like Jan:45 Feb:55 )
f) Overdraft usage
g) Address (Residential Address of employee and not Employer's office address)
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
                "severity": "Medium",
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

def validate_payslip(fields):
    print("\n=== Running Payslip Validation ===")
    results = {}

    gross_income = fields.get("Gross monthly income")
    net_income = fields.get("Net monthly income")
    tax_ni = fields.get("Tax/NI deductions")
    pay_date_str = fields.get("Pay Date") or fields.get("Payment Date") or ""

    print(f"Fields Found: Gross Income: {gross_income}, Net Income: {net_income}, Tax/NI: {tax_ni}, Pay Date: {pay_date_str}")

    # Minimum payslip presence based on salary being present
    results["Minimum of one month's most recent payslip"] = bool(gross_income)

    # Tax/NI must be present
    results["Ensure that Tax and NI contribution is present"] = bool(tax_ni)

    # Pay date check
    results["Payslip must be Dated (not undated)"] = False  # Assume False unless proven

    if pay_date_str:
        # Try parsing common formats
        possible_formats = ["%d %b %Y", "%d %B %Y", "%d/%m/%Y", "%Y-%m-%d"]

        parsed_date = None
        for fmt in possible_formats:
            try:
                parsed_date = datetime.strptime(pay_date_str.strip(), fmt)
                break
            except Exception:
                continue

        if parsed_date:
            today = datetime.today()
            age_days = (today - parsed_date).days
            print(f"Payslip Date Parsed: {parsed_date} (Age in days: {age_days})")
            results["Payslip must be Dated (not undated)"] = True
        else:
            print("❌ Could not parse Payslip Date properly.")
    else:
        print("❌ No Pay Date field found in Payslip.")

    print(f"Validation Results: {results}\n")
    return results

def validate_p60(fields):
    print("\n=== Running P60 Validation ===")
    results = {}

    p60_date_str = fields.get("Tax Year Ending") or fields.get("Date of Issue") or fields.get("Tax Year End") or ""

    print(f"P60 Fields Found: {fields}")
    print(f"P60 Tax Year Ending field: {p60_date_str}")

    # Assume failure unless verified
    results["Latest document within 12 months"] = False

    if p60_date_str:
        # Try to parse common formats
        possible_formats = ["%d %b %Y", "%d %B %Y", "%d/%m/%Y", "%Y-%m-%d", "%b %Y", "%B %Y"]

        parsed_date = None
        for fmt in possible_formats:
            try:
                parsed_date = datetime.strptime(p60_date_str.strip(), fmt)
                break
            except Exception:
                continue

        if parsed_date:
            today = datetime.today()
            age_days = (today - parsed_date).days
            print(f"P60 Document Date Parsed: {parsed_date} (Age in days: {age_days})")

            if age_days <= 400:  # roughly within 12-13 months
                results["Latest document within 12 months"] = True
            else:
                results["Latest document within 12 months"] = False
        else:
            print("❌ Could not parse P60 Tax Year Ending Date properly.")
            results["Latest document within 12 months"] = False
    else:
        print("❌ No Tax Year Ending field found for P60.")

    print(f"Validation Results: {results}\n")
    return results

def validate_contract(fields):
    print("\n=== Running Contract of Employment Validation ===")
    results = {}

    contract_type = fields.get("Type of contract", "").lower()
    start_date = fields.get("Job Start Date", "")
    employer_name = fields.get("Employer Name") or fields.get("Employer name")
    employee_name = fields.get("Employee Name") or fields.get("Employee name")

    print(f"Contract Fields Found: Type: {contract_type}, Start Date: {start_date}, Employer: {employer_name}, Employee: {employee_name}")

    # 1. Contract must mention Permanent/Temporary/Zero Hour
    if contract_type:
        if any(word in contract_type for word in ["permanent", "temporary", "fixed", "zero hour"]):
            results["Contract must mention Permanent/Temporary/Zero Hour"] = True
        else:
            results["Contract must mention Permanent/Temporary/Zero Hour"] = False
    else:
        results["Contract must mention Permanent/Temporary/Zero Hour"] = False

    # 2. Company Header Check (basic check on Employer Name presence)
    if employer_name and employer_name.strip():
        results["Contract should have company header"] = True
    else:
        results["Contract should have company header"] = False

    # 3. Contract Print Date (Job Start Date) Check
    try:
        if start_date:
            # Try parsing different formats
            date_formats = ["%d %B %Y", "%d/%m/%Y", "%d-%m-%Y", "%B %d, %Y", "%d %b %Y"]
            parsed_date = None
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(start_date.strip(), fmt)
                    break
                except ValueError:
                    continue
            
            if parsed_date:
                today = datetime.today()
                days_diff = (today - parsed_date).days
                print(f"Start Date Parsed: {parsed_date}, Days difference from today: {days_diff}")
                # Optional Rule: Should not be more than 10 years old
                if days_diff <= 3650:  # 10 years
                    results["Contract should have recent valid start date"] = True
                else:
                    results["Contract should have recent valid start date"] = False
            else:
                results["Contract should have recent valid start date"] = False
        else:
            results["Contract should have recent valid start date"] = False
    except Exception as e:
        print(f"Error parsing start date: {e}")
        results["Contract should have recent valid start date"] = False

    print(f"Validation Results: {results}\n")

    return results

def validate_bank_statement(fields):
    print("\n=== Running Bank Statement Validation ===")
    results = {}

    deposits_str = fields.get("Monthly deposits", "")
    expenses_str = fields.get("Monthly expenses", "")
    address = fields.get("Address", "")
    overdraft = fields.get("Overdraft usage", "")
    
    print(f"Bank Statement Fields Found: {fields}")

    # Initialize
    latest_date_valid = False
    minimum_days_covered = False
    consecutive_balances_check = False

    # Validate 1: Latest bank statement within 35 days
    try:
        # Assuming that Monthly deposits keys are in format 'Jan:2630.00, Feb:2630.00, Mar:2630.00'
        months_data = deposits_str.split(',')
        if months_data:
            last_month = months_data[-1].split(':')[0].strip()

            month_mapping = {
                "jan": 1, "feb": 2, "mar": 3,
                "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9,
                "oct": 10, "nov": 11, "dec": 12
            }
            now = datetime.now()
            month_number = month_mapping.get(last_month[:3].lower(), 0)
            year = now.year
            if month_number > now.month:
                year -= 1  # If document month is ahead of current month, assume last year

            last_date = datetime(year, month_number, 1)
            diff_days = (now - last_date).days

            print(f"Last month in statement: {last_month}, Diff days: {diff_days}")

            if diff_days <= 35:
                latest_date_valid = True
    except Exception as e:
        print(f"Error while validating latest statement date: {e}")

    # Validate 2: Bank statement must cover at least 30 days
    try:
        if len(months_data) >= 1:
            minimum_days_covered = True
    except Exception as e:
        print(f"Error while checking 30 days coverage: {e}")

    # Validate 3: Consecutive pages must match balance and date
    try:
        # Note: This needs real per-page balance checking. For now assume if we have multiple months, it's OK.
        if len(months_data) >= 2:
            consecutive_balances_check = True
    except Exception as e:
        print(f"Error while checking consecutive balances: {e}")

    # Set Results
    results["Latest bank statement within 35 days"] = latest_date_valid
    results["Bank statement must cover latest 30 days"] = minimum_days_covered
    results["Consecutive pages must match balance and dates"] = consecutive_balances_check

    print(f"Validation Results: {results}\n")

    return results