# (The first part remains the same)
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

import os
from collections import defaultdict
import traceback

from .models import CustomerDocumentUpload, PageAnalysis
from .serializers import CustomerDocumentUploadSerializer
from .utils import (
    split_pdf_into_pages,
    easyocr_text_from_pdf,
    merge_pdfs,
    check_page_quality,
    check_ocr_completeness,
    detect_cross_document_anomalies,
    llm_full_page_analysis,
    generate_memo_from_fields
)

# ====== Testing Flag ======
TESTING_MODE = False  # 🔵 Set to False for production

# Dummy document types to assign in testing mode
DUMMY_DOCUMENT_TYPES = [
    "ID Proof(Passport, Driving License)",
    "Bank Statement",
    "Contract of Employment",
    "Contract of Employment",
    "P60",
    "Payslip"
]

def normalize_document_type(doc_type):
    """
    Normalize document type variations to a standard.
    """
    doc_type = doc_type.lower()
    if "contract of employement" in doc_type or "contract of employment" in doc_type:
        return "Contract of Employment"
    if "id proof" in doc_type or "passport" in doc_type or "driving license" in doc_type:
        return "ID Proof(Passport, Driving License)"
    if "bank statement" in doc_type:
        return "Bank Statement"
    if "p60" in doc_type:
        return "P60"
    if "payslip" in doc_type:
        return "Payslip"
    return doc_type.title()

class CustomerDocumentUploadViewSet(viewsets.ModelViewSet):
    queryset = CustomerDocumentUpload.objects.all()
    serializer_class = CustomerDocumentUploadSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return self.queryset.filter(user=self.request.user)

    def perform_create(self, serializer):
        instance = serializer.save(user=self.request.user)

        print("\n🛠️  === STARTING DOCUMENT PROCESSING ===\n")

        original_pdf_path = instance.original_file.path
        output_folder = os.path.join(settings.MEDIA_ROOT, 'uploads', 'splits', f"{instance.id}")
        os.makedirs(output_folder, exist_ok=True)

        split_pages = split_pdf_into_pages(original_pdf_path, output_folder)

        page_data = []

        for idx, page_path in enumerate(split_pages):
            extracted_text = easyocr_text_from_pdf(page_path)

            if TESTING_MODE:
                document_type = DUMMY_DOCUMENT_TYPES[idx % len(DUMMY_DOCUMENT_TYPES)]
            else:
                analysis = llm_full_page_analysis(extracted_text)
                document_type = analysis.get('document_type', 'unknown')

            normalized_doc_type = normalize_document_type(document_type)

            print(f"📄 Page {idx+1} ({os.path.basename(page_path)}): Classified as ➡️  [{normalized_doc_type}]")

            page_data.append({
                "page_number": idx + 1,
                "page_path": os.path.relpath(page_path, settings.MEDIA_ROOT),
                "document_type": normalized_doc_type,
                "ocr_text": extracted_text,
            })

        # Group pages by normalized document type
        grouped_docs = defaultdict(list)
        for page in page_data:
            grouped_docs[page["document_type"]].append(page)

        print("\n📚 === GROUPED DOCUMENTS SUMMARY ===")
        for doc_type, pages in grouped_docs.items():
            page_names = [os.path.basename(p['page_path']) for p in pages]
            print(f"✅ {doc_type}: {len(pages)} page(s) -> {page_names}")
        print("======================================\n")

        # For each document type, merge text and call LLM once
        for document_type, pages in grouped_docs.items():
            combined_text = "\n\n".join([p["ocr_text"] for p in pages])

            if TESTING_MODE:
                analysis = {
                    "document_type": document_type,
                    "extracted_fields": {
                        "Full Name": "John Doe",
                        "Salary Amount": "5000"
                    },
                    "confidence_scores": {
                        "Full Name": 0.95,
                        "Salary Amount": 0.92
                    },
                    "missing_fields": []
                }
            else:
                analysis = llm_full_page_analysis(combined_text)

            for page in pages:
                PageAnalysis.objects.create(
                    document=instance,
                    page_number=page["page_number"],
                    page_path=page["page_path"],
                    document_type=document_type,
                    extracted_fields=analysis.get('extracted_fields', {}),
                    confidence_scores=analysis.get('confidence_scores', {}),
                    missing_fields=analysis.get('missing_fields', []),
                    ocr_text=page["ocr_text"]
                )

        instance.processed = True
        instance.save()

        print("\n✅ === UPLOAD PROCESSED AND SAVED SUCCESSFULLY ===\n")

    @action(detail=True, methods=["get"], url_path="quality-check")
    def quality_check(self, request, pk=None):
        try:
            document = self.get_object()
            quality_results = check_page_quality(document.original_file.path)

            return Response({
                "document_id": document.id,
                "file_name": document.original_file.name,
                "total_pages": len(quality_results),
                "quality_report": quality_results
            }, status=status.HTTP_200_OK)

        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=["get"], url_path="ocr-check")
    def ocr_check(self, request, pk=None):
        try:
            document = self.get_object()
            pages = document.pages.all()

            classified_pages = []
            for page in pages:
                classified_pages.append({
                    "page_path": page.page_path,
                    "document_type": page.document_type,
                    "missing_fields": page.missing_fields or []
                })

            return Response({
                "document_id": document.id,
                "file_name": document.original_file.name,
                "ocr_check_report": classified_pages
            }, status=status.HTTP_200_OK)

        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    
    @action(detail=True, methods=["get"], url_path="anomaly-check")
    def anomaly_check(self, request, pk=None):
        try:
            document = self.get_object()
            pages = document.pages.all()

            extracted_data = {}
            for page in pages:
                doc_type = page.document_type or f"unknown_page_{page.page_number}"
                extracted_data[doc_type] = page.extracted_fields or {}

            # === 🔥 New split here ===
            inter_document_checks_result, intra_document_anomalies = detect_cross_document_anomalies(extracted_data)

            main_document_type = pages.first().document_type if pages.exists() else "Unknown"

            return Response({
                "document_id": document.id,
                "document_type": main_document_type,
                "inter_document_checks": inter_document_checks_result,
                "intra_document_anomalies": intra_document_anomalies,
            }, status=status.HTTP_200_OK)

        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=["get"], url_path="field-extraction")
    def field_extraction(self, request, pk=None):
        try:
            document = self.get_object()
            pages = document.pages.all()

            extracted_results = []
            for page in pages:
                extracted_results.append({
                    "page_number": page.page_number,
                    "page_path": page.page_path,
                    "document_type": page.document_type,
                    "fields_extracted": page.extracted_fields or {},
                    "confidence_scores": page.confidence_scores or {}
                })

            return Response({
                "document_id": document.id,
                "file_name": document.original_file.name,
                "field_extraction_report": extracted_results
            }, status=status.HTTP_200_OK)

        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    @action(detail=True, methods=["get"], url_path="memo")
    def generate_memo(self, request, pk=None):
        try:
            document = self.get_object()
            pages = document.pages.all()

            # Organize data for memo generation
            extracted_data = {}
            for page in pages:
                doc_type = page.document_type or f"unknown_page_{page.page_number}"
                extracted_data[doc_type] = page.extracted_fields or {}

            # Call the utility function to build the memo
            memo_text = generate_memo_from_fields(extracted_data)

            return Response({
                "document_id": document.id,
                "memo_text": memo_text
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    @action(detail=True, methods=["get"], url_path="data-validation")
    def data_validation(self, request, pk=None):
        try:
            document = self.get_object()
            pages = document.pages.all()

            extracted_data = {}
            for page in pages:
                doc_type = page.document_type or f"unknown_page_{page.page_number}"
                extracted_data[doc_type] = page.extracted_fields or {}

            payslip_data = extracted_data.get("Payslip", {})
            contract_data = extracted_data.get("Contract of Employment", {})

            gross_monthly_str = payslip_data.get("Gross monthly income", "")
            contract_annual_str = contract_data.get("Annual Salary", "")

            from .utils import clean_salary_value  # make sure it's imported

            gross_monthly = clean_salary_value(gross_monthly_str)
            contract_annual = clean_salary_value(contract_annual_str)

            if gross_monthly is None or contract_annual is None:
                return Response({
                    "document_id": document.id,
                    "status": "Incomplete",
                    "note": "Missing gross monthly income or annual contract salary data."
                }, status=status.HTTP_200_OK)

            calculated_annual = round(gross_monthly * 12, 2)
            status_str = "Verified ✅" if abs(calculated_annual - contract_annual) < 100 else "Mismatch ❌"

            validation_note = (
                f"Total gross pay is £{gross_monthly:.2f}. When calculated annually, "
                f"it is £{calculated_annual:.2f}. Income on contract is £{contract_annual:.2f}. "
                f"Hence verified using contract. ({status_str})"
            )

            return Response({
                "document_id": document.id,
                "gross_monthly": f"£{gross_monthly:.2f}",
                "calculated_annual": f"£{calculated_annual:.2f}",
                "contract_annual": f"£{contract_annual:.2f}",
                "status": status_str,
                "note": validation_note
            }, status=status.HTTP_200_OK)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)