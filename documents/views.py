from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from django.conf import settings
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import status
import os

from .models import CustomerDocumentUpload, PageAnalysis
from .serializers import CustomerDocumentUploadSerializer
from .utils import (
    split_pdf_into_pages, easyocr_text_from_pdf, paddle_ocr_text_from_pdf,
    classify_text_with_llm, merge_pdfs, check_page_quality, is_blurry, is_blank,
    check_ocr_completeness, llm_extract_fields_with_gemini, detect_cross_document_anomalies,
    llm_full_page_analysis
)

class CustomerDocumentUploadViewSet(viewsets.ModelViewSet):
    queryset = CustomerDocumentUpload.objects.all()
    serializer_class = CustomerDocumentUploadSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return self.queryset.filter(user=self.request.user)

    def perform_create(self, serializer):
        instance = serializer.save(user=self.request.user)

        original_pdf_path = instance.original_file.path
        output_folder = os.path.join(settings.MEDIA_ROOT, 'uploads', 'splits', f"{instance.id}")
        os.makedirs(output_folder, exist_ok=True)

        split_pages = split_pdf_into_pages(original_pdf_path, output_folder)

        # New logic: OCR + Full Gemini analysis once
        for idx, page_path in enumerate(split_pages):
            extracted_text = easyocr_text_from_pdf(page_path)
            analysis = llm_full_page_analysis(extracted_text)

            PageAnalysis.objects.create(
                document=instance,
                page_number=idx + 1,
                page_path=os.path.relpath(page_path, settings.MEDIA_ROOT),
                document_type=analysis.get('document_type', 'unknown'),
                extracted_fields=analysis.get('extracted_fields', {}),
                confidence_scores=analysis.get('confidence_scores', {}),
                missing_fields=analysis.get('missing_fields', []),
                ocr_text=extracted_text
            )

        instance.processed = True
        instance.save()
        print("\nUpload processed and saved successfully!\n")

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
            import traceback
            traceback.print_exc()
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=["get"], url_path="ocr-check")
    def ocr_check(self, request, pk=None):
        """
        Fetch OCR missing fields from pre-saved PageAnalysis instead of re-running OCR.
        """
        try:
            document = self.get_object()
            pages = document.pages.all()

            classified_pages = []
            for page in pages:
                classified_pages.append({
                    "page_path": page.page_path,
                    "document_type": page.document_type,
                    "missing_fields": page.missing_fields
                })

            return Response({
                "document_id": document.id,
                "file_name": document.original_file.name,
                "ocr_check_report": classified_pages
            }, status=status.HTTP_200_OK)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=["get"], url_path="anomaly-check")
    def anomaly_check(self, request, pk=None):
        """
        Perform anomaly detection from pre-saved extracted fields.
        """
        try:
            document = self.get_object()
            pages = document.pages.all()

            extracted_data = {}

            for page in pages:
                doc_type = page.document_type or f"unknown_page_{page.page_number}"
                extracted_data[doc_type] = page.extracted_fields or {}

            anomalies = detect_cross_document_anomalies(extracted_data)

            return Response({
                "document_id": document.id,
                "anomalies_detected": anomalies,
                "extracted_fields": extracted_data
            }, status=status.HTTP_200_OK)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=["get"], url_path="field-extraction")
    def field_extraction(self, request, pk=None):
        """
        Fetch detailed field extraction with confidence scores from pre-saved analysis.
        """
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
            import traceback
            traceback.print_exc()
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)