from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class CustomerDocumentUpload(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    original_file = models.FileField(upload_to='uploads/originals/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)

    # After classification, these fields will be populated
    payslip_file = models.FileField(upload_to='uploads/payslips/', null=True, blank=True)
    contract_file = models.FileField(upload_to='uploads/contracts/', null=True, blank=True)
    bank_statement_file = models.FileField(upload_to='uploads/bank_statements/', null=True, blank=True)
    id_proof_file = models.FileField(upload_to='uploads/id_proofs/', null=True, blank=True)
    p60_file = models.FileField(upload_to='uploads/p60_forms/', null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} - Uploaded on {self.uploaded_at}"
    
class PageAnalysis(models.Model):
    document = models.ForeignKey('CustomerDocumentUpload', on_delete=models.CASCADE, related_name="pages")
    page_number = models.IntegerField()
    page_path = models.CharField(max_length=500)
    document_type = models.CharField(max_length=100)
    extracted_fields = models.JSONField(null=True, blank=True)
    confidence_scores = models.JSONField(null=True, blank=True)
    missing_fields = models.JSONField(null=True, blank=True)
    ocr_text = models.TextField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Page {self.page_number} - {self.document_type} (DocID: {self.document.id})"