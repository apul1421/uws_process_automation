from rest_framework import serializers
from .models import CustomerDocumentUpload

class CustomerDocumentUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomerDocumentUpload
        fields = '__all__'
        read_only_fields = ['user', 'uploaded_at', 'processed',
                            'payslip_file', 'contract_file', 'bank_statement_file', 'id_proof_file', 'p60_file']