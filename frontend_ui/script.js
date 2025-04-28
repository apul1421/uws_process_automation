const backendUrl = "http://127.0.0.1:8000/api/v1";
const authToken = "8b065598c06c59ae0f2398907add571604058167";

let uploadedDocuments = [];
let currentDocumentId = null;

// Upload multiple files
document.getElementById("uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const files = document.getElementById("fileInput").files;
  if (files.length === 0) {
    alert("Please select at least one file!");
    return;
  }
  for (let file of files) {
    await uploadSingleFile(file);
  }
});

// Upload single file
async function uploadSingleFile(file) {
  const formData = new FormData();
  formData.append("original_file", file);

  document.getElementById("uploadStatus").innerText = `Uploading ${file.name}...`;

  try {
    const response = await fetch(`${backendUrl}/documents/`, {
      method: "POST",
      headers: { "Authorization": `Token ${authToken}` },
      body: formData
    });

    if (!response.ok) throw new Error("Upload failed!");

    const data = await response.json();
    uploadedDocuments.push({ id: data.id, name: file.name });
    updateUploadedFilesList();
    document.getElementById("uploadStatus").innerText = "Upload successful!";
  } catch (error) {
    console.error(error);
    alert(`Failed to upload ${file.name}`);
  }
}

// Sidebar list
function updateUploadedFilesList() {
  const list = document.getElementById("uploadedFilesList");
  list.innerHTML = "";

  uploadedDocuments.forEach(doc => {
    const li = document.createElement("li");
    li.className = "list-group-item";
    li.innerHTML = `${doc.name}`;
    li.onclick = () => {
      currentDocumentId = doc.id;
      loadAllTabResults();
    };
    list.appendChild(li);
  });
}

// Load all tabs
function loadAllTabResults() {
  fetchAnomalies();
  fetchOCRCheck();
  fetchFieldExtraction();
  fetchQualityReport();
  fetchMemo();
  fetchDataValidation();
}

// API Caller
async function callApi(endpoint) {
  try {
    const res = await fetch(`${backendUrl}${endpoint}`, {
      headers: { "Authorization": `Token ${authToken}` }
    });
    if (!res.ok) throw new Error("API call failed");
    return await res.json();
  } catch (error) {
    console.error(error);
    return {};
  }
}

// Set loading spinner
function setLoading(tabId) {
  const container = document.getElementById(`${tabId}Content`);
  container.innerHTML = `<div class="loading-spinner"></div>`;
}

// Fetch for each tab
async function fetchAnomalies() {
  if (!currentDocumentId) return;
  setLoading("anomaly");
  const data = await callApi(`/documents/${currentDocumentId}/anomaly-check/`);
  renderAnomalies(data);
}

async function fetchOCRCheck() {
  if (!currentDocumentId) return;
  setLoading("ocr");
  const data = await callApi(`/documents/${currentDocumentId}/ocr-check/`);
  renderOCRCheck(data);
}

async function fetchFieldExtraction() {
  if (!currentDocumentId) return;
  setLoading("field");
  const data = await callApi(`/documents/${currentDocumentId}/field-extraction/`);
  renderFieldExtraction(data);
}

async function fetchQualityReport() {
  if (!currentDocumentId) return;
  setLoading("quality");
  const data = await callApi(`/documents/${currentDocumentId}/quality-check/`);
  renderQualityReport(data);
}

function renderAnomalies(data) {
    const container = document.getElementById("anomalyContent");
    container.innerHTML = "";
  
    const { document_type, inter_document_checks, intra_document_anomalies } = data;
  
    // === DOCUMENT NAME Heading ===
    const docHeading = document.createElement("h4");
    docHeading.className = "mb-3";
    docHeading.innerHTML = `📄 <strong>Document:</strong> ${document_type || "Unknown"}`;
    container.appendChild(docHeading);
  
    // === INTER DOCUMENT CHECKS ===
    const interHeading = document.createElement("h5");
    interHeading.innerHTML = "✅ Inter Document Checks";
    container.appendChild(interHeading);
  
    const interList = document.createElement("ul");
    interList.className = "list-group";
  
    for (let check in inter_document_checks) {
      const li = document.createElement("li");
      li.className = "list-group-item d-flex justify-content-between align-items-center";
  
      const checkPassed = inter_document_checks[check];
      li.innerHTML = `
        <span>${check}</span>
        <span class="badge ${checkPassed ? 'bg-success' : 'bg-danger'}">
          ${checkPassed ? '✔️ Passed' : '❌ Failed'}
        </span>
      `;
      interList.appendChild(li);
    }
  
    container.appendChild(interList);
  
    // === INTRA DOCUMENT ANOMALIES ===
    const intraHeading = document.createElement("h5");
    intraHeading.className = "mt-4";
    intraHeading.innerHTML = "⚠️ Intra Document Anomalies";
    container.appendChild(intraHeading);
  
    if (!intra_document_anomalies || intra_document_anomalies.length === 0) {
      const noAnomalies = document.createElement("p");
      noAnomalies.className = "text-success";
      noAnomalies.innerHTML = "✅ No intra-document anomalies detected!";
      container.appendChild(noAnomalies);
    } else {
      const anomalyList = document.createElement("ul");
      anomalyList.className = "list-group";
  
      intra_document_anomalies.forEach(anomaly => {
        const li = document.createElement("li");
        li.className = "list-group-item list-group-item-warning";
        li.innerHTML = `
          🟠 <strong>${anomaly.severity} Severity</strong><br>
          <strong>${anomaly.type}:</strong> ${anomaly.details}
        `;
        anomalyList.appendChild(li);
      });
  
      container.appendChild(anomalyList);
    }
  }

// Render OCR Check
function renderOCRCheck(data) {
  const container = document.getElementById("ocrContent");
  container.innerHTML = "";

  const expectedDocs = [
    "ID Proof(Passport, Driving License)",
    "Bank Statement",
    "Contract of Employment",
    "Payslip",
    "P60"
  ];

  const foundDocs = data.ocr_check_report.map(page => page.document_type);

  const list = document.createElement("ul");
  list.className = "list-group";

  expectedDocs.forEach(docType => {
    const li = document.createElement("li");
    if (foundDocs.includes(docType)) {
      li.className = "list-group-item list-group-item-success";
      li.innerHTML = `✅ ${docType}`;
    } else {
      li.className = "list-group-item list-group-item-danger";
      li.innerHTML = `❌ ${docType}`;
    }
    list.appendChild(li);
  });

  container.appendChild(list);
}

// Render Field Extraction (grouped properly)
function renderFieldExtraction(data) {
  const container = document.getElementById("fieldContent");
  container.innerHTML = "";

  const grouped = {};

  data.field_extraction_report.forEach(page => {
    const docType = page.document_type;

    if (!grouped[docType]) {
      grouped[docType] = { fields: {}, confidences: {} };
    }

    const fields = page.fields_extracted || {};
    const scores = page.confidence_scores || {};

    for (let key in fields) {
      if (!grouped[docType].fields[key] || fields[key]) {
        grouped[docType].fields[key] = fields[key];
        grouped[docType].confidences[key] = scores[key];
      }
    }
  });

  Object.keys(grouped).forEach(docType => {
    const section = document.createElement("div");
    section.className = "mb-4";

    const heading = document.createElement("h5");
    heading.textContent = `${docType}`;
    section.appendChild(heading);

    const list = document.createElement("ul");
    list.className = "list-group";

    const fields = grouped[docType].fields;
    const scores = grouped[docType].confidences;

    for (let field in fields) {
      const value = fields[field];
      const confidence = scores[field] !== undefined ? (scores[field] * 100).toFixed(1) : "N/A";

      const li = document.createElement("li");
      li.className = "list-group-item";

      if (!value || value === "null" || value === null) {
        li.innerHTML = `<span class="field-missing">${field}: Not Present</span> [Confidence: ${confidence}%]`;
      } else {
        li.innerHTML = `<span class="field-present">${field}: ${value}</span> [Confidence: ${confidence}%]`;
      }
      list.appendChild(li);
    }

    section.appendChild(list);
    container.appendChild(section);
  });
}

// Render Quality Report
function renderQualityReport(data) {
  const container = document.getElementById("qualityContent");
  container.innerHTML = "";

  if (!data.quality_report || data.quality_report.length === 0) {
    container.innerHTML = "<p>No quality report available.</p>";
    return;
  }

  const pageToDocument = {
    1: "ID Proof",
    2: "Bank Statement",
    3: "Contract of Employment",
    4: "Contract of Employment",
    5: "P60",
    6: "Payslip"
  };

  const table = document.createElement("table");
  table.className = "table table-striped table-bordered";

  const thead = document.createElement("thead");
  thead.innerHTML = `
    <tr>
      <th>Document</th>
      <th>Blurry</th>
      <th>Blur Quality</th>
      <th>Blank</th>
      <th>Blank Quality</th>
    </tr>
  `;
  table.appendChild(thead);

  const tbody = document.createElement("tbody");

  data.quality_report.forEach(page => {
    const tr = document.createElement("tr");

    const blurQuality = scoreToQuality(page.blur_score);
    const blankQuality = scoreToQuality(page.blank_score * 100);

    tr.innerHTML = `
      <td>${pageToDocument[page.page] || `Page ${page.page}`}</td>
      <td class="${page.blurry ? 'text-danger' : 'text-success'}">${page.blurry ? 'Yes' : 'No'}</td>
      <td>${blurQuality}</td>
      <td class="${page.blank ? 'text-danger' : 'text-success'}">${page.blank ? 'Yes' : 'No'}</td>
      <td>${blankQuality}</td>
    `;
    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  container.appendChild(table);
}

// Helper to convert score to label
function scoreToQuality(score) {
  if (score >= 90) return "Excellent";
  if (score >= 70) return "Good";
  if (score >= 50) return "Average";
  return "Poor";
}

async function fetchMemo() {
    if (!currentDocumentId) return;
    setLoading("memo");
  
    try {
      const response = await fetch(`${backendUrl}/documents/${currentDocumentId}/memo/`, {
        headers: { Authorization: `Token ${authToken}` }
      });
  
      if (!response.ok) throw new Error("Failed to fetch memo.");
  
      const data = await response.json();
      renderMemo(data);
    } catch (error) {
      console.error("Error fetching memo:", error);
      document.getElementById("memoContent").innerHTML = `<p class="text-danger">Error loading memo.</p>`;
    }
  }

function renderMemo(data) {
    const container = document.getElementById("memoContent");
    container.innerHTML = "";
  
    const textarea = document.createElement("textarea");
    textarea.className = "form-control";
    textarea.rows = 20;
  
    // Use entire memo text from backend
    textarea.value = data.memo_text || "No memo generated.";
  
    container.appendChild(textarea);
}

async function fetchDataValidation() {
    if (!currentDocumentId) return;
    setLoading("validation");
  
    try {
      const response = await fetch(`${backendUrl}/documents/${currentDocumentId}/data-validation/`, {
        headers: { Authorization: `Token ${authToken}` }
      });
  
      if (!response.ok) throw new Error("Failed to fetch data validation report.");
  
      const data = await response.json();
      renderDataValidation(data);
    } catch (error) {
      console.error("Error fetching data validation:", error);
      document.getElementById("validationContent").innerHTML = `<p class="text-danger">Error loading data validation.</p>`;
    }
  }
  
  function renderDataValidation(data) {
    const container = document.getElementById("validationContent");
    container.innerHTML = "";
  
    if (data.error) {
      container.innerHTML = `<p class="text-danger">${data.error}</p>`;
      return;
    }
  
    const card = document.createElement("div");
    card.className = "card p-4 shadow-sm";
  
    card.innerHTML = `
      <h5 class="mb-3">✅ Gross Salary Validation</h5>
      <p>${data.note}</p>
      <ul class="list-group">
        <li class="list-group-item"><strong>Gross Monthly Income:</strong> ${data.gross_monthly}</li>
        <li class="list-group-item"><strong>Calculated Annual Income:</strong> ${data.calculated_annual}</li>
        <li class="list-group-item"><strong>Contract Annual Income:</strong> ${data.contract_annual}</li>
      </ul>
    `;
  
    container.appendChild(card);
  }