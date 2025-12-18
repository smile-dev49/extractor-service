# RCS Extraction Service

Microservice for extracting structured data from RCS (Registre du Commerce et des Sociétés) PDF forms.

## Overview

This service processes T3 and T4 RCS PDF forms and extracts structured data including:
- RCS numbers
- Filing IDs and dates
- Company information
- Shareholders, managers, directors
- Addresses and other structured fields

## Quick Start

### Using Docker (Recommended)

```bash
# Build and run
docker-compose up -d

# Check health
curl http://localhost:8091/api/v1/health
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set working directory to mappings folder
cd mappings

# Run the service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8091 --reload
```

## API Endpoints

### Health Check
```bash
GET /api/v1/health
```

### Extract Single PDF
```bash
POST /api/v1/extract
Content-Type: multipart/form-data

file: <PDF file>
rcs_number: B123456 (optional)
filing_id: L230124830 (optional)
```

**Response:**
```json
{
  "status": "success",
  "form_type": "t3",
  "extracted_data": { ... },
  "metadata": {
    "form_type": "t3",
    "file_name": "document.pdf",
    "file_size_bytes": 524288,
    "processing_time_seconds": 2.5,
    "page_count": 5,
    "rcs_number": "B123456",
    "filing_id": "L230124830",
    "filing_date": "2024-01-01"
  }
}
```

### Batch Extraction
```bash
POST /api/v1/extract/batch
Content-Type: multipart/form-data

files: <PDF file 1>, <PDF file 2>, ...
```

**Response:**
```json
{
  "status": "completed",
  "total": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "file_name": "doc1.pdf",
      "status": "success",
      "extracted_data": { ... },
      "processing_time_seconds": 2.3
    },
    ...
  ]
}
```

## Integration with Scraper Service

After your scraper service downloads PDFs, you can call this service:

```python
import requests

def extract_pdf(pdf_path, rcs_number):
    """Extract data from PDF using extraction service"""
    with open(pdf_path, 'rb') as f:
        files = {'file': f}
        data = {'rcs_number': rcs_number}
        response = requests.post(
            'http://localhost:8091/api/v1/extract',
            files=files,
            data=data,
            timeout=300  # 5 minutes for large PDFs
        )
    return response.json()
```

## Configuration

Environment variables:

- `EXTRACTION_SERVICE_PORT` - Service port (default: 8091)
- `EXTRACTION_SERVICE_HOST` - Bind host (default: 0.0.0.0)
- `MAPPINGS_DIR` - Directory containing JSON mapping files (default: ./mappings)
- `MAX_FILE_SIZE_MB` - Maximum PDF file size in MB (default: 50)
- `LOG_LEVEL` - Logging level (default: INFO)

## Required Files

The service requires these JSON mapping files in the `mappings` directory:

- `extracted_data_mapping.json` - Maps field names to data points
- `section_mapping.json` - Maps sections to database tables
- `table_dp_mapping.json` - Maps tables to data points

## API Documentation

Once the service is running, visit:
- Swagger UI: http://localhost:8091/docs
- ReDoc: http://localhost:8091/redoc

## Error Handling

The service returns appropriate HTTP status codes:

- `200 OK` - Successful extraction
- `400 Bad Request` - Invalid file or parameters
- `500 Internal Server Error` - Processing error

Error responses include a `error` field with details:

```json
{
  "status": "error",
  "error": "Failed to process PDF: ..."
}
```

## Development

### Project Structure
```
rcs_extraction_service/
├── app/
│   ├── main.py                          # FastAPI application
│   └── bsharp1001_rcs_extraction_engine/
│       ├── __init__.py
│       ├── gateway.py                   # Main processing functions
│       ├── t3.py                        # T3 form scraper
│       ├── t4.py                        # T4 form scraper
│       └── helpers.py                   # Data processing classes
├── mappings/                            # JSON mapping files
│   ├── extracted_data_mapping.json
│   ├── section_mapping.json
│   └── table_dp_mapping.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Notes

- The service automatically detects form type (T3 or T4) based on checkbox presence
- Processing time varies based on PDF complexity (typically 1-5 seconds)
- Maximum file size is configurable (default 50MB)
- The service is stateless and can be scaled horizontally

