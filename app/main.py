"""
RCS Extraction Service - FastAPI Application
Extracts structured data from RCS PDF forms (T3 and T4)
"""
import os
import logging
import tempfile
import time
import requests
import traceback
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Union
from urllib.parse import urlparse, urlunparse
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import pymupdf

from .bsharp1001_rcs_extraction_engine.gateway import processForm, determineFormType
from .bsharp1001_rcs_extraction_engine.helpers import T3ExtractedForm, T4ExtractedForm
from .bsharp1001_rcs_extraction_engine.t3 import T3BsharpFormScrapper
from .bsharp1001_rcs_extraction_engine.t4 import T4BsharpFormScrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SERVICE_NAME = "RCS Extraction Service"
SERVICE_VERSION = "1.0.0"
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAPPINGS_DIR = os.getenv("MAPPINGS_DIR", "./mappings")

# Change working directory to mappings directory if it exists
if os.path.exists(MAPPINGS_DIR):
    os.chdir(MAPPINGS_DIR)
    logger.info(f"Changed working directory to {MAPPINGS_DIR} for mapping files")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    logger.info(f"Starting {SERVICE_NAME} v{SERVICE_VERSION}")
    
    # Verify mapping files exist
    required_files = [
        "extracted_data_mapping.json",
        "section_mapping.json",
        "table_dp_mapping.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Missing mapping files: {missing_files}")
        logger.warning("Service will start but extraction may fail")
    else:
        logger.info("All required mapping files found")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {SERVICE_NAME}")


# Create FastAPI app
app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description="Microservice for extracting structured data from RCS PDF forms",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class ExtractionResponse(BaseModel):
    status: str
    form_type: Optional[str] = None
    extracted_data: Optional[dict] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None


class BatchExtractionResult(BaseModel):
    file_name: str
    status: str
    extracted_data: Optional[dict] = None
    error: Optional[str] = None
    processing_time_seconds: Optional[float] = None


class BatchExtractionResponse(BaseModel):
    status: str
    total: int
    successful: int
    failed: int
    results: List[BatchExtractionResult]


class HealthResponse(BaseModel):
    status: str
    version: str
    mappings_loaded: bool
    mappings_dir: str


# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    required_files = [
        "extracted_data_mapping.json",
        "section_mapping.json",
        "table_dp_mapping.json"
    ]
    
    mappings_loaded = all(os.path.exists(f) for f in required_files)
    
    return HealthResponse(
        status="healthy",
        version=SERVICE_VERSION,
        mappings_loaded=mappings_loaded,
        mappings_dir=os.getcwd()
    )


@app.post("/api/v1/extract", response_model=ExtractionResponse, tags=["Extraction"])
async def extract_pdf(
    file: UploadFile = File(..., description="PDF file to extract"),
    rcs_number: Optional[str] = Form(None, description="RCS number (optional metadata)"),
    filing_id: Optional[str] = Form(None, description="Filing ID (optional metadata)")
):
    """
    Extract structured data from a single PDF file.
    
    - **file**: PDF file (multipart/form-data)
    - **rcs_number**: Optional RCS number for metadata
    - **filing_id**: Optional filing ID for metadata
    
    Returns extracted structured data in JSON format.
    """
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a PDF"
            )
        
        # Read file content
        content = await file.read()
        
        # Check file size
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Determine form type
            form_type = determineFormType(tmp_path)
            logger.info(f"Form type determined: {form_type} for file: {file.filename}")
            
            # Process PDF
            pdf_path, extracted_data, result_status = processForm(tmp_path)
            
            processing_time = time.time() - start_time
            
            if result_status == "success" and extracted_data:
                # Add metadata
                metadata = {
                    "form_type": form_type,
                    "file_name": file.filename,
                    "file_size_bytes": len(content),
                    "processing_time_seconds": round(processing_time, 2),
                    "page_count": extracted_data.get("page_count"),
                    "rcs_number": extracted_data.get("rcs_number") or rcs_number,
                    "filing_id": extracted_data.get("filing_id") or filing_id,
                    "filing_date": extracted_data.get("filing_date")
                }
                
                return ExtractionResponse(
                    status="success",
                    form_type=form_type,
                    extracted_data=extracted_data,
                    metadata=metadata
                )
            else:
                return ExtractionResponse(
                    status="error",
                    error=result_status or "Unknown error during extraction"
                )
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        return ExtractionResponse(
            status="error",
            error=f"Failed to process PDF: {str(e)}"
        )


@app.post("/api/v1/extract/batch", response_model=BatchExtractionResponse, tags=["Extraction"])
async def extract_batch(
    files: List[UploadFile] = File(..., description="PDF files to extract (max 10)")
):
    """
    Extract structured data from multiple PDF files (batch processing).
    
    - **files**: List of PDF files (multipart/form-data)
    
    Maximum 10 files per request.
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 files allowed per batch request"
        )
    
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        file_start_time = time.time()
        
        try:
            # Validate file
            if not file.filename or not file.filename.lower().endswith('.pdf'):
                results.append(BatchExtractionResult(
                    file_name=file.filename or "unknown",
                    status="error",
                    error="Invalid file type"
                ))
                failed += 1
                continue
            
            # Read file content
            content = await file.read()
            
            # Check file size
            if len(content) > MAX_FILE_SIZE_BYTES:
                results.append(BatchExtractionResult(
                    file_name=file.filename,
                    status="error",
                    error=f"File size exceeds {MAX_FILE_SIZE_MB}MB"
                ))
                failed += 1
                continue
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                # Process PDF
                pdf_path, extracted_data, result_status = processForm(tmp_path)
                processing_time = time.time() - file_start_time
                
                if result_status == "success" and extracted_data:
                    results.append(BatchExtractionResult(
                        file_name=file.filename,
                        status="success",
                        extracted_data=extracted_data,
                        processing_time_seconds=round(processing_time, 2)
                    ))
                    successful += 1
                else:
                    results.append(BatchExtractionResult(
                        file_name=file.filename,
                        status="error",
                        error=result_status or "Unknown error"
                    ))
                    failed += 1
            
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
            results.append(BatchExtractionResult(
                file_name=file.filename or "unknown",
                status="error",
                error=f"Failed to process: {str(e)}"
            ))
            failed += 1
    
    return BatchExtractionResponse(
        status="completed",
        total=len(files),
        successful=successful,
        failed=failed,
        results=results
    )


def process_pdf_from_url(pdf_url: str, rcs_number: str = None, filing_id: str = None) -> tuple:
    """
    Process PDF directly from URL using in-memory processing (no disk writes).
    
    Args:
        pdf_url: URL to the PDF file
        rcs_number: Optional RCS number for metadata
        filing_id: Optional filing ID for metadata
    
    Returns:
        tuple: (pdf_url, extracted_data, status)
    """
    start_time = time.time()
    extracted = False
    
    try:
        logger.info(f"Processing PDF from URL: {pdf_url}")
        
        # Download PDF content into memory
        pdf_response = requests.get(pdf_url, timeout=60, stream=True)
        pdf_response.raise_for_status()
        
        # Read PDF content into BytesIO
        pdf_bytes = BytesIO(pdf_response.content)
        file_size = len(pdf_response.content)
        logger.debug(f"Downloaded PDF into memory", extra={'pdf_url': pdf_url, 'file_size_bytes': file_size})
        
        # Determine form type using BytesIO
        pdf_bytes.seek(0)  # Reset to beginning
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            first_page = doc[0]
            drawings = first_page.get_cdrawings(extended=True)
            checkboxes = [x for x in drawings 
                         if x.keys().__contains__("items") and x.keys().__contains__("rect") 
                         and len(x['items']) == 1 
                         and abs((x['rect'][2] - x['rect'][0]) - (x['rect'][3] - x['rect'][1])) <= 5]
            form_type = "t3" if len(checkboxes) > 0 else "t4"
        
        logger.info(f"Form type determined: {form_type}", extra={'pdf_url': pdf_url, 'form_type': form_type})
        
        # Extract data using appropriate scraper
        # Reset BytesIO for scraper
        pdf_bytes.seek(0)
        extraction_start = time.time()
        
        # Create a temporary file path for the scraper (it expects a path)
        # But we'll use BytesIO directly if possible
        # Note: The scrapers (T3/T4) currently expect file paths, so we need to save temporarily
        # However, we can minimize disk I/O by using memory-mapped files or keeping it in memory
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_bytes.read())
            tmp_path = tmp_file.name
        
        try:
            # Extract data using file path (scrapers need file paths currently)
            data = T4BsharpFormScrapper(tmp_path) if form_type == 't4' else T3BsharpFormScrapper(tmp_path)
            extraction_duration = time.time() - extraction_start
            
            # Create appropriate form object
            data = T4ExtractedForm(
                form_id = data[0], 
                page_count = data[1], 
                existingSections = data[2], 
                absentSections = data[3], 
                rcs_number = data[4] or rcs_number, 
                filing_id = data[5] or filing_id, 
                filing_date = data[6], 
                title = data[7], 
                subtitle = data[8], 
                extracted_sections = data[9], 
            ) if form_type == 't4' else T3ExtractedForm(
                form_id = data[0], 
                page_count = data[1], 
                existingSections = data[2], 
                absentSections = data[3], 
                rcs_number = data[4] or rcs_number, 
                filing_id = data[5] or filing_id, 
                filing_date = data[6], 
                title = data[7], 
                subtitle = data[8], 
                extracted_sections = data[9],
            )
            
            extracted = True
            
            # Process the extracted data
            processing_start = time.time()
            data.preprocessSections()
            data.restructureDataforDB([])
            data.postprocessSections()
            data.nested_del()
            data.remove_empty()
            hh = data.nested_rename()
            data.leftovers = hh
            processing_duration = time.time() - processing_start
            
            # Serialize to JSON
            serialization_start = time.time()
            json_ = data.serializeSelf()
            serialization_duration = time.time() - serialization_start
            
            total_duration = time.time() - start_time
            
            logger.info(
                f"PDF processing completed successfully from URL",
                extra={'pdf_url': pdf_url,
                'form_type': form_type,
                'total_duration_seconds': total_duration,
                'success': True
                }
            )
            
            return (pdf_url, json_, "success")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as error:
        duration = time.time() - start_time
        logger.error(
            f"PDF processing failed from URL",
            extra={'pdf_url': pdf_url,
            'error': str(error),
            'duration_seconds': duration,
            'extracted': extracted,
            'exception': traceback.format_exc()
            }
        )
        return (pdf_url, None, f"failed processing... extracted: {extracted}, Exception: {str(error)}")


@app.post("/api/v1/extract/job/{job_id}", tags=["Extraction"])
async def extract_job_pdfs(
    job_id: str,
    scraper_service_url: Optional[str] = Form(None, description="Scraper service URL to fetch PDFs")
):
    """
    Extract data from all PDFs in a scraper job using PDF URLs directly.
    PDFs are processed in-memory without saving to disk.
    
    - **job_id**: The job ID from scraper service
    - **scraper_service_url**: URL of scraper service (default: http://localhost:8090)
    
    Fetches detail.json from scraper service, then extracts data from all PDFs using their URLs.
    """
    # Handle scraper_service_url - check if it's None, empty, or invalid
    if scraper_service_url and scraper_service_url.strip() and scraper_service_url != "string":
        scraper_url = scraper_service_url.strip()
    else:
        scraper_url = os.getenv("SCRAPER_SERVICE_URL", "http://localhost:8090")
    
    # Ensure URL has scheme
    if not scraper_url.startswith(("http://", "https://")):
        scraper_url = f"http://{scraper_url}"
    
    logger.info(f"Using scraper service URL: {scraper_url}")
    
    try:
        # 1. Fetch job details from scraper service
        logger.info(f"Fetching job details for {job_id} from scraper service")
        detail_response = requests.get(
            f"{scraper_url}/api/v1/job/{job_id}",
            timeout=30
        )
        
        if detail_response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found in scraper service"
            )
        
        job_data = detail_response.json()
        documents = job_data.get("documents", [])
        rcs_number = job_data.get("rcs_number")
        
        if not documents:
            return {
                "status": "error",
                "error": "No documents found in job"
            }
        
        logger.info(f"Found {len(documents)} documents to extract for job {job_id}")
        
        # 2. Extract data from each PDF using URLs directly
        results = []
        successful = 0
        failed = 0
        
        for doc in documents:
            pdf_url = doc.get("pdf_path")
            deposit_number = doc.get("deposit_number")
            filing_id = doc.get("filing_id")
            
            if not pdf_url:
                logger.warning(f"No PDF URL for document {deposit_number}")
                failed += 1
                results.append({
                    "deposit_number": deposit_number,
                    "status": "error",
                    "error": "No PDF URL"
                })
                continue
            
            # Fix PDF URL if it points to wrong host/port
            # Replace any host:port in the PDF URL with the scraper service URL
            try:
                parsed_pdf = urlparse(pdf_url)
                parsed_scraper = urlparse(scraper_url)
                
                # If PDF URL has different host/port than scraper service, replace it
                if parsed_pdf.netloc != parsed_scraper.netloc:
                    logger.info(f"Replacing PDF URL host from {parsed_pdf.netloc} to {parsed_scraper.netloc}")
                    # Reconstruct URL with scraper service host/port
                    pdf_url = urlunparse((
                        parsed_scraper.scheme,  # Use scraper service scheme
                        parsed_scraper.netloc,  # Use scraper service host:port
                        parsed_pdf.path,        # Keep original path
                        parsed_pdf.params,      # Keep original params
                        parsed_pdf.query,       # Keep original query
                        parsed_pdf.fragment     # Keep original fragment
                    ))
                    logger.info(f"Updated PDF URL: {pdf_url}")
            except Exception as e:
                logger.warning(f"Could not parse/fix PDF URL {pdf_url}: {e}")
                # Continue with original URL
            
            try:
                # Process PDF directly from URL (in-memory, no disk save)
                pdf_path, extracted_data, result_status = process_pdf_from_url(
                    pdf_url, 
                    rcs_number=rcs_number, 
                    filing_id=filing_id
                )
                
                if result_status == "success" and extracted_data:
                    results.append({
                        "deposit_number": deposit_number,
                        "filing_id": filing_id,
                        "status": "success",
                        "extracted_data": extracted_data,
                        "metadata": {
                            "rcs_number": rcs_number,
                            "deposit_number": deposit_number,
                            "filing_id": filing_id,
                            "filing_date": doc.get("filing_date"),
                            "page_count": extracted_data.get("page_count"),
                            "pdf_url": pdf_url
                        }
                    })
                    successful += 1
                else:
                    results.append({
                        "deposit_number": deposit_number,
                        "filing_id": filing_id,
                        "status": "error",
                        "error": result_status
                    })
                    failed += 1
            
            except Exception as e:
                logger.error(f"Error processing {deposit_number}: {str(e)}", exc_info=True)
                results.append({
                    "deposit_number": deposit_number,
                    "filing_id": filing_id,
                    "status": "error",
                    "error": str(e)
                })
                failed += 1
        
        return {
            "status": "completed",
            "job_id": job_id,
            "rcs_number": rcs_number,
            "total": len(documents),
            "successful": successful,
            "failed": failed,
            "results": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process job: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.getenv("EXTRACTION_SERVICE_PORT", "8091"))
    host = os.getenv("EXTRACTION_SERVICE_HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

