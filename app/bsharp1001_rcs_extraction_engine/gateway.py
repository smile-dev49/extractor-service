import math
import logging
import time
import os
import traceback

import pymupdf
from .helpers import T3ExtractedForm, T4ExtractedForm, extractedForm
from .t3 import T3BsharpFormScrapper
from .t4 import T4BsharpFormScrapper

# Setup logging
logger = logging.getLogger(__name__)

def isCheckboxType3(drawing, verticalStart=0, verticalEnd= math.inf):
    return drawing.keys().__contains__("items") and drawing.keys().__contains__("rect")  and drawing['rect'][1] > verticalStart and drawing['rect'][3] < verticalEnd and len(drawing['items']) == 1 and abs((drawing['rect'][2] - drawing['rect'][0]) - (drawing['rect'][3] - drawing['rect'][1]) ) <= 5

def extractCheckboxes(drawings, type=3, verticalStart=0, verticalEnd= math.inf, banArea: pymupdf.Rect = None):
    """Extract checkbox drawings from PDF page drawings"""
    logger.debug(f"Extracting checkboxes from {len(drawings)} drawings", 
                extra={'vertical_start': verticalStart, 'vertical_end': verticalEnd, 'ban_area': banArea is not None})
    
    l = []
    rs = []
    for x in drawings:
        if isCheckboxType3(x, verticalStart=verticalStart, verticalEnd=verticalEnd) and x['rect'] not in rs:
            if banArea is None or (banArea is not None and banArea.contains(x['rect']) == False):
                l.append(x)
                rs.append(x['rect'])

    logger.debug(f"Found {len(l)} checkboxes", extra={'checkbox_count': len(l)})
    return l

def determineFormType(pdf: str):
    """Determine PDF form type (T3 or T4) based on checkbox detection"""
    logger.debug(f"Determining form type for PDF: {pdf}")
    start_time = time.time()
    
    try:
        with pymupdf.open(pdf) as doc: 
            first_page = doc[0]
            drawings = first_page.get_cdrawings(extended=True)
            checkboxes = extractCheckboxes(drawings, verticalStart=0)
            
            form_type = "t3" if len(checkboxes) > 0 else "t4"
            duration = time.time() - start_time
            
            logger.info(
                f"Form type determined: {form_type}",
                extra={
                    'pdf_path': pdf,
                    'form_type': form_type,
                    'checkbox_count': len(checkboxes),
                    'duration_seconds': duration
                }
            )
            
            return form_type
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Failed to determine form type for PDF: {pdf}",
            extra={'pdf_path': pdf,
            'error': str(e),
            'duration_seconds': duration,
            'exception': traceback.format_exc()
            }
        )
        raise

def processForm(pdf: str) -> tuple[str, extractedForm, str]:
    """Process PDF form and extract structured data"""
    logger.info(f"Starting PDF form processing: {pdf}")
    start_time = time.time()
    extracted = False
    
    try:
        # Check if PDF file exists
        if not os.path.exists(pdf):
            raise FileNotFoundError(f"PDF file not found: {pdf}")
        
        file_size = os.path.getsize(pdf)
        logger.debug(f"Processing PDF file", extra={'pdf_path': pdf, 'file_size_bytes': file_size})
        
        # Determine form type
        formtype = determineFormType(pdf)
        logger.info(f"Using {formtype} form scraper", extra={'pdf_path': pdf, 'form_type': formtype})
        
        # Extract data using appropriate scraper
        extraction_start = time.time()
        data = T4BsharpFormScrapper(pdf) if formtype == 't4' else T3BsharpFormScrapper(pdf)
        extraction_duration = time.time() - extraction_start
        
        logger.info(
            f"Raw data extraction completed",
            extra={'pdf_path': pdf,
            'form_type': formtype,
            'extraction_duration_seconds': extraction_duration,
            'page_count': data[1],
            'rcs_number': data[4],
            'filing_id': data[5]
            }
        )
        
        # Create appropriate form object
        data = T4ExtractedForm(
            form_id = data[0], 
            page_count = data[1], 
            existingSections = data[2], 
            absentSections = data[3], 
            rcs_number = data[4], 
            filing_id = data[5], 
            filing_date = data[6], 
            title = data[7], 
            subtitle = data[8], 
            extracted_sections = data[9], 
            ) if formtype == 't4' else T3ExtractedForm(
                form_id = data[0], 
                page_count = data[1], 
                existingSections = data[2], 
                absentSections = data[3], 
                rcs_number = data[4], 
                filing_id = data[5], 
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
        
        logger.info(
            f"Data processing completed",
            extra={'pdf_path': pdf,
            'processing_duration_seconds': processing_duration,
            'sections_found': len(data.existingSections),
            'sections_missing': len(data.absentSections)
            }
        )
        
        # Serialize to JSON
        serialization_start = time.time()
        json_ = data.serializeSelf()
        serialization_duration = time.time() - serialization_start
        
        total_duration = time.time() - start_time
        
        logger.info(
            f"PDF form processing completed successfully",
            extra={'pdf_path': pdf,
            'form_type': formtype,
            'total_duration_seconds': total_duration,
            'extraction_duration_seconds': extraction_duration,
            'processing_duration_seconds': processing_duration,
            'serialization_duration_seconds': serialization_duration,
            'success': True
            }
        )
        
        return (pdf, json_, "success")
        
    except Exception as error:
        duration = time.time() - start_time
        logger.error(
            f"PDF form processing failed",
            extra={'pdf_path': pdf,
            'error': str(error),
            'duration_seconds': duration,
            'extracted': extracted,
            'exception': traceback.format_exc()
            }
        )
        return (pdf, None, f"failed processing... extracted: {extracted}, Exception: {str(error)}\n-----------------------------------------------------------")

def processEntityDocument(entityDocument):
    """Process EntityDocument PDF and extract structured data"""
    extracted = False
    pdf = entityDocument.pdf.path
    entity_id = getattr(entityDocument, 'entity_id', 'unknown')
    filing_id = getattr(entityDocument, 'filing_id', 'unknown')
    
    logger.info(
        f"Processing EntityDocument PDF",
        extra={'pdf_path': pdf,
        'entity_id': entity_id,
        'filing_id': filing_id
        }
    )
    
    start_time = time.time()
    
    try:
        # Check if PDF file exists
        if not os.path.exists(pdf):
            raise FileNotFoundError(f"PDF file not found: {pdf}")
        
        file_size = os.path.getsize(pdf)
        logger.debug(f"Processing EntityDocument PDF", extra={'pdf_path': pdf, 'file_size_bytes': file_size})
        
        # Determine form type and extract data
        formtype = determineFormType(pdf)
        data = T4BsharpFormScrapper(pdf) if formtype == 't4' else T3BsharpFormScrapper(pdf)
        
        # Create appropriate form object
        data = T4ExtractedForm(
                form_id = data[0], 
                page_count = data[1], 
                existingSections = data[2], 
                absentSections = data[3], 
                rcs_number = data[4], 
                filing_id = data[5], 
                filing_date = data[6], 
                title = data[7], 
                subtitle = data[8], 
                extracted_sections = data[9],
            ) if formtype == 't4' else T3ExtractedForm(
                form_id = data[0], 
                page_count = data[1], 
                existingSections = data[2], 
                absentSections = data[3], 
                rcs_number = data[4], 
                filing_id = data[5], 
                filing_date = data[6], 
                title = data[7], 
                subtitle = data[8], 
                extracted_sections = data[9],
            )
        
        extracted = True
        
        # Process the extracted data
        processing_start = time.time()
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
            f"EntityDocument processing completed successfully",
            extra={'pdf_path': pdf,
            'entity_id': entity_id,
            'filing_id': filing_id,
            'form_type': formtype,
            'total_duration_seconds': total_duration,
            'processing_duration_seconds': processing_duration,
            'serialization_duration_seconds': serialization_duration,
            'success': True
            }
        )
        
        return (entityDocument, json_, "success")
        
    except Exception as error:
        duration = time.time() - start_time
        logger.error(
            f"EntityDocument processing failed",
            extra={'pdf_path': pdf,
            'entity_id': entity_id,
            'filing_id': filing_id,
            'error': str(error),
            'duration_seconds': duration,
            'extracted': extracted,
            'exception': traceback.format_exc()
            }
        )
        return (entityDocument, None, f"failed processing... extracted: {extracted}, Exception: {str(error)}\n-----------------------------------------------------------")
