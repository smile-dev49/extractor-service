
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import itertools
import math
from billiard.pool import get_context
import billiard
import billiard.pool
import pymupdf
import regex
import logging
import time
import os
import traceback

SEGSEP='#____SEP____#'

# Setup logging
logger = logging.getLogger(__name__)

MODIFIED_SUBTITLES = regex.compile(r'(Modification\s*[/]?[\s*Renouvellement]?|Ab[aä]nderungs*[/]?[\s*Erneuerung]?)', flags= regex.IGNORECASE)
NEW_SUBTITLES = regex.compile(r'(Inscription?|Neueintragung?|Einschreibung?)', flags= regex.IGNORECASE)
DELETED_SUBTITLES = regex.compile(r'(Radiation?|L[öo]schung)', flags= regex.IGNORECASE)
RESIGNED_SUBTITLES = regex.compile(r'^\s*D[ée]mission\s*\w+\s*', flags= regex.IGNORECASE)

def rec_dd():
    return defaultdict(rec_dd)

def extractPageCount(footerText):
    h = regex.findall(r'.+\d+.*\d+', str(footerText).split("\n")[0].split("Pag")[0], flags=regex.IGNORECASE | regex.DOTALL)
    s = regex.split(r'.*\d+\D\/', h[0].strip(), flags=regex.IGNORECASE | regex.DOTALL, maxsplit=1)
    h = regex.sub(r'\D', '', s[1])
    h = int(h)
    return h

def addAttribsToSpan(span, lineID, blockID, pageNum):
    span["blockId"] = blockID
    span["lineId"] = lineID
    span["pageNum"] = pageNum
    return span

def flatSpans(blocks, pageNum) -> list:
    return [[addAttribsToSpan(span, block['lines'].index(line_), blocks['blocks'].index(block), pageNum) for line_ in block['lines'] for span in line_['spans']] for block in blocks['blocks'] if block.keys().__contains__("lines")]

def isUnderLined(spanRect: pymupdf.Rect, drawings: list):
    # for each point check:
    # 1- start point
    # 2- end point 
    # 3- width
    # 4- distance from title
    for drawing in drawings:
        if drawing.keys().__contains__("rect") == False:
            continue
        # 1- start point
        c1 = abs(drawing['rect'][0] - spanRect[0]) <= 5
        # 2- end point
        c2 = abs(drawing['rect'][2] - spanRect[2]) >= 0
        # 3- width
        c3 = abs((drawing['rect'][2] - drawing['rect'][0]) - (spanRect[2] - spanRect[0])) >= 0
        # 4- distance from title
        c4 = abs(spanRect[3] - drawing['rect'][1]) <= 4
        if (c1 and c2 and c3 and c4):
            return (True, drawings.index(drawing))
    return (False, -1)

def isTitle(span):
    return span['size'] == 12 and span['font'].lower().__contains__("bold")

def isSubsection(span, drawings): # variable names of individuals
    return (span['size'] == 10 and span['bbox'][0] < 75 and span['font'].lower().__contains__("bold")) and isUnderLined(span['bbox'], drawings)[0] == False

def isDP(span):
    return span['size'] == 10 and (span['font'].lower().__contains__("regular") or (span['font'].lower().__contains__("bold") and span['bbox'][0] > 80) or span['font'].lower().__contains__("it"))

def isLabel(span): # grey labels of DPs 
    return (span['size'] == 8.5 or (span['size'] <= 10.0 and span['bbox'][0] < 75 and span['color'] == -8355712)) and span['font'].lower().__contains__("regular")

def isSubtitle(span, drawings): # eg: inscription, radiation, modification
    return isTitle(span) and isUnderLined(span['bbox'], drawings)[0] == False

def isSection(span, drawings): #eg: Denomination, form juridique, associe, etc...
    return isTitle(span) and isUnderLined(span['bbox'], drawings)[0]

def getFooterLine(blocks: list, pageWidth):
    """
    Extract footer line from PDF blocks.
    Returns the bottommost image/line that spans 75-100% of page width.
    
    Args:
        blocks: List of PDF blocks
        pageWidth: Width of the page
        
    Returns:
        Bounding box of footer line, or None if not found
    """
    imgs = []
    for x in blocks:
        if x.keys().__contains__("ext") and x.keys().__contains__("colorspace") and x.keys().__contains__("type") and x['type'] == 1:
            img = x['bbox']
            startXPoint = x['bbox'][0]
            endXPoint = x['bbox'][2]
            length = endXPoint - startXPoint 
            percentage = math.floor(length * 100 / pageWidth) 
            if percentage >= 75 and percentage <= 100:
                imgs.append(img)
    
    if not imgs:
        logger.warning("No footer line found in PDF blocks")
        return None
    
    imgs = sorted(imgs, key=lambda img: img[1])
    return imgs[-1]

def getFooterText(footerLineY, page: pymupdf.Page):
    return (page.get_textbox((0.0, footerLineY, page.rect.x1,  page.rect.y1,)), pymupdf.Rect(0.0, footerLineY, page.rect.x1,  page.rect.y1,))

def getPageSpans(x: int, page):
    ss =  flatSpans(page.get_text("dict",sort=True), x)
    return [ss, page.get_cdrawings(extended=True)]

def mergeSpansText(spans):
    return " ".join([span['text'] for span in spans])

def mergeSpansofSpansText(spansofspans):
    k = ""
    for s in spansofspans:
        k += f"{mergeSpansText(s)} ////"
    return k

def splitSpans(spansofspans, drawings, predicateFunc, returnSpansBeforeFirstPredicateMatch=False):
    indices = [i for i, spans in enumerate(spansofspans) if predicateFunc(spans[0], drawings[spans[0]['pageNum']] if len(drawings) > 0 else [] )]
    if len(indices) == 0:
        return spansofspans
    indices.append(len(spansofspans))
    spans2 = [[mergeSpansText(spansofspans[x]), spansofspans[x+1: indices[indices.index(x)+1]]] for x in indices if x != len(spansofspans)]
    if not returnSpansBeforeFirstPredicateMatch:
        return spans2
    else:
        indexOfFirstPredicateMatch = indices[0]
        if indexOfFirstPredicateMatch == len(spansofspans) or indexOfFirstPredicateMatch == 0:
            return spans2
        else:
            spans2 = spansofspans[0: spansofspans[indexOfFirstPredicateMatch]] + spans2
            return spans2

def spanIsContinuation(first_span, inspected_span):
    c1 = first_span['size'] == inspected_span['size']
    c2 = first_span['color'] == inspected_span['color']
    c3 = first_span['blockId'] == inspected_span['blockId']
    c4 = first_span['pageNum'] == inspected_span['pageNum']
    c5 = inspected_span['bbox'][1] - first_span['bbox'][3] < 3.0
    if c1 and c2 and c3 and c4 and c5:
        return True
    else:
        return False

def mergeSpans(first_span, inspected_span):
    first_span['bbox'] = (first_span['bbox'][0], first_span['bbox'][1], inspected_span['bbox'][2], inspected_span['bbox'][3])
    first_span['text'] += f" {inspected_span['text']}"
    return first_span

def makeSingleLevel(spans, arr=[], height = 0, merge = True):
    dd = arr
    for s in spans:
        if isinstance(s, list):
            makeSingleLevel(s, arr, height)
        else:
            factor = s['pageNum'] * height
            s['bbox'] = (s['bbox'][0], s['bbox'][1]+factor, s['bbox'][2], s['bbox'][3]+factor)
            if merge:
                matches = list(filter(lambda x: isLabel(x) and spanIsContinuation(x, s), dd))
                if len(matches) > 0:
                    index = dd.index(matches[0])
                    mergeSpans(matches[0], s)
                    dd[index] = matches[0]
                else:
                    #factor = s['pageNum'] * height
                    #s['bbox'] = (s['bbox'][0], s['bbox'][1]+factor, s['bbox'][2], s['bbox'][3]+factor)
                    dd.append(s)
            else:
                #factor = s['pageNum'] * height
                #s['bbox'] = (s['bbox'][0], s['bbox'][1]+factor, s['bbox'][2], s['bbox'][3]+factor)
                dd.append(s)
            # adding height multiplied by page number standarize the y axis making it independent from page
            # so we simplify span linking to parent labels when spans traverse multiple pages
            # so we don't have to add complexity of comparing page numbers
            #factor = s['pageNum'] * height
            #s['bbox'] = (s['bbox'][0], s['bbox'][1]+factor, s['bbox'][2], s['bbox'][3]+factor)
            #dd.append(s)
    return dd

def getNearestLabel(span, labels):
    for i, lbl in enumerate(labels):
        # (belonging) has 3 conditions: 
        # 1- in all situations: span.x0 - parent_label.x0 > 20
        c1 = span['bbox'][0] - lbl['bbox'][0] > 20
        # 2- in between 2 labels:
        # (abs(parent_label.y0 - span.y0) <= 5) and (span.y0 < next_label.y0) 
        if lbl != labels[-1]:
            c2 = span['bbox'][1] - lbl['bbox'][1] >= -5 and labels[i+1]['bbox'][1] - span['bbox'][1] >= 10
        #OR
        else:
            # 3- if it's the last label then abs(parent_label.y0 - span.y0) <= 5) should be enough
            c2 = span['bbox'][1] - lbl['bbox'][1] >= -5
        if c1 and c2:
            return (i, lbl)
    return (-1, None)

PERSONNE_MORALE_LUXEMBOURGEOISE_REGEX = regex.compile(r'B\d{5,20}$')
PERSONNE_MORALE_étrangère_REGEX = regex.compile(r'.*(sous\s*le\s*num[ée]ro\s*|unter\s*der\s*Nummer)(?=.+)', flags=regex.IGNORECASE)
DATE_YEAR_REGEX = regex.compile(r'(?<=\w+)\s+(?=\d{2}/\d{2}/\d{4})')
DATE_MONTH_REGEX = regex.compile(r'(?<=\w+)\s+(?=\d{2}/\d{2})')

def postProcessStatements(statement, individual):
    # check type of person
    if individual == False:
        return statement
    moral_person = regex.findall(PERSONNE_MORALE_LUXEMBOURGEOISE_REGEX, statement)
    if len(moral_person) > 0:
        return {
            "Type de personne": "Personne morale luxembourgeoise",
            "N° d'immatriculation au RCS": moral_person[0],
        }
    else:
        moral_foreign_person = str(statement).split(",")
        if len(moral_foreign_person) >= 3:
            reg_number = regex.sub(PERSONNE_MORALE_étrangère_REGEX, "", moral_foreign_person[0].strip())
            reg_number = regex.sub(r'\s*', "", reg_number)
            reg_country = moral_foreign_person[-1]
            name_of_registry = ", ".join(moral_foreign_person[1:-1])
            return {
                "Type de personne": "Personne morale étrangère",
                "N° d'immatriculation au RCS": reg_number,
                "Personne morale étrangère": {
                    "Pays": reg_country,
                    "Nom du registre": name_of_registry,
                }
            }
        else:
            return {
                "Type de personne": "Personne physique",
            }

def processDPs(dp):
    #date processing
    date_dp = regex.split(DATE_YEAR_REGEX, dp)
    if len(date_dp) == 2:
        return {date_dp[0]: date_dp[1]}
    else:
        date_dp = regex.split(DATE_MONTH_REGEX, dp)
        if len(date_dp) == 2:
            return {date_dp[0]: date_dp[1]}
        else:
            return dp

def normalizeShares(shares: list[str]):
    shares = shares.split(":")
    shares[0] = f"0000 \n {shares[0]}"
    # New issue arised: sometimes the type of shares is omitted as the deefaukt is (part sociales) so a patch is needed
    # Hypothesis: shares statement always ends with the amount of shares input
    # Application: we can split the data first of joined shares into share groups (share groups = [+/- type of shares, amount of shares])
    # 1- prepare an empty dict of dicts of share groups (dict[share_group]) 
    # 2- iterate collected shares data list split by ":", starting from index n=1
    # 2.1- prepare an empty dict of share group (dict{+/-type: text, amount: number}) with each iteration
    # 2.2- look behind by -1 for the title of the input (split by last newline \n)
    # 2.3- check the current data input (split by first \n)
    # 2.3.1- if it's numbers only -> amount of shares (i.e end of share group):
    #   append default type of shares: part sociales to dict of share group (created on step No. 2.1), 
    #   append amount of shares to dict of share group (created on step No. 2.1), 
    #   append to dict of dicts (created on step No. 2) 
    #   step loop by 1
    # 2.3.2- else-> type of shares (i.e start of share group, must have amount of shares after it):
    #   that's what's currently implemented: 
    #       check the next element, same processing as 2.3 since it's expected to be a data input
    #       deal with our data input as 2.2 since it's n-1 for the next data point
    #       step loop by 2
    
    # 1- prepare an empty dict of dicts of share groups (dict[share_group]) 
    shares_obj = {}
    cc = 0
    
    x = 1
    # 2- iterate collected shares data list split by ":", starting from index n=1
    while x < len(shares):
        # 2.1- prepare an empty dict of share group (dict{+/-type: text, amount: number}) with each iteration
        shares_obj[f"item{cc}"] = {}
        # 2.2- look behind by -1 for the title of the input (split by last newline \n)
        label_of_current_input = shares[x-1].rsplit('\n')[1].strip()
        
        # 2.3- check the current data input (split by first \n)
        data_input_possibly_with_label_of_next_input = shares[x].rsplit("\n", 1)
        data_input = data_input_possibly_with_label_of_next_input[0].strip()
        
        # 2.3.1- if it's numbers only -> amount of shares (i.e end of share group):
        if regex.sub(r'[\s*\W]', '', data_input).isnumeric():
            #   append default type of shares: part sociales to dict of share group (created on step No. 2.1),
            #   append amount of shares to dict of share group (created on step No. 2.1), 
            #   append to dict of dicts (created on step No. 2)
            # according to language
            if label_of_current_input.__contains__("parts"): # french
                shares_obj[f"item{cc}"] = {"Type de parts": "Parts sociales", label_of_current_input: data_input }
            else: # german
                shares_obj[f"item{cc}"] = {"Art der Anteile": "anteile", label_of_current_input: data_input }
            
            #  step loop by 1
            x+=1
            
        # 2.3.2- else-> type of shares (i.e start of share group, must have amount of shares after it):
        else:
            #   that's what's currently implemented: 
            #   check the next element, same processing as 2.3 since it's expected to be a data input
            title_of_next_input = data_input_possibly_with_label_of_next_input[1].strip()
            shares_obj[f"item{cc}"] = {
                label_of_current_input: data_input, 
                #   deal with our data input as 2.2 since it's n-1 for the next data point
                title_of_next_input: shares[x+1].split('\n')[0].strip()
            }
            
            #   step loop by 2
            x+=2
        
        cc+=1
    return shares_obj

def deepUpdateDict(updatableDict, newDict):
    udk = list(updatableDict.keys())
    for key in list(newDict.keys()):
        if key in udk:
            if isinstance(newDict[key], dict) or isinstance(newDict[key], defaultdict):
                deepUpdateDict(updatableDict[key], newDict[key])
            else: 
                updatableDict[key] = f'{updatableDict[key]} \n {newDict[key]}'
        else:
            updatableDict[key] = newDict[key]

def processSectionExtractedData(spans: list, height, section, individual):
    data = rec_dd()
    # 2- flat all spans, no merging should happen now as there will be spans from different blocks and spanning multiple pages
    spans = makeSingleLevel(spans, [], height)
    # 3- get nearest of the grey labels
    # 3.1- separate grey labels
    greyLbls = []
    idx = []
    for i, s in enumerate(spans):
        if isLabel(s):
            greyLbls.append(s)
            idx.append(i)
    idx.reverse()
    for i in idx:
        spans.pop(i)
    # 3.2- sort by y0 
    greyLbls = sorted(greyLbls, key=lambda x: (x['pageNum'],x['bbox'][1]))
    # 3.3- get 10.0 sized spans belonging to grey labels
    statements = False
    statement_combined = ''
    for s in spans:
        if isDP(s) == False:
            continue
        index, label = getNearestLabel(s, greyLbls)
        if label == None:
            key = section
            value = postProcessStatements(s['text'], individual=individual)
            if individual:
                statement_combined = statement_combined+s['text']
            if type(value) is str:
                key += f" {value}"
            statements = True
        else:
            key = label['text']
            value = processDPs(s['text'])
        if list(data.keys()).__contains__(key):
            if isinstance(data[key], dict):
                deepUpdateDict(data[key], value)
            else:
                data[key] = f'{data[key]} \n {value}'
        else:
            if isinstance(value, dict) == False:
                data[key] = value
            else:
                if key == section:
                    data = {**data, **value}
                else:
                    data[key] = {**data[key], **value}
    if individual and (statements == False or statement_combined != ''):
        data = {**data, **postProcessStatements(statement_combined, individual=individual)}
    if individual:
        for key in data:
            if data[key].__contains__(":"):
                data[key] = normalizeShares(data[key])
    return data
    # 4- for date values split if date hs text before making them subtitles, for shares split by (:) check
    # 5- for duplicated labels, make sure to have update_or_add kinda function check

def handleIndividualdsWithoutSubtitles(individualsWithoutSubtitle): # for demission
    subtitles = defaultdict(list)
    for i in individualsWithoutSubtitle:
        if len(regex.findall(RESIGNED_SUBTITLES, i[0])) > 0:
            i[0] = regex.sub(RESIGNED_SUBTITLES, '', i[0])
            subtitles["Démission"].append(i)
    individualsWithoutSubtitle = list(subtitles.items())
    return individualsWithoutSubtitle

def processSection(secSpansPair, height):
    sectionName, spans = secSpansPair
    section = rec_dd()
    # take each data array and:
    # 1- split yet again by subtitles check
    spans = splitSpans(spans, [], isSubtitle, True)   
    
    individualsWithoutSubtitle = splitSpans([s for s in spans if isinstance(s[0], dict)], [], isSubsection) 
    appendIndividualsWithoutSubtitle = False
    # indivivduals without subtitiles
    if isinstance(spans[0][0], dict) and isinstance(individualsWithoutSubtitle[0][0], str):
        individualsWithoutSubtitle = handleIndividualdsWithoutSubtitles(individualsWithoutSubtitle)
        appendIndividualsWithoutSubtitle = True
        spans = [s for s in spans if not isinstance(s[0], dict)]
        spans = spans + individualsWithoutSubtitle
        
    # normal DPs, process directly
    if isinstance(spans[0][0], dict) and not appendIndividualsWithoutSubtitle:
        sectionData = processSectionExtractedData(spans, height, sectionName+" statement", individual=False)
        section[sectionName] = sectionData
    #individuals section
    else:
        for individualGroup in spans:
            individuals = splitSpans(individualGroup[1], [], isSubsection) if isinstance(individualGroup[1][0][0], dict) else individualGroup[1]
            for individual in individuals:
                individualData = processSectionExtractedData(individual[1], height, sectionName+" statement", individual=True)
                section[sectionName+SEGSEP+individual[0]] = individualData
                state = individualGroup[0]
                section[sectionName+SEGSEP+individual[0]]['state'] = state
                
                if section[sectionName+SEGSEP+individual[0]]['Type de personne'] == 'Personne physique':
                    fullName = str(individual[0]).split(" ")
                    section[sectionName+SEGSEP+individual[0]]['Vorname'] = fullName.pop(0)
                    section[sectionName+SEGSEP+individual[0]]['Name'] = " ".join(fullName)
                if section[sectionName+SEGSEP+individual[0]]['Type de personne'] == 'Personne morale étrangère':
                    nonLuName = str(individual[0])
                    section[sectionName+SEGSEP+individual[0]]["Personne morale étrangère"]['Dénomination'] = nonLuName
                    legal_form_key = [key for key in section[sectionName+SEGSEP+individual[0]].keys() if len(regex.findall(r'(Form[e]\s*juridique|Rechtsform)', key.strip(), flags=regex.IGNORECASE)) > 0]
                    if len(legal_form_key) == 1:
                        section[sectionName+SEGSEP+individual[0]]["Personne morale étrangère"]['Forme juridique'] = section[sectionName+SEGSEP+individual[0]][legal_form_key[0]]
                    
                    siege_key = [key for key in section[sectionName+SEGSEP+individual[0]].keys() if len(regex.findall(r'(Si[èe]ge\s*[social]|Sitz\s*[der\s*Gesellschaft])', key.strip(), flags=regex.IGNORECASE)) > 0]
                    if len(siege_key) == 1:
                        section[sectionName+SEGSEP+individual[0]]["Personne morale étrangère"]['Siège social'] = section[sectionName+SEGSEP+individual[0]][siege_key[0]]
                
                if regex.findall(MODIFIED_SUBTITLES, state):
                    section[sectionName+SEGSEP+individual[0]]['Rayer'] = False
                    section[sectionName+SEGSEP+individual[0]]['Modifier'] = True
                elif regex.findall(DELETED_SUBTITLES, state):
                    section[sectionName+SEGSEP+individual[0]]['Rayer'] = True
                    section[sectionName+SEGSEP+individual[0]]['Modifier'] = False
                elif state == "Démission":
                    section[sectionName+SEGSEP+individual[0]]['Démission'] = True
                    section[sectionName+SEGSEP+individual[0]]['Rayer'] = None
                    section[sectionName+SEGSEP+individual[0]]['Modifier'] = None
                else:
                    section[sectionName+SEGSEP+individual[0]]['Rayer'] = None
                    section[sectionName+SEGSEP+individual[0]]['Modifier'] = None
    
    return section

def T4BsharpFormScrapper(pdf) -> tuple:
    """T4 Form Scraper - Extract data from T4 type PDF forms"""
    logger.info(f"Starting T4 form scraping for PDF: {pdf}")
    start_time = time.time()
    
    # 1- get number of pages from footer check
    # 2- Parallel: loop over pages, grabbing text check
    # 3- split per sections as in T3 check
    # 4- Parallel: loop over splitted sections children (blocks in between), processing grey labels or subtitles
    page_height = 0
    form_id = None
    page_count = 0
    existingSections = []
    absentSections = []
    rcs_number = None
    filing_id = None
    filing_date = None
    title = None
    subtitle = None
    extracted_sections = []
    
    try:
        # Check if PDF file exists
        if not os.path.exists(pdf):
            raise FileNotFoundError(f"PDF file not found: {pdf}")
        
        file_size = os.path.getsize(pdf)
        logger.debug(f"Processing T4 PDF file", extra={'pdf_path': pdf, 'file_size_bytes': file_size})
        
        with pymupdf.open(pdf) as doc: 
            fffs = pdf.replace(".pdf", "").replace("Prospect_SRC/", "")
            form_id=fffs
            logger.debug(f"Processing first page", extra={'pdf_path': pdf, 'form_id': form_id})
            
            first_page: pymupdf.Page = doc[0]
            page_height = first_page.rect.height
            textBlocks = first_page.get_text("dict",sort=True)
            footer_line = getFooterLine(textBlocks['blocks'], first_page.rect.width)
            
            if footer_line is None:
                logger.warning("Footer line not found, using page bottom as fallback", extra={'pdf_path': pdf})
                footer_line_y = first_page.rect.y1 - 50  # Use bottom 50 pixels as fallback
            else:
                footer_line_y = footer_line[1]
            
            (footerText, footerRect) = getFooterText(footer_line_y, first_page)
            page_count = extractPageCount(footerText)
            existingSections = []
            absentSections  = []
            spans = flatSpans(textBlocks, 0)
            ids = makeSingleLevel(spans)
            gg = list(filter(lambda x: x['size'] == 14 and x['font'].lower().__contains__("bold"), ids))
            ids = "###".join([s['text'] for s in ids[0:ids.index(gg[-1])]])
            ids = regex.findall(r'B\d{1,}|L\d{1,}|\d{2}/\d{2}/\d{4,}',ids)
            rcs_number = ids[0]
            filing_id = ids[1]
            filing_date = ids[2]
            
            logger.info(f"Extracted IDs from first page", 
                      extra={'pdf_path': pdf,
                      'rcs_number': rcs_number,
                      'filing_id': filing_id,
                      'filing_date': filing_date,
                      'page_count': page_count
                      })
            #data.title = getFullTitleBlockText(block, mode=1)#['lines'][0]['spans'][0]['text']
            #data.subtitle = textBlocks['blocks'][1]['lines'][0]['spans'][0]['text']
            drawings = [first_page.get_cdrawings(extended=True)]
            
            logger.info(f"Processing additional pages", 
                      extra={'pdf_path': pdf,
                      'pages_to_process': page_count-1})
            
            for x in range(1, page_count):
                logger.debug(f"Processing page {x}", extra={'pdf_path': pdf, 'page_number': x})
                p = getPageSpans(x, doc[x])
                spans.extend(p[0])
                drawings.append(p[1])
            #doc.close()
        
        logger.info(f"Processing sections", extra={'pdf_path': pdf, 'total_spans': len(spans)})
        spans = splitSpans(spans, drawings, isSection)
        
        logger.info(f"Split into {len(spans)} sections", 
                  extra={'pdf_path': pdf,
                  'sections_count': len(spans)})
        
        # here (spans) var represents a pair of the section title 
        # and its spans array without being spanned on multiple pages
        # next, we need to process each section's data in parallel
        # TODO: bring back parallel processing through celery
        #with billiard.pool.Pool(context=get_context(method="fork")) as exe:#with ProcessPoolExecutor() as exe: #billiard.pool.Pool(context=get_context(method="fork"))
            #res = exe.map(processSection, spans, itertools.repeat(page_height)) 
            #res = exe.starmap(processSection, [[s, page_height] for s in spans])
        
        logger.info(f"Processing sections in parallel", extra={'pdf_path': pdf})
        res = [processSection(s, page_height) for s in spans]
        
        extracted_sections = {}
        for r in res:
            extracted_sections = {**extracted_sections, **r}
        existingSections = list(extracted_sections.keys())
        
        total_duration = time.time() - start_time
        
        logger.info(f"T4 form scraping completed successfully", 
                  extra={'pdf_path': pdf,
                  'form_id': form_id,
                  'page_count': page_count,
                  'rcs_number': rcs_number,
                  'filing_id': filing_id,
                  'filing_date': filing_date,
                  'title': title,
                  'subtitle': subtitle,
                  'existing_sections_count': len(existingSections),
                  'absent_sections_count': len(absentSections),
                  'extracted_sections_count': len(extracted_sections),
                  'total_duration_seconds': total_duration,
                  'success': True
                  })

        return (form_id, page_count, existingSections, absentSections, rcs_number, filing_id, filing_date, title, subtitle, extracted_sections)
    
    except Exception as error:
        duration = time.time() - start_time
        logger.error(f"T4 form scraping failed", 
                   extra={'pdf_path': pdf,
                   'error': str(error),
                   'duration_seconds': duration,
                   'exception': traceback.format_exc()
                   })
        raise