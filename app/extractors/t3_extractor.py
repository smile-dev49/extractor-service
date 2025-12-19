import numpy as np
import pymupdf
import pymupdf.pymupdf
import regex
import math
import bisect
import logging
import time
import os
import traceback

# text extraction method: as in the initial entry the text is extracted in blocks with sor from top left enabled
# and then the spans are fused with predefined separator (e.g.: ||sep||) and lines are fused with single newlinee character (\n)

# 1) first of all we need to know the titles present for extraction and total pages:
# this is done by:
#       1.1) extracting marked checkboxes on first page and grabbing the marked titles:
#           - for registration type 2: checkboxes are drawing with 'items' attribute having 3 shapes (re,l,l) with their coressponding points 
#               so we need to have checkNearest logic based on bboxes for the checkbox and text block
#           - for type 3 docs: checkboxes mark are detected in text, no need to apply checkNearest logic
#       1.2) get footer line coords, this is the same for all pages:
#           - the footer line separator is a drawing of type line with width == page width below which all text is footer text
#           - from the footer text:we extract the pattern (page \d .+ \d)i and check the second \d
# 
# 2) for each page: 
#   2.0) extract all text blocks dict to have the tree of blocks -> lines -> spans each with its coressponding bbox anf font data,
#   with minding the footer text:
#       - the footer line separator is a drawing of type line with width == page width below which all text is footer text
#   2.1) check if any -more than one- titles from the marked is present:
#       2.1.1) check the titles by: text regex matching, bold font, and > 12-14 font size
#       2.1.2) pluck the part of the blocks dict from the found title until next found title or till end of page, but be sure to check for 
#              text including "page \d / \d" which signifies footer text and store it to a temp var like: currentlyExtractingTitleDataDict = DataDict
#       2.1.3) add the title to temporary var like: currentlyExtractingTitle = title to signify the title in working so if the content
#              spilled to the next page you know you're still extracting this title and any found textboxes before the new title belong to it
#
#       2.2) for each title :
#           2.2.1) extract all textboxes using the 2 conditional lines specified below
#           2.2.2) if type 2, extract checkboxes; else, check if right sign "✔" is present in text span

SEGSEP='#____SEP____#'

# Setup logging
logger = logging.getLogger(__name__)

def mark_drawings(drawings, page):
    for s in drawings:
        if s.keys().__contains__('rect'):
            page.draw_rect(s['rect'], fill=(0,0,0), width=6, fill_opacity=0.2)
        else:
            page.draw_rect(s['scissor'], fill=(0,0,0), width=6, fill_opacity=0.2)

#---------------------------------------------------------------#
#--------------------First Page Functions-----------------------#
#---------------------------------------------------------------#

def getFooterLine(drawings, pageWidth):
    pageWidth = pageWidth - 10
    for x in drawings:
        if x.keys().__contains__("items") and len(x['items']) == 1 and x['items'] and x['items'][0][0] == "l":
            line = x['items'][0]
            startXPoint = line[1][0]
            endXPoint = line[2][0]
            length = endXPoint - startXPoint 
            if length >= pageWidth:
                return line

def getFooterText(footerLineY, page: pymupdf.Page):
    return (page.get_textbox((0.0, footerLineY, page.rect.x1,  page.rect.y1,)), pymupdf.Rect(0.0, footerLineY, page.rect.x1,  page.rect.y1,))

def extractPageCount(footerText):
    h = regex.findall(r'.+\d+.*\d+\s*$', footerText, flags=regex.IGNORECASE | regex.DOTALL)
    s = regex.split(r'.+1\D', h[0], flags=regex.IGNORECASE | regex.DOTALL)
    h = regex.sub(r'\D', '', s[1])
    h = int(h)
    return h

#---------------------------------------------------------------#
#------------------End of First Page Functions------------------#
#---------------------------------------------------------------#


#---------------------------------------------------------------#
#--------------------List Detection Functions-------------------#
#---------------------------------------------------------------#

def isLine(drawing):
    bb = drawing.keys().__contains__('rect') and drawing.keys().__contains__('items') and len(drawing['items']) == 1 and drawing['type'] == "s" and drawing['items'][0][0] == "l"
    #print(f"isLine{bb}")
    return bb

def getSlope(m1, m2):
    nom = m2 - m1
    denom = m1*m2
    denom = denom+1
    tanAngle = math.atan(nom/denom)
    return abs(tanAngle)

def arePerpendicular(line1, line2):
    ((x1, y1,), (x2, y2),) = line1
    ((x3, y3,), (x4, y4)) = line2
    pp = False
    # Both lines have infinite slope
    if (x2 - x1 == 0 and x4 - x3 == 0):
        pp = False

    # Only line 1 has infinite slope
    elif (x2 - x1 == 0):
        m2 = (y4 - y3) / (x4 - x3)

        if (m2 == 0):
            pp = True
        else:
            pp = False

    # Only line 2 has infinite slope
    elif (x4 - x3 == 0):
        m1 = (y2 - y1) / (x2 - x1);

        if (m1 == 0):
            pp = True
        else:
            pp = False

    else:
        # Find slopes of the lines
        m1 = (y2 - y1) / (x2 - x1)
        m2 = (y4 - y3) / (x4 - x3)

        # Check if their product is -1
        if (m1 * m2 == -1):
            pp = True
        else:
            angle = getSlope(m1, m2)
            if 88.0 <= angle and angle <= 90:
                pp = True
            else:
                pp = False
    #print(f'arePerpendicular {pp}')
    return pp

def isSquare(line1, line2):
    ((x1, y1,), (x2, y2),) = line1
    ((x3, y3,), (x4, y4)) = line2
    h1 = y2-y1
    w1 = x2-x1
    h2 = y4-y3
    w2 = x4-x3
    if abs(w2-h1) <= 5 and abs(h2-w1) <=5:
        return True
    else: 
        return False

def createSquare(drawing, ndrawing) -> pymupdf.Rect:
    (x0, y0, x1, y1) = drawing['rect']
    (x2, y2, x3, y3) = ndrawing['rect']
    Xs = [x0, x1, x2, x3]
    Ys = [y0, y1, y2, y3]
    rect = pymupdf.Rect(Xs[np.argmin(Xs)], Ys[np.argmin(Ys)], Xs[np.argmax(Xs)], Ys[np.argmax(Ys)])
    return rect

def isListItem(dr, nextDr, txt):
    drp1= (dr['items'][0][1], dr['items'][0][2])
    ndrp1=(nextDr['items'][0][1], nextDr['items'][0][2])
    return isSquare(drp1, ndrp1) and arePerpendicular(drp1, ndrp1) and len(regex.regex.findall(r'^\s*\d{1,3}\s*$', txt)) == 1

#---------------------------------------------------------------#
#----------------End of List Detection Functions----------------#
#---------------------------------------------------------------#

def removeFooter(cropRect: pymupdf.Rect, page: pymupdf.Page):
    page.set_cropbox(cropRect)

def isCheckboxType2(drawing, verticalStart=0, verticalEnd= math.inf):
    #TODO: add a check for the type of 3 items, the bbox for the rect and lines and the intersecting angle of the 2 lines (45)
    return drawing.keys().__contains__("items") and drawing.keys().__contains__("rect")  and len(drawing['items']) == 3 and drawing['rect'][1] > verticalStart and drawing['rect'][3] < verticalEnd and abs((drawing['rect'][2] - drawing['rect'][0]) - (drawing['rect'][3] - drawing['rect'][1]) ) <= 5

def isCheckboxType3(drawing, verticalStart=0, verticalEnd= math.inf):
    return drawing.keys().__contains__("items") and drawing.keys().__contains__("rect") and (drawing['items'][0][0] == "re" or len(drawing['items'][0][0]) == 4) and drawing['rect'][1] > verticalStart and drawing['rect'][3] < verticalEnd and len(drawing['items']) == 1 and abs((drawing['rect'][2] - drawing['rect'][0]) - (drawing['rect'][3] - drawing['rect'][1]) ) <= 5

def extractCheckboxes(drawings, type=3, verticalStart=0, verticalEnd= math.inf, banArea: pymupdf.Rect = None):
    l = []
    rs = []
    if type == 2:
        for x in drawings:
            #TODO: add a check for the type of 3 items, the bbox for the rect and lines and the intersecting angle of the 2 lines (45)
            if isCheckboxType2(x, verticalStart=verticalStart, verticalEnd=verticalEnd) and x['rect'] not in rs:
                if banArea is None or (banArea is not None and banArea.contains(x['rect']) == False):
                    l.append(x)
                    rs.append(x['rect'])
    elif type == 3:
        for x in drawings:
            if isCheckboxType3(x, verticalStart=verticalStart, verticalEnd=verticalEnd) and x['rect'] not in rs:
                if banArea is None or (banArea is not None and banArea.contains(x['rect']) == False):
                    l.append(x)
                    rs.append(x['rect'])

    return l

def isTextboxType2(drawing, drawing2, drawing3, verticalStart=0, verticalEnd= math.inf):
    # this condition detects three-line text boxes (i.e.: incomplete textboxes) in registration docs (type 2)
    return drawing.keys().__contains__("rect") and drawing['type'] == "s" and drawing.keys().__contains__("items") and len(drawing['items']) == 1 and drawing['items'][0][0] == "l" and drawing['rect'][1] > verticalStart and drawing['rect'][3] < verticalEnd

def isTextboxType3(drawing, verticalStart=0, verticalEnd= math.inf):
    return drawing.keys().__contains__("items") and drawing.keys().__contains__("rect") and (drawing['type'] == "fs") and len(drawing['items']) == 1  and drawing['items'][0][0] == "re" and drawing['rect'][1] > verticalStart and drawing['rect'][3] < verticalEnd

#TODO:
def mergeThreeLinesIntoTextbox(l1, l2, l3):
    return l1

def extractTextboxes(drawings, type=2, verticalStart=0, verticalEnd= math.inf, banArea: pymupdf.Rect = None):
    l = []
    rs = []
    if type == 2:
        for i, x in enumerate(drawings):
            if isTextboxType2(x, drawings[i+1], drawings[i+2], verticalStart=verticalStart, verticalEnd=verticalEnd) and x['rect'] not in rs:
                if banArea is None or (banArea is not None and banArea.contains(x['rect']) == False):
                    l.append(mergeThreeLinesIntoTextbox(x, drawings[i+1], drawings[i+2]))
                    rs.append(x['rect'])
        l = mergeThreeLinesIntoTextbox(l)
        return l
    elif type == 3:
        for x in drawings:
            if isTextboxType3(x, verticalStart=verticalStart, verticalEnd=verticalEnd) and x['rect'] not in rs:
                if banArea is None or (banArea is not None and banArea.contains(x['rect']) == False):
                    l.append(x)
                    rs.append(x['rect'])

        return l
    else:
        raise RuntimeError("Unsupported Document Type")

def extractDrawings(drawings, type=3, verticalStart=0, verticalEnd= math.inf, banArea: pymupdf.Rect = None, cb=True, txb=True):
    l = []
    if type == 2:
        for i, x in enumerate(drawings):
            if txb and isTextboxType2(x, drawings[i+1], drawings[i+2], verticalStart=verticalStart, verticalEnd=verticalEnd):
                l.append((1, mergeThreeLinesIntoTextbox(x, drawings[i+1], drawings[i+2])))
            if cb and isCheckboxType2(x, verticalStart=verticalStart, verticalEnd=verticalEnd):
                l.append((0, x))
        return l
    elif type == 3:
        for x in drawings:
            if (txb and isTextboxType3(x, verticalStart=verticalStart, verticalEnd=verticalEnd)):
                if banArea is not None:
                    tt = banArea.contains(x['rect'])
                if banArea is None or (banArea is not None and banArea.contains(x['rect']) == False):
                    l.append((1, x))
            if (cb and isCheckboxType3(x, verticalStart=verticalStart, verticalEnd=verticalEnd)):
                if banArea is None or (banArea is not None and banArea.contains(x['rect']) == False):
                    l.append((0, x))

        return l

def getTextBox(rect, spans, join=""):
    if type(rect) == tuple:
        rect = pymupdf.Rect(rect)
    rect.include_rect(pymupdf.Rect(rect.x0-2, rect.y0-2, rect.x1+2, rect.y1+2)) 
    return join.join([span['text'] for span in spans if span.keys().__contains__("bbox") and rect.contains(span['bbox'])])

def getNearestPoint(points, point):
    distances = [math.dist(d, point) for d in points]
    closest = np.argmin(distances)
    return (closest, distances[closest])

def getNearestSpan(spans: list, target, onlyBefore = False, onlyAfter = False, verticalComp=False, HorizontalComp=False, pop=True):
    # get coords of spans
    if type(spans[0]) is dict and spans[0].keys().__contains__("bbox"):
        boxes = [[span['bbox'][0], span['bbox'][1], span['bbox'][2], span['bbox'][3], i] for i,span in enumerate(spans)]
    elif type(spans[0]) is tuple and type(spans[0][2]) is tuple and len(spans[0][2]) == 4:
        boxes = [[span[2][0], span[2][1], span[2][2], span[2][3], i] for i,span in enumerate(spans)]
    else:
        raise TypeError("the type of list is not supported and doesn't have coordinations field")

    # add the top left of the checkbox
    if target.keys().__contains__("bbox"):
        targetRect = target['bbox']
    else:
        targetRect = target['items'][0][1]
    targetRectFormalized = [targetRect[0], targetRect[1], targetRect[2], targetRect[3], -1]
    boxes.append(targetRectFormalized)

    #convert to np array and sort per coords from top left to bottom right
    #boxes = np.array(boxes)
    #indcs = np.lexsort((boxes[:,0], math.sqrt(math.pow(boxes[:,0], 2) + math.pow(boxes[:,1], 2)) ))
    #boxes = boxes[indcs]
    boxes.sort(key= lambda b: (b[3], b[0]) )

    #pprint.pprint("targetRect")
    #pprint.pprint(targetRectFormalized)
    #pprint.pprint("--------------------------------")
    ixd = boxes.index(targetRectFormalized)
    ## for before compare end coords of before with start of target 
    before = boxes[:ixd]
    
    ## for after compare start coords of after with end of target 
    after = boxes[ixd+1:]

    #comparison indices: usually compare horizontally top right of before with top left of after
    #x1, y1 for after
    x1 = 0
    y1 = 1
    #x2, y2 for before
    x2 = 2
    y2 = 1
    
    
    # if vertical comparison: comapare bottom left of before with top left of after
    #if verticalComp:
    #x1, y1 for after
    #vertical coords
    xv1 = 0
    yv1 = 1
    #x2, y2 for before
    xv2 = 0
    yv2 = 3
    
    j = None
    
    if onlyBefore == True or len(after) == 0:
        if HorizontalComp == True and verticalComp == False:
            nbeforeh = getNearestPoint([(x[x2], x[y2]) for x in before], (targetRect[x1], targetRect[y1]))
            j = int(before[nbeforeh[0]][-1])
        elif HorizontalComp == False and verticalComp == True:
            nbeforev = getNearestPoint([(x[xv2], x[yv2]) for x in before], (targetRect[xv1], targetRect[yv1]))
            j = int(before[nbeforev[0]][-1])
        else:
            nbeforeh = getNearestPoint([(x[x2], x[y2]) for x in before], (targetRect[x1], targetRect[y1]))
            nbeforev = getNearestPoint([(x[xv2], x[yv2]) for x in before], (targetRect[xv1], targetRect[yv1]))
            min = np.argmin([nbeforeh[1], nbeforev[1]])
            if min == 0:
                j = int(before[nbeforeh[0]][-1])
            elif min == 1:
                j = int(before[nbeforev[0]][-1])    
    
    elif onlyAfter == True or len(before) == 0:
        if HorizontalComp == True and verticalComp == False:
            nafterh = getNearestPoint([(x[x1], x[y1]) for x in after], (targetRect[x2], targetRect[y2]))
            j = int(after[nafterh[0]][-1])
        elif HorizontalComp == False and verticalComp == True:
            nafterv = getNearestPoint([(x[xv1], x[yv1]) for x in after], (targetRect[xv2], targetRect[yv2]))
            j = int(after[nafterv[0]][-1])
        else:
            nafterh = getNearestPoint([(x[x1], x[y1]) for x in after], (targetRect[x2], targetRect[y2]))
            nafterv = getNearestPoint([(x[xv1], x[yv1]) for x in after], (targetRect[xv2], targetRect[yv2]))
            min = np.argmin([nafterh[1], nafterv[1]])
            if min == 0:
                j = int(after[nafterh[0]][-1])
            elif min == 1:
                j = int(after[nafterv[0]][-1])
    
    
    else:
        if HorizontalComp == False and verticalComp == True:
            nbeforev = getNearestPoint([(x[xv2], x[yv2]) for x in before], (targetRect[xv1], targetRect[yv1]))
            nafterv = getNearestPoint([(x[xv1], x[yv1]) for x in after], (targetRect[xv2], targetRect[yv2]))
            
            min = np.argmin([nbeforev[1], nafterv[1]])
            if min == 0:
                j = int(before[nbeforev[0]][-1])
            elif min == 1:
                j = int(after[nafterv[0]][-1])

        elif HorizontalComp == True and verticalComp == False:
            nbefore = getNearestPoint([(x[x2], x[y2]) for x in before], (targetRect[x1], targetRect[y1]))
            nafter = getNearestPoint([(x[x1], x[y1]) for x in after], (targetRect[x2], targetRect[y2]))

            min = np.argmin([nbefore[1], nafter[1]])
            if min == 0:
                j = int(before[nbefore[0]][-1])
            elif min == 1:
                j = int(after[nafter[0]][-1])

        else:
            nbefore = getNearestPoint([(x[x2], x[y2]) for x in before], (targetRect[x1], targetRect[y1]))
            nbeforev = getNearestPoint([(x[xv2], x[yv2]) for x in before], (targetRect[xv1], targetRect[yv1]))
            nafter = getNearestPoint([(x[x1], x[y1]) for x in after], (targetRect[x2], targetRect[y2]))
            nafterv = getNearestPoint([(x[xv1], x[yv1]) for x in after], (targetRect[xv2], targetRect[yv2]))

            min = np.argmin([nbefore[1], nbeforev[1], nafter[1], nafterv[1]])
            if min == 0:
                j = int(before[nbefore[0]][-1])
            elif min == 1:
                j = int(before[nbeforev[0]][-1])
            elif min == 2:
                j = int(after[nafter[0]][-1])
            elif min == 3:
                j = int(after[nafterv[0]][-1])
    if pop:
        return [spans.pop(j), j]
    else:
        return [spans[j], j]

def addAttribsToSpan(span, lineID, blockID):
    span["blockId"] = blockID
    span["lineId"] = lineID
    return span

def flatSpans(blocks, containCheckMark=False):
    if containCheckMark:
        return [addAttribsToSpan(span, block['lines'].index(line_), blocks['blocks'].index(block)) for block in blocks['blocks'] if block.keys().__contains__("lines") for line_ in block['lines'] for span in line_['spans']]
    else:
        return [addAttribsToSpan(span, block['lines'].index(line_), blocks['blocks'].index(block)) for block in blocks['blocks'] if block.keys().__contains__("lines") for line_ in block['lines'] for span in line_['spans'] if span['font'] != "AdobePiStd"]

def getFullTitleBlockText(block, mode = 0):
    # modes 0, 1 or 2
    #title
    if mode == 0:
        return
    #subtitle
    elif mode == 1:
        s = [span['text'] for line in block['lines'] for span in line['spans'] if span['size'] >= 13 and 'bold' in span['font'].lower() and regex.match(r'^\s*\d+\s*$', span['text']) == None]
        return regex.sub(r'\s{2,}', ' ', ' '.join(s).strip())

def getMultilineText(spans, text, title, start=0):
    start_of_text = 0
    first_captured = False
    for id, span in enumerate(spans):
        if span['size'] >= 13 and 'bold' in span['font'].lower() and regex.match(r'^\s*\d+\s*$', span['text']) == None:
            if first_captured == False:
                start_of_text = id
                first_captured = True
            text += regex.sub(r'\s+', '', span['text'].strip())
            if title == text:
                return (start_of_text+start, text)
    return False

def getSpans(spans, vstart=0, vend=math.inf):
    return [span for span in spans if span.keys().__contains__('bbox') and span['bbox'][1] >= vstart and span['bbox'][3] <= vend]

def grabDP(page: pymupdf.Page, dp, spanList):
    #checkboxes
    if dp[0] == 0:
        dp = dp[1]
        value = page.get_textbox(dp['rect'])
        value = True if value != "" and value != None else False
        label = getNearestSpan(spanList, dp)
    #textboxes
    elif dp[0] == 1:
        dp = dp[1]
        value = page.get_textbox(dp['rect'])
        label = getNearestSpan(spanList, dp, onlyBefore=True, verticalComp=True)
    
    return (label, value)

def formDictKey(nearestLabelbbox, index, text):
    distance = math.sqrt(math.pow(nearestLabelbbox[0], 2) + math.pow(nearestLabelbbox[1], 2))
    labelKey = (index, distance, nearestLabelbbox, text)
    labelKey = SEGSEP.join([str(l) for l in labelKey])
    return labelKey

def getListItemChkbxes(dps, spanWithChecks, spansNoCheck, count, first=False):
    ttt = {}
    lbls = []
    for dp in dps:
        if dp[0] == 0:
            dp = dp[1]
            value = getTextBox(dp['rect'], spanWithChecks)
            value = True if value != "" and value != None else False
            label = getNearestSpan(spansNoCheck, dp, pop=False, onlyBefore=True, verticalComp=True)
            lbls.append(label[0])
        #textboxes
        elif dp[0] == 1:
            dp = dp[1]
            value = getTextBox(dp['rect'], spanWithChecks)
            label = getNearestSpan(spansNoCheck, dp, onlyBefore=True, verticalComp=True, pop=False)
            lbls.append(label[0])
        labelKey = formDictKey(label[0]['bbox'], count, label[0]['text'])
        ttt[labelKey] = value
    if first:
        return [ttt,lbls]
    return [ttt]

#TODO: detect_list function should separate list items on each item with "1" as text as this signifies start of a new list and return list of lists
def detect_list(page: pymupdf.Page, drawings, spans, verticalStart=0, verticalEnd= math.inf):
    listItemNumbersSquares: list[pymupdf.Rect] = []
    firstD = None
    firstS = None
    for i, s in enumerate(drawings):
        if isLine(s) and i+1 < len(drawings) and isLine(drawings[i+1]) and s['rect'][1] >= verticalStart and s['rect'][3] <= verticalEnd  and drawings[i+1]['rect'][1] >= verticalStart and drawings[i+1]['rect'][3] <= verticalEnd:
            sq = createSquare(s, drawings[i+1])
            sq = sq.round().rect
            txt = getTextBox(sq, spans)
            if isListItem(s, drawings[i+1], txt):
                listItemNumbersSquares.append(sq)
    if (len(listItemNumbersSquares) == 0): # no lists detected
        return listItemNumbersSquares
    
    #find end of list
    for d in drawings:
        if d.keys().__contains__('rect') and d['rect'][1] > listItemNumbersSquares[-1].y1 and d['rect'][0] < listItemNumbersSquares[-1].x0: 
            firstD = pymupdf.Rect(d['rect'])
            break

    for d in spans:
        if d.keys().__contains__('bbox') and d['bbox'][1] > listItemNumbersSquares[-1].y1 and d['bbox'][0] < listItemNumbersSquares[-1].x0: 
            firstS = pymupdf.Rect(d['bbox'])
            break

    if firstS is not None and firstD is not None:
        if firstS.y0 < firstD.y0:
            listItemNumbersSquares.append(firstS)
        else:
            listItemNumbersSquares.append(firstD)
    elif firstD is None:
        listItemNumbersSquares.append(firstS)
    else:
        listItemNumbersSquares.append(firstD)
    return listItemNumbersSquares

#TODO: send last detected item and check if startindex > 0 [which means this is orphaned data]
# and add a check for text preceding the first found list element, if found update the last element with it
def extract_list(page: pymupdf.Page, detected_list, section, drawings, spans, spansWithChecks, tx=None, startIndex=0, preObtainedheaders=None):
    listItemNumbersSquares: list[pymupdf.Rect] = detected_list
    if (len(listItemNumbersSquares) == 0): # no lists detected
        return (None, None, None)
    # segment listItemNumbersSquares
    #check if text or textbox list from first item

    aoi = (page.bound().x0, listItemNumbersSquares[0].y0 , page.bound().x1, listItemNumbersSquares[1].y0)
    txtbxes = extractTextboxes(drawings, type=3, verticalStart=aoi[1]-3, verticalEnd=aoi[3]+3)
    
    #------------------------------------
    #data list
    list_ = {}
    if len(txtbxes) > 0:
        datapoints = extractDrawings(drawings, type=3, verticalStart=aoi[1]-5, verticalEnd=aoi[3]+5)
        spans = [s for s in spans if listItemNumbersSquares[0].contains(s['bbox']) == False]
        headers = set()
        if preObtainedheaders is not None:
            headers = preObtainedheaders.copy()
            for h in headers:
                ttop = listItemNumbersSquares[0].y0 - 10
                height_diff = h['bbox'][3] - h['bbox'][1]
                h['bbox'] = (h['bbox'][0], ttop, h['bbox'][2], ttop + height_diff)
        for dp in datapoints:
            if dp[0] == 0:
                dp = dp[1]
                value = getTextBox(dp['rect'], spansWithChecks)
                value = True if value != "" and value != None else False
                if preObtainedheaders is None:
                    label = getNearestSpan(spans, dp, pop=False, onlyBefore=True)
                else:
                    label = getNearestSpan(headers, dp, pop=False, onlyBefore=True)
            #textboxes
            elif dp[0] == 1:
                dp = dp[1]
                value = getTextBox(dp['rect'], spansWithChecks)
                if preObtainedheaders is None:
                    label = getNearestSpan(spans, dp, onlyBefore=True, verticalComp=True, pop=False)
                else:
                    label = getNearestSpan(headers, dp, onlyBefore=True, verticalComp=True, pop=False)
                
            if preObtainedheaders is None:
                headers.add(label[1])
            label = label[0]
            labelKey = formDictKey(label['bbox'], 0, label['text'])

            if list_.keys().__contains__("item"+str(startIndex)):
                list_["item"+str(startIndex)][labelKey] = value
            else:
                list_["item"+str(startIndex)] = {labelKey: value}
            if value != "" or value != False:
                trect = pymupdf.Rect(dp['rect'])
                spans = [s for s in spans if trect.contains(s['bbox']) == False]
        if preObtainedheaders is None:
            hvs = [spans[i] for i in headers]
            for i in headers:
                spans[i] = None
            spans = [s for s in spans if s is not None]
        else:
            hvs = headers
        # repeat for each element
        listItemNumbersSquares.pop(0)
        for i in range(0, len(listItemNumbersSquares) - 1):
            aoi = (page.bound().x0, listItemNumbersSquares[i].y0 , page.bound().x1, listItemNumbersSquares[i+1].y0)
            txtbxes = extractTextboxes(drawings, type=3, verticalStart=aoi[1]-5, verticalEnd=aoi[3]+5)
            #data list
            if len(txtbxes) > 0:
                #get checkboxes
                spans = [s for s in spans if listItemNumbersSquares[i].contains(s['bbox']) == False]
                datapoints = extractDrawings(drawings, type=3, verticalStart=aoi[1]-5, verticalEnd=aoi[3]+5)
                
                for dpx, dp in enumerate(datapoints):
                    if dp[0] == 0:
                        dp = dp[1]
                        value = getTextBox(dp['rect'], spansWithChecks)
                        value = True if value != "" and value != None else False
                        if value != "" or value != False:
                            trect = pymupdf.Rect(dp['rect'])
                            spans = [s for s in spans if trect.contains(s['bbox']) == False]
                        #pprint.pprint(dp)
                        label = getNearestSpan(hvs, dp, pop=False, onlyBefore=True)
                    #textboxes
                    elif dp[0] == 1:
                        dp = dp[1]
                        value = getTextBox(dp['rect'], spansWithChecks)
                        if value != "" or value != False:
                            trect = pymupdf.Rect(dp['rect'])
                            spans = [s for s in spans if trect.contains(s['bbox']) == False]
                        label = getNearestSpan(hvs, dp, onlyBefore=True, verticalComp=True, pop=False)
                    
                    
                    label = label[0]
                    labelKey = formDictKey(label['bbox'], i, label['text'])
                    if list_.keys().__contains__("item"+str(i+1+startIndex)):
                        list_["item"+str(i+1+startIndex)][label['text']] = value
                    else:
                        list_["item"+str(i+1+startIndex)] = {label['text']: value}
                    
        minY = [s['bbox'][1] for s in hvs]
        minY = minY[np.argmin(minY)]
        extent = pymupdf.Rect(x0=page.bound().x0, x1=page.bound().x1, y0=minY, y1=listItemNumbersSquares[-1].y1)
        
        return (extent, hvs, list_)
    
    #------------------------------------
    #subtitles list
    else:
        subtitles = []
        
        if startIndex > 0:
            preSpans = getSpans(spans, vstart=0, vend=aoi[1] - 5)
            foundSubs=[s['text'] for s in preSpans if s['size'] >= 10]
            dps = extractDrawings(drawings, type=3, verticalStart=0, verticalEnd=aoi[1] - 5)
            subtitles.append([0, regex.sub(r'\s{2,}', ' ',' '.join(foundSubs).strip()), dps])
        
        tts = getSpans(spans, vstart= aoi[1] - 5, vend=aoi[3] + 5)
        foundSubs=[s['text'] for s in tts if s['size'] >= 10]
        dps = extractDrawings(drawings, type=3, verticalStart=aoi[1]-3, verticalEnd=aoi[3]+3)
        subtitles.append([0+startIndex, section+SEGSEP+regex.sub(r'\s{2,}', ' ',' '.join(foundSubs).strip()), dps])
            #if foundSubs[-1]['color'] == 21148:
            #    subtitles.append(foundSubs[-1])
            #elif foundSubs[-1]['color'] == 6908265:
            #    ignored_subs.append()
        
        ss = listItemNumbersSquares.pop(0)
        for i in range(0, len(listItemNumbersSquares) - 1):
            aoi = (page.bound().x0, listItemNumbersSquares[i].y0 , page.bound().x1, listItemNumbersSquares[i+1].y0)
            dps = extractDrawings(drawings, type=3, verticalStart=aoi[1]-3, verticalEnd=aoi[3]+3)
            tts = getSpans(spans, vstart=aoi[1]-5, vend=aoi[3]+5)
            foundSubs=[s['text'] for s in tts if s['size'] >= 10]
            subtitles.append([i+1+startIndex, section+SEGSEP+regex.sub(r'\s{2,}', ' ',' '.join(foundSubs).strip()), dps])
        #pprint.pprint(subtitles)
        
        extent = pymupdf.Rect(x0=page.bound().x0, y0=ss.y0 - 5, x1=page.bound().x1, y1=listItemNumbersSquares[-1].y1)
        return (extent, "subtitles", subtitles)

def scrapPageData(page: pymupdf.Page, count, section, drawings, spanList, spanWithChecks, vlimits, banArea=None, subtitles_list_chkbxes_dps=None):
    secDataDict = {}
    subsections = []
    extent = banArea
    headers = None
    List_data = None
    #TODO: detect_list function should separate list items on each item with "1" as text as this signifies start of a new list and return list of lists
    detected_lists = detect_list(page,  drawings, spanList, verticalStart=vlimits[0], verticalEnd=vlimits[1])
    if len(detected_lists) > 0 and getTextBox(detected_lists[0], spanWithChecks).strip() == "1":
        res = extract_list(page, detected_lists, section, drawings, spanList, spanWithChecks)
        extent = res[0]
        headers = res[1]
        List_data = res[2]
    datapoints = extractDrawings(drawings, type=3, verticalStart=vlimits[0], verticalEnd=vlimits[1], banArea=extent)
    if len(detected_lists) == 0 and len(datapoints) == 0:
        return (None, None)
    
    spansNoCheck = [span if span['font'] != "AdobePiStd" else {"bbox": (math.inf, math.inf, math.inf, math.inf)} for span in spanWithChecks ]

    #saving page lists formed keys in array for the possibility of presence of more than 1 list
    listLabelKeys = []
    if List_data is not None:
        listKey = headers+"_list" if headers == "subtitles" else "data_list"
        
        if headers == "subtitles":
            for subi, subtitle in enumerate(List_data):
                ttt = getListItemChkbxes(subtitle[2], spanWithChecks, spansNoCheck, count, first= subi == 0)
                if subi == 0:
                    spansNoCheck = ttt[1]
                List_data[subi][2] = ttt[0]
            
            subsections = List_data
            secDataDict[listKey] = List_data
        
        else:
            listLabelKey = formDictKey(headers[0]['bbox'], -1, listKey)
            listLabelKeys.append(listLabelKey)
            secDataDict[listLabelKey] = List_data
            secDataDict[listLabelKey]['headers'] = headers
    
    inc = 0
    for dpx, dp in enumerate(datapoints):
        if dp[0] == 0:
            dp = dp[1]
            value = getTextBox(dp['rect'], spanWithChecks)
            value = True if value != "" and value != None else False
            label = getNearestSpan([s for s in spanList if s['font'] != "AdobePiStd"], dp, pop=False)
        #textboxes
        elif dp[0] == 1:
            dp = dp[1]
            value = getTextBox(dp['rect'], spanWithChecks)
            label = getNearestSpan(spanList, dp, onlyBefore=True, verticalComp=True, pop=False)
        for lkeyi in listLabelKeys:
            lkeyi = lkeyi.split(SEGSEP)
            if len(lkeyi) != 4: 
                continue
            y0 = lkeyi[2].split(",")[1]
            if float(y0) <= label[0]['bbox'][1]:
                hh = secDataDict[SEGSEP.join(lkeyi)].copy()
                lkeyi[0] = str(dpx+inc)
                secDataDict[SEGSEP.join(lkeyi)] = hh
                lkeyi[0] = str(-1)
                del secDataDict[SEGSEP.join(lkeyi)]
                listLabelKeys.remove(SEGSEP.join(lkeyi))
                inc += 1
        labelKey = formDictKey(label[0]['bbox'], count, label[0]['text'])
        secDataDict[labelKey] = value
        
        if value != "" or value != False:
            trect = pymupdf.Rect(dp['rect'])
            spanList = [s for s in spanList if trect.contains(s['bbox']) == False]
    
    latest_dpx = len(datapoints)
    for lkeyi in listLabelKeys:
            lkeyi = lkeyi.split(SEGSEP)
            if len(lkeyi) != 4: 
                continue
            hh = secDataDict[SEGSEP.join(lkeyi)].copy()
            lkeyi[0] = str(latest_dpx+inc)
            secDataDict[SEGSEP.join(lkeyi)] = hh
            lkeyi[0] = str(-1)
            del secDataDict[SEGSEP.join(lkeyi)]
            listLabelKeys.remove(SEGSEP.join(lkeyi))
            inc += 1
    #pprint.pprint(secsData)
    
    return (secDataDict, subsections)

def assignSubtitlesToData(spanList, secDataDict: dict):
    if len(spanList) > 0 :
        #print("remainder found, checking if subtitles...")
        secKs = list(secDataDict.keys())
        #print(secKs)
        for i in range(len(secKs)):
            k = secKs[i].split(SEGSEP)
            if len(k) != 4: 
                continue
            secKs[i] = (int(k[0]), float(k[1]), tuple([float(s.replace('(', "").replace(')', "")) for s in k[2].split(',')]), k[3])
        #pprint.pprint(secKs)
        ids = [(span, getNearestSpan(secKs, span, onlyAfter=True, pop=False)) for span in spanList if span['size'] >= 10 and 'bold' in span['font'].lower()]
        segmentedKeys = {}
        base_o=0
        secKs = list(secDataDict.keys())
        for x, id in enumerate(ids):
            subtitle = id[0]
            current_ = id[1][0]
            current_ = SEGSEP.join([str(l) for l in current_])
            closest = next(i for i in range(len(secKs)) if secKs[i] == current_)
            if x+1 < len(ids):
                next_ = ids[x+1][1][0]
                next_ = SEGSEP.join([str(l) for l in next_])
                closestAfter = next(i for i in range(len(secKs)) if secKs[i] == next_)
            else:
                closestAfter = len(secKs)
            #print(closest, closestAfter)
            #print(secKs[-1])
            if x == 0:
                base_o = id[1][0][0]
            distance = math.sqrt(math.pow(subtitle['bbox'][0], 2) + math.pow(subtitle['bbox'][1], 2))
            subtitleKey = (base_o, distance, subtitle['bbox'], subtitle['text']+"_subtitle")
            subtitleKey = SEGSEP.join([str(l) for l in subtitleKey])
            if regex.findall(r'.*person.*', subtitle["text"], flags= regex.IGNORECASE | regex.MULTILINE | regex.DOTALL) and subtitle['size'] == 13:
                secDataDict[SEGSEP.join([str(l) for l in (base_o, distance+10, subtitle['bbox'], "Type de personne_backup")])] = {"Type de personne_backup": subtitle["text"]}
            segmentedKeys[subtitleKey] = secKs[closest:closestAfter]
            #base_o += 1
            
        for subtitleKey in segmentedKeys.keys():
            secKeys = segmentedKeys[subtitleKey]
            secDataDict[subtitleKey] = {}
            for kiki in secKeys:
                if secDataDict.keys().__contains__(kiki):
                    secDataDict[subtitleKey][kiki] = secDataDict[kiki]                
                    del secDataDict[kiki] 
        del segmentedKeys
    return secDataDict

def assignOrphanedSections(doc: pymupdf.Document, orphanedData: dict, pageSecs: dict, data: dict, subsections: list):
    pageNumbersContainingSecs = list(pageSecs.keys())
    pageNumbersContainingSecs.sort()
    ssde = subsections
    for k in orphanedData.keys():
        numOfPageOwningOrphanedData = bisect.bisect_left(pageNumbersContainingSecs, k)
        numOfPageOwningOrphanedData = numOfPageOwningOrphanedData - 1
        LastDetectedSectionInPageOwningOrphanedData = pageSecs[pageNumbersContainingSecs[numOfPageOwningOrphanedData]][-1]
        section = LastDetectedSectionInPageOwningOrphanedData
        page = doc[k]
        drawings = page.get_cdrawings(extended=True)
        orphanedDataSpansWithCheckS = orphanedData[k]
        orphanedDataSpans = [span for span in orphanedDataSpansWithCheckS if span['font'] != "AdobePiStd"]
        endlimit = orphanedDataSpans[-1]['bbox'][3]+10
        vlimits = [orphanedDataSpans[0]['bbox'][1]-10, endlimit]
        listNumSquares = detect_list(page, drawings, orphanedDataSpans, verticalStart= vlimits[0], verticalEnd=vlimits[1])
        sectionKeys = list(data[LastDetectedSectionInPageOwningOrphanedData].keys())
        keyOfLastElement: str = sectionKeys[-1]
        extentb=None
        headers = None
        if len(listNumSquares) > 0 and int(getTextBox(listNumSquares[0], orphanedDataSpansWithCheckS).strip()) >= 2:
            if keyOfLastElement.__contains__("data_list"):
                startIndex = len(data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement].keys())
                headers = data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement]['headers']
            if keyOfLastElement.__contains__("subtitles_list"):
                startIndex = len(data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement])
            elif keyOfLastElement.__contains__("subtitle"):
                subtitleKeys = list(data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement].keys())
                subtitleKeys.reverse()
                for key in subtitleKeys:
                    if key.__contains__("data_list"):
                        startIndex = len(data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement][key].keys())
                        headers = data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement][key]['headers']

                        break
                    elif key.__contains__("subtitles_list"):
                        startIndex = len(data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement][key])
                        break
            (extent, h, List_data) = extract_list(page, listNumSquares, section, drawings, orphanedDataSpans, orphanedDataSpansWithCheckS, startIndex=startIndex, preObtainedheaders=headers)
            extentb=extent
            if keyOfLastElement.__contains__("data_list"):
                data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement].update(List_data)
                #data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement] = completeList
            elif keyOfLastElement.__contains__("subtitles_list"):
                spansNoCheck = [span if span['font'] != "AdobePiStd" else {"bbox": (math.inf, math.inf, math.inf, math.inf)} for span in orphanedDataSpansWithCheckS ]
                if len(List_data) > len(listNumSquares):
                    diff = len(List_data) - len(listNumSquares)
                    overflowingElementsFromPrevPage = List_data[:diff]
                    List_data = List_data[diff:]
                    for x, ss in enumerate(List_data):
                        ttt = getListItemChkbxes(ss[2], orphanedDataSpansWithCheckS, spansNoCheck, k, first= x == 0)
                        if x == 0:
                            spansNoCheck = ttt[1]
                        List_data[x][2] = ttt[0]
                    data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement].reverse()
                    subsections.reverse()
                    for i in range(0, len(overflowingElementsFromPrevPage)):
                        ttt = getListItemChkbxes(overflowingElementsFromPrevPage[i][2], orphanedDataSpansWithCheckS, spansNoCheck, k)
                        List_data[x][2] = ttt[0]
                        data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement][i] = [data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement][i][0], data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement][i][1] + " " + overflowingElementsFromPrevPage[i][1], None, ttt[0]]
                        subsections[i] = [subsections[i][0], subsections[i][1] + " " + overflowingElementsFromPrevPage[i][1], None, ttt[0]]
                    data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement].reverse()
                    subsections.reverse()

                data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement].extend(List_data)
                subsections.extend(List_data)
                #data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement] = completeList
            elif keyOfLastElement.__contains__("subtitle"):
                subtitleKeys = list(data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement].keys())
                subtitleKeys.reverse()
                for key in subtitleKeys:
                    if key.__contains__("data_list"):
                        data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement][key].update(List_data)
                        #data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement][key] = completeList
                        break
                    elif keyOfLastElement.__contains__("subtitles_list"):
                        spansNoCheck = [span if span['font'] != "AdobePiStd" else {"bbox": (math.inf, math.inf, math.inf, math.inf)} for span in orphanedDataSpansWithCheckS ]
                        if len(List_data) > len(listNumSquares):
                            diff = len(List_data) - len(listNumSquares)
                            overflowingElementsFromPrevPage = List_data[:diff]
                            List_data = List_data[diff:]
                            for x, ss in enumerate(List_data):
                                ttt = getListItemChkbxes(ss[2], orphanedDataSpansWithCheckS, spansNoCheck, k, first= x == 0)
                                if x == 0:
                                    spansNoCheck = ttt[1]
                                List_data[x][2] = ttt[0]
                            data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement].reverse()
                            subsections.reverse()
                            for i in range(0, len(overflowingElementsFromPrevPage)):
                                ttt = getListItemChkbxes(overflowingElementsFromPrevPage[i][2], orphanedDataSpansWithCheckS, spansNoCheck, k)
                                List_data[x][2] = ttt[0]
                                data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement][i] = [data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement][i][0], data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement][i][1] + " " + overflowingElementsFromPrevPage[i][1], None, ttt[0]]
                                subsections[i] = [subsections[i][0], subsections[i][1] + " " + overflowingElementsFromPrevPage[i][1], None, ttt[0]]
                            data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement].reverse()
                            subsections.reverse()
        
                        data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement].extend(List_data)
                        subsections.extend(List_data)
                        #data[LastDetectedSectionInPageOwningOrphanedData][keyOfLastElement][key] = completeList
                        break
        
        (secDataDict, ss) = scrapPageData(page, k, section, drawings, orphanedDataSpans, orphanedDataSpansWithCheckS, vlimits,banArea=extentb)
        # change it to check if empty not none
        if (secDataDict is None or len(secDataDict.keys()) == 0) and (ss is None or len(ss) == 0):
            continue
        subtitledOrphanedData = assignSubtitlesToData(orphanedDataSpans, secDataDict)
        if keyOfLastElement.endswith("_subtitle") == False:
            data[LastDetectedSectionInPageOwningOrphanedData].update(subtitledOrphanedData)
        else:
            nonSubtitledDPsKeys = []
            for orphanedDataKey in subtitledOrphanedData.keys():
                if orphanedDataKey.endswith("_subtitle") == False:
                    nonSubtitledDPsKeys.append(orphanedDataKey)
                    data[LastDetectedSectionInPageOwningOrphanedData][orphanedDataKey] = subtitledOrphanedData[orphanedDataKey]
            for key in nonSubtitledDPsKeys:
                del subtitledOrphanedData[orphanedDataKey]
        data[LastDetectedSectionInPageOwningOrphanedData].update(subtitledOrphanedData)
    return (subsections, data)

def T3BsharpFormScrapper(pdf) -> tuple:
    """T3 Form Scraper - Extract data from T3 type PDF forms"""
    logger.info(f"Starting T3 form scraping for PDF: {pdf}")
    start_time = time.time()
    
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
        logger.debug(f"Processing T3 PDF file", extra={'pdf_path': pdf, 'file_size_bytes': file_size})
        
        with pymupdf.open(pdf) as doc: 
            fffs = pdf.replace(".pdf", "").replace("Prospect_SRC/", "")
            form_id=fffs
            logger.debug(f"Processing first page", extra={'pdf_path': pdf, 'form_id': form_id})
            
            # deal with first page
            first_page = doc[0]
            drawings = first_page.get_cdrawings(extended=True)
            footer_line = getFooterLine(drawings, first_page.rect.width)
            textBlocks = first_page.get_text("dict",sort=True, flags=pymupdf.TEXT_INHIBIT_SPACES)
            (footerText, footerRect) = getFooterText(footer_line[1][1], first_page)
            
            logger.debug(f"First page processing completed", 
                        extra={'pdf_path': pdf, 
                        'drawings_count': len(drawings),
                        'text_blocks_count': len(textBlocks['blocks'])})
        
            #get title, subtitle, ids and crop the text blocks to start from the title
            logger.debug(f"Extracting IDs and metadata from first page", extra={'pdf_path': pdf})
            for block in textBlocks['blocks']:
                if (block.keys().__contains__("lines") 
                    and 'bold' in block['lines'][0]['spans'][0]['font'].lower() 
                    and block['lines'][0]['spans'][0]['text'].lower().__contains__("helpdesk")
                    and (block['lines'][0]['spans'][0]['size'] < 12 and block['lines'][0]['spans'][0]['size'] >= 7)
                    ):
                    helpdeskLine = block['lines'][0]
                    yy = getNearestSpan([(i,0,d['rect']) for i,d in enumerate(drawings) if d.keys().__contains__("rect") and d['type'] == 's'], helpdeskLine, onlyBefore=True)[0]
                    idsBox = drawings[yy[0]]
                    fspansWithids = flatSpans({'blocks': textBlocks['blocks'][0:textBlocks['blocks'].index(block)+1]}, containCheckMark=True)
                    ids = getTextBox(idsBox['rect'], fspansWithids, join='###')
                    ids = regex.findall(r'B\d{1,}|L\d{1,}|\d{2}/\d{2}/\d{4,}',ids)
                    rcs_number = ids[0]
                    filing_id = ids[1]
                    filing_date = ids[2]
                    textBlocks['blocks'] = textBlocks['blocks'][textBlocks['blocks'].index(block):]
                    
                    logger.info(f"Extracted IDs from first page", 
                              extra={'pdf_path': pdf,
                              'rcs_number': rcs_number,
                              'filing_id': filing_id,
                              'filing_date': filing_date
                              })
                    break

            for block in textBlocks['blocks']:
                if block.keys().__contains__("lines") and 'bold' in block['lines'][0]['spans'][0]['font'].lower() and block['lines'][0]['spans'][0]['size'] > 12:
                    textBlocks['blocks'] = textBlocks['blocks'][textBlocks['blocks'].index(block):]
                    title = getFullTitleBlockText(block, mode=1)#['lines'][0]['spans'][0]['text']
                    subtitle = textBlocks['blocks'][1]['lines'][0]['spans'][0]['text']
                    
                    logger.info(f"Extracted title and subtitle", 
                              extra={'pdf_path': pdf,
                              'title': title,
                              'subtitle': subtitle})
                    break

            fspansWithCheck = flatSpans(textBlocks, containCheckMark=True)
            spans = flatSpans(textBlocks)
            page_count = extractPageCount(footerText)
            existingSections = []
            absentSections  = []
            checkboxes = extractCheckboxes(drawings, verticalStart=textBlocks['blocks'][0]['bbox'][1])

            logger.info(f"Processing checkboxes and sections", 
                      extra={'pdf_path': pdf,
                      'page_count': page_count,
                      'checkbox_count': len(checkboxes)})

            for bx in checkboxes:
                section = getNearestSpan(spans, bx)
                dd = getTextBox(bx['items'][0][1], fspansWithCheck)
                if dd != "" and dd != None:
                    existingSections.append(section[0]['text'])
                else:
                    absentSections.append(section[0]['text'])
            
            logger.info(f"Section analysis completed", 
                      extra={'pdf_path': pdf,
                      'existing_sections_count': len(existingSections),
                      'absent_sections_count': len(absentSections),
                      'existing_sections': existingSections,
                      'absent_sections': absentSections
                      })

            subsections = []
            #start page looping
            secsData = {}
            pagesSecs = {}
            orphaned_page_data = {}
            last_detected_list_section = -1
            
            logger.info(f"Starting page processing loop", 
                      extra={'pdf_path': pdf,
                      'total_pages': page_count,
                      'pages_to_process': page_count-1})
            
            for x in range(1, page_count):
                page = doc[x]
                logger.debug(f"Processing page {x}", extra={'pdf_path': pdf, 'page_number': x})
                
                #remove footer
                removeFooter(pymupdf.Rect(0.0, 0.0, page.rect.x1, footerRect.y0), page)
                blocks = page.get_text("dict", sort=True, flags=pymupdf.TEXT_INHIBIT_SPACES)
                spans = flatSpans(blocks, containCheckMark=True)
                spansWithCheck = flatSpans(blocks, containCheckMark=True)
                drawings = page.get_cdrawings(extended=True)
                sections = []
                
                # get sections of interest that contain data
                noneIds = []
                for id, span in enumerate(spans):
                    if span['text'] in existingSections and span['size'] >= 14 and 'bold' in span['font'].lower():
                        sections.append((None, id, None, None))
                    for xx, subsection in enumerate(subsections):
                        ffd = blocks['blocks'][span['blockId']]
                        ssd = getFullTitleBlockText(ffd, mode=1)
                        ssd = ssd.replace(" ", '')
                        
                        # multiline title/subtitle search update: before checking for subsection existence
                        # we first check if it has been already captured as this may be a span in the next line
                        # if that's the ase then update the id of the section of interest to the last id resembling 
                        # the last span in the last line of this multiline title/subtitle
                        #if (
                        #    len(sections) > 0 and sections[-1][0] is not None 
                        #    and (
                        #        sections[-1][0].split(SEGSEP)[-1].strip().replace(" ", '') == ssd or 
                        #            (
                        #            sections[-1][0].split(SEGSEP)[-1].replace(" ", '').__contains__(ssd) 
                        #            and regex.match(r'^\s*\d+\s*$', ssd) == None
                        #            )
                        #        ) 
                        #    and span['size'] >= 13 and 'bold' in span['font'].lower() 
                        #    and sections[-1][2] is not None and sections[-1][2] == last_detected_list_section
                        #    ):
                        #    sections[-1] = (sections[-1][0], id, last_detected_list_section)
                        #    continue
                        
                        # duplicate title addition
                        #check if the number detected (can be detected under same font conditions of the subtitle) == subtitle index - 1
                        # if so, update the last_detected_list_section
                        # detection happen on 2 parts
                        # 1. detect number based on font conditions 
                        # 2. get the area of page containing the number with full width and some wider Height
                        # e.g (0, spany0-10, page width, spany1+10) and detect a list box just like in detecting a list
                        # in page, then check if it contains the number
                        
                        # 1. detect number based on font conditions 
                        if span['size'] >= 13 and 'bold' in span['font'].lower() and regex.match(r'^\s*\d+\s*$',  span['text']) != None:
                            # 2. get the area of page containing the number with full width and some wider Height
                            # e.g (0, spany0-5, page width, spany1+5) and detect a list box just like in detecting a list
                            # in page, then check if it contains the number
                            list_num_Square = detect_list(page, drawings, [span], verticalStart=0, verticalEnd=span['bbox'][3] + 30)
                            if len(list_num_Square) > 0 and getTextBox(list_num_Square[0], [span]).strip() == span['text'].strip():
                                last_detected_list_section = int(span['text'].strip()) - 1
                        # end of duplicate title addition
                        
                        #if not then continue the normal flow of searching in subsections
                        if subsection is not None and span['size'] >= 13 and 'bold' in span['font'].lower() and regex.match(r'^\s*\d+\s*$', span['text']) == None:
                            # first check if concating block spans' text is enough
                            ssubsec = subsection[1].split(SEGSEP)[-1].strip().replace(" ", '')
                            #if ssubsec == ssd and subsection[0] == last_detected_list_section:
                            #    sections.append((subsection[1], id, xx))
                            #    noneIds.append(xx)
                            if ssubsec == regex.sub(r'\s+', '', span['text'].strip()):
                                sections.append((subsection[1], id, last_detected_list_section, subsection[2]))
                                noneIds.append(xx)
                            #if not, apply iteration function for more general search 
                            elif ssubsec.startswith(regex.sub(r'\s+', '', span['text'].strip())):
                                resss = getMultilineText(spans[id+1:], regex.sub(r'\s+', '', span['text'].strip()), ssubsec, start=id+1)
                                if resss != False and subsection[0] == last_detected_list_section:
                                    (lid, fullSubtitle) = resss
                                    sections.append((subsection[1], id, last_detected_list_section, subsection[2]))
                                    noneIds.append(xx)
                
                for sid in noneIds:
                    subsections[sid] = None
                # if sections found in page: add a reference with page number as key and sections titles arr in order of presence
                # this will be used later to assign orphaned data to the last found section in previous pages (not necessarily the immediate before)
                if len(sections) > 0:
                    pagesSecs[x] = [spans[section[1]]['text'] if section[0] is None else section[0] for section in sections ]
                            
                #orphaned_data represents overflowed data points
                if len(sections) == 0 or (len(sections) > 1 and sections[0][1] > 0):
                    idx = len(spans) if len(sections) == 0 else sections[0][1]
                    orphaned_data = spansWithCheck[0: idx]
                    orphaned_page_data[x] = orphaned_data
                    (oS, secsData) = assignOrphanedSections(doc, orphaned_page_data, pagesSecs, secsData, subsections)
                    del orphaned_page_data[x]
                
                sections.append((None, len(spans) - 1, None, None))

                if len(sections) == 1 and sections[-1][1] == len(spans)-1:
                    continue
                else:
                    spans2 = [ 
                                [sections[i][0], spans[sections[i][1]:sections[i+1][1]], sections[i][3]] for i in range(0, len(sections)-1)
                            ]
                    #for i in range(0, len(sections)-1):
                    #    tempspans = spans[sections[i][1]:sections[i+1][1]]
                    #    # to contain all possible data: add a coordinates check within the limits of the extracted spans, 
                    #    # so if some datapoints are missed due to reading order (e.g: deleted / modified check boxes),
                    #    # they are added by coordinates check
                    #    sss = tempspans[0]['bbox']
                    #    coordssspans = [ s for s in spans[0:sections[i][1] + 1] if (s['bbox'][1] >= sss[1]-3 and s['bbox'][3] <= sss[3]+3) ]
                    #    tempspans = coordssspans + tempspans
                    #    spans2.append( 
                    #                [sections[i][0], tempspans] 
                    #    )
                    spans2[-1][1].append(spans[-1])
                    spans = spans2
                for iii, spanSecPair in enumerate(spans):
                    spanList = spanSecPair[1]
                    #print(f'iii: {iii}, ')
                    endlimit = spans[iii+1][1][0]['bbox'][1] if iii+1 < len(spans) else page.rect.y1
                    vlimits = [spanList[0]['bbox'][1], endlimit]
                    startSpan = spanList.pop(0)
                    if spanSecPair[0] is None:
                        section = startSpan['text']
                    else:
                        section = spanSecPair[0]
                    
                    (secDataDict, extractedSubsections) = scrapPageData(page, x, section, drawings, spanList, spansWithCheck, vlimits, subtitles_list_chkbxes_dps=spanSecPair[2])
                    if secDataDict is None and extractedSubsections is None:
                        continue
                    subsections.extend(extractedSubsections)
                    secsData[section] = assignSubtitlesToData(spanList, secDataDict)
                    if spanSecPair[2] is not None and spanSecPair[2] != []:
                        tt = {}
                        for k in spanSecPair[2].keys():
                            l = k.split(SEGSEP)
                            l[0] = str(x)
                            l = SEGSEP.join(l)
                            tt[l] = spanSecPair[2][k]
                        secsData[section] = {**tt, **secsData[section]}
            #assignOrphanedSections(doc, orphaned_page_data, pagesSecs, secsData)
            #doc.save(pdf.replace('.pdf', '_list_marked.pdf'))
            extracted_sections = secsData
            
            total_duration = time.time() - start_time
            
            logger.info(f"T3 form scraping completed successfully", 
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
            #save doc if needed eg for visualizing boxes
                # mark_drawings(textboxes, page)
            #doc.save(pdf.replace('.pdf', '_marked.pdf'))
    
    except Exception as error:
        duration = time.time() - start_time
        logger.error(f"T3 form scraping failed", 
                   extra={'pdf_path': pdf,
                   'error': str(error),
                   'duration_seconds': duration,
                   'exception': traceback.format_exc()
                   })
        raise
