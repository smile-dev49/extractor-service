# Translation Guide - RCS Extraction Service

## What Needs Translation?

### ✅ Already Handled (No Translation Needed)

1. **Field Name Matching** - Already multilingual
   - The mapping files (`extracted_data_mapping.json`, `section_mapping.json`) already contain field names in multiple languages:
     - **French**: "Dénomination", "Forme juridique", "Siège social"
     - **German**: "Bezeichnung", "Rechtsform", "Sitz der Gesellschaft"
     - **English**: Some English terms
   - The extraction engine matches fields regardless of language

2. **Section Detection** - Already multilingual
   - The engine recognizes sections in all languages:
     - "Modification" (FR) / "Abänderung" (DE)
     - "Inscription" (FR) / "Neueintragung" (DE)
     - "Radiation" (FR) / "Löschung" (DE)

### ❌ What You DON'T Need to Translate

1. **Extracted Data Values** - Keep as-is
   - Company names, addresses, dates, numbers
   - These are extracted exactly as they appear in PDFs
   - No translation needed - they're factual data

2. **API Responses** - Optional
   - Error messages, status messages
   - Typically kept in English for APIs
   - Can be translated if needed for user-facing applications

## Current Multilingual Support

### Field Matching (Already Works)

The extraction engine matches fields in any language:

```json
{
  "dp_id": "DP_003",
  "standard_name": "Entity_BusinessName",
  "field_name": [
    "Dénomination",                    // French
    "Dénomination ou raison sociale",   // French
    "Bezeichnung der Gesellschaft",    // German
    "Bezeichnung"                      // German
  ]
}
```

The engine will find "Dénomination" (FR) or "Bezeichnung" (DE) and map both to the same standard field.

### Section Recognition (Already Works)

The engine recognizes sections in multiple languages:

```python
# From t4.py
MODIFIED_SUBTITLES = regex.compile(
    r'(Modification\s*[/]?[\s*Renouvellement]?|Ab[aä]nderungs*[/]?[\s*Erneuerung]?)',
    flags=regex.IGNORECASE
)
# Matches: "Modification" (FR) or "Abänderung" (DE)
```

## What Translation Would Be Needed?

### Option 1: Translate Extracted Values (Optional)

If you want to translate the actual extracted data values:

```python
# Example: Translate company legal form
extracted_data = {
    "legal_form": "Société à responsabilité limitée"  # French
}

# Translate to English
translated = {
    "legal_form": "Limited Liability Company"  # English
}
```

**This is typically NOT needed** because:
- Company names should stay in original language
- Addresses should stay in original format
- Legal forms are standardized codes

### Option 2: Translate API Error Messages (Optional)

If you want user-facing error messages in different languages:

```python
# Current (English)
{"error": "PDF file not found"}

# Translated (French)
{"error": "Fichier PDF introuvable"}

# Translated (German)
{"error": "PDF-Datei nicht gefunden"}
```

**This is optional** - APIs typically use English.

## Recommendation

### ✅ Keep Current Approach (Recommended)

**No translation needed** because:

1. **Field matching already works** - The engine finds fields regardless of language
2. **Data values should stay original** - Company names, addresses are factual
3. **Standard names are language-neutral** - Output uses standard field names like `Entity_BusinessName`

### Example Output

```json
{
  "rcs_number": "B231239",
  "extracted_sections": {
    "Dénomination": {  // Field name in PDF (French)
      "Entity_BusinessName": "LZMP S.à r.l."  // Standard name (language-neutral)
    },
    "Forme juridique": {  // Field name in PDF (French)
      "Entity_LegalForm": "SARL"  // Standard code
    }
  }
}
```

The output uses:
- **Standard field names** (language-neutral): `Entity_BusinessName`
- **Original values** (as in PDF): "LZMP S.à r.l."
- **Field matching** (works in any language): Finds "Dénomination" or "Bezeichnung"

## If You Need Translation

### For Extracted Values

Add a translation service:

```python
# In helpers.py or separate translation module
def translate_extracted_data(extracted_data: dict, target_language: str = "en"):
    """Translate extracted data values to target language"""
    # Use translation service (e.g., Google Translate API, DeepL)
    translated = {}
    for key, value in extracted_data.items():
        if isinstance(value, str) and needs_translation(key):
            translated[key] = translate_text(value, target_language)
        else:
            translated[key] = value
    return translated
```

### For API Responses

Add language parameter to API:

```python
@app.post("/api/v1/extract")
async def extract_pdf(
    file: UploadFile,
    language: Optional[str] = Form("en")  # en, fr, de
):
    # ... extraction logic ...
    
    if language != "en":
        # Translate error messages
        error_msg = translate_error(result.get("error"), language)
```

## Summary

**Current Status:**
- ✅ Field matching: Already multilingual (FR/DE/EN)
- ✅ Section detection: Already multilingual
- ✅ Data extraction: Works with any language
- ❌ Translation: Not needed for core functionality

**Translation is only needed if:**
1. You want to translate extracted data values (not recommended)
2. You want API error messages in different languages (optional)

**Recommendation:** Keep the current approach - the engine already handles multilingual PDFs correctly without translation.

