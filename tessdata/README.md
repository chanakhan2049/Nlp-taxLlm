# Tesseract language data

This directory contains the `tha` and `eng` trained-data files used by the
ingestion pipeline. The Thai model comes from the official
`tesseract-ocr/tessdata_fast` repository. The English model is distributed
with the UB Mannheim Tesseract 5 Windows build.

The pipeline passes this directory explicitly to Tesseract so every machine
uses the same Thai and English language models.
