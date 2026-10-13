# PDF to Supabase Embeddings

ชุดไฟล์นี้ใช้แปลง PDF ภาษาไทยเป็น chunks สร้าง vector embeddings และบันทึกเป็น JSONL
โดยสามารถเลือก upload เข้า Supabase `pgvector` ได้ด้วย

## 1. เตรียม Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

การเรียกใช้งานครั้งแรกต้องใช้อินเทอร์เน็ตเพื่อดาวน์โหลดโมเดล `BAAI/bge-m3`

## 2. สร้างไฟล์ embedding ในเครื่อง

```powershell
python .\create_embeddings.py `
  --pdf "C:\path\to\tax.pdf" `
  --title "ชื่อเอกสารภาษี" `
  --tax-year 2568 `
  --source-url "https://www.rd.go.th/..." `
  --document-type "tax_form_instruction"
```

ไฟล์ผลลัพธ์จะอยู่ที่ `output/<ชื่อไฟล์>_embeddings.jsonl` และแต่ละบรรทัดจะมี
ข้อความ เลขหน้า metadata และ embedding ของหนึ่ง chunk

หากไฟล์ผลลัพธ์มีอยู่แล้ว ให้เพิ่ม `--overwrite` เมื่อต้องการสร้างใหม่

## 3. Upload เข้า Supabase

1. เปิด Supabase SQL Editor แล้วรัน `supabase_schema.sql`
2. คัดลอก `.env.example` เป็น `.env`
3. ใส่ `SUPABASE_URL` และ `SUPABASE_SERVICE_KEY`
4. เรียกคำสั่งเดิมโดยเพิ่ม `--upload-supabase`

```powershell
python .\create_embeddings.py `
  --pdf "C:\path\to\tax.pdf" `
  --title "ชื่อเอกสารภาษี" `
  --tax-year 2568 `
  --source-url "https://www.rd.go.th/..." `
  --document-type "tax_form_instruction" `
  --upload-supabase
```

ห้ามนำ `.env` หรือ service-role key ขึ้น GitHub

## 4. นำเข้าเอกสารทั้งโฟลเดอร์

ใช้ `batch_ingest.py` เมื่อต้องการสร้าง embeddings จากไฟล์ PDF/TXT หลายไฟล์
โดยสคริปต์จะโหลดโมเดลเพียงครั้งเดียว บันทึก JSONL ทุกเอกสาร แล้วจึง upload เข้า Supabase

```powershell
.\.venv\Scripts\python.exe -X utf8 .\batch_ingest.py `
  --source-dir "C:\path\to\tax" `
  --output-dir .\output `
  --default-tax-year 2568 `
  --batch-size 8 `
  --upload-supabase `
  --overwrite
```

ชื่อไฟล์ที่มีวันที่รูปแบบ `DDMMYY` จะถูกใช้อนุมานปี พ.ศ. เช่น `241268` เป็นปี 2568
และการรันซ้ำจะ upsert ด้วย hash ของเอกสารและรหัส chunk จึงไม่สร้างข้อมูลซ้ำ

## PDF สแกน

สคริปต์จะแจ้งเลขหน้าที่มีข้อความน้อยผิดปกติ หากเกือบทุกหน้าได้รับคำเตือน ให้ OCR ก่อน เช่น

```powershell
ocrmypdf -l tha+eng --deskew input.pdf output_ocr.pdf
```

เอกสารภาษีที่มีตารางควรตรวจตัวเลขและลำดับคอลัมน์ด้วยคนก่อนนำมาเป็นฐานความรู้
