# Thai Tax RAG with Supabase

โปรเจกต์นี้นำเอกสารภาษี PDF/TXT มาแบ่งเป็น chunks สร้าง embeddings ด้วย
`BAAI/bge-m3` และเก็บใน Supabase `pgvector` เพื่อค้นข้อมูลด้วย cosine similarity

## Pipeline

```text
PDF/TXT -> Clean text -> Chunk + overlap -> BGE-M3 embedding
        -> Supabase pgvector -> Top-k retrieval
```

ค่าที่ใช้:

- Chunk size: 1,800 ตัวอักษร
- Chunk overlap: 300 ตัวอักษร
- Embedding model: `BAAI/bge-m3`
- Embedding dimension: 1,024
- Similarity: cosine similarity
- Retrieval: `k = 5`
- `document_year`: ปีที่เผยแพร่เอกสาร โดยอนุมานจากวันที่ในชื่อไฟล์
- `tax_year`: ปีภาษีที่เอกสารยังใช้เป็นข้อมูลอ้างอิง (ปัจจุบันกำหนดเป็น 2568)

## ไฟล์สำคัญ

- `create_embeddings.py` — ฟังก์ชันอ่านเอกสาร, chunk, embedding, upload และ retrieval
- `batch_ingest.py` — เรียก pipeline กับเอกสารทุกไฟล์
- `supabase_schema.sql` — ตาราง `documents`, `document_chunks` และฟังก์ชันค้นหา

## วิธีใช้งาน

### 1. ติดตั้ง dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. เตรียม Supabase

1. รัน `supabase_schema.sql` ใน Supabase SQL Editor (รันซ้ำได้เพื่อเพิ่ม
   `document_year` ให้ฐานข้อมูลเดิม)
2. คัดลอก `.env.example` เป็น `.env`
3. ใส่ Project API URL และ service-role key

```env
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_SERVICE_KEY=YOUR_SERVICE_ROLE_KEY
```

ห้ามส่งไฟล์ `.env` หรือ service-role key ให้อื่น และห้ามนำขึ้น GitHub

### 3. นำเข้าเอกสาร

วางเอกสารไว้ในโฟลเดอร์ `data` แล้วรัน:

```powershell
python .\batch_ingest.py
```

หรือระบุโฟลเดอร์เอกสารเอง:

```powershell
python .\batch_ingest.py "C:\path\to\tax-documents"
```

เมื่อ upload เสร็จ สคริปต์จะทดสอบ retrieval ด้วยคำถามตัวอย่างและแสดง
similarity score, ชื่อเอกสาร และเลขหน้า

ควรรัน ingestion ใหม่หลังเปลี่ยน metadata เพื่อให้ชื่อเอกสาร ปีที่เผยแพร่
ปีภาษีที่ยังใช้อ้างอิง และ embeddings สอดคล้องกันทั้งหมด

หากไม่มีไฟล์ต้นฉบับแล้ว สามารถสร้าง embeddings ใหม่จากข้อความที่เก็บไว้ใน
Supabase ได้ โดยข้อความที่ตรวจพบว่าเสียจะถูกทำเครื่องหมายเพื่อไม่ให้ retrieval ใช้:

```powershell
python .\refresh_embeddings.py
```

### 4. ถามคำถามด้วย Ollama

เปิด Ollama และดาวน์โหลดโมเดลก่อนใช้งานครั้งแรก:

```powershell
ollama pull qwen2.5:7b
python .\ask.py "ผู้มีเงินได้เฉพาะเงินเดือนต้องยื่นแบบอะไร"
```

ค่า timeout เริ่มต้นคือ 10 วินาทีสำหรับการเชื่อมต่อ และ 300 วินาทีสำหรับ
การรอคำตอบ หากโมเดลใช้เวลาโหลดนานสามารถตั้ง `OLLAMA_READ_TIMEOUT` ใน `.env`
ให้สูงขึ้นได้

คำตอบจะใช้ถ้อยคำและโครงสร้างที่หลากหลายขึ้นโดยค่าเริ่มต้น หากต้องการปรับระดับ
ความหลากหลายให้ตั้ง `OLLAMA_TEMPERATURE` ใน `.env` เป็นค่าระหว่าง `0` ถึง `2`
(ค่าเริ่มต้น `0.7`) ค่าต่ำจะให้คำตอบคงที่กว่า ส่วนค่าสูงจะหลากหลายกว่า

### 5. ประเมินผล

ทดสอบ retrieval ด้วยคำถามมาตรฐาน 20 ข้อ:

```powershell
python .\evaluate.py
```

รวมการสร้างคำตอบโดยไม่แสดงหมายเลขอ้างอิงในวงเล็บเหลี่ยม:

```powershell
python .\evaluate.py --with-llm
```

## หมายเหตุ

- ชื่อไฟล์ที่มีวันที่ `DDMMYY` ใช้อนุมานปีที่เผยแพร่เอกสาร เช่น `241268`
  หมายถึงเอกสารเผยแพร่ปี 2568 ไม่ใช่ปีภาษีที่เอกสารใช้ได้
- รองรับไฟล์ PDF, TXT และ Markdown (`.md`)
- pipeline ทำ Unicode normalization และล้างช่องว่าง/อักขระล่องหนก่อนแบ่ง chunk
- PDF ที่เป็นภาพหรือมีข้อความเสียจาก custom font จะใช้ Tesseract OCR (`tha+eng`) อัตโนมัติ
- ติดตั้ง Tesseract 5 ก่อน ingestion; บน Windows ใช้ `winget install UB-Mannheim.TesseractOCR`
- หาก `tesseract` ไม่อยู่ใน PATH ให้กำหนด `TESSERACT_CMD` ใน `.env`
- โมเดลภาษา OCR อยู่ใน `tessdata/` และ metadata ของแต่ละ chunk ระบุ `extraction_method`
- ควรตรวจตัวเลขและตารางภาษีด้วยคนก่อนใช้เป็นฐานความรู้
