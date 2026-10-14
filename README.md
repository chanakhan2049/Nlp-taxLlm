# Tax LLM

ระบบถาม-ตอบข้อมูลภาษีเงินได้บุคคลธรรมดา โดยค้นข้อมูลจากฐานความรู้ใน
Supabase แล้วส่งเอกสารที่เกี่ยวข้องให้โมเดลภาษาบน Ollama สร้างคำตอบภาษาไทย

ฐานความรู้และ embeddings ถูกจัดเตรียมไว้ใน Supabase แล้ว ผู้ใช้งานทั่วไป
ไม่ต้องวางไฟล์ในโฟลเดอร์ `data`, ไม่ต้อง train โมเดล และไม่ต้องรัน
`batch_ingest.py`

## เริ่มใช้งาน

### 1. ติดตั้ง Python dependencies

เปิด PowerShell ในโฟลเดอร์โปรเจกต์แล้วรัน:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. ติดตั้งและเตรียม Ollama

ติดตั้ง Ollama บนเครื่อง แล้วดาวน์โหลดโมเดลครั้งแรก:

```powershell
ollama pull qwen2.5:7b
```

เปิด Ollama ไว้ขณะใช้งาน หาก service ไม่ทำงานให้รัน `ollama serve`

### 3. สร้างไฟล์ `.env`

สร้างไฟล์ชื่อ `.env` ที่โฟลเดอร์หลักของโปรเจกต์ และรับค่า Supabase จาก
ผู้ดูแลโปรเจกต์ผ่านช่องทางที่ปลอดภัย:

```env
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_SERVICE_KEY=YOUR_SERVICE_ROLE_KEY

HF_TOKEN=hf_YOUR_TOKEN

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_NUM_CTX=8192
OLLAMA_CONNECT_TIMEOUT=10
OLLAMA_READ_TIMEOUT=300
OLLAMA_TEMPERATURE=0.5
```

`HF_TOKEN` ไม่บังคับ เพราะ `BAAI/bge-m3` เป็นโมเดลสาธารณะ แต่การใส่ token
ช่วยเพิ่ม rate limit และลดคำเตือนจาก Hugging Face ตอนดาวน์โหลดครั้งแรก

`localhost` หมายถึง Ollama ที่ทำงานอยู่บนเครื่องของผู้ใช้แต่ละคน จึงไม่ต้อง
เปลี่ยนเป็น IP ของเครื่องผู้พัฒนา

ห้าม commit ไฟล์ `.env` หรือเผยแพร่ Supabase service-role key และ token
ขึ้น GitHub

### 4. ถามคำถาม

ถามหนึ่งคำถามจาก command line:

```powershell
.\.venv\Scripts\python.exe .\ask.py "ผู้มีเงินได้เฉพาะเงินเดือนต้องยื่นแบบอะไร"
```

หรือเปิดโหมดถามต่อเนื่อง:

```powershell
.\.venv\Scripts\python.exe .\ask.py
```

ครั้งแรกอาจใช้เวลาสักครู่เพื่อดาวน์โหลดและโหลด embedding model
`BAAI/bge-m3` หลังจากนั้นระบบจะค้น Top 5 chunks เพื่อสร้างคำตอบ และแสดง
แหล่งอ้างอิง 2 รายการแรก

## ประเมินผล

ทดสอบ retrieval ด้วยคำถามมาตรฐาน 20 ข้อ:

```powershell
.\.venv\Scripts\python.exe .\evaluate.py
```

ทดสอบ retrieval พร้อมให้ Ollama สร้างคำตอบ:

```powershell
.\.venv\Scripts\python.exe .\evaluate.py --with-llm
```

## การทำงานของระบบ

```text
คำถาม -> BGE-M3 embedding -> Supabase pgvector
       -> hybrid reranking -> Top 5 chunks -> Ollama -> คำตอบภาษาไทย
```

ค่าหลักของระบบ:

- Embedding model: `BAAI/bge-m3`
- Embedding dimension: 1,024
- Vector similarity: cosine similarity
- Retrieval: Top 5 chunks
- แหล่งอ้างอิงที่แสดง: 2 รายการ
- Ollama model: `qwen2.5:7b`
- Temperature: `0.5`
- ปีภาษีเริ่มต้น: 2568

## ไฟล์สำคัญ

- `ask.py` — รับคำถาม ค้นเอกสาร และสร้างคำตอบด้วย Ollama
- `create_embeddings.py` — อ่านเอกสาร สร้าง embeddings อัปโหลด และค้นข้อมูล
- `evaluate.py` — ประเมิน retrieval และทดลองสร้างคำตอบ
- `evaluation_questions.json` — ชุดคำถามมาตรฐานสำหรับประเมินผล
- `supabase_schema.sql` — schema และฟังก์ชันค้นหาใน Supabase
- `batch_ingest.py` — นำเข้าเอกสารใหม่สำหรับผู้ดูแลฐานข้อมูล
- `refresh_embeddings.py` — สร้าง embeddings ใหม่จากข้อความใน Supabase

## สำหรับผู้ดูแล: อัปเดตฐานความรู้

ส่วนนี้ไม่จำเป็นสำหรับผู้ใช้งานทั่วไป ใช้เฉพาะเมื่อมีการเพิ่ม แก้ไข หรือทำความ
สะอาดเอกสารต้นฉบับในโฟลเดอร์ `llm things`

ติดตั้ง Tesseract 5 ก่อน ingestion:

```powershell
winget install UB-Mannheim.TesseractOCR
```

จากนั้นนำเข้าเอกสารทั้งหมดใหม่:

```powershell
.\.venv\Scripts\python.exe .\batch_ingest.py ".\llm things"
```

pipeline รองรับ PDF, TXT และ Markdown ทำ Unicode normalization และใช้
Tesseract OCR (`tha+eng`) อัตโนมัติเมื่อ PDF เป็นภาพหรือมี text layer เสียจาก
custom font โดยโมเดลภาษา OCR อยู่ในโฟลเดอร์ `tessdata`

หากต้องสร้าง embeddings ใหม่จากข้อความเดิมใน Supabase โดยไม่อ่านไฟล์ต้นฉบับ:

```powershell
.\.venv\Scripts\python.exe .\refresh_embeddings.py
```

ควรตรวจตัวเลข ตารางภาษี และผล OCR ด้วยคนก่อนนำข้อมูลใหม่ไปใช้งานจริง
