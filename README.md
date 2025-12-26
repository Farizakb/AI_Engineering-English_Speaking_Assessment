# 📚 English Speaking Assessment - MVP

A simple, working web app for teachers to review students' spoken English submissions.

**Features:**
- 🎤 Upload WhatsApp voice notes (.ogg, .opus, .mp3, .m4a, .wav, .webm)
- 🔄 Auto-normalize audio to 16kHz mono WAV
- 🎯 Self-hosted STT using openai-whisper
- 💡 AI-powered feedback from OpenAI
- 📊 Separate views for teachers and students
- ⚡ Fast, minimal MVP implementation

---

## 📋 Quick Start

### Prerequisites

- Docker & Docker Compose (recommended)
- OR: Python 3.10+, Node.js 18+, FFmpeg

### Option A: Docker (Easiest)

```bash
# 1. Clone or download this project
cd english-speaking-assessment

# 2. Set up environment
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-your-actual-key-here

# 3. Run with Docker Compose
docker-compose up --build

# 4. Open browser
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
```

The backend will automatically download the Whisper model on first transcription (15-20s wait).

### Option B: Local Development

#### Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file in project root
cp ../.env.example ../.env
# Edit and add your OpenAI API key

# Run backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Requirements:**
- FFmpeg must be installed: `brew install ffmpeg` (Mac) or `apt-get install ffmpeg` (Linux)
- On Windows: Download from https://ffmpeg.org/download.html

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
REACT_APP_API_URL=http://localhost:8000 npm start
```

Open http://localhost:3000 in your browser.

---

## 🔑 Configuration

Create a `.env` file in the project root:

```env
# Required: OpenAI API key for feedback generation
OPENAI_API_KEY=sk-xxxxxxxxxxxxx

# Optional: whisper model size (tiny|base|small|medium|large)
# Default: small (good balance of speed/quality)
STT_MODEL_SIZE=small

# Optional: max audio duration in minutes
# Default: 10
MAX_AUDIO_MINUTES=10
```

**Model sizes:**
- `tiny`: Fastest, ~1GB RAM, lower accuracy
- `base`: Fast, ~1GB RAM
- `small`: Recommended MVP, ~2GB RAM
- `medium`: Slow, ~5GB RAM, high accuracy
- `large`: Very slow, ~10GB RAM, best accuracy

---

## 🚀 API Endpoints

### 1. Upload Submission
```bash
POST /api/submission
Content-Type: multipart/form-data

Parameters:
  - audio_file: (binary) Audio file
  - task_topic: (string) Speaking task/prompt
  - dialect: (string, optional) "US" or "UK" [default: US]

Response:
{
  "submission_id": "a1b2c3d4e5f6",
  "status": "uploaded"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/submission \
  -F "audio_file=@voice_note.ogg" \
  -F "task_topic=Describe your weekend" \
  -F "dialect=US"
```

### 2. Transcribe Audio
```bash
POST /api/submission/{submission_id}/transcribe

Response:
{
  "submission_id": "a1b2c3d4e5f6",
  "transcript": "I spent my weekend...",
  "duration_seconds": 45.2,
  "word_count": 120,
  "confidence": "high",
  "status": "transcribed"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/submission/a1b2c3d4e5f6/transcribe
```

### 3. Generate Feedback
```bash
POST /api/submission/{submission_id}/feedback

Response:
{
  "submission_id": "a1b2c3d4e5f6",
  "feedback": {
    "task_topic": "Describe your weekend",
    "student_level_guess": "B1",
    "teacher_summary": {
      "overall": "Good effort with clear pronunciation. Some grammar inconsistencies.",
      "strengths": [
        "Clear, natural pacing",
        "Good use of past tense",
        "Fluent delivery"
      ],
      "focus_next": [
        "Subject-verb agreement",
        "Article usage (a/the)",
        "More varied vocabulary"
      ]
    },
    "student_feedback": {
      "quick_message": "Nice work! Your pronunciation is clear. Try to focus on article usage.",
      "top_fixes": [
        {
          "original": "I go to beach",
          "better": "I went to the beach",
          "why": "Past tense + article needed"
        }
      ],
      "better_version": "I went to the beach and had a great time with my family."
    }
  },
  "status": "feedback_generated"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/submission/a1b2c3d4e5f6/feedback
```

### 4. Get Full Submission
```bash
GET /api/submission/{submission_id}

Response:
{
  "submission_id": "a1b2c3d4e5f6",
  "metadata": {
    "task_topic": "Describe your weekend",
    "dialect": "US",
    "created_at": "2024-12-21T10:30:00",
    "original_filename": "voice_note.ogg",
    "original_size_bytes": 45000
  },
  "transcript": { ... },
  "feedback": { ... }
}
```

**Example:**
```bash
curl http://localhost:8000/api/submission/a1b2c3d4e5f6
```

### 5. Health Check
```bash
GET /api/health

Response:
{
  "status": "ok",
  "ffmpeg": "installed",
  "openai_key": "set",
  "stt_model": "small",
  "max_audio_minutes": 10
}
```

---

## 📂 File Structure

```
english-speaking-assessment/
├── backend/
│   ├── main.py                 # FastAPI app with all endpoints
│   └── requirements.txt          # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js               # Main app component
│   │   ├── App.css              # Styling
│   │   ├── index.js             # React entry point
│   │   └── components/
│   │       ├── UploadForm.js     # File upload component
│   │       ├── TranscriptDisplay.js  # Transcript view
│   │       ├── FeedbackDisplay.js    # Feedback view (teacher + student tabs)
│   │       └── ErrorAlert.js     # Error messages
│   ├── public/
│   │   └── index.html
│   └── package.json
├── data/
│   └── submissions/             # Stores all submission data
│       └── {submission_id}/
│           ├── original.*       # Original uploaded audio
│           ├── normalized.wav   # Processed audio
│           ├── metadata.json    # Submission metadata
│           ├── transcript.json  # STT result
│           └── feedback.json    # LLM feedback
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── .env.example
└── README.md
```

---

## 🎯 Workflow Example

### Step 1: Teacher Uploads Audio
```
Teacher selects:
- Speaking task: "Describe your favorite movie"
- Audio file: recording.ogg (WhatsApp voice note)

System returns: submission_id = "abc123def456"
```

### Step 2: System Transcribes
```
Backend processes:
1. Validates audio format
2. Normalizes to 16kHz mono WAV
3. Runs openai-whisper (small model) on normalized audio
4. Stores transcript JSON
```

### Step 3: System Generates Feedback
```
LLM processes:
1. Analyzes transcript + task_topic
2. Estimates CEFR level (A2|B1|B2|C1)
3. Creates teacher summary (overall + strengths + focus areas)
4. Creates student feedback (message + corrections + better version)
5. Stores feedback JSON
```

### Step 4: Teacher Reviews
```
Teacher sees:
- Transcribed text (verbatim)
- Teacher Summary tab:
  * Overall assessment
  * 3 strengths
  * 3 areas for focus
- Student Feedback tab:
  * Friendly message for student
  * Top 6 grammatical/pronunciation corrections
  * Natural "better version" of response

Teacher can:
- Export PDF (TODO)
- Add notes (TODO)
- Send to student via WhatsApp (TODO)
```

---

## ⚙️ Technical Details

### Audio Processing
- **Supported formats:** .ogg, .opus, .mp3, .m4a, .wav, .webm
- **Normalization:** FFmpeg converts all to 16kHz mono WAV
- **Long audio handling:** If >6 minutes, splits into 90s chunks, transcribes each, merges results
- **Silent audio rejection:** Returns error if audio is too short or contains only silence

### Speech-to-Text (STT)
- **Provider:** openai-whisper (OpenAI Whisper package)
- **Model download:** ~9GB for "small" model (auto-downloaded on first use)
- **Confidence scoring:**
  - High: 1.5-3.5 words/second + 20+ words
  - Medium: 10-20 words
  - Low: <10 words or unusual pacing

### Feedback Generation
- **LLM:** OpenAI gpt-4o-mini
- **Prompt engineering:** Strict JSON format + no-invention rules
- **CEFR levels:** A2 (elementary), B1 (intermediate), B2 (upper-intermediate), C1 (advanced)
- **Corrections:** Only top 6, with exact quotes from transcript
- **Fillers:** Not penalized unless very frequent

### Storage
- **Location:** `./data/submissions/{submission_id}/`
- **Format:** JSON files (human-readable, easy to migrate)
- **Persistence:** All data saved to disk (survives container restart)

---

## 🔧 Troubleshooting

### "ffmpeg not found"
**Solution:** Install FFmpeg
```bash
# Mac
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
# Add to PATH environment variable
```

### "OPENAI_API_KEY not set"
**Solution:** Create `.env` file with key
```bash
echo "OPENAI_API_KEY=sk-your-key" > .env
```

### Backend container won't start
```bash
# Check logs
docker-compose logs backend

# Rebuild
docker-compose up --build backend

# Common issue: Port 8000 already in use
# Solution: Change port in docker-compose.yml or kill process
```

### Audio normalization fails
- Audio format not supported → try converting to .wav first
- Audio file corrupted → re-record or use different file
- FFmpeg not in container → rebuild Dockerfile.backend

### LLM feedback is empty
- OpenAI API key invalid → verify in `.env`
- Rate limit hit → wait a minute and retry
- Transcript too short (<10 words) → results in minimal feedback

### Transcription is garbled
- Audio quality very low → ask student to re-record
- Non-English speech → set `dialect` parameter
- Model confidence is "low" → output will reflect this in teacher summary

---

## 📊 Sample Data

### Sample Submission Workflow

```bash
# 1. Upload
curl -X POST http://localhost:8000/api/submission \
  -F "audio_file=@sample.ogg" \
  -F "task_topic=Tell me about your morning routine" \
  -F "dialect=US"

# Response:
# { "submission_id": "xyz789", "status": "uploaded" }

# 2. Transcribe
curl -X POST http://localhost:8000/api/submission/xyz789/transcribe

# 3. Get Feedback
curl -X POST http://localhost:8000/api/submission/xyz789/feedback

# 4. View All
curl http://localhost:8000/api/submission/xyz789
```

---

## 🔮 Future Enhancements (TODO)

**Core Features:**
- [ ] Teacher authentication & student management
- [ ] Async queue for transcription (Celery/RabbitMQ)
- [ ] Diarization for multi-speaker audio
- [ ] Per-student progress tracking & analytics dashboard
- [ ] Batch transcription/feedback generation

**Providers:**
- [ ] Alternative STT: Google Cloud Speech-to-Text
- [ ] Alternative STT: Azure Cognitive Services
- [ ] Alternative LLM: Anthropic Claude, open-source models

**UI/UX:**
- [ ] Dark mode
- [ ] Export to PDF
- [ ] Teacher notes per submission
- [ ] Student progress chart over time
- [ ] Mobile app (React Native)

**Integration:**
- [ ] WhatsApp Bot API (auto-receive submissions)
- [ ] Google Classroom integration
- [ ] Gradescope integration

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         React Frontend (3000)           │
│  - Upload → Transcribe → Feedback UI    │
└──────────────┬──────────────────────────┘
               │ REST API (JSON)
┌──────────────▼──────────────────────────┐
│   FastAPI Backend (8000)                │
│  ┌─────────────────────────────────┐   │
│  │ Endpoints:                      │   │
│  │ POST /api/submission            │   │
│  │ POST /api/submission/{id}/tr    │   │
│  │ POST /api/submission/{id}/fb    │   │
│  │ GET /api/submission/{id}        │   │
│  │ GET /api/health                 │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ Audio Processing:               │   │
│  │ - FFmpeg normalization          │   │
│  │ - openai-whisper STT            │   │
│  │ - long audio chunking           │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ LLM Integration:                │   │
│  │ - OpenAI gpt-4o-mini            │   │
│  │ - Structured feedback JSON      │   │
│  └─────────────────────────────────┘   │
└──────────────┬──────────────────────────┘
               │ Read/Write
┌──────────────▼──────────────────────────┐
│     Local Storage (/data)               │
│  - Original audio files                 │
│  - Normalized WAV                       │
│  - Transcripts (JSON)                   │
│  - Feedback (JSON)                      │
└─────────────────────────────────────────┘
```

---

## 📝 License

MIT - Use freely for educational purposes.

---

## ❓ FAQ

**Q: Can I use this offline?**
A: No, the LLM feedback requires OpenAI API. STT can work offline if you use a local LLM.

**Q: How much does it cost?**
A: OpenAI charges ~$0.001 per submission (feedback only). STT is free (local).

**Q: Can I store data to cloud?**
A: Currently local storage. TODO: Add S3/Google Cloud support.

**Q: How long does feedback take?**
A: Transcription: 60-300s depending on audio length. Feedback: 2-5s. Total: 70-310s.

**Q: Can teachers edit feedback?**
A: Not in current MVP. TODO: Add editing interface.

**Q: Does it support other languages?**
A: Whisper supports 99 languages. LLM feedback assumes English task. Modify prompt for other languages.

---

**Built with ❤️ for English teachers**
