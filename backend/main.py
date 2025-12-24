"""
MVP Backend for English Speaking Assessment
- Audio upload and normalization (ffmpeg)
- Speech-to-Text transcription (faster-whisper preferred, whisper fallback)
- Simple feedback generation (OpenAI API via openai==1.3.0)
"""

import os
import json
import uuid
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("speaking_mvp")

# ----------------------------
# Env
# ----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "small").strip()
STT_BACKEND = os.getenv("STT_BACKEND", "whisper").strip().lower()
MAX_AUDIO_MINUTES = int(os.getenv("MAX_AUDIO_MINUTES", "10"))
CHUNK_DURATION = int(os.getenv("CHUNK_DURATION", "30"))  # seconds
SPLIT_THRESHOLD_SECONDS = int(os.getenv("SPLIT_THRESHOLD_SECONDS", "180"))  # 3 min

# ----------------------------
# Storage
# ----------------------------
DATA_DIR = Path("./data/submissions")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(title="English Speaking Assessment API (MVP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# OpenAI Client (v1.x)
# ----------------------------
client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ============================================================================
# Helper Functions
# ============================================================================

def get_submission_dir(submission_id: str) -> Path:
    sub_dir = DATA_DIR / submission_id
    sub_dir.mkdir(parents=True, exist_ok=True)
    return sub_dir


def _tool_exists(cmd: List[str]) -> bool:
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


def check_ffmpeg() -> Tuple[bool, bool]:
    ffmpeg_ok = _tool_exists(["ffmpeg", "-version"])
    ffprobe_ok = _tool_exists(["ffprobe", "-version"])
    return ffmpeg_ok, ffprobe_ok


def normalize_audio(input_path: str, output_path: str) -> bool:
    """
    Normalize to 16kHz mono WAV using ffmpeg.
    Returns False if input seems empty/invalid.
    """
    try:
        if not os.path.exists(input_path) or os.path.getsize(input_path) < 1000:
            return False

        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-ac", "1",
            "-ar", "16000",
            "-vn",
            "-acodec", "pcm_s16le",
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)

        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000

    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode(errors="ignore")
        logger.error(f"FFmpeg normalize error: {err}")
        return False
    except Exception as e:
        logger.error(f"Normalization error: {e}")
        return False


def get_audio_duration(file_path: str) -> Optional[float]:
    """Get duration of audio file in seconds."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
            check=True
        )
        duration = float(result.stdout.decode().strip())
        return duration
    except Exception as e:
        logger.error(f"Error getting duration: {str(e)}")
        # Log stderr too (helps debugging)
        try:
            logger.error(f"ffprobe stderr: {result.stderr.decode(errors='ignore')}")
        except Exception:
            pass
        return None


def split_wav_to_chunks(wav_path: str, duration: float, chunk_seconds: int) -> List[str]:
    """
    Split normalized wav into chunks by re-encoding (stable).
    Returns list of chunk file paths.
    """
    chunks = []
    chunk_count = int(duration / chunk_seconds) + 1

    for i in range(chunk_count):
        start_time = i * chunk_seconds
        end_time = min((i + 1) * chunk_seconds, duration)
        if end_time <= start_time:
            continue

        chunk_path = wav_path.replace(".wav", f"_chunk_{i}.wav")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-to", str(end_time),
            "-i", wav_path,
            "-ac", "1",
            "-ar", "16000",
            "-vn",
            "-acodec", "pcm_s16le",
            chunk_path
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 1000:
            chunks.append(chunk_path)

    return chunks


def transcribe_with_faster_whisper(wav_path: str) -> Optional[str]:
    try:
        from faster_whisper import WhisperModel  # type: ignore

        try:
            model = WhisperModel(STT_MODEL_SIZE, device="cpu", compute_type="int8")
        except Exception:
            model = WhisperModel(STT_MODEL_SIZE, device="cpu")

        segments, _info = model.transcribe(wav_path, language="en")
        text = " ".join([seg.text.strip() for seg in segments if seg.text and seg.text.strip()])
        return text.strip() if text.strip() else ""
    except Exception as e:
        logger.info(f"faster-whisper not available/failed: {e}")
        return None


def transcribe_with_whisper(wav_path: str) -> Optional[str]:
    try:
        import whisper  # type: ignore

        model_size = STT_MODEL_SIZE or "small"
        model = whisper.load_model(model_size)
        result = model.transcribe(
            wav_path,
            language="en",
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=False,
            beam_size=1,
            best_of=1,
            fp16=False,  # CPU için daha güvenli
        )
        text = (result.get("text") or "").strip()
        return text
    except Exception as e:
        logger.info(f"whisper not available/failed: {e}")
        return None


def transcribe_audio(wav_path: str) -> str:
    """
    Transcribe using faster-whisper if available, else whisper.
    Chunk if longer than SPLIT_THRESHOLD_SECONDS.
    Returns transcript or raises errors via sentinel strings.
    """
    duration = get_audio_duration(wav_path)
    if duration is None:
        return "__DURATION_FAILED__"

    backend = STT_BACKEND

    def transcribe_one(path: str) -> Optional[str]:
        if backend == "faster-whisper":
            return transcribe_with_faster_whisper(path)
        elif backend == "whisper":
            return transcribe_with_whisper(path)
        else:
            # invalid config
            return None
        
        # Validate backend selection early (no silent fallback)
    if backend not in ("whisper", "faster-whisper"):
        logger.error(f"Invalid STT_BACKEND='{backend}'. Use 'whisper' or 'faster-whisper'.")
        return "__STT_NOT_AVAILABLE__"

    # Chunking
    # Chunking
    if duration > SPLIT_THRESHOLD_SECONDS:
        logger.info(f"Audio {duration:.1f}s > {SPLIT_THRESHOLD_SECONDS}s, chunking... (backend={backend})")
        chunks = split_wav_to_chunks(wav_path, duration, CHUNK_DURATION)
        if not chunks:
            return "__CHUNKING_FAILED__"

        parts: List[str] = []
        for c in chunks:
            try:
                t = transcribe_one(c)
                if t is None:
                    # selected backend missing / failed
                    return "__STT_NOT_AVAILABLE__"
                if t.strip():
                    parts.append(t.strip())
            finally:
                try:
                    os.remove(c)
                except Exception:
                    pass

        return " ".join(parts).strip() if parts else ""

    # Short audio
    logger.info(f"Transcribing short audio (backend={backend})")
    t = transcribe_one(wav_path)

    if t is None:
        return "__STT_NOT_AVAILABLE__"

    return t.strip()


def generate_feedback(transcript: str, task_topic: str, dialect: str = "US") -> Optional[dict]:
    if not transcript or not transcript.strip():
        return None
    if client is None:
        logger.error("OPENAI_API_KEY not set")
        return None

    system_prompt = (
        "You are an experienced English teacher.\n"
        "Provide brief, actionable feedback for a student's spoken English.\n"
        "Return ONLY a single valid JSON object. No markdown. No extra text.\n"
        "Do not invent content. Use exact quotes from the transcript.\n"
        "Do not over-penalize filler words unless very frequent.\n"
        f"Dialect preference: {dialect}.\n"
    )

    user_prompt = f"""
        Task topic: {task_topic}
        Transcript: {transcript}

        Return EXACTLY this JSON shape (fill values). Keep it SHORT and SIMPLE.

        {{
        "task_topic": "{task_topic}",
        "student_level_guess": "A2|B1|B2|C1",
        "teacher_summary": {{
            "overall": "1-2 sentences",
            "strengths": ["max 3 short bullets"],
            "focus_next": ["max 3 short bullets"]
        }},
        "student_feedback": {{
            "quick_message": "2-4 short sentences, friendly tone",
            "top_fixes": [
            {{
                "original": "short exact quote from transcript",
                "better": "corrected version",
                "why": "max 12 words"
            }}
            ],
            "better_version": "A cleaned, more natural version (short)."
        }}
        }}

        Rules:
        - If transcript has <30 words, set top_fixes to [].
        - top_fixes: 3-6 items max.
        - If parts were unclear, mention in teacher_summary.overall.
        - Use exact quotes from the transcript.
    """

    try:
        # openai==1.3.0 client usage
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=900,
        )

        text = (resp.choices[0].message.content or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"LLM returned non-JSON: {text}")
            return None

    except Exception as e:
        logger.error(f"LLM feedback error: {e}")
        return None


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/submission")
async def upload_submission(
    audio_file: UploadFile = File(...),
    task_topic: str = Form(...),
    dialect: Optional[str] = Form("US"),
):
    if not task_topic or len(task_topic.strip()) < 3:
        raise HTTPException(status_code=400, detail="task_topic must be at least 3 characters")

    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    allowed_extensions = {".ogg", ".opus", ".mp3", ".m4a", ".wav", ".webm"}
    ext = Path(audio_file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}. Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    submission_id = str(uuid.uuid4())[:12]
    sub_dir = get_submission_dir(submission_id)

    original_path = sub_dir / f"original{ext}"
    try:
        with open(original_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    size_bytes = original_path.stat().st_size if original_path.exists() else 0

    metadata = {
        "submission_id": submission_id,
        "task_topic": task_topic.strip(),
        "dialect": (dialect or "US").strip(),
        "created_at": datetime.now().isoformat(),
        "original_filename": audio_file.filename,
        "original_size_bytes": size_bytes,
    }
    with open(sub_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Uploaded {submission_id} ({size_bytes} bytes)")
    return {"submission_id": submission_id, "status": "uploaded"}


@app.post("/api/submission/{submission_id}/transcribe")
async def transcribe_submission(submission_id: str):
    sub_dir = get_submission_dir(submission_id)

    original_files = list(sub_dir.glob("original*"))
    if not original_files:
        raise HTTPException(status_code=404, detail="Submission not found")

    ffmpeg_ok, ffprobe_ok = check_ffmpeg()
    if not ffmpeg_ok:
        raise HTTPException(status_code=500, detail="ffmpeg not installed on server")
    if not ffprobe_ok:
        raise HTTPException(status_code=500, detail="ffprobe not installed on server")

    original_path = str(original_files[0])
    normalized_path = str(sub_dir / "normalized.wav")

    if not normalize_audio(original_path, normalized_path):
        raise HTTPException(status_code=400, detail="Audio normalization failed. Audio may be empty/invalid.")

    duration = get_audio_duration(normalized_path)
    if duration is None or duration <= 0:
        raise HTTPException(status_code=400, detail="Could not determine audio duration")

    if duration > MAX_AUDIO_MINUTES * 60:
        raise HTTPException(status_code=400, detail=f"Audio exceeds maximum {MAX_AUDIO_MINUTES} minutes")

    logger.info(f"Transcribing {submission_id}: {duration:.1f}s using STT_MODEL_SIZE={STT_MODEL_SIZE}")
    transcript = transcribe_audio(normalized_path)

    if transcript == "__STT_NOT_AVAILABLE__":
        raise HTTPException(
            status_code=500,
            detail=(
                "No STT backend available inside this container. "
                "Install faster-whisper or whisper+torch in the image."
            ),
        )
    if transcript in ("__DURATION_FAILED__", "__CHUNKING_FAILED__"):
        raise HTTPException(status_code=500, detail=f"Transcription internal error: {transcript}")

    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Audio appears silent or no speech detected")

    # Save transcript
    transcript_data = {
        "submission_id": submission_id,
        "transcript": transcript.strip(),
        "duration_seconds": round(duration, 2),
        "model_size": STT_MODEL_SIZE,
        "transcribed_at": datetime.now().isoformat(),
    }
    with open(sub_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)

    # Simple confidence estimate
    word_count = len(transcript.split())
    wps = word_count / duration if duration > 0 else 0.0
    if 1.5 <= wps <= 3.5 and word_count >= 20:
        confidence = "high"
    elif word_count >= 10:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "submission_id": submission_id,
        "transcript": transcript.strip(),
        "duration_seconds": round(duration, 2),
        "word_count": word_count,
        "confidence": confidence,
        "status": "transcribed",
    }


@app.post("/api/submission/{submission_id}/feedback")
async def generate_submission_feedback(submission_id: str):
    sub_dir = get_submission_dir(submission_id)

    transcript_path = sub_dir / "transcript.json"
    if not transcript_path.exists():
        raise HTTPException(status_code=400, detail="Transcript not found. Call /transcribe first.")

    metadata_path = sub_dir / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    transcript = (transcript_data.get("transcript") or "").strip()
    task_topic = (metadata.get("task_topic") or "").strip()
    dialect = (metadata.get("dialect") or "US").strip()

    logger.info(f"Generating feedback for {submission_id} (model={os.getenv('OPENAI_MODEL','gpt-4o-mini')})")
    feedback = generate_feedback(transcript, task_topic, dialect=dialect)
    if feedback is None:
        raise HTTPException(status_code=500, detail="Feedback generation failed (LLM returned invalid JSON?)")

    feedback_data = {
        "submission_id": submission_id,
        "feedback": feedback,
        "generated_at": datetime.now().isoformat(),
    }
    with open(sub_dir / "feedback.json", "w", encoding="utf-8") as f:
        json.dump(feedback_data, f, indent=2, ensure_ascii=False)

    return {"submission_id": submission_id, "feedback": feedback, "status": "feedback_generated"}


@app.get("/api/submission/{submission_id}")
async def get_submission(submission_id: str):
    sub_dir = get_submission_dir(submission_id)
    if not (sub_dir / "metadata.json").exists():
        raise HTTPException(status_code=404, detail="Submission not found")

    with open(sub_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    result = {
        "submission_id": submission_id,
        "metadata": metadata,
        "transcript": None,
        "feedback": None,
    }

    if (sub_dir / "transcript.json").exists():
        with open(sub_dir / "transcript.json", "r", encoding="utf-8") as f:
            result["transcript"] = json.load(f)

    if (sub_dir / "feedback.json").exists():
        with open(sub_dir / "feedback.json", "r", encoding="utf-8") as f:
            result["feedback"] = json.load(f)

    return result


@app.get("/api/health")
async def health_check():
    ffmpeg_ok, ffprobe_ok = check_ffmpeg()
    return {
        "status": "ok",
        "ffmpeg": "installed" if ffmpeg_ok else "not_found",
        "ffprobe": "installed" if ffprobe_ok else "not_found",
        "openai_key": "set" if bool(OPENAI_API_KEY) else "not_set",
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "stt_model": STT_MODEL_SIZE,
        "max_audio_minutes": MAX_AUDIO_MINUTES,
        "split_threshold_seconds": SPLIT_THRESHOLD_SECONDS,
        "chunk_duration": CHUNK_DURATION,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
