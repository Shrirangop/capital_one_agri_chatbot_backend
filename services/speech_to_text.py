import whisper
import soundfile as sf
import io
from pydub import AudioSegment

# Load the model
model = whisper.load_model("base")

def speech_to_english(audio_bytes: bytes) -> str:
    """
    Converts audio bytes to English text.
    Uses pydub to handle OGG/Opus and converts to WAV in memory
    before passing to soundfile and Whisper.
    """
    # 1. Load the OGG/Opus audio bytes with pydub
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))

    # 2. Set Whisper's required sample rate (16000Hz)
    audio_segment = audio_segment.set_frame_rate(16000)
    
    # 3. Export to a WAV format in an in-memory buffer
    wav_buffer = io.BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0) # Rewind the buffer to the beginning

    # 4. Read the WAV data with soundfile
    audio_data, sample_rate = sf.read(wav_buffer, dtype='float32')

    # 5. Transcribe with Whisper
    result = model.transcribe(audio_data, task="translate")
    
    return result["text"].strip()