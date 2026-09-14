import soundfile as sf
from backend_cpp.asr.transcribe_engine import TranscribeEngine

audio_path = 'debug_audio/22d54213-6418-4985-86db-0d1fa892f2dc/00_ingress_stream.wav'
data, sr = sf.read(audio_path, start=16000*5, stop=16000*15, dtype='float32')

for key in ['sensevoice-small', 'nemotron-3.5-streaming']:
    print(f'\n=== Testing TranscribeEngine: {key} ===')
    engine = TranscribeEngine(key)
    res = engine._run_inference(data, is_commit=True)
    print(f'Result: "{res}"')
