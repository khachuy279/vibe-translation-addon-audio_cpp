import soundfile as sf
import transcribe_cpp
from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.asr.family_adapter import normalize_language_for_family, build_family_options

audio_path = 'debug_audio/22d54213-6418-4985-86db-0d1fa892f2dc/00_ingress_stream.wav'
# Read 10s of audio with actual Japanese speech (from 5s to 15s)
data, sr = sf.read(audio_path, start=16000*5, stop=16000*15, dtype='float32')

mgr = ASRModelManager()
model = mgr.ensure_model('nemotron-3.5-streaming')
session = mgr.ensure_session(model)

norm_lang = normalize_language_for_family('ja', 'nemotron')
print('Normalized lang for nemotron:', norm_lang)

for right in [0, 3, 6, 13]:
    print(f'\n--- Testing att_context_right={right} ---')
    opts = build_family_options('nemotron', {'att_context_right': right}, model=model, slot='stream')
    try:
        with session.stream(language=norm_lang, family=opts) as stream:
            stream.feed(data)
            stream.finalize()
            text = stream.text().full
            print(f'SUCCESS (right={right}): "{text}"')
    except Exception as e:
        print(f'FAILED (right={right}): {e}')
