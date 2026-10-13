import asyncio
from core.schemas import AudioMetadata, PreprocessedAudio, AcousticFeatures, AnalysisResult
from ingestion.base import BaseAudioPreprocessor
from features.base import BaseFeatureExtractor
from engine.base import BaseVoiceEngine
from pipeline import AsyncVoicePipeline

from ingestion.realisation import SoxAudioPreprocessor
from features.realisation import LibrosaMFCCExtractor

from engine.real_class import RealASREngine, RealBiometricEngine, RealEmotionEngine

import soundfile as sf

# --- Реализация фейковых компонентов для теста ---
class MockPreprocessor(BaseAudioPreprocessor):
    async def process(self, audio_data: bytes, metadata: AudioMetadata) -> PreprocessedAudio:
        await asyncio.sleep(0.5) # Имитация работы алгоритма шумоподавления
        return PreprocessedAudio(raw_bytes=audio_data, metadata=metadata)

class MockExtractor(BaseFeatureExtractor):
    async def extract(self, audio: PreprocessedAudio) -> AcousticFeatures:
        await asyncio.sleep(0.3) # Имитация расчета MFCC
        return AcousticFeatures(features_matrix=[[0.1, 0.2], [0.3, 0.4]], metadata=audio.metadata)

class MockASREngine(BaseVoiceEngine):
    async def analyze(self, features: AcousticFeatures) -> AnalysisResult:
        await asyncio.sleep(1.0) # Имитация инференса Whisper/Conformer
        return AnalysisResult(text="Привет, мир", confidence=0.98)

class MockEmotionEngine(BaseVoiceEngine):
    async def analyze(self, features: AcousticFeatures) -> AnalysisResult:
        await asyncio.sleep(0.8) # Имитация работы классификатора эмоций
        return AnalysisResult(emotion="Спокойствие", confidence=0.85)

# --- Главный асинхронный процесс ---
async def main():
    # Инициализируем инфраструктуру
    '''
    pipeline = AsyncVoicePipeline(
        preprocessor=MockPreprocessor(),
        extractor=MockExtractor(),
        engines=[MockASREngine(), MockEmotionEngine()] # Добавляем модули анализа
    )
    '''
    pipeline = AsyncVoicePipeline(
        preprocessor=MockPreprocessor(),#preprocessor=SoxAudioPreprocessor(),
        extractor=LibrosaMFCCExtractor(),
        engines=[RealASREngine(), RealBiometricEngine(), RealEmotionEngine()] # Добавляем модули анализа
    )
    # Эмулируем входные данные
    fake_audio_bytes = b"\x00\x01\x02\x03"
    meta = AudioMetadata(sample_rate=16000, channels=1)
    
    audio_path = "data/voice.mp3"
    with open(audio_path, "rb") as audio_file:
        #Read all contents into a bytes object
        audio_bytes = audio_file.read()
    # Читает файл напрямую в массив numpy
    
    #audio_bytes, sample_rate = sf.read("data/voice.mp3")
    
    #print("🚀 Запуск асинхронного анализа голоса...")
    #results = await pipeline.run_pipeline(fake_audio_bytes, meta)
    results = await pipeline.run_pipeline(audio_bytes, meta)
    
    print("\n✅ Анализ успешно завершен! Результаты:")
    for res in results:
        print(res.model_dump_json(indent=2, exclude_none=True))

1
if __name__ == "__main__":
    asyncio.run(main())
