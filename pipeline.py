import asyncio
from typing import List
from core.schemas import AudioMetadata, AnalysisResult
from ingestion.base import BaseAudioPreprocessor
from features.base import BaseFeatureExtractor
from engine.base import BaseVoiceEngine



class AsyncVoicePipeline:
    """Оркестратор, управляющий полным циклом обработки голоса"""
    
    def __init__(
        self, 
        preprocessor: BaseAudioPreprocessor, 
        extractor: BaseFeatureExtractor, 
        engines: List[BaseVoiceEngine]
    ):
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.engines = engines

    async def run_pipeline(self, raw_audio: bytes, metadata: AudioMetadata) -> List[AnalysisResult]:
        # 1. Предобработка аудио (очистка, VAD)
        clean_audio = await self.preprocessor.process(raw_audio, metadata)
        
        # 2. Извлечение признаков (MFCC/Спектрограммы)
        features = await self.extractor.extract(clean_audio)
        
        # 3. Асинхронный параллельный запуск всех подключенных нейросетей (ASR, Биометрия, Эмоции)
        tasks = [engine.analyze(features) for engine in self.engines]
        results = await asyncio.gather(*tasks)
        
        return list(results)
