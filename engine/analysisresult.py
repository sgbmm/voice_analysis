import asyncio
from typing import List
from engine.base import BaseVoiceEngine
from engine.abstract_class import ASREngine,BiometricSpeakerIDEngine,EmotionRecognitionEngine
from core.schemas import AcousticFeatures, AnalysisResult

class VoiceAnalyticPipeline:
    """Оркестратор, запускающий все аналитические движки параллельно."""

    def __init__(self):
        self.engines: List[BaseVoiceEngine] = [
            ASREngine(),
            BiometricSpeakerIDEngine(),
            EmotionRecognitionEngine()
        ]

    async def process_audio(self, features: AcousticFeatures) -> AnalysisResult:
        # Запускаем инференс всех моделей параллельно
        tasks = [engine.analyze(features) for engine in self.engines]
        results = await asyncio.gather(*tasks)
        
        # Собираем отдельные ответы в единый контракт AnalysisResult
        final_result = AnalysisResult(
            text=next((r.text for r in results if r.text is not None), None),
            speaker_id=next((r.speaker_id for r in results if r.speaker_id is not None), None),
            emotion=next((r.emotion for r in results if r.emotion is not None), None),
            # Рассчитываем среднюю уверенность или берем логику взвешивания
            confidence=round(sum(r.confidence for r in results) / len(results), 2)
        )
        
        return final_result
