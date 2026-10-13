from abc import ABC, abstractmethod
from core.schemas import AcousticFeatures, AnalysisResult

class BaseVoiceEngine(ABC):
    """Базовый интерфейс для всех аналитических ядер (ASR, Bio, Emotion)"""
    
    @abstractmethod
    async def analyze(self, features: AcousticFeatures) -> AnalysisResult:
        """Проводит инференс нейросети на основе извлеченных признаков"""
        pass
