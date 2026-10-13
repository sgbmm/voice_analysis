from abc import ABC, abstractmethod
from core.schemas import PreprocessedAudio, AcousticFeatures

class BaseFeatureExtractor(ABC):
    """Абстрактный класс для расчета математических признаков (MFCC, Spectrogram)"""
    
    @abstractmethod
    async def extract(self, audio: PreprocessedAudio) -> AcousticFeatures:
        """Преобразует очищенный аудиосигнал в матрицу признаков для моделей"""
        pass
            