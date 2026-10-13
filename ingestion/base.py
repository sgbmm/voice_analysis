from abc import ABC, abstractmethod
from core.schemas import PreprocessedAudio, AudioMetadata

class BaseAudioPreprocessor(ABC):
    """Абстрактный класс для захвата звука и удаления фонового шума (VAD/Denoise)"""
    
    @abstractmethod
    async def process(self, audio_data: bytes, metadata: AudioMetadata) -> PreprocessedAudio:
        """Принимает сырые байты, возвращает валидированный и очищенный аудиопоток"""
        pass
