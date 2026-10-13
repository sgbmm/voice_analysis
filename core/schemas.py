from pydantic import BaseModel, Field
from typing import List, Optional

# Входные метаданные аудиофайла/потока
class AudioMetadata(BaseModel):
    sample_rate: int = Field(default=16000, description="Частота дискретизации в Гц")
    channels: int = Field(default=1, description="Количество аудиоканалов")
    bit_depth: int = Field(default=16, description="Глубина звука в битах")

# Результат предобработки аудио
class PreprocessedAudio(BaseModel):
    raw_bytes: bytes = Field(..., description="Очищенный от шумов бинарный аудиосигнал")
    metadata: AudioMetadata

# Результат извлечения признаков (вектора)
class AcousticFeatures(BaseModel):
    features_matrix: List[List[float]] = Field(..., description="Матрица признаков (например, MFCC или спектрограмма)")
    metadata: AudioMetadata

# Единый результат комплексного анализа
class AnalysisResult(BaseModel):
    text: Optional[str] = Field(None, description="Распознанный текст (ASR)")
    speaker_id: Optional[str] = Field(None, description="ID идентифицированного спикера")
    emotion: Optional[str] = Field(None, description="Определенная эмоция")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность модели в результате")
