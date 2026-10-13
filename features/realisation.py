import asyncio
import numpy as np
import librosa
from abc import ABC, abstractmethod
from typing import List

# Предположим, что импорты схем выглядят так (в соответствии с вашим условием)
from core.schemas import PreprocessedAudio, AcousticFeatures, AudioMetadata
from features.base import BaseFeatureExtractor # Если BaseFeatureExtractor лежит там


class LibrosaMFCCExtractor(BaseFeatureExtractor):
    """
    Реализация экстрактора признаков на основе библиотеки librosa.
    Извлекает коэффициенты MFCC из сырых байт аудиосигнала.
    """

    def __init__(self, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512):
        """
        Args:
            n_mfcc: Количество извлекаемых коэффициентов MFCC.
            n_fft: Длина окна для БПФ (Быстрого Преобразования Фурье).
            hop_length: Шаг смещения окна в фреймах.
        """
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    async def extract(self, audio: PreprocessedAudio) -> AcousticFeatures:
        """
        Преобразует очищенный аудиосигнал в матрицу признаков MFCC.
        Процесс выполняется в неблокирующем потоке для предотвращения зависания event loop.
        """
        # Запускаем тяжелые математические вычисления в отдельном потоке (ThreadPool)
        # так как librosa и numpy — это синхронный CPU-bound код.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._extract_sync, audio)

    def _extract_sync(self, audio: PreprocessedAudio) -> AcousticFeatures:
        metadata = audio.metadata
        
        # 1. Определяем тип данных на основе bit_depth
        if metadata.bit_depth == 16:
            dtype = np.int16
        elif metadata.bit_depth == 32:
            dtype = np.int32
        elif metadata.bit_depth == 8:
            dtype = np.uint8
        else:
            raise ValueError(f"Поддерживается только глубина звука 8, 16 или 32 бит. Передано: {metadata.bit_depth}")

        # 2. Конвертируем сырые байты в массив numpy
        audio_data = np.frombuffer(audio.raw_bytes, dtype=dtype)

        # 3. Нормализуем массив в high-precision float (от -1.0 до 1.0), как требует librosa
        if dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif dtype == np.uint8:
            audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0

        # 4. Если аудио многоканальное, усредняем каналы до моно
        if metadata.channels > 1:
            # Изменяем форму массива под количество каналов и берем среднее
            audio_data = audio_data.reshape(-1, metadata.channels).mean(axis=1)

        # 5. Рассчитываем MFCC
        # Результат имеет форму (n_mfcc, t), транспонируем в (t, n_mfcc) для удобства моделей
        mfcc = librosa.feature.mfcc(
            y=audio_data, 
            sr=metadata.sample_rate, 
            n_mfcc=self.n_mfcc,
            n_fft=min(self.n_fft, len(audio_data)), # защита от слишком коротких аудио
            hop_length=self.hop_length
        ).T

        # 6. Преобразуем numpy array в обычный List[List[float]] для Pydantic
        features_matrix = mfcc.tolist()

        return AcousticFeatures(
            features_matrix=features_matrix,
            metadata=metadata
        )
