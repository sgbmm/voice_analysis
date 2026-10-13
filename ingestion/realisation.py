import asyncio
from core.schemas import PreprocessedAudio, AudioMetadata
from ingestion.base import BaseAudioPreprocessor  # Предполагаем, что базовый класс лежит тут

class SoxAudioPreprocessor(BaseAudioPreprocessor):
    """
    Конкретная реализация предобработки аудио с использованием системной утилиты SoX.
    Выполняет шумоподавление, ресемплинг до 16кГц и конвертацию в моно.
    """

    def __init__(self, noise_profile_path: str = "noise.prof"):
        self.noise_profile_path = noise_profile_path

    async def process(self, audio_data: bytes, metadata: AudioMetadata) -> PreprocessedAudio:
        # 1. Формируем аргументы для команды sox
        # Входной формат (динамически берем из метаданных)
        in_args = [
            "-t", "raw",
            "-r", str(metadata.sample_rate),
            "-c", str(metadata.channels),
            "-b", str(metadata.bit_depth),
            "-e", "signed-integer",
            "-",
        ]
        
        # Выходной формат (приводим к стандарту: 16кГц, моно, 16бит)
        target_sample_rate = 16000
        target_channels = 1
        target_bit_depth = 16

        out_args = [
            "-t", "raw",
            "-r", str(target_sample_rate),
            "-c", str(target_channels),
            "-b", str(target_bit_depth),
            "-e", "signed-integer",
            "-",
            "noisered", self.noise_profile_path, "0.21"  # Фильтр шумоподавления
        ]

        # 2. Запускаем асинхронный подпроцесс SoX
        process = await asyncio.create_subprocess_exec(
            "sox", *(in_args + out_args),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # 3. Передаем сырые байты на stdin и читаем обработанный поток с stdout
        stdout, stderr = await process.communicate(input=audio_data)

        if process.returncode != 0:
            raise RuntimeError(f"Ошибка обработки SoX: {stderr.decode('utf-8', errors='ignore')}")

        # 4. Формируем новые метаданные для измененного аудиопотока
        output_metadata = AudioMetadata(
            sample_rate=target_sample_rate,
            channels=target_channels,
            bit_depth=target_bit_depth
        )

        # 5. Возвращаем валидированный Pydantic-объект согласно контракту
        return PreprocessedAudio(
            raw_bytes=stdout,
            metadata=output_metadata
        )
