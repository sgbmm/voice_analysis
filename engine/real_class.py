import asyncio
import numpy as np
import torch
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification
from speechbrain.inference.speaker import SpeakerRecognition
from core.schemas import AcousticFeatures, AnalysisResult
from engine.base import BaseVoiceEngine
import os

# Настройка устройства
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Способ 1: Через переменную окружения (наиболее чистый подход)
# Безопасное получение токена
HF_TOKEN = os.getenv("HF_TOKEN")

class RealASREngine(BaseVoiceEngine):
    """Асинхронный движок ASR на базе OpenAI Whisper."""
    
    def __init__(self):
        # Инициализация пайплайна (скачивание/загрузка весов)
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition", 
            model="openai/whisper-tiny", 
            device=DEVICE
        )

    def _sync_inference(self, audio_array: np.ndarray, sample_rate: int) -> dict:
        """Синхронная функция инференса, которая будет запущена в потоке."""
        payload = {
            "sampling_rate": sample_rate,
            "raw": audio_array
        }
        return self.asr_pipeline(payload)

    async def analyze(self, features: AcousticFeatures) -> AnalysisResult:
        # Предварительная подготовка данных (быстрая CPU-операция)
        audio_array = np.array(features.features_matrix, dtype=np.float32).flatten()
        
        # Переносим блокирующий инференс нейросети в отдельный поток
        prediction = await asyncio.to_thread(
            self._sync_inference, 
            audio_array, 
            features.metadata.sample_rate
        )
        
        return AnalysisResult(
            text=prediction.get("text", "").strip(),
            speaker_id=None,
            emotion=None,
            confidence=1.0
        )


class RealBiometricEngine(BaseVoiceEngine):
    """Асинхронный движок верификации спикера на базе SpeechBrain ECAPA-TDNN."""
    
    def __init__(self):
        self.verification_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": DEVICE}
        )
        # Мок-база данных эмбеддингов спикеров {speaker_id: tensor}
        self.speaker_db = {} 

    def _sync_inference(self, audio_tensor: torch.Tensor) -> tuple[str | None, float]:
        """Синхронный расчет эмбеддингов и поиск по базе в отдельном потоке."""
        with torch.no_grad():
            embedding = self.verification_model.encode_batch(audio_tensor)
            embedding = embedding.mean(dim=1) 
            
        best_speaker_id = None
        max_score = 0.0
        
        for speaker_id, reference_embedding in self.speaker_db.items():
            score, _ = self.verification_model.verify_embeddings(embedding, reference_embedding)
            score_val = float(score.item())
            
            if score_val > max_score and score_val > 0.5:
                max_score = score_val
                best_speaker_id = speaker_id
                
        return best_speaker_id, max_score

    async def analyze(self, features: AcousticFeatures) -> AnalysisResult:
        audio_array = np.array(features.features_matrix, dtype=np.float32).flatten()
        audio_tensor = torch.tensor(audio_array)
        
        # Выносим тяжелые матричные вычисления из event loop
        speaker_id, confidence = await asyncio.to_thread(self._sync_inference, audio_tensor)
        
        return AnalysisResult(
            text=None,
            speaker_id=speaker_id if speaker_id else "unknown_speaker",
            emotion=None,
            confidence=round(confidence, 2)
        )


class RealEmotionEngine(BaseVoiceEngine):
    """Асинхронный движок классификации эмоций на базе Hubert/Wav2Vec2."""
    
    def __init__(self):
        self.model_name = "Wiam/wav2vec2-lg-xlsr-en-speech-emotion-recognition-finetuned-ravdess-v8"
        self.extractor = AutoFeatureExtractor.from_pretrained(self.model_name, token=HF_TOKEN)
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_name).to(DEVICE)
        self.model.eval()

    def _sync_inference(self, audio_array: np.ndarray, sample_rate: int) -> tuple[str, float]:
        """Синхронный инференс классификатора эмоций."""
        inputs = self.extractor(
            audio_array, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            scores = torch.softmax(logits, dim=-1)
            
        best_class_idx = torch.argmax(scores, dim=-1).item()
        confidence = scores[0][best_class_idx].item()
        
        labels = self.model.config.id2label
        detected_emotion = labels.get(best_class_idx, "unknown")
        
        return detected_emotion, confidence

    async def analyze(self, features: AcousticFeatures) -> AnalysisResult:
        audio_array = np.array(features.features_matrix, dtype=np.float32).flatten()
        
        # Исполняем инференс в пуле потоков
        emotion, confidence = await asyncio.to_thread(
            self._sync_inference, 
            audio_array, 
            features.metadata.sample_rate
        )
        
        return AnalysisResult(
            text=None,
            speaker_id=None,
            emotion=emotion,
            confidence=round(confidence, 2)
        )
