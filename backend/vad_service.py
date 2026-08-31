import torch
import numpy as np
from config import SPEECH_CONFIDENCE_THRESHOLD
import noisereduce as nr


def preprocess_audio(audio, sample_rate=16000):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    audio = nr.reduce_noise(
        y=audio,
        sr=sample_rate,
        stationary=False,
        prop_decrease=0.7
    )

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    return audio.astype(np.float32)

class SileroVAD:

    def __init__(self):
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        (self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks) = self.utils

    def __call__(self, audio):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        confidences = []

        for i in range(0, len(audio), 512):
            chunk = audio[i:i + 512]

            if len(chunk) < 512:
                chunk = np.pad(chunk, (0, 512 - len(chunk)))

            confidence = self.model(
                torch.from_numpy(chunk.copy()),
                16000
            ).item()

            confidences.append(confidence)

        if not confidences:
            return False, 0.0

        speech_count = sum(
            c >= SPEECH_CONFIDENCE_THRESHOLD
            for c in confidences
        )

        speech_ratio = speech_count / len(confidences)

        is_speech = speech_ratio >= 0.25

        return is_speech, max(confidences)

    # def __call__(self, audio):
    #     audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    #     confidences = []

    #     for i in range(0, len(audio), 512):
    #         chunk = audio[i:i + 512]

    #         if len(chunk) < 512:
    #             chunk = np.pad(chunk, (0, 512 - len(chunk)))

    #         confidence = self.model(torch.from_numpy(chunk.copy()), 16000).item()
    #         confidences.append(confidence)

    #     confidence = max(confidences) if confidences else 0.0
    #     return confidence >= SPEECH_CONFIDENCE_THRESHOLD, confidence


silero_vad = SileroVAD()