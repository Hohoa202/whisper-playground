import torch
import numpy as np
from config import SPEECH_CONFIDENCE_THRESHOLD


class SileroVAD:

    def __init__(self):
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        (self.get_speech_timestamps, self.save_audio, self.read_audio,
         self.VADIterator, self.collect_chunks) = self.utils

    def __call__(self, audio):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        confidences = []

        for i in range(0, len(audio), 512):
            chunk = audio[i:i + 512]

            if len(chunk) < 512:
                chunk = np.pad(chunk, (0, 512 - len(chunk)))

            confidence = self.model(torch.from_numpy(chunk.copy()), 16000).item()
            confidences.append(confidence)

        confidence = max(confidences) if confidences else 0.0
        return confidence >= SPEECH_CONFIDENCE_THRESHOLD, confidence


silero_vad = SileroVAD()