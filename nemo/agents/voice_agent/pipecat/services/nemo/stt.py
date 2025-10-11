# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from typing import AsyncGenerator, List, Optional

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pydantic import BaseModel

from nemo.agents.voice_agent.pipecat.services.nemo.audio_logger import AudioLogger
from nemo.agents.voice_agent.pipecat.services.nemo.legacy_asr import NemoLegacyASRService

try:
    # disable nemo logging
    from nemo.utils import logging

    level = logging.getEffectiveLevel()
    logging.setLevel(logging.CRITICAL)


except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use NVIDIA NeMo STT, you need to `pip install "nemo_toolkit[all]"`.')
    raise Exception(f"Missing module: {e}")


class NeMoSTTInputParams(BaseModel):
    language: Optional[Language] = Language.EN_US
    att_context_size: Optional[List] = [70, 1]
    frame_len_in_secs: Optional[float] = 0.08  # 80ms for FastConformer model
    config_path: Optional[str] = None  # path to the Niva ASR config file
    raw_audio_frame_len_in_secs: Optional[float] = 0.016  # 16ms for websocket transport
    buffer_size: Optional[int] = 5  # number of audio frames to buffer, 1 frame is 16ms


class NemoSTTService(STTService):
    def __init__(
        self,
        *,
        model: Optional[str] = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi",
        device: Optional[str] = "cuda:0",
        sample_rate: Optional[int] = 16000,
        params: Optional[NeMoSTTInputParams] = None,
        has_turn_taking: bool = False,
        backend: Optional[str] = "legacy",
        decoder_type: Optional[str] = "rnnt",
        record_audio_data: Optional[bool] = False,
        audio_logger: Optional[AudioLogger] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._queue = asyncio.Queue()
        self._sample_rate = sample_rate
        params.buffer_size = params.frame_len_in_secs // params.raw_audio_frame_len_in_secs
        self._params = params
        self._model_name = model
        self._has_turn_taking = has_turn_taking
        self._backend = backend
        self._decoder_type = decoder_type
        self._record_audio_data = record_audio_data
        self._audio_logger = audio_logger
        if not params:
            raise ValueError("params is required")

        self._device = device

        self._load_model()

        self.audio_buffer = []
        # Buffers for accumulating audio and transcriptions for a complete turn (for logging)
        self._turn_audio_buffer = []
        self._turn_transcription_buffer = []

    def _load_model(self):
        if self._backend == "legacy":
            self._model = NemoLegacyASRService(self._model_name, device=self._device, decoder_type=self._decoder_type)
        else:
            raise ValueError(f"Invalid ASR backend: {self._backend}")

    def can_generate_metrics(self) -> bool:
        """Indicates whether this service can generate metrics.

        Returns:
            bool: True, as this service supports metric generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Handle service start.

        Args:
            frame: StartFrame containing initial configuration
        """
        await super().start(frame)

        # Initialize the model if not already done
        if not hasattr(self, "_model"):
            self._load_model()

    async def stop(self, frame: EndFrame):
        """Handle service stop.

        Args:
            frame: EndFrame that triggered this method
        """
        await super().stop(frame)
        # Clear any internal state if needed
        await self._queue.put(None)  # Signal to stop processing

    async def cancel(self, frame: CancelFrame):
        """Handle service cancellation.

        Args:
            frame: CancelFrame that triggered this method
        """
        await super().cancel(frame)
        # Clear any internal state
        await self._queue.put(None)  # Signal to stop processing
        self._queue = asyncio.Queue()  # Reset the queue

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio data and generate transcription frames.

        Args:
            audio: Raw audio bytes to transcribe

        Yields:
            Frame: Transcription frames containing the results
        """
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

        try:
            is_final = False
            transcription = None
            self.audio_buffer.append(audio)
            if len(self.audio_buffer) >= self._params.buffer_size:
                audio = b"".join(self.audio_buffer)
                self.audio_buffer = []

                transcription, is_final = self._model.transcribe(audio)
                if self._record_audio_data and self._audio_logger:
                    self._turn_audio_buffer.append(audio)
                    # Accumulate transcriptions for turn-based logging
                    if transcription:
                        self._turn_transcription_buffer.append(transcription)
                        self._stage_turn_audio_and_transcription()

                await self.stop_ttfb_metrics()
                await self.stop_processing_metrics()

            if transcription:
                logger.debug(f"Transcription (is_final={is_final}): `{transcription}`")

                # Get the language from params or default to EN_US
                language = self._params.language if self._params else Language.EN_US

                # Create and push the transcription frame
                if self._has_turn_taking or not is_final:
                    frame_type = InterimTranscriptionFrame
                else:
                    frame_type = TranscriptionFrame
                await self.push_frame(
                    frame_type(
                        transcription,
                        "",  # No speaker ID in this implementation
                        time_now_iso8601(),
                        language,
                        result={"text": transcription},
                    )
                )

                # Handle the transcription
                await self._handle_transcription(
                    transcript=transcription,
                    is_final=is_final,
                    language=language,
                )

            yield None

        except Exception as e:
            logger.error(f"Error in NeMo STT processing: {e}")
            await self.push_frame(
                ErrorFrame(
                    str(e),
                    time_now_iso8601(),
                )
            )
            yield None

    def _stage_turn_audio_and_transcription(self):
        """
        Stage the complete turn audio and accumulated transcriptions.
        
        This method is called when a final transcription is received.
        It joins all accumulated audio and transcription chunks and logs them together.
        """
        if not self._turn_audio_buffer or not self._turn_transcription_buffer:
            logger.debug("No audio or transcription to log")
            return
        
        try:
            # Join all accumulated audio and transcriptions for this turn
            complete_turn_audio = b"".join(self._turn_audio_buffer)
            complete_transcription = "".join(self._turn_transcription_buffer)
            
            logger.debug(f"Staging a turn with: {len(self._turn_audio_buffer)} audio chunks, "
                        f"{len(self._turn_transcription_buffer)} transcription chunks")
            
            self._audio_logger.stage_user_audio(
                audio_data=complete_turn_audio,
                transcription=complete_transcription,
                sample_rate=self._sample_rate,
                num_channels=1,
                is_final=True,
                additional_metadata={
                    "model": self._model_name,
                    "backend": self._backend,
                    "audio_duration_sec": len(complete_turn_audio) / (self._sample_rate * 2),
                    "num_transcription_chunks": len(self._turn_transcription_buffer),
                    "num_audio_chunks": len(self._turn_audio_buffer),
                },
            )
            
            logger.info(f"Staged the audio and transcription for turn: '{complete_transcription[:50]}...'")
            
        except Exception as e:
            logger.warning(f"Failed to log user audio: {e}")


    @traced_stt
    async def _handle_transcription(self, transcript: str, is_final: bool, language: Optional[str] = None):
        """Handle a transcription result.

        Args:
            transcript: The transcribed text
            is_final: Whether this is a final transcription
            language: The language of the transcription
        """
        pass  # Base implementation - can be extended for specific handling needs

    async def set_language(self, language: Language):
        """Update the service's recognition language.

        Args:
            language: New language for recognition
        """
        if self._params:
            self._params.language = language
        else:
            self._params = NeMoSTTInputParams(language=language)

        logger.info(f"Switching STT language to: {language}")

    async def set_model(self, model: str):
        """Update the service's model.

        Args:
            model: New model name/path to use
        """
        await super().set_model(model)
        self._model_name = model
        self._load_model()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, VADUserStoppedSpeakingFrame) and isinstance(self._model, NemoLegacyASRService):
            # manualy reset the state of the model when end of utterance is detected by VAD
            logger.debug("Resetting state of the model due to VADUserStoppedSpeakingFrame")
            self._model.reset_state()
            # Clear turn buffers if logging wasn't completed (e.g., no final transcription)
            if self._turn_audio_buffer or self._turn_transcription_buffer:
                logger.debug("Clearing turn audio and transcription buffers due to VAD user stopped speaking")
                self._turn_audio_buffer = []
                self._turn_transcription_buffer = []
        await super().process_frame(frame, direction)
