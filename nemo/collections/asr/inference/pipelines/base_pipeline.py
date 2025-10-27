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

from __future__ import annotations

import re
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Iterable

from omegaconf import DictConfig

from nemo.collections.asr.inference.model_wrappers.asr_inference_wrapper import ASRInferenceWrapper
from nemo.collections.asr.inference.pipelines.pipeline_interface import PipelineInterface
from nemo.collections.asr.inference.streaming.buffering.audio_bufferer import BatchedAudioBufferer
from nemo.collections.asr.inference.streaming.buffering.cache_feature_bufferer import BatchedCacheFeatureBufferer
from nemo.collections.asr.inference.streaming.buffering.feature_bufferer import BatchedFeatureBufferer
from nemo.collections.asr.inference.streaming.framing.multi_stream import ContinuousBatchedRequestStreamer
from nemo.collections.asr.inference.streaming.framing.request import FeatureBuffer, Frame, Request
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.streaming.text.text_processing import StreamingTextProcessor
from nemo.collections.asr.inference.utils.bpe_decoder import BPEDecoder
from nemo.collections.asr.inference.utils.context_manager import CacheAwareContextManager
from nemo.collections.asr.inference.utils.enums import RequestType
from nemo.collections.asr.inference.utils.pipeline_utils import (
    check_existance_of_required_attributes,
    get_leading_punctuation_regex_pattern,
)
from nemo.collections.asr.inference.utils.progressbar import ProgressBar
from nemo.collections.asr.inference.utils.text_segment import TextSegment
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

if TYPE_CHECKING:
    from nemo.collections.asr.inference.itn.inverse_normalizer import AlignmentPreservingInverseNormalizer
    from nemo.collections.asr.inference.pnc.punctuation_capitalizer import PunctuationCapitalizer


class PipelineOutput:
    """
    Class to store the output of the pipeline. Contains transcriptions and text segments.
    """

    def __init__(self, texts: list[str] | None = None, segments: list[list[TextSegment]] | None = None):
        """
        Initialize the PipelineOutput.
        Args:
            texts: (list[str] | None) List of transcriptions.
            segments: (list[list[TextSegment]] | None) List of corresponding text segments.
        """
        if texts is None and segments is None:
            raise ValueError("At least one of the 'texts' or 'segments' should be provided.")
        self.texts = texts
        self.segments = segments


class BasePipeline(PipelineInterface):
    """
    Base class for all pipelines.
    """

    def __init__(self):
        """Initialize state pool to store the state for each stream"""
        self._state_pool: dict[int, Any] = {}

    def get_state(self, stream_id: int) -> Any:
        """Retrieve state for a given stream ID."""
        return self._state_pool.get(stream_id, None)

    def get_states(self, stream_ids: Iterable[int]) -> list[Any]:
        """Retrieve states for a list of stream IDs."""
        return [self.get_state(stream_id) for stream_id in stream_ids]

    def delete_state(self, stream_id: int) -> None:
        """Delete the state from the state pool."""
        if stream_id in self._state_pool:
            del self._state_pool[stream_id]

    def delete_states(self, stream_ids: Iterable[int]) -> None:
        """Delete states for a list of stream IDs."""
        for stream_id in stream_ids:
            self.delete_state(stream_id)

    def init_state(self, stream_id: int, options: ASRRequestOptions) -> Any:
        """Initialize the state of the stream"""
        if stream_id not in self._state_pool:
            state = self.create_state(options)
            self._state_pool[stream_id] = state
        return self._state_pool[stream_id]

    def reset_session(self) -> None:
        """Reset the frame buffer and internal state pool"""
        self._state_pool.clear()

    def open_session(self) -> None:
        """Start a new session by resetting the internal state pool"""
        self.reset_session()

    def close_session(self) -> None:
        """Close the session by resetting the internal state pool"""
        self.reset_session()

    @abstractmethod
    def transcribe_step_for_frames(self, frames: list[Frame]) -> None:
        """Transcribe a step for frames"""
        pass

    @abstractmethod
    def transcribe_step_for_feature_buffers(self, fbuffers: list[FeatureBuffer]) -> None:
        """Transcribe a step for feature buffers"""
        pass

    @abstractmethod
    def get_request_generator(self) -> ContinuousBatchedRequestStreamer:
        """Return the request generator."""
        pass

    @abstractmethod
    def get_sep(self) -> str:
        """Return the separator for the text postprocessor."""
        pass

    def transcribe_step(self, requests: list[Request]) -> None:
        """Transcribe a step"""
        if isinstance(requests[0], Frame):
            self.transcribe_step_for_frames(frames=requests)
        elif isinstance(requests[0], FeatureBuffer):
            self.transcribe_step_for_feature_buffers(fbuffers=requests)
        else:
            raise ValueError(f"Invalid request type: {type(requests[0])}")

    def copy_asr_model_attributes(self, asr_model: ASRInferenceWrapper) -> None:
        """
        Copy the attributes from the ASR model
        Args:
            asr_model (ASRInferenceWrapper): ASR model to copy the attributes from.
        """
        self.asr_model = asr_model
        self.tokenizer = asr_model.tokenizer
        self.device = asr_model.device
        self.supports_punctuation = asr_model.supports_punctuation()
        self.asr_supported_puncts = asr_model.supported_punctuation()
        self.leading_regex_pattern = get_leading_punctuation_regex_pattern(self.asr_supported_puncts)
        self.blank_id = asr_model.get_blank_id()
        self.vocabulary = asr_model.get_vocabulary()
        self.sep = asr_model.word_separator
        self.segment_separators = asr_model.segment_separators
        self.underscore_id = asr_model.underscore_id
        self.punctuation_ids = asr_model.punctuation_ids
        self.language_token_ids = asr_model.language_token_ids
        self.preprocessor, self.preprocessor_config = asr_model.create_preprocessor()
        self.subsampling_factor = asr_model.get_subsampling_factor()
        self.window_stride = asr_model.get_window_stride()
        self.model_stride_in_secs = asr_model.get_model_stride(in_secs=True)
        self.model_stride_in_milliseconds = asr_model.get_model_stride(in_milliseconds=True)

    def update_partial_transcript(
        self, requests: list[Request], tokenizer: TokenizerSpec, leading_regex_pattern: str
    ) -> None:
        """
        Update partial transcript from the state.
        Args:
            requests (list[Request]): List of Request objects.
            tokenizer (TokenizerSpec): Used to convert tokens into text
            leading_regex_pattern (str): Regex pattern for the punctuation marks.
        """
        for request in requests:
            state = self.get_state(request.stream_id)
            # state tokens represent all tokens accumulated since the EOU
            # incomplete segment tokens are the remaining tokens on the right side of the buffer after EOU
            all_tokens = state.tokens + state.incomplete_segment_tokens
            if len(all_tokens) > 0:
                pt_string = tokenizer.ids_to_text(all_tokens)
                if leading_regex_pattern:
                    pt_string = re.sub(leading_regex_pattern, r'\1', pt_string)
                state.partial_transcript = pt_string
            else:
                state.partial_transcript = ""

    def init_bpe_decoder(self) -> None:
        """Initialize the BPE decoder"""
        check_existance_of_required_attributes(
            self,
            [
                'vocabulary',
                'tokenizer',
                'confidence_aggregator',
                'asr_supported_puncts',
                'word_boundary_tolerance',
                'model_stride_in_secs',
            ],
        )

        self.bpe_decoder = BPEDecoder(
            vocabulary=self.vocabulary,
            tokenizer=self.tokenizer,
            confidence_aggregator=self.confidence_aggregator,
            asr_supported_puncts=self.asr_supported_puncts,
            word_boundary_tolerance=self.word_boundary_tolerance,
            token_duration_in_secs=self.model_stride_in_secs,
        )

    def init_text_processor(
        self,
        cfg: DictConfig,
        pnc_model: PunctuationCapitalizer | None,
        itn_model: AlignmentPreservingInverseNormalizer | None,
    ) -> None:
        """Initialize the text processor."""
        check_existance_of_required_attributes(
            self,
            [
                'asr_supported_puncts',
                'supports_punctuation',
                'confidence_aggregator',
                'sep',
                'segment_separators',
            ],
        )

        self.text_processor = StreamingTextProcessor(
            pnc_cfg=cfg.pnc,
            itn_cfg=cfg.itn,
            pnc_model=pnc_model,
            itn_model=itn_model,
            asr_supported_puncts=self.asr_supported_puncts,
            asr_supports_punctuation=self.supports_punctuation,
            confidence_aggregator=self.confidence_aggregator,
            sep=self.sep,
            segment_separators=self.segment_separators,
            enable_pnc=cfg.enable_pnc,
            enable_itn=cfg.enable_itn,
        )

    def init_bufferer_for_buffered_streaming(self) -> None:
        """Initialize the bufferer."""
        check_existance_of_required_attributes(
            self,
            [
                'request_type',
                'sample_rate',
                'buffer_size_in_secs',
                'preprocessor_config',
                'device',
            ],
        )

        if self.request_type is RequestType.FEATURE_BUFFER:
            # Feature buffering: It will be used when the input is feature buffers
            self.bufferer = BatchedFeatureBufferer(
                sample_rate=self.sample_rate,
                buffer_size_in_secs=self.buffer_size_in_secs,
                preprocessor_cfg=self.preprocessor_config,
                device=self.device,
            )
        elif self.request_type is RequestType.FRAME:
            # Audio buffering: It will be used when the input is audio frames
            self.bufferer = BatchedAudioBufferer(
                sample_rate=self.sample_rate, buffer_size_in_secs=self.buffer_size_in_secs
            )
        else:
            raise ValueError(f"Unknown request type: {self.request_type}")

    def init_bufferer_for_cache_aware_streaming(self) -> None:
        """Initialize the bufferer for cache-aware streaming."""
        check_existance_of_required_attributes(
            self,
            [
                'use_feat_cache',
                'chunk_size_in_secs',
                'buffer_size_in_secs',
                'sample_rate',
                'preprocessor_config',
                'device',
            ],
        )

        if self.use_feat_cache:
            # Only calculate mel-spec features for last chunk
            chunk_size_for_feature_buffer = self.chunk_size_in_secs
        else:
            # Calculate mel-spec features for the whole buffer
            chunk_size_for_feature_buffer = self.buffer_size_in_secs

        self.bufferer = BatchedCacheFeatureBufferer(
            sample_rate=self.sample_rate,
            buffer_size_in_secs=self.buffer_size_in_secs,
            chunk_size_in_secs=chunk_size_for_feature_buffer,
            preprocessor_cfg=self.preprocessor_config,
            device=self.device,
        )

    def init_context_manager(self) -> None:
        """Initialize the context manager."""
        check_existance_of_required_attributes(self, ['asr_model', 'num_slots', 'use_cache'])
        self.context_manager = CacheAwareContextManager(
            cache_aware_model=self.asr_model, num_slots=self.num_slots, use_cache=self.use_cache
        )

    def run(
        self,
        audio_filepaths: list[str],
        options: list[ASRRequestOptions] | None = None,
        progress_bar: ProgressBar | None = None,
    ) -> PipelineOutput:
        """
        Orchestrates reading from audio_filepaths in a streaming manner,
        transcribes them, and packs the results into a PipelineOutput.
        Args:
            audio_filepaths (list[str]): List of audio filepaths to transcribe.
            options (list[ASRRequestOptions] | None): List of RequestOptions for each stream.
            progress_bar (ProgressBar | None): Progress bar to show the progress. Default is None.
        Returns:
            PipelineOutput: A dataclass containing transcriptions and segments.
        """
        if progress_bar is not None and not isinstance(progress_bar, ProgressBar):
            raise ValueError("progress_bar must be an instance of ProgressBar.")

        if options is None:
            # Use default options if not provided
            options = [ASRRequestOptions() for _ in audio_filepaths]

        if len(options) != len(audio_filepaths):
            raise ValueError("options must be the same length as audio_filepaths")

        request_generator = self.get_request_generator()
        request_generator.set_audio_filepaths(audio_filepaths, options)
        request_generator.set_progress_bar(progress_bar)

        self.open_session()
        for requests in request_generator:
            for request in requests:
                if request.is_first:
                    self.init_state(request.stream_id, request.options)
            self.transcribe_step(requests)
        output = self.pack_output()
        self.close_session()
        return output

    def pack_output(self) -> PipelineOutput:
        """Pack the output from the internal state pool."""
        texts, segments = [], []
        for stream_id in sorted(self._state_pool):
            state = self.get_state(stream_id)
            if state.options.is_word_level_output():
                attr_name = "itn_words" if state.options.enable_itn else "pnc_words"
                state_segments = getattr(state, attr_name)
                state_text = self.get_sep().join(word.text for word in state_segments)
            else:
                # Segment-level output branch
                state_segments = getattr(state, "segments")
                state_text = self.get_sep().join(segment.text for segment in state_segments)
            texts.append(state_text)
            segments.append(state_segments)

        return PipelineOutput(texts=texts, segments=segments)
