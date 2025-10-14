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

import math
from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig
from torch import Tensor

from nemo.collections.asr.inference.model_wrappers.rnnt_inference_wrapper import RNNTInferenceWrapper
from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline
from nemo.collections.asr.inference.streaming.buffering.audio_bufferer import BatchedAudioBufferer
from nemo.collections.asr.inference.streaming.buffering.feature_bufferer import BatchedFeatureBufferer
from nemo.collections.asr.inference.streaming.decoders.greedy.greedy_rnnt_decoder import ClippedRNNTGreedyDecoder
from nemo.collections.asr.inference.streaming.endpointing.greedy.greedy_rnnt_endpointing import RNNTGreedyEndpointing
from nemo.collections.asr.inference.streaming.framing.multi_stream import ContinuousBatchedRequestStreamer
from nemo.collections.asr.inference.streaming.framing.request import FeatureBuffer, Frame, Request
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.streaming.state.rnnt_state import RNNTStreamingState
from nemo.collections.asr.inference.streaming.text.text_processing import StreamingTextPostprocessor
from nemo.collections.asr.inference.utils.bpe_decoder import BPEDecoder
from nemo.collections.asr.inference.utils.enums import FeatureBufferPaddingMode, RequestType
from nemo.collections.asr.inference.utils.pipeline_utils import (
    adjust_vad_segments,
    drop_trailing_features,
    get_confidence_utils,
    normalize_features,
    update_punctuation_and_language_tokens_timestamps,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis as NemoHypothesis

if TYPE_CHECKING:
    from nemo.collections.asr.inference.itn.inverse_normalizer import AlignmentPreservingInverseNormalizer
    from nemo.collections.asr.inference.pnc.punctuation_capitalizer import PunctuationCapitalizer


class BufferedRNNTPipeline(BasePipeline):

    def __init__(
        self,
        cfg: DictConfig,
        asr_model: RNNTInferenceWrapper,
        pnc_model: PunctuationCapitalizer | None = None,
        itn_model: AlignmentPreservingInverseNormalizer | None = None,
    ):

        self.copy_asr_model_attributes(asr_model)

        self.tokens_to_move = self.punctuation_ids.union(self.language_token_ids)
        self.asr_output_granularity = cfg.asr_output_granularity

        # Streaming related fields
        self.streaming_cfg = cfg.streaming
        self.sample_rate = self.streaming_cfg.sample_rate
        self.stateful = self.streaming_cfg.stateful
        self.stateless = not self.stateful

        self.chunk_size = self.streaming_cfg.chunk_size
        self.left_padding_size = self.streaming_cfg.left_padding_size
        self.right_padding_size = self.streaming_cfg.right_padding_size
        self.buffer_size_in_secs = self.chunk_size + self.left_padding_size + self.right_padding_size
        self.expected_feature_buffer_len = int(self.buffer_size_in_secs / self.window_stride)

        self.mid_delay = math.ceil((self.chunk_size + self.right_padding_size) / self.model_stride_in_secs)
        self.tokens_per_frame_float = self.chunk_size / self.model_stride_in_secs
        self.tokens_per_left_padding_float = self.left_padding_size / self.model_stride_in_secs
        self.tokens_per_right_padding_float = self.right_padding_size / self.model_stride_in_secs
        self.tokens_per_frame = math.ceil(self.tokens_per_frame_float)
        self.tokens_per_left_padding = math.ceil(self.tokens_per_left_padding_float)
        self.tokens_per_right_padding = math.ceil(self.tokens_per_right_padding_float)

        if self.stateful:
            effective_buffer_size_in_secs = self.chunk_size + self.right_padding_size
            self.initial_delay = self.right_padding_size / self.model_stride_in_secs
        else:
            effective_buffer_size_in_secs = self.buffer_size_in_secs
            self.initial_delay = (self.left_padding_size + self.right_padding_size) / self.model_stride_in_secs

        if self.stateful and (
            abs(self.tokens_per_frame_float - self.tokens_per_frame) > 1e-5
            or abs(self.tokens_per_left_padding_float - self.tokens_per_left_padding) > 1e-5
            or abs(self.tokens_per_right_padding_float - self.tokens_per_right_padding) > 1e-5
        ):
            self.tokens_per_frame_float = self.tokens_per_frame
            self.tokens_per_left_padding_float = self.tokens_per_left_padding
            self.left_padding_size = self.tokens_per_left_padding * self.model_stride_in_secs
            self.chunk_size = self.tokens_per_frame * self.model_stride_in_secs
            self.right_padding_size = self.tokens_per_right_padding * self.model_stride_in_secs
            self.buffer_size_in_secs = self.chunk_size + self.left_padding_size + self.right_padding_size
            self.tokens_per_buffer_float = self.buffer_size_in_secs / self.model_stride_in_secs
            self.streaming_cfg.left_padding_size = self.left_padding_size
            self.streaming_cfg.chunk_size = self.chunk_size

        # Request type
        self.request_type = RequestType.from_str(self.streaming_cfg.request_type)
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

        # Confidence related fields
        self.conf_func, self.confidence_aggregator = get_confidence_utils(cfg.confidence)

        # Endpointing related fields
        self.stop_history_eou_in_millisecs = cfg.endpointing.stop_history_eou
        self.endpointer = RNNTGreedyEndpointing(
            vocabulary=self.vocabulary,
            ms_per_timestep=self.model_stride_in_milliseconds,
            effective_buffer_size_in_secs=effective_buffer_size_in_secs,
            stop_history_eou=self.stop_history_eou_in_millisecs,
            residue_tokens_at_end=cfg.endpointing.residue_tokens_at_end,
        )

        # Greedy RNNT decoder
        self.greedy_rnnt_decoder = ClippedRNNTGreedyDecoder(
            vocabulary=self.vocabulary,
            conf_func=self.conf_func,
            endpointer=self.endpointer,
            tokens_per_frame=self.tokens_per_frame,
        )
        self.return_tail_result = cfg.return_tail_result

        # BPE Decoder
        self.bpe_decoder = BPEDecoder(
            vocabulary=self.vocabulary,
            tokenizer=self.tokenizer,
            confidence_aggregator=self.confidence_aggregator,
            asr_supported_puncts=self.asr_supported_puncts,
            word_boundary_tolerance=self.streaming_cfg.word_boundary_tolerance,
            token_duration_in_secs=self.model_stride_in_secs,
        )

        # Decoding computer
        self.decoding_computer = None
        if self.stateful:
            self.decoding_computer = self.asr_model.asr_model.decoding.decoding.decoding_computer

        # PnC and ITN related fields
        self.text_postprocessor = StreamingTextPostprocessor(
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

        self.padding_mode = FeatureBufferPaddingMode.from_str(self.streaming_cfg.padding_mode)
        self.right_padding = self.padding_mode is FeatureBufferPaddingMode.RIGHT
        self.tail_padding_in_samples = int(self.chunk_size * self.sample_rate * 0.45)
        self.tail_padding_in_samples = max(self.tail_padding_in_samples, 6400)
        self.zero_encoded = None
        if self.right_padding:
            self.zero_encoded = self.init_zero_enc()

        super().__init__()

    def init_zero_enc(self) -> Tensor:
        """Initialize the encoder output for the zero buffer."""
        buffer_size_in_samples = int(self.buffer_size_in_secs * self.sample_rate)
        zero_buffer = torch.zeros(1, buffer_size_in_samples, device=self.device)
        zero_features, zero_features_len = self.preprocess(
            buffers=zero_buffer,
            buffer_lens=torch.tensor([zero_buffer.shape[1]], device=self.device),
            expected_feature_buffer_len=self.expected_feature_buffer_len,
        )
        zero_encoded, _ = self.asr_model.encode(
            processed_signal=zero_features, processed_signal_length=zero_features_len
        )
        return zero_encoded[0]

    def reset_session(self) -> None:
        """Reset the frame buffer and internal state pool."""
        super().reset_session()

    def create_state(self, options: ASRRequestOptions) -> RNNTStreamingState:
        """Create new empty state."""
        state = RNNTStreamingState()
        state.set_global_offset(-self.initial_delay)
        new_options = options.augment_with_defaults(
            default_enable_itn=self.text_postprocessor.is_itn_enabled(),
            default_enable_pnc=self.text_postprocessor.is_pnc_enabled(),
            default_stop_history_eou=self.stop_history_eou_in_millisecs,
            default_asr_output_granularity=self.asr_output_granularity,
        )
        state.set_options(new_options)
        return state

    def get_sep(self) -> str:
        """Return the separator for the text postprocessor."""
        return self.sep

    def preprocess(
        self, buffers: Tensor, buffer_lens: Tensor, expected_feature_buffer_len: int
    ) -> tuple[Tensor, Tensor]:
        """Preprocess the buffered frames and extract features."""
        feature_buffers, feature_buffer_lens = self.preprocessor(input_signal=buffers, length=buffer_lens)
        feature_buffers = drop_trailing_features(feature_buffers, expected_feature_buffer_len)
        feature_buffers = normalize_features(feature_buffers, feature_buffer_lens)
        feature_buffer_lens = feature_buffer_lens.clamp(max=feature_buffers.shape[2])
        return feature_buffers, feature_buffer_lens

    def get_cut_off_range(self, T: int, is_last: bool) -> tuple[int, int]:
        """Compute the start and end indices to clip the log probs."""
        start = max(T - 1 - self.mid_delay, 0)
        end = T if is_last else min(start + self.tokens_per_frame, T)
        return start, end

    def encode_raw_signals(
        self, frames: list[Frame], raw_signals: list[Tensor], left_paddings: list[int]
    ) -> tuple[Tensor, Tensor]:
        """Run Encoder part on the raw buffered frames."""

        if self.right_padding:
            left_paddings = torch.tensor(left_paddings, dtype=torch.int64, device=self.device)

        buffers = []
        for i in range(len(raw_signals)):
            buffer = raw_signals[i]
            if self.right_padding:
                # Roll the buffered frames to the left by the left padding
                # This is done to avoid the padding at the beginning of the buffered frames
                # which can cause the performance degradation
                lpad = left_paddings[i].item()
                if lpad > 0:
                    buffer = buffer.roll(shifts=-lpad)
            buffers.append(buffer.unsqueeze_(0))

        # Only final frames have right padding
        # Keep some amount of extra padding to avoid the performance degradation
        right_paddings = torch.tensor(
            [frame.size - frame.valid_size - self.extra_padding_in_samples for frame in frames], device=self.device
        ).clamp(min=0)

        # Create and adjust the buffer lens
        buffer_lens = torch.tensor([buffers[0].size(1)] * len(buffers), device=self.device)
        buffer_lens = buffer_lens - right_paddings
        if self.right_padding:
            buffer_lens = buffer_lens - left_paddings

        feature_buffers, feature_buffer_lens = self.preprocess(
            buffers=torch.cat(buffers).to(self.device),
            buffer_lens=buffer_lens,
            expected_feature_buffer_len=self.expected_feature_buffer_len,
        )

        encoded, encoded_len = self.asr_model.encode(
            processed_signal=feature_buffers, processed_signal_length=feature_buffer_lens
        )
        encoded = encoded.clone()
        encoded_len = encoded_len.clone()

        # Roll back the encoded signals to the right
        if self.right_padding:
            for i in range(encoded.shape[0]):
                lpad = left_paddings[i]
                if lpad > 0:
                    lpad = int(lpad / self.sample_rate / self.model_stride_in_secs)
                    encoded[i] = encoded[i].roll(lpad, dims=1)
                    encoded[i][:, :lpad] = self.zero_encoded[:, :lpad]
                    encoded_len[i] = encoded_len[i] + lpad

        return encoded, encoded_len

    def encode_processed_signals(
        self, fbuffers: list[FeatureBuffer], processed_signals: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Run Encoder part on the processed buffered frames."""

        processed_signals = torch.cat([sig.unsqueeze_(0) for sig in processed_signals]).to(self.device)
        processed_signals = drop_trailing_features(processed_signals, self.expected_feature_buffer_len)
        processed_signal_lengths = torch.tensor([f.valid_size for f in fbuffers], device=self.device)
        processed_signals = normalize_features(processed_signals, processed_signal_lengths)
        processed_signal_lengths = processed_signal_lengths.clamp(max=processed_signals.shape[2])

        encoded, encoded_len = self.asr_model.encode(
            processed_signal=processed_signals, processed_signal_length=processed_signal_lengths
        )
        encoded = encoded.clone()
        encoded_len = encoded_len.clone()

        if self.right_padding:
            for i in range(encoded.shape[0]):
                lpad = int(fbuffers[i].roll_size / self.subsampling_factor)
                if lpad > 0:
                    encoded[i] = encoded[i].roll(lpad, dims=1)
                    encoded[i][:, :lpad] = self.zero_encoded[:, :lpad]
                    encoded_len[i] = encoded_len[i] + lpad
        return encoded, encoded_len

    def encode_frames(self, frames: list[Frame]) -> tuple[Tensor, Tensor]:
        """Encode the frames using the Encoder part of the ASR model."""
        raw_signals, left_paddings = self.bufferer.update(frames)
        encs, enc_lens = None, None
        if len(raw_signals) > 0:
            encs, enc_lens = self.encode_raw_signals(frames, raw_signals, left_paddings)
        return encs, enc_lens

    def encode_feature_buffers(self, fbuffers: list[FeatureBuffer]) -> tuple[Tensor, Tensor]:
        """Encode the feature buffers using the Encoder part of the ASR model."""
        processed_signals = self.bufferer.update(fbuffers)
        encs, enc_lens = None, None
        if len(processed_signals) > 0:
            encs, enc_lens = self.encode_processed_signals(fbuffers, processed_signals)
        return encs, enc_lens

    def run_greedy_decoder(
        self,
        state: RNNTStreamingState,
        request: Request,
        timesteps: torch.Tensor,
        tokens: torch.Tensor,
        start: int,
        end: int,
        alignment_length: int,
        timestamp_offset: int = 0,
        vad_segments: torch.Tensor = None,
    ) -> bool:
        """Greedy RNN-T decoder."""
        if self.stateful and vad_segments is not None:
            vad_segments = adjust_vad_segments(vad_segments, self.left_padding_size)

        clipped_output, tail_output, eou_detected, start_idx, end_idx = self.greedy_rnnt_decoder(
            global_timesteps=timesteps,
            tokens=tokens,
            alignment_length=alignment_length,
            clip_start=start,
            clip_end=end,
            is_last=request.is_last,
            is_start=request.is_first,
            return_tail_result=self.return_tail_result,
            state_start_idx=state.decoder_start_idx,
            state_end_idx=state.decoder_end_idx,
            timestamp_offset=timestamp_offset,
            vad_segments=vad_segments,
            stop_history_eou=state.options.stop_history_eou,
        )
        state.update_state(clipped_output, eou_detected)
        state.update_from_decoder_results(start_idx, end_idx)
        if self.stateless:
            # For stateless mode, we need to set the last token, it will be used for filtering duplicate token
            state.set_last_token(clipped_output["last_token"], clipped_output["last_token_idx"])
            # For stateless mode, we need to increment the global offset
            state.increment_global_offset(self.tokens_per_frame_float)
        state.set_incomplete_segment_tokens(tail_output["tokens"])
        return eou_detected

    def stateless_transcribe_step(
        self, requests: list[Request], encs: Tensor, enc_lens: Tensor, ready_state_ids: set
    ) -> None:
        """
        Transcribe the frames in a stateless manner.
        Stateless assumes that we don't keep track of partial hypotheses (partial_hypotheses=None).
        """
        states = [self.get_state(request.stream_id) for request in requests]
        best_hyp = self.asr_model.decode(encs, enc_lens, partial_hypotheses=None)
        # For stateless mode, use zero timestamp offsets since we don't track timestamps
        ready_states = self.decode_step(best_hyp, requests, states)
        ready_state_ids.update(ready_states)

    def stateful_transcribe_step(
        self, requests: list[Request], encs: Tensor, enc_lens_chunk: Tensor, enc_lens: Tensor, ready_state_ids: set
    ) -> None:
        """
        Transcribe the frames in a stateful manner.
        """
        states = [self.get_state(request.stream_id) for request in requests]
        partial_hypotheses, rnnt_states = [], []
        all_rnnt_states_are_none = True
        for state in states:
            hyp_state = state.hyp_decoding_state
            if hyp_state is not None:
                partial_hypotheses.append(
                    NemoHypothesis(score=0.0, y_sequence=torch.zeros([0], dtype=torch.long), dec_state=hyp_state)
                )
                rnnt_states.append(hyp_state)
                all_rnnt_states_are_none = False
            else:
                partial_hypotheses.append(None)
                rnnt_states.append(None)

        batched_rnnt_states = None
        if not all_rnnt_states_are_none:
            batched_rnnt_states = self.decoding_computer.merge_to_batched_state(rnnt_states)

        if self.tokens_per_right_padding > 0:
            with torch.inference_mode(), torch.no_grad():
                best_hyp_chunk, alignments, batched_state = self.decoding_computer(
                    encs.transpose(1, 2), enc_lens_chunk, batched_rnnt_states
                )

        best_hyp = self.asr_model.decode(encs, enc_lens, partial_hypotheses=partial_hypotheses)
        if self.tokens_per_right_padding > 0:
            for state, rnnt_state in zip(states, self.decoding_computer.split_batched_state(batched_state)):
                state.hyp_decoding_state = rnnt_state
        else:
            for state, hyp in zip(states, best_hyp):
                state.hyp_decoding_state = hyp.dec_state

        ready_states = self.decode_step(best_hyp, requests, states)
        for curr_state in states:
            curr_state.timestamp_offset += self.tokens_per_frame_float
        ready_state_ids.update(ready_states)

    def decode_step(self, best_hyp: list, requests: list[Request], states: list[RNNTStreamingState]) -> set:
        """
        Perform greedy RNNT decoding to get the best hypothesis and update the state.
        If EOU is detected, push the words to the state and cleanup the state.
        """
        ready_state_ids = set()
        for idx, hyp in enumerate(best_hyp):
            state = states[idx]
            request = requests[idx]
            # Perform timestamp based decoding for the hypothesis
            if self.stateful:
                alignment_length = self.tokens_per_right_padding + self.tokens_per_frame
            else:
                if self.request_type is RequestType.FEATURE_BUFFER:
                    alignment_length = math.ceil(request.size / self.subsampling_factor)
                else:  # RequestType.FRAME
                    alignment_length = math.ceil(self.expected_feature_buffer_len / self.subsampling_factor)

            if self.stateful:
                start, end = 0, self.tokens_per_frame
            else:
                # For stateless mode
                if request.is_first and request.is_last:
                    start, end = 0, alignment_length
                else:
                    start, end = self.get_cut_off_range(alignment_length, request.is_last)

            timestamp = hyp.timestamp
            tokens = hyp.y_sequence
            timestamp = torch.tensor(timestamp) if isinstance(timestamp, list) else timestamp
            tokens = torch.tensor(tokens) if isinstance(tokens, list) else tokens
            timestamp = update_punctuation_and_language_tokens_timestamps(
                tokens, timestamp, self.tokens_to_move, self.underscore_id
            )
            vad_segments = request.vad_segments
            eou_detected = self.run_greedy_decoder(
                state=state,
                request=request,
                timesteps=timestamp,
                tokens=tokens,
                start=start,
                end=end,
                alignment_length=alignment_length,
                timestamp_offset=state.timestamp_offset,
                vad_segments=vad_segments,
            )

            if eou_detected:
                self.bpe_decoder.decode_bpe_tokens(state)
                state.cleanup_after_eou()
                ready_state_ids.add(request.stream_id)
        return ready_state_ids

    def shared_transcribe_step_stateful(self, requests: list[Request], encs: Tensor, enc_lens: Tensor) -> None:
        """
        Transcribe a step for frames in a stateful manner.
        """
        tokens_per_left_padding_tensor = torch.tensor(self.tokens_per_left_padding, device=self.device)
        tokens_per_frame_tensor = torch.tensor(self.tokens_per_frame, device=self.device)
        postponed_requests = [(ridx, request.stream_id) for ridx, request in enumerate(requests)]
        next_postponed_requests = []
        ready_state_ids = set()
        while len(postponed_requests) > 0:
            request_ids_to_process = []
            for ridx, stream_id in postponed_requests:
                if stream_id in ready_state_ids:
                    next_postponed_requests.append((ridx, stream_id))
                    continue
                request_ids_to_process.append(ridx)
            if len(request_ids_to_process) > 0:
                requests_to_process = [requests[jdx] for jdx in request_ids_to_process]
                request_is_last = torch.tensor(
                    [request.is_last for request in requests_to_process], dtype=torch.bool, device=self.device
                )
                enc_lens_dec = enc_lens - tokens_per_left_padding_tensor
                enc_lens_dec_trimmed = torch.where(
                    request_is_last,
                    enc_lens_dec,
                    torch.minimum(enc_lens_dec, tokens_per_frame_tensor.expand_as(enc_lens_dec)),
                )
                self.stateful_transcribe_step(
                    requests_to_process,
                    encs[request_ids_to_process][:, :, self.tokens_per_left_padding :],
                    enc_lens_dec_trimmed,
                    enc_lens_dec,
                    ready_state_ids,
                )
            if len(ready_state_ids) > 0:
                self.text_postprocessor.process([self.get_state(stream_id) for stream_id in ready_state_ids])
                ready_state_ids.clear()
            postponed_requests = next_postponed_requests.copy()
            next_postponed_requests.clear()

        self.update_partial_transcript(requests, self.tokenizer, self.leading_regex_pattern)

    def shared_transcribe_step(self, requests: list[Request], encs: Tensor, enc_lens: Tensor) -> None:
        """
        Transcribes the frames in a streaming manner.
        After detecting EOU, it updates the state and run text postprocessor.
        If there are multiple streams, it waits until all stated are ready to run text postprocessor.
        """
        postponed_requests = [(ridx, request.stream_id) for ridx, request in enumerate(requests)]
        next_postponed_requests = []
        ready_state_ids = set()

        while len(postponed_requests) > 0:

            request_ids_to_process = []
            for ridx, stream_id in postponed_requests:

                if stream_id in ready_state_ids:
                    # Skip if the state is already ready
                    next_postponed_requests.append((ridx, stream_id))
                    continue

                request_ids_to_process.append(ridx)

            if len(request_ids_to_process) > 0:
                requests_to_process = [requests[jdx] for jdx in request_ids_to_process]
                self.stateless_transcribe_step(
                    requests_to_process,
                    encs=encs[request_ids_to_process],
                    enc_lens=enc_lens[request_ids_to_process],
                    ready_state_ids=ready_state_ids,
                )

            if len(ready_state_ids) > 0:
                self.text_postprocessor.process([self.get_state(stream_id) for stream_id in ready_state_ids])
                ready_state_ids.clear()

            postponed_requests = next_postponed_requests.copy()
            next_postponed_requests.clear()

        self.update_partial_transcript(requests, self.tokenizer, self.leading_regex_pattern)

    def transcribe_step_for_feature_buffers(self, fbuffers: list[FeatureBuffer]) -> None:
        """
        Transcribe a step for feature buffers.
        Args:
            fbuffers: List of feature buffers to transcribe.
        """
        encs, enc_lens = self.encode_feature_buffers(fbuffers)
        if encs is not None:
            if self.stateful:
                self.shared_transcribe_step_stateful(requests=fbuffers, encs=encs, enc_lens=enc_lens)
            else:
                self.shared_transcribe_step(requests=fbuffers, encs=encs, enc_lens=enc_lens)

    def transcribe_step_for_frames(self, frames: list[Frame]) -> None:
        """
        Transcribe a step for frames.
        Args:
            frames: List of frames to transcribe.
        """
        encs, enc_lens = self.encode_frames(frames)
        if encs is not None:
            if self.stateful:
                self.shared_transcribe_step_stateful(requests=frames, encs=encs, enc_lens=enc_lens)
            else:
                self.shared_transcribe_step(requests=frames, encs=encs, enc_lens=enc_lens)

    def get_request_generator(self) -> ContinuousBatchedRequestStreamer:
        """Initialize the request generator."""
        request_generator = ContinuousBatchedRequestStreamer(
            n_frames_per_stream=1,
            frame_size_in_secs=self.chunk_size,
            sample_rate=self.sample_rate,
            batch_size=self.streaming_cfg.batch_size,
            request_type=self.request_type,
            preprocessor=self.preprocessor,
            buffer_size_in_secs=self.buffer_size_in_secs,
            device=self.device,
            pad_last_frame=True,
            right_pad_features=self.right_padding,
            tail_padding_in_samples=self.tail_padding_in_samples,
        )
        return request_generator
