# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import re
from typing import Iterable, List, Optional, Tuple, Union

import editdistance
import jiwer
import torch
from torchmetrics import Metric
from unicodedata import normalize

from nemo.collections.asr.parts.submodules.ctc_decoding import AbstractCTCDecoding
from nemo.collections.asr.parts.submodules.multitask_decoding import (
    AbstractMultiTaskDecoding,
)
from nemo.collections.asr.parts.submodules.rnnt_decoding import AbstractRNNTDecoding
from nemo.utils import logging

__all__ = ["word_error_rate", "word_error_rate_detail", "WER"]


def move_dimension_to_the_front(tensor, dim_index):
    all_dims = list(range(tensor.ndim))
    return tensor.permute(
        *([dim_index] + all_dims[:dim_index] + all_dims[dim_index + 1 :])
    )


def word_error_rate(
    hypotheses: List[str], references: List[str], use_cer=False
) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string.

    Hypotheses and references must have same length.

    Args:
        hypotheses (list): list of hypotheses
        references(list) : list of references
        use_cer (bool): set True to enable cer

    Returns:
        wer (float): average word error rate
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        # May deprecate using editdistance in future release for here and rest of codebase
        # once we confirm jiwer is reliable.
        scores += editdistance.eval(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float("inf")
    return wer


def word_error_rate_detail(
    hypotheses: List[str], references: List[str], use_cer=False
) -> Tuple[float, int, float, float, float]:
    """
    Computes Average Word Error Rate with details (insertion rate, deletion rate, substitution rate)
    between two texts represented as corresponding lists of string.

    Hypotheses and references must have same length.

    Args:
        hypotheses (list): list of hypotheses
        references(list) : list of references
        use_cer (bool): set True to enable cer

    Returns:
        wer (float): average word error rate
        words (int):  Total number of words/charactors of given reference texts
        ins_rate (float): average insertion error rate
        del_rate (float): average deletion error rate
        sub_rate (float): average substitution error rate
    """
    scores = 0
    words = 0
    ops_count = {"substitutions": 0, "insertions": 0, "deletions": 0}

    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )

    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()

        # To get rid of the issue that jiwer does not allow empty string
        if len(r_list) == 0:
            if len(h_list) != 0:
                errors = len(h_list)
                ops_count["insertions"] += errors
            else:
                errors = 0
        else:
            if use_cer:
                measures = jiwer.cer(r, h, return_dict=True)
            else:
                measures = jiwer.compute_measures(r, h)

            errors = (
                measures["insertions"]
                + measures["deletions"]
                + measures["substitutions"]
            )
            ops_count["insertions"] += measures["insertions"]
            ops_count["deletions"] += measures["deletions"]
            ops_count["substitutions"] += measures["substitutions"]

        scores += errors
        words += len(r_list)

    if words != 0:
        wer = 1.0 * scores / words
        ins_rate = 1.0 * ops_count["insertions"] / words
        del_rate = 1.0 * ops_count["deletions"] / words
        sub_rate = 1.0 * ops_count["substitutions"] / words
    else:
        wer, ins_rate, del_rate, sub_rate = (
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
        )

    return wer, words, ins_rate, del_rate, sub_rate


def word_error_rate_per_utt(
    hypotheses: List[str], references: List[str], use_cer=False
) -> Tuple[List[float], float]:
    """
    Computes Word Error Rate per utterance and the average WER
    between two texts represented as corresponding lists of string.

    Hypotheses and references must have same length.

    Args:
        hypotheses (list): list of hypotheses
        references(list) : list of references
        use_cer (bool): set True to enable cer

    Returns:
        wer_per_utt (List[float]): word error rate per utterance
        avg_wer (float): average word error rate
    """
    scores = 0
    words = 0
    wer_per_utt = []

    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )

    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()

        # To get rid of the issue that jiwer does not allow empty string
        if len(r_list) == 0:
            if len(h_list) != 0:
                errors = len(h_list)
                wer_per_utt.append(float("inf"))
        else:
            if use_cer:
                measures = jiwer.cer(r, h, return_dict=True)
                er = measures["cer"]
            else:
                measures = jiwer.compute_measures(r, h)
                er = measures["wer"]

            errors = (
                measures["insertions"]
                + measures["deletions"]
                + measures["substitutions"]
            )
            wer_per_utt.append(er)

        scores += errors
        words += len(r_list)

    if words != 0:
        avg_wer = 1.0 * scores / words
    else:
        avg_wer = float("inf")

    return wer_per_utt, avg_wer


DEFAULT_CONVERSION_DICT = {
    "aa": "å",
    "ğ": "g",
    "ñ": "n",
    "ń": "n",
    "è": "e",
    "kg": " kilo ",
    "μg": " mikrogram ",
    "-": " minus ",
    "+": " plus ",
    "μ": " mikro ",
    "§": " paragraf ",
    "%": " procent ",
    "‰": " promille ",
    "ú": "u",
    "ş": "s",
    "ê": "e",
    "ã": "a",
    "ë": "e",
    "ć": "c",
    "ä": "æ",
    "í": "i",
    "š": "s",
    "î": "i",
    "ě": "e",
    "ð": "d",
    "á": "a",
    "ó": "o",
    "þ": "th",
    "ı": "i",
    "ö": "ø",
    "ç": "c",
    "ș": "s",
    "\u0301": " ",  # Empty whitespace symbol
    "\u200b": " ",  # Empty whitespace symbol
}

NUMERAL_REGEX = re.compile(r"\b(0|[1-9]\d{0,2}(?:(?:\.\d{3})*|\d*)(?:,\d+)?)\b")


def convert_numeral_to_words(numeral: str, inside_larger_numeral: bool = False) -> str:
    """Convert numerals to words.

    Args:
        numeral:
            The numeral to convert.
        inside_larger_numeral (optional):
            Whether the numeral is inside a larger numeral. For instance, if `numeral`
            is 10, but is part of the larger numeral 1,010, then this should be `True`.

    Returns:
        The text with numerals converted to words.
    """
    if re.fullmatch(pattern=NUMERAL_REGEX, string=numeral) is None:
        return numeral

    numeral = numeral.replace(".", "")

    if "," in numeral:
        assert numeral.count(",") == 1, f"Too many commas in {numeral!r}"
        major, minor = numeral.split(",")
        major = convert_numeral_to_words(numeral=major)
        minor = " ".join(convert_numeral_to_words(numeral=char) for char in minor)
        return f"{major} komma {minor.replace('en', 'et')}"

    match len(numeral):
        case 1:
            mapping = {
                "0": "nul",
                "1": "en",
                "2": "to",
                "3": "tre",
                "4": "fire",
                "5": "fem",
                "6": "seks",
                "7": "syv",
                "8": "otte",
                "9": "ni",
            }
            result = mapping[numeral]

        case 2:
            mapping = {
                "10": "ti",
                "11": "elleve",
                "12": "tolv",
                "13": "tretten",
                "14": "fjorten",
                "15": "femten",
                "16": "seksten",
                "17": "sytten",
                "18": "atten",
                "19": "nitten",
                "20": "tyve",
                "30": "tredive",
                "40": "fyrre",
                "50": "halvtreds",
                "60": "tres",
                "70": "halvfjerds",
                "80": "firs",
                "90": "halvfems",
            }
            if numeral in mapping:
                return mapping[numeral]
            minor = convert_numeral_to_words(
                numeral=numeral[1], inside_larger_numeral=True
            )
            major = convert_numeral_to_words(
                numeral=numeral[0] + "0", inside_larger_numeral=True
            )
            result = f"{minor}og{major}"

        case 3:
            mapping = {"100": "hundrede"}
            if not inside_larger_numeral and numeral in mapping:
                return mapping[numeral]
            major = convert_numeral_to_words(
                numeral=numeral[0], inside_larger_numeral=True
            ).replace("en", "et")
            minor = convert_numeral_to_words(
                numeral=numeral[1:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "hundrede"
            if minor:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case 4:
            mapping = {"1000": "tusind"}
            if not inside_larger_numeral and numeral in mapping:
                return mapping[numeral]
            major = convert_numeral_to_words(
                numeral=numeral[0], inside_larger_numeral=True
            ).replace("en", "et")
            minor = convert_numeral_to_words(
                numeral=numeral[1:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "tusind"
            if minor and len(str(int(numeral[1:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}".strip()

        case 5:
            major = convert_numeral_to_words(
                numeral=numeral[:2], inside_larger_numeral=True
            )
            minor = convert_numeral_to_words(
                numeral=numeral[2:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "tusind"
            if minor and len(str(int(numeral[2:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case 6:
            major = convert_numeral_to_words(
                numeral=numeral[:3], inside_larger_numeral=True
            )
            minor = convert_numeral_to_words(
                numeral=numeral[3:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "tusind"
            if minor and len(str(int(numeral[3:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case 7:
            major = convert_numeral_to_words(
                numeral=numeral[0], inside_larger_numeral=True
            )
            minor = convert_numeral_to_words(
                numeral=numeral[1:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "million" if int(numeral[0]) == 1 else "millioner"
            if minor and len(str(int(numeral[1:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case 8:
            major = convert_numeral_to_words(
                numeral=numeral[:2], inside_larger_numeral=True
            )
            minor = convert_numeral_to_words(
                numeral=numeral[2:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "millioner"
            if minor and len(str(int(numeral[2:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case 9:
            major = convert_numeral_to_words(
                numeral=numeral[:3], inside_larger_numeral=True
            )
            minor = convert_numeral_to_words(
                numeral=numeral[3:].lstrip("0"), inside_larger_numeral=True
            )
            infix = "millioner"
            if minor and len(str(int(numeral[3:]))) <= 2:
                infix += " og"
            result = f"{major} {infix} {minor}"

        case _:
            return numeral

    return re.sub(r" +", " ", result).strip()


def process_text_example(
    text: str,
    characters_to_keep: Iterable[str] | None,
    conversion_dict: dict[str, str] = DEFAULT_CONVERSION_DICT,
    clean_text: bool = True,
    lower_case: bool = True,
    convert_numerals: bool = True,
):
    if convert_numerals and re.search(pattern=NUMERAL_REGEX, string=text):
        text = "".join(
            convert_numeral_to_words(numeral=maybe_numeral)
            for maybe_numeral in re.split(pattern=NUMERAL_REGEX, string=text)
            if maybe_numeral is not None
        )

    if lower_case:
        text = text.lower()

    # Normalise the transcription, which uniformises the characters. For instance, the
    # "long dash" (－) is converted to the normal dash (-).
    if clean_text:
        text = normalize("NFKC", text)

        for key, value in conversion_dict.items():
            text = text.replace(key, value)

        # Remove all non-standard characters
        if characters_to_keep is not None:
            characters_to_keep = "".join(char for char in characters_to_keep)
            if lower_case:
                characters_to_keep = characters_to_keep.lower()
            else:
                characters_to_keep = (
                    characters_to_keep.upper() + characters_to_keep.lower()
                )
            non_standard_characters_regex = re.compile(
                f"[^{re.escape(characters_to_keep + ' |')}]"
            )
            text = re.sub(non_standard_characters_regex, " ", text.strip())

        # Replace superfluous spaces
        text = re.sub(r" +", " ", text)

        # Strip each newline
        text = "\n".join([line.strip() for line in text.split("\n")]).strip("\n")
    return text


CHARACTERS_TO_KEEP = "abcdefghijklmnopqrstuvwxyzæøå0123456789éü"


class WER(Metric):
    """
    This metric computes numerator and denominator for Overall Word Error Rate (WER) between prediction and reference
    texts. When doing distributed training/evaluation the result of ``res=WER(predictions, predictions_lengths, targets, target_lengths)``
    calls will be all-reduced between all workers using SUM operations. Here ``res`` contains three numbers
    ``res=[wer, total_levenstein_distance, total_number_of_words]``.

    If used with PytorchLightning LightningModule, include wer_numerator and wer_denominators inside validation_step
    results. Then aggregate (sum) then at the end of validation epoch to correctly compute validation WER.

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            wer_num, wer_denom = self.__wer(predictions, predictions_len, transcript, transcript_len)
            self.val_outputs = {'val_loss': loss_value, 'val_wer_num': wer_num, 'val_wer_denom': wer_denom}
            return self.val_outputs

        def on_validation_epoch_end(self):
            ...
            wer_num = torch.stack([x['val_wer_num'] for x in self.val_outputs]).sum()
            wer_denom = torch.stack([x['val_wer_denom'] for x in self.val_outputs]).sum()
            tensorboard_logs = {'validation_loss': val_loss_mean, 'validation_avg_wer': wer_num / wer_denom}
            self.val_outputs.clear()  # free memory
            return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    Args:
        decoding: An instance of CTCDecoding or RNNTDecoding.
        use_cer: Whether to use Character Error Rate instead of Word Error Rate.
        log_prediction: Whether to log a single decoded sample per call.
        batch_dim_index: Index corresponding to batch dimension. (For RNNT.)
        dist_dync_on_step: Whether to perform reduction on forward pass of metric.

    Returns:
        res: a tuple of 3 zero dimensional float32 ``torch.Tensor` objects: a WER score, a sum of Levenstein's
            distances for all prediction - reference pairs, total number of words in all references.
    """

    full_state_update: bool = True

    def __init__(
        self,
        decoding: Union[
            AbstractCTCDecoding, AbstractRNNTDecoding, AbstractMultiTaskDecoding
        ],
        use_cer=False,
        log_prediction=True,
        fold_consecutive=True,
        batch_dim_index=0,
        dist_sync_on_step=False,
        sync_on_compute=True,
        **kwargs,
    ):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step, sync_on_compute=sync_on_compute
        )

        self.decoding = decoding
        self.use_cer = use_cer
        self.log_prediction = log_prediction
        self.fold_consecutive = fold_consecutive
        self.batch_dim_index = batch_dim_index

        self.decode = None
        if isinstance(self.decoding, AbstractRNNTDecoding):
            self.decode = (
                lambda predictions,
                predictions_lengths,
                predictions_mask,
                input_ids: self.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=predictions, encoded_lengths=predictions_lengths
                )
            )
        elif isinstance(self.decoding, AbstractCTCDecoding):
            self.decode = (
                lambda predictions,
                predictions_lengths,
                predictions_mask,
                input_ids: self.decoding.ctc_decoder_predictions_tensor(
                    decoder_outputs=predictions,
                    decoder_lengths=predictions_lengths,
                    fold_consecutive=self.fold_consecutive,
                )
            )
        elif isinstance(self.decoding, AbstractMultiTaskDecoding):
            self.decode = (
                lambda predictions,
                prediction_lengths,
                predictions_mask,
                input_ids: self.decoding.decode_predictions_tensor(
                    encoder_hidden_states=predictions,
                    encoder_input_mask=predictions_mask,
                    decoder_input_ids=input_ids,
                    return_hypotheses=False,
                )
            )
        else:
            raise TypeError(
                f"WER metric does not support decoding of type {type(self.decoding)}"
            )

        self.add_state(
            "scores", default=torch.tensor(0), dist_reduce_fx="sum", persistent=False
        )
        self.add_state(
            "words", default=torch.tensor(0), dist_reduce_fx="sum", persistent=False
        )

    def update(
        self,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        targets: torch.Tensor,
        targets_lengths: torch.Tensor,
        predictions_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,  # To allow easy swapping of metrics without worrying about var alignment.
    ):
        """
        Updates metric state.
        Args:
            predictions: an integer torch.Tensor of shape ``[Batch, Time, {Vocabulary}]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            prediction_lengths: an integer torch.Tensor of shape ``[Batch]``
            targets: an integer torch.Tensor of shape ``[Batch, Time]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            target_lengths: an integer torch.Tensor of shape ``[Batch]``
            predictions_lengths: an integer torch.Tensor of shape ``[Batch]``
        """
        words = 0
        scores = 0
        references = []

        with torch.no_grad():
            tgt_lenths_cpu_tensor = targets_lengths.long().cpu()
            targets_cpu_tensor = targets.long().cpu()
            # check batch_dim_index is first dim
            if self.batch_dim_index != 0:
                targets_cpu_tensor = move_dimension_to_the_front(
                    targets_cpu_tensor, self.batch_dim_index
                )
            # iterate over batch
            for ind in range(targets_cpu_tensor.shape[0]):
                tgt_len = tgt_lenths_cpu_tensor[ind].item()
                target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
                reference = self.decoding.decode_ids_to_str(target)
                references.append(reference)
            hypotheses = (
                self.decode(
                    predictions, predictions_lengths, predictions_mask, input_ids
                )
                if predictions.numel() > 0
                else []
            )

        processed_references = []
        processed_hypotheses = []
        for ref, hyp in zip(references, hypotheses):
            processed_ref = process_text_example(
                text=ref,
                characters_to_keep=CHARACTERS_TO_KEEP,
                conversion_dict=DEFAULT_CONVERSION_DICT,
                clean_text=True,
                lower_case=True,
                convert_numerals=True,
            )
            processed_hyp = process_text_example(
                text=hyp.text,
                characters_to_keep=CHARACTERS_TO_KEEP,
                conversion_dict=DEFAULT_CONVERSION_DICT,
                clean_text=True,
                lower_case=True,
                convert_numerals=True,
            )
            processed_references.append(processed_ref)
            processed_hypotheses.append(processed_hyp)

        if hypotheses and self.log_prediction:
            logging.info("\n")
            logging.info(f"WER reference:{processed_references[0]}")
            logging.info(f"WER predicted:{processed_hypotheses[0]}")

        for h, r in zip(processed_hypotheses, processed_references):
            if isinstance(h, list):
                h = h[0]
            if self.use_cer:
                h_list = list(h)
                r_list = list(r)
            else:
                h_list = h.split()
                r_list = r.split()
            words += len(r_list)
            # Compute Levenstein's distance
            scores += editdistance.eval(h_list, r_list)

        self.scores = torch.tensor(
            scores, device=self.scores.device, dtype=self.scores.dtype
        )
        self.words = torch.tensor(
            words, device=self.words.device, dtype=self.words.dtype
        )

    def compute(self):
        scores = self.scores.detach().float()
        words = self.words.detach().float()
        return scores / words, scores, words
