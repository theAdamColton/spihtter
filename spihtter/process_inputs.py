"""
Convert interleaved texts/images into input_ids and metadata ids.
"""

import numpy as np
from dataclasses import dataclass
import torch
from typing import List, Optional
from html.parser import HTMLParser


import spiht
from transformers import PreTrainedTokenizerFast

from .spiht_streaming_decoder import SpihtExhausted, SpihtStreamingDecoder
from .spiht_image import SpihtImage
from .spiht_configuration import SpihtConfiguration


class SpihtHTMLParser(HTMLParser):
    _in_spiht_tag = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "spiht":
            return
        # if self._in_spiht_tag:
        # raise ValueError("Multiple open spiht tags detected in a text block")
        self.spiht_attrs = {k: v for k, v in attrs}
        self._in_spiht_tag = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "spiht":
            self._in_spiht_tag = False

    def in_spiht_tag(self):
        return self._in_spiht_tag

    def get_spiht_attrs(self):
        return getattr(self, "spiht_attrs", None)


def parse_spiht_tag_attrs(text):
    parser = SpihtHTMLParser()
    parser.feed(text)
    return getattr(parser, "spiht_attrs", None)


class InProgressImage:
    def __init__(self, spiht_image: SpihtImage):
        if spiht_image._encoded_bits is not None:
            self.bits = [bool(b) for b in spiht_image.encoded_bits]
        else:
            assert spiht_image._encoded_bytes is None
            self.bits = []

        self.streaming_decoder = SpihtStreamingDecoder(
            spiht_image.spiht_configuration,
            spiht_image,
            self.bits,
        )
        self.spiht_image = spiht_image
        self._exhausted = False

    def push_bit(self, bit):
        self.bits.append(bit)

    def pull_metadata(self, n: int):
        metadata = []
        for _ in range(n):
            try:
                metadata.append(next(self.streaming_decoder))
            except StopIteration:
                if not self._exhausted:
                    print(
                        "Spiht exhausted when attempting to pull metadata. Continuing with padding"
                    )
                self._exhausted = True
                metadata.extend(
                    [torch.zeros(8, dtype=torch.long)] * (n - len(metadata))
                )
        return torch.stack(metadata)

    def render(self):
        try:
            next(self.streaming_decoder)
            print(
                "Warning! rendering image while there are still bits to be consumed in the streaming_decoder!"
            )
        except StopIteration:
            pass
        except SpihtExhausted:
            pass

        rec_arr = self.streaming_decoder.rec_arr
        spiht_configuration = self.spiht_image.spiht_configuration
        image = spiht.spiht_wrapper.decode_from_rec_arr(
            rec_arr,
            self.spiht_image.height,
            self.spiht_image.width,
            self.spiht_image.level,
            spiht_configuration.spiht_settings,
        )
        return image


@dataclass
class FinishedImage:
    image: np.ndarray


class InputProcessorCache:
    def __init__(self):
        self.last_token: Optional[str] = None
        self.in_progress_image: Optional[InProgressImage] = None
        self.finished_images: List[FinishedImage] = []
        self.html_parser: SpihtHTMLParser = SpihtHTMLParser()

    def get_last_image(self):
        self.finish_image()
        return self.finished_images[-1].image

    def finish_image(self):
        if self.in_progress_image is not None:
            print("Finishing image...")
            image = self.in_progress_image.render()
            self.in_progress_image = None
            self.finished_images.append(FinishedImage(image))


def find_until_not_in(seq, _set, start=0):
    out = -1
    for i in range(start, len(seq)):
        if seq[i] not in _set:
            return i
    return out


def find_until_in(seq, _set, start=0):
    out = -1
    for i in range(start, len(seq)):
        if seq[i] in _set:
            return i
    return out


class SpihtInputProcessor:
    def __init__(
        self,
        tokenizer,
        spiht_configuration: SpihtConfiguration,
    ):
        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        self.tokenizer = tokenizer
        self.spiht_configuration = spiht_configuration
        on_bit_token = spiht_configuration.on_bit_token
        off_bit_token = spiht_configuration.off_bit_token
        self.on_bit_token_id = tokenizer(
            on_bit_token, add_special_tokens=False
        ).input_ids[-1]
        self.off_bit_token_id = tokenizer(
            off_bit_token, add_special_tokens=False
        ).input_ids[-1]
        self.on_bit_token = on_bit_token
        self.off_bit_token = off_bit_token

        assert tokenizer.decode(self.off_bit_token_id) == off_bit_token
        assert tokenizer.decode(self.on_bit_token_id) == on_bit_token

    def process_normalized_images_texts(
        self,
        images: List[Optional[SpihtImage]],
        texts: List[Optional[str]],
    ):
        assert images[0] is None

        n_turns = len(images)
        assert n_turns == len(texts)
        for i in range(n_turns - 1):
            # adds the spiht images as HTML to the texts
            text, next_image = texts[i], images[i + 1]
            if text is None:
                continue

            assert next_image is not None

            texts[i] = text + next_image.to_html_opening_tag()

            at_third_to_last = i + 3 >= n_turns
            if not at_third_to_last:
                next_next_text = texts[i + 2]
                assert next_next_text is not None
                texts[i + 2] = "</spiht>" + next_next_text

        text_ids = []
        for i, text in enumerate(texts):
            if text is None:
                text_ids.append(None)
            else:
                tokenizer_output = self.tokenizer(
                    text, return_tensors="pt", add_special_tokens=True
                )
                text_ids.append(tokenizer_output.input_ids.squeeze(0))

        input_ids = []
        metadata_ids = []

        _metadata_pad = torch.zeros(8, dtype=torch.long).unsqueeze(0)
        for i, (text_ids_section, image) in enumerate(zip(text_ids, images)):
            at_last_element = i + 1 >= n_turns
            if text_ids_section is not None:
                if at_last_element:
                    pad_amount = len(text_ids_section)
                else:
                    pad_amount = len(text_ids_section) - 1
                metadata_ids.extend([_metadata_pad] * pad_amount)
                input_ids.append(text_ids_section)
            else:
                metadata_ids.append(image.get_metadata())
                image_ids = self._tokenize_image_ids(image.encoded_bits)
                input_ids.append(image_ids)

        input_ids = torch.cat(input_ids)
        metadata_ids = torch.cat(metadata_ids)

        return input_ids, metadata_ids

    def process_input_ids_with_cache(
        self,
        input_ids: torch.LongTensor,
        past_input_processor_cache: Optional[InputProcessorCache] = None,
        output_input_processor_cache: bool = False,
    ):
        s = input_ids.shape
        if past_input_processor_cache is None:
            past_input_processor_cache = InputProcessorCache()

        # text_tokens is a list of strings
        text_tokens_raw = self.tokenizer.convert_ids_to_tokens(input_ids)

        # Tokenizers have special logic regarding space characters, when joining tokens
        # We need to correctly decode space characters because the HTML parser
        # is sensitive to them.
        # So we use a hacky solution to get the tokenizer decoder to
        # not remove prefix whitespace

        # This only works if the tokenizer uses the metaspace character: ▁
        text_tokens = []
        for tok in text_tokens_raw:
            if tok.startswith("▁"):
                tok = " " + self.tokenizer.convert_tokens_to_string([tok])
            else:
                tok = self.tokenizer.convert_tokens_to_string([tok])
            text_tokens.append(tok)

        metadata_ids = []
        _metadata_pad = torch.zeros(8, dtype=torch.long).unsqueeze(0)

        # at each iteration of this for loop,
        # exactly one row of metadata_ids is appended to metadata_ids
        for token, id in zip(text_tokens, input_ids):
            if past_input_processor_cache.html_parser.in_spiht_tag():
                # parse image tokens

                assert past_input_processor_cache.in_progress_image is not None

                # feed a token. then check if we're still in a spiht tag
                past_input_processor_cache.html_parser.feed(token)

                if not past_input_processor_cache.html_parser.in_spiht_tag():
                    # switch back to parsing text tokens
                    past_input_processor_cache.finish_image()
                elif token not in self.on_bit_token + self.off_bit_token:
                    # we are in a spiht bit tag, but the token is not a valid
                    # spiht bit.
                    # We can either throw an exception,
                    # or push a padding metadata_id.
                    # Or better, we can push a valid non-padding metadata_id if
                    # there is a previous one to use
                    if len(metadata_ids) > 0:
                        metadata_ids.append(metadata_ids[-1])
                    else:
                        metadata_ids.append(_metadata_pad)
                    continue
                else:
                    # parse this token as a spiht bit
                    # and then update the cache
                    # and then continue
                    past_input_processor_cache.in_progress_image.push_bit(
                        self._parse_bits(id)
                    )
                    metadata_ids.append(
                        past_input_processor_cache.in_progress_image.pull_metadata(1)
                    )
                    continue

            # continue parsing text tokens until a valid spiht html opening tag
            # is found
            past_input_processor_cache.html_parser.feed(token)
            spiht_attrs = past_input_processor_cache.html_parser.get_spiht_attrs()

            if spiht_attrs is None:
                metadata_ids.append(_metadata_pad)
                continue

            try:
                spiht_image = SpihtImage.from_html_opening_tag_attrs(
                    spiht_attrs, self.spiht_configuration
                )
            except Exception as e:
                print("Couldn't parse spiht image from html attrs", spiht_attrs, e)
                metadata_ids.append(_metadata_pad)
                continue

            # so we have a good spiht_image
            # so now we can pull the first metadata row
            # and start trying to parse spiht bits
            past_input_processor_cache.in_progress_image = InProgressImage(spiht_image)
            metadata_ids.append(
                past_input_processor_cache.in_progress_image.pull_metadata(1)
            )

        metadata_ids = torch.cat(metadata_ids, 0)

        if not output_input_processor_cache:
            return metadata_ids
        return metadata_ids, past_input_processor_cache

    def _parse_spiht_attrs(self, spiht_attrs):
        height = spiht_attrs.get("h")
        width = spiht_attrs.get("w")
        max_n = spiht_attrs.get("n")
        return height, width, max_n

    def _parse_bits(self, ids: torch.LongTensor):
        """
        ids is assumed to be self.on_bit_token_id or self.off_bit_token_id
        """
        return (ids == self.on_bit_token_id).tolist()

    def _tokenize_image_ids(self, bits) -> torch.LongTensor:
        if isinstance(bits, np.ndarray):
            bits = torch.from_numpy(bits).to(torch.bool)
        else:
            bits = torch.BoolTensor(bits)
        image_ids = torch.full(bits.shape, self.off_bit_token_id, dtype=torch.long)
        image_ids.masked_fill_(bits, self.on_bit_token_id)
        return image_ids
