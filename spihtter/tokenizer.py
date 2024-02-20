"""
Any tokenizer can be used to encode and decode SPIHT data
if it satisfies some simple requirements.

The tokenizer needs to be able to encode/decode HTML tags because that is what
syntax is used to specify the image attributes.

It also needs 2 tokens, for the bits 0 and 1.
By default these are the \x00 byte and the \x01 byte
"""

import transformers
import tokenizers
import tokenizers.models
import tokenizers.trainers
import tokenizers


def get_simple_tokenizer(vocab_size=32, extra_data=[" h=", " w=", " n=", *[str(i) for i in range(10)]]):
    """
    constructs a very simple tokenizer that can tokenize limited spiht
    formatted as html
    """
    off_token = "\x00"
    on_token = "\x01"
    required_tokens = [off_token, on_token, "<spiht", ">"]
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    trainer = tokenizers.trainers.BpeTrainer(
        special_tokens=["<s>", "</s>"], show_progress=False, vocab_size=vocab_size,
    )
    tokenizer.add_special_tokens(["<s>", "</s>"])
    tokenizer.add_tokens(required_tokens)
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [
            tokenizers.pre_tokenizers.Digits(individual_digits=True),
            tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tokenizer.train_from_iterator(extra_data, trainer)
    tokenizer.post_processor = tokenizers.processors.Sequence(
        [
            tokenizers.processors.ByteLevel(trim_offsets=True),
            tokenizers.processors.TemplateProcessing(
                single="<s> $A",
                special_tokens=[
                    ("<s>", tokenizer.get_vocab()["<s>"]),
                ],
            ),
        ]
    )
    tokenizer.decoder = tokenizers.decoders.ByteLevel()

    tokenizer_tf = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer_tf.eos_token = "</s>"
    tokenizer_tf.bos_token = "</s>"
    tokenizer_tf.bos_token_id = tokenizer.get_vocab()["<s>"]
    tokenizer_tf.eos_token_id = tokenizer.get_vocab()["</s>"]
    tokenizer_tf.pad_token = tokenizer_tf.eos_token

    return tokenizer_tf
