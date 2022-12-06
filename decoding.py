import torch
from sacremoses import MosesDetokenizer, MosesPunctNormalizer
from tokenizers import Tokenizer
from typing import List
from model import TranslationModel

# it's a surprise tool that will help you later
detok = MosesDetokenizer(lang="en")
mpn = MosesPunctNormalizer()


def generate_mask(sz):
    mask = torch.triu(torch.ones((sz, sz)) == 1, diagonal=1)
    mask = torch.zeros((sz, sz)).masked_fill(mask, float('-inf'))
    return mask


def _greedy_decode(
    model: TranslationModel,
    src: torch.Tensor,
    max_len: int,
    tgt_tokenizer: Tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Given a batch of source sequences, predict its translations with greedy search.
    The decoding procedure terminates once either max_len steps have passed
    or the "end of sequence" token has been reached for all sentences in the batch.
    :param model: the model to use for translation
    :param src: a (batch, time) tensor of source sentence tokens
    :param max_len: the maximum length of predictions
    :param tgt_tokenizer: target language tokenizer
    :param device: device that the model runs on
    :return: a (batch, time) tensor with predictions
    """
    src = src.to(device)
    model.to(device)
    model.eval()
    pad = tgt_tokenizer.token_to_id("[PAD]")
    bos = tgt_tokenizer.token_to_id("[BOS]")
    eos = tgt_tokenizer.token_to_id("[EOS]")
    src_mask = (src == pad)
    memory = model.encode(src, src_mask.to(device))
    batch_sz = src.shape[0]
    res = torch.ones(batch_sz, 1).fill_(bos).to(device)
    for i in range(max_len-1):
        tgt_mask = generate_mask(res.size(1)).to(device)
        out = model.decode(res, memory, tgt_mask)
        next_word = torch.argmax(out, dim=-1)
        res = torch.cat([res, next_word[None]], dim=1)
        if torch.all(next_word == eos):
            break
    return res


def _beam_search_decode(
    model: TranslationModel,
    src: torch.Tensor,
    max_len: int,
    tgt_tokenizer: Tokenizer,
    device: torch.device,
    beam_size: int,
) -> torch.Tensor:
    """
    Given a batch of source sequences, predict its translations with beam search.
    The decoding procedure terminates once max_len steps have passed.
    :param model: the model to use for translation
    :param src: a (batch, time) tensor of source sentence tokens
    :param max_len: the maximum length of predictions
    :param tgt_tokenizer: target language tokenizer
    :param device: device that the model runs on
    :param beam_size: the number of hypotheses
    :return: a (batch, time) tensor with predictions
    """
    pass


@torch.inference_mode()
def translate(
    model: torch.nn.Module,
    src_sentences: List[str],
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    translation_mode: str,
    device: torch.device,
) -> List[str]:
    """
    Given a list of sentences, generate their translations.
    :param model: the model to use for translation
    :param src_sentences: untokenized source sentences
    :param src_tokenizer: source language tokenizer
    :param tgt_tokenizer: target language tokenizer
    :param translation_mode: either "greedy", "beam" or anything more advanced
    :param device: device that the model runs on
    """
    src_pad = src_tokenizer.token_to_id("[PAD]")
    src = []
    for line in src_sentences:
        src.append((torch.tensor(src_tokenizer.encode(line).ids).type(torch.long)))
    src = torch.nn.utils.rnn.pad_sequence(
        sequences=src,
        batch_first=True,
        padding_value=src_pad
    )
    max_len = src.shape[1]
    if translation_mode == "greedy":
        out = _greedy_decode(model, src, max_len, tgt_tokenizer, device)
    elif translation_mode == "beam":
        out = _beam_search_decode(model, src, max_len, tgt_tokenizer, device)
    out = tgt_tokenizer.decode_batch(out).text
    return out
