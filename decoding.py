import torch
from sacremoses import MosesDetokenizer, MosesPunctNormalizer
from tokenizers import Tokenizer
from typing import List
from model import TranslationModel
from collections import defaultdict
import numpy as np

# it's a surprise tool that will help you later
detok = MosesDetokenizer(lang="en")
mpn = MosesPunctNormalizer()


def generate_mask(sz):
    mask = torch.triu(torch.ones((sz, sz)) == 1, diagonal=1)
    mask = torch.zeros((sz, sz)).masked_fill(mask, float('-inf'))
    return mask.type(torch.bool)


def _greedy_decode(
        model: TranslationModel,
        src: torch.Tensor,
        max_len: int,
        tgt_tokenizer: Tokenizer,
        src_mask,
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
    bos = tgt_tokenizer.token_to_id("[BOS]")
    eos = tgt_tokenizer.token_to_id("[EOS]")
    memory = model.encode(src, src_mask.to(device))
    batch_sz = src.shape[0]
    res = torch.ones(batch_sz, 1).fill_(bos).type(torch.long).to(device)
    for i in range(max_len - 1):
        tgt_mask = generate_mask(res.size(1)).to(device)
        out = model.decode(res, memory, tgt_mask)
        next_word = torch.argmax(out, dim=-1)
        res = torch.cat([res, next_word[None]], dim=1)
        if torch.all(next_word == eos):
            break
    return res


def _beam_search_decode_one_batch(
        model: TranslationModel,
        src: torch.Tensor,
        max_len: int,
        tgt_tokenizer: Tokenizer,
        device: torch.device,
        src_mask,
        beam_size: int,
) -> torch.Tensor:
    src = src.to(device)
    model.to(device)
    model.eval()
    pad = tgt_tokenizer.token_to_id("[PAD]")
    bos = tgt_tokenizer.token_to_id("[BOS]")
    eos = tgt_tokenizer.token_to_id("[EOS]")
    src = src.repeat(beam_size, 1)
    src_mask = src_mask.repeat(beam_size, 1)
    memory = model.encode(src, src_mask.to(device))
    res = torch.ones(beam_size, 1).fill_(bos).type(torch.long).to(device)
    probs = torch.ones(beam_size, 1).type(torch.long).to(device)
    for j in range(max_len - 1):
        tgt_mask = generate_mask(res.size(1)).to(device)
        out = torch.log_softmax(model.decode(res, memory, tgt_mask), dim=-1)
        prob_k, next_word = torch.topk(out, beam_size, dim=-1)
        probs, res = get_beams(res, probs, prob_k, next_word, beam_size, device)
        if torch.all(res[:, -1] == eos):
            break
    return res.type(torch.long)


def get_beams(res, probs, prob_k, next_word, beam_size, device):
    all_res = []
    all_probs = []
    for i, cur_res in enumerate(res):
        for j, word in enumerate(next_word[i, :]):
            all_res.append(torch.cat([cur_res, word[None]], dim=-1).tolist())
            all_probs.append(probs[i] * prob_k[i][j].tolist())
    all_probs = torch.tensor(all_probs).to(device)
    all_res = torch.tensor(all_res).to(device)
    new_prob, idx = torch.topk(all_probs, beam_size)
    new_res = all_res[idx]

    # res = res.repeat_interleave(beam_size, dim=-1)
    # probs = probs.repeat_interleave(beam_size, dim=-1)
    # all_probs = probs + prob_k
    # new_res = torch.cat([res, next_word[:,None]], dim=-1)
    # _, idx = torch.topk(all_probs, beam_size, dim=-1)
    # new_prob = torch.gather(all_probs, -1, idx)
    # new_res = torch.gather(new_res, -1, idx[None].repeat())

    # #print(new_res[0], res[0] , next_word.view(beam_size**2)[0])
    return new_prob, new_res


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
    # пока берем ток первый бим
    if translation_mode == "greedy":
        src_mask = src == src_pad
        out = _greedy_decode(model, src, max_len, tgt_tokenizer, src_mask, device)
        out = tgt_tokenizer.decode_batch(list(out.cpu().numpy()))

    elif translation_mode == "beam":
        src_mask = src == src_pad
        out = _beam_search_decode_one_batch(model, src, max_len, tgt_tokenizer, device, src_mask, beam_size=5)
        o = out[0].unsqueeze(0)
        out = tgt_tokenizer.decode_batch(list(o.cpu().numpy()))
    return out
