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


def _beam_search_decode(
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
    bos = tgt_tokenizer.token_to_id("[BOS]")
    eos = tgt_tokenizer.token_to_id("[EOS]")
    batch_sz = src.shape[0]
    memory = model.encode(src, src_mask.to(device))
    res = torch.ones(batch_sz, 1, 1).fill_(bos).type(torch.long).to(device)
    probs = torch.zeros(batch_sz, 1).type(torch.long).to(device)
    for j in range(max_len - 1):
        beam_sz = res.shape[1]
        memory_beam = memory.repeat_interleave(beam_sz, dim=0)
        tgt_mask = generate_mask(res.size(-1)).to(device)
        out = torch.log_softmax(model.decode(res.view(beam_sz*batch_sz, -1), memory_beam, tgt_mask), dim=-1)
        #prob_k, next_word = torch.topk(out.view(batch_sz, beam_sz, -1), beam_size, dim=-1)
        probs, res = get_beams_batch(out, res, probs, beam_size, batch_sz, device)
        if torch.all(res[:, :, -1] == eos):
            break
    return res[:, 0, :].type(torch.long)


def get_beams_batch(out, res, probs, beam_size, batch_sz, device):
    beam_sz = res.shape[1]
    #print(probs.shape, out.shape)
    all_probs = probs.unsqueeze(2) + out.view(batch_sz, beam_sz, -1)
    ln = all_probs.shape[-1]
    all_probs = all_probs.view(batch_sz, -1)
    new_prob, ind = torch.topk(all_probs, beam_size, dim=-1)
    res_ind = (ind // ln).unsqueeze(2).expand(-1, -1, res.shape[-1])
    word_ind = (ind % ln).unsqueeze(2)

    new_res = res.gather(1, res_ind)

    new_res = torch.cat([new_res, word_ind], dim=-1)
    #print(new_prob.shape, new_res.shape)
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

    if translation_mode == "greedy":
        src_mask = src == src_pad
        out = _greedy_decode(model, src, max_len, tgt_tokenizer, src_mask, device)
        out = tgt_tokenizer.decode_batch(list(out.cpu().numpy()))
        detoks = []
        for o in out:
            o = mpn.normalize(o)
            detoks.append(detok.detokenize(o.split()))
        return detoks
    elif translation_mode == "beam":
        src_mask = src == src_pad
        beam_size = 5
        out = _beam_search_decode(model, src, max_len, tgt_tokenizer, device, src_mask, beam_size=beam_size)
        out = tgt_tokenizer.decode_batch(list(out.cpu().numpy()))
        detoks = []
        for o in out:
            o = mpn.normalize(o)
            detoks.append(detok.detokenize(o.split()))
        return detoks
