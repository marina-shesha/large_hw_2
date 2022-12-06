from argparse import ArgumentParser
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU
from tokenizers import Tokenizer
from tqdm import trange

from data import TranslationDataset
from decoding import translate, generate_mask
from model import TranslationModel
import wandb

from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

class ModelConfig:
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    emb_size: int = 128
    dim_feedforward: int = 512
    n_head: int = 8
    dropout_prob: float = 0.1
    max_len: int = 256
    batch_size: int = 64
    lr: float = 1e-3


def train_epoch(
    model: TranslationModel,
    train_dataloader,
    scheduler,
    optimizer,
    criterion,
    device,
    src_tokenizer,
    tgt_tokenizer,
):
    # train the model for one epoch
    # you can obviously add new arguments or change the API if it does not suit you
    model.train()
    model.to(device)
    src_pad = src_tokenizer.token_to_id("[PAD]")
    tgt_pad = tgt_tokenizer.token_to_id("[PAD]")
    losses = 0
    cnt = 0
    i = 0
    save_period = 5
    for batch in tqdm(train_dataloader):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_input = tgt[:, :-1]
        tgt_mask = generate_mask(tgt_input.shape[1]).to(device)
        src_padding_mask = (src == src_pad).to(device)
        tgt_padding_mask = (tgt_input == tgt_pad).to(device)

        logits = model(src, tgt_input, tgt_mask,
        src_padding_mask, tgt_padding_mask)

        optimizer.zero_grad()
        tgt_out = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses += loss.item() * src.shape[0]
        cnt += src.shape[0]
        if i % save_period == 0:
            wandb.log({"loss": loss.cpu().item(), "lr": scheduler.get_last_lr()[0]})
        i += 1
    return losses / cnt


@torch.inference_mode()
def evaluate(model: TranslationModel, val_dataloader, criterion, device, src_tokenizer,
    tgt_tokenizer):
    # compute the loss over the entire validation subset
    model.eval()
    model.to(device)
    src_pad = src_tokenizer.token_to_id("[PAD]")
    tgt_pad = tgt_tokenizer.token_to_id("[PAD]")
    losses = 0
    cnt = 0
    for batch in tqdm(val_dataloader):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_input = tgt[:, :-1]
        tgt_mask = generate_mask(tgt_input.shape[1]).to(device)
        src_padding_mask = (src == src_pad).to(device)
        tgt_padding_mask = (tgt_input == tgt_pad).to(device)

        logits = model(src, tgt_input, tgt_mask,
        src_padding_mask, tgt_padding_mask)

        tgt_out = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        losses += loss.item() * src.shape[0]
        cnt += src.shape[0]

    return losses / cnt


def train_model(data_dir, tokenizer_path, num_epochs):
    src_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer_de.json"))
    tgt_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer_en.json"))

    train_dataset = TranslationDataset(
        data_dir / "train.de.txt",
        data_dir / "train.en.txt",
        src_tokenizer,
        tgt_tokenizer,
        max_len=128,  # might be enough at first
    )
    val_dataset = TranslationDataset(
        data_dir / "val.de.txt",
        data_dir / "val.en.txt",
        src_tokenizer,
        tgt_tokenizer,
        max_len=128,
    )
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    src_pad = src_tokenizer.token_to_id("[PAD]")
    tgt_pad = tgt_tokenizer.token_to_id("[PAD]")

    wandb.init(project="BDZ2")

    model = TranslationModel(
        config.num_encoder_layers,
        config.num_decoder_layers,
        config.emb_size,
        config.dim_feedforward,
        config.n_head,
        src_tokenizer.get_vocab_size(),
        tgt_tokenizer.get_vocab_size(),
        config.dropout_prob,
        config.max_len,
        src_pad,
        tgt_pad,
    )
    model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("count_parametrs:", pytorch_total_params)

    # create loss, optimizer, scheduler objects, dataloaders etc.
    # don't forget about collate_fn
    # if you intend to use AMP, you might need something else

    min_val_loss = float("inf")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        config.batch_size,
        collate_fn=train_dataset.collate_translation_data,
        shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        config.batch_size,
        collate_fn=val_dataset.collate_translation_data,
        shuffle=False
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_pad)
    optimizer = Adam(model.parameters(), config.lr)
    scheduler = OneCycleLR(
        optimizer, max_lr=config.lr, steps_per_epoch=len(train_dataloader), epochs=num_epochs, anneal_strategy="cos", pct_start=0.1
    )

    for epoch in trange(1, num_epochs + 1):
        train_loss = train_epoch(
            model,
            train_dataloader,
            scheduler,
            optimizer,
            criterion,
            device,
            src_tokenizer,
            tgt_tokenizer
        )
        val_loss = evaluate(
            model,
            val_dataloader,
            criterion,
            device,
            src_tokenizer,
            tgt_tokenizer
        )

        # might be useful to translate some sentences from validation to check your decoding implementation
        wandb.log({"val_loss": val_loss, "train_loss": train_loss, "epoch": epoch})

        # also, save the best checkpoint somewhere around here
        if val_loss < min_val_loss:
            print("New best loss! Saving checkpoint")
            torch.save(model.state_dict(), "checkpoint_best.pth")
            min_val_loss = val_loss

        # and the last one in case you need to recover
        # by the way, is this sufficient?
        torch.save(model.state_dict(), "checkpoint_last.pth")

    # load the best checkpoint
    model.load_state_dict(torch.load("checkpoint_best.pth"))
    return model


def translate_test_set(model: TranslationModel, data_dir, tokenizer_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    greedy_translations = []

    src_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer_de.json"))
    tgt_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer_en.json"))

    with open(data_dir / "test.de.txt") as input_file, open(
        "answers_greedy.txt", "w+"
    ) as output_file:
        for src in input_file:
            out = translate(
                model,
                [src.strip()],
                src_tokenizer,
                tgt_tokenizer,
                translation_mode='greedy',
                device=device,
            )
            greedy_translations.append(out[0])
            output_file.write(out[0])

    beam_translations = []
    with open(data_dir / "test.de.txt") as input_file, open(
        "answers_beam.txt", "w+"
    ) as output_file:
        # translate with beam search
        pass

    with open(data_dir / "test.en.txt") as input_file:
        references = [line.strip() for line in input_file]

    bleu = BLEU()
    bleu_greedy = bleu.corpus_score(greedy_translations, [references]).score

    # we're recreating the object, as it might cache some stats
    bleu = BLEU()
    bleu_beam = bleu.corpus_score(beam_translations, [references]).score

    print(f"BLEU with greedy search: {bleu_greedy}, with beam search: {bleu_beam}")
    # maybe log to wandb/comet/neptune as well


if __name__ == "__main__":
    parser = ArgumentParser()
    data_group = parser.add_argument_group("Data paths")
    data_group.add_argument(
        "--data-dir", type=Path, help="Path to the directory containing processed data"
    )
    data_group.add_argument(
        "--tokenizer-path", type=Path, help="Path to the trained tokenizer files"
    )

    # argument groups are useful for separating semantically different parameters
    hparams_group = parser.add_argument_group("Training hyperparameters")
    hparams_group.add_argument(
        "--num-epochs", type=int, default=50, help="Number of training epochs"
    )

    args = parser.parse_args()

    model = train_model(args.data_dir, args.tokenizer_path, args.num_epochs)
    translate_test_set(model, args.data_dir, args.tokenizer_path)
