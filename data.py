from enum import Enum
from pathlib import Path
import xml.etree.ElementTree as ET
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch
from tokenizers.processors import TemplateProcessing


def process_training_file(input_path: Path, output_path: Path):
    """
    Processes raw training files ("train.tags.SRC-TGT.*"), saving the output as a sequence of unformatted examples
    (.txt file, one example per line).
    :param input_path: Path to the file with the input data (formatted examples)
    :param output_path: Path to the file with the output data (one example per line)
    """
    output = []
    with input_path.open() as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("<"):
                continue
            output.append(line)
    output = '\n'.join(output)
    with output_path.open("w") as f:
        f.write(output)


def process_evaluation_file(input_path: Path, output_path: Path):
    """
    Processes raw validation and testing files ("IWSLT17.TED.{dev,test}2010.SRC-TGT.*.xml"),
    saving the output as a sequence of unformatted examples (.txt file, one example per line).
    :param input_path: Path to the file with the input data (formatted examples)
    :param output_path: Path to the file with the output data (one example per line)
    """
    output = []
    tree = ET.parse(input_path)
    root = tree.getroot()
    for child in root:
        for doc in child:
            for item in doc:
                if item.tag == 'seg':
                    output.append(item.text.strip())
    output = '\n'.join(output)
    with output_path.open("w") as f:
        f.write(output)


def convert_files(base_path: Path, output_path: Path):
    """
    Given a directory containing all the dataset files, convert each one into the "one example per line" format.
    :param base_path: Path containing files with original data
    :param output_path: Path containing files with processed data
    """

    for language in "de", "en":
        process_training_file(
            base_path / f"train.tags.de-en.{language}",
            output_path / f"train.{language}.txt",
        )
        process_evaluation_file(
            base_path / f"IWSLT17.TED.dev2010.de-en.{language}.xml",
            output_path / f"val.{language}.txt",
        )
        process_evaluation_file(
            base_path / f"IWSLT17.TED.tst2010.de-en.{language}.xml",
            output_path / f"test.{language}.txt",
        )


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_file_path,
        tgt_file_path,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        max_len=32,
    ):
        """
        Loads the training dataset and parses it into separate tokenized training examples.
        No padding should be applied at this stage
        :param src_file_path: Path to the source language training data
        :param tgt_file_path: Path to the target language training data
        :param src_tokenizer: Trained tokenizer for the source language
        :param tgt_tokenizer: Trained tokenizer for the target language
        :param max_len: Maximum length of source and target sentences for each example:
        if either of the parts contains more tokens, it needs to be filtered.
        """
        # your code here
        self.src_file_path = src_file_path
        self.tgt_file_path = tgt_file_path
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_tokenizer.enable_truncation(max_length=max_len)
        self.tgt_tokenizer.enable_truncation(max_length=max_len)

        self.src_tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", self.src_tokenizer.token_to_id("[BOS]")),
                ("[EOS]", self.src_tokenizer.token_to_id("[EOS]")),
            ],
        )

        self.tgt_tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", self.tgt_tokenizer.token_to_id("[BOS]")),
                ("[EOS]", self.tgt_tokenizer.token_to_id("[EOS]")),
            ],
        )

        self.src_lines = []
        self.src_ids = []
        with src_file_path.open() as f:
            for line in f.readlines():
                line = line.strip()
                self.src_lines.append(line)
                self.src_ids.append(torch.tensor(self.src_tokenizer.encode(line).ids))

        self.tgt_lines = []
        self.tgt_ids = []
        with tgt_file_path.open() as f:
            for line in f.readlines():
                line = line.strip()
                self.tgt_lines.append(line)
                self.tgt_ids.append(torch.tensor(self.tgt_tokenizer.encode(line).ids))

    def __len__(self):
        return len(self.tgt_lines)

    def __getitem__(self, i):
        return self.src_ids[i], self.tgt_ids[i]

    def collate_translation_data(self, batch):
        """
        Given a batch of examples with varying length, collate it into `source` and `target` tensors for the model.
        This method is meant to be used when instantiating the DataLoader class for training and validation datasets in your pipeline.
        """
        src_ids = []
        tgt_ids = []
        for src, tgt in batch:
            src_ids.append(src)
            tgt_ids.append(tgt)

        batch = {}
        batch["src"] = torch.nn.utils.rnn.pad_sequence(
                sequences=src_ids,
                batch_first=True,
                padding_value=self.src_tokenizer.token_to_id("[PAD]")
            )
        batch["tgt"] = torch.nn.utils.rnn.pad_sequence(
            sequences=tgt_ids,
            batch_first=True,
            padding_value=self.tgt_tokenizer.token_to_id("[PAD]")
        )
        return batch


class SpecialTokens(Enum):
    UNKNOWN = "[UNK]"
    PADDING = "[PAD]"
    BEGINNING = "[BOS]"
    END = "[EOS]"


def train_tokenizers(base_dir: Path, save_dir: Path):
    """
    Trains tokenizers for source and target languages and saves them to `save_dir`.
    :param base_dir: Directory containing processed training and validation data (.txt files from `convert_files`)
    :param save_dir: Directory for storing trained tokenizer data (two files: `tokenizer_de.json` and `tokenizer_en.json`)
    """
    for language in "de", "en":
        tokenizer = Tokenizer(BPE(unk_token=SpecialTokens.UNKNOWN.value))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=[tok.value for tok in SpecialTokens], vocab_size=30000)
        files = [f"{base_dir}/train.{language}.txt", f"{base_dir}/val.{language}.txt"]
        tokenizer.train(files, trainer)
        tokenizer.save(f"{save_dir}/tokenizer_{language}.json")

