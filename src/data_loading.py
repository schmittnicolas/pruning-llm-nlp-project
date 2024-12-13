from datasets import load_dataset
import random
import torch
from torch.utils.data import Dataset


def get_c4(nsamples, seed, seqlen, tokenizer, batch_size=8):
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Group samples into batches
    trainloader = [
        trainloader[i : i + batch_size] for i in range(0, len(trainloader), batch_size)
    ]

    # Prepare validation dataset
    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    return trainloader, valenc


def get_wikitext2_unstructured(nsamples, seed, seqlen, tokenizer, batch_size=8):
    # Load train and test datasets
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # # Group samples into batches
    # trainloader = [
    #     trainloader[i:i + batch_size]
    #     for i in range(0, len(trainloader), batch_size)
    # ]

    return trainloader, testenc


def get_wikitext2(seq_len, tokenizer):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return traindata, testdata


class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)


def process_data(samples, tokenizer, seq_len, field_name):
    test_ids = tokenizer(
        "\n\n".join(samples[field_name]), return_tensors="pt"
    ).input_ids[0]
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len) : ((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)


def get_loaders(name, tokenizer, seq_len=2048, batch_size=8):
    if "wikitext2" in name:
        train_data, test_data = get_wikitext2(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, "text")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_data, test_loader
