import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    return (pd,)


@app.cell
def _():
    MAX_INPUT_LENGTH = 30
    MAX_OUTPUT_LEN = 10
    return MAX_INPUT_LENGTH, MAX_OUTPUT_LEN


@app.cell
def _(pd):
    df = pd.read_csv('../data_generator/bodmas.csv')
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(MAX_INPUT_LENGTH, MAX_OUTPUT_LEN):
    VOCAB = [
        "<PAD>",
        "<SOS>",
        "<EOS>",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "+", "-", "*",
        "(", ")",
        " "
    ]

    char_to_id = {ch:i for i, ch in enumerate(VOCAB)}
    id_to_char = {i:ch for i, ch in enumerate(VOCAB)}

    def encode_input(s: str) -> list:
        ids = [char_to_id["<SOS>"]]
        ids += [char_to_id[c] for c in s]
        ids += [char_to_id["<EOS>"]]

        if len(ids) > MAX_INPUT_LENGTH:
            raise ValueError(f"Input string is too long: {s}")

        ids += [char_to_id["<PAD>"]] * (MAX_INPUT_LENGTH - len(ids))
        return ids

    def encode_output(s: str) -> list:
        ids = [char_to_id["<SOS>"]]
        ids += [char_to_id[ch] for ch in s]
        ids += [char_to_id["<EOS>"]]

        if len(ids) > MAX_OUTPUT_LEN:
            raise ValueError("Output too long")

        ids += [char_to_id["<PAD>"]] * (MAX_OUTPUT_LEN - len(ids))
        return ids

    def decode(ids: list) -> str:
        chars = [id_to_char[i] for i in ids if i != char_to_id["<PAD>"]]
        return "".join(chars).replace("<SOS>", "").replace("<EOS>", "")
    return VOCAB, char_to_id, decode, encode_input, encode_output


@app.cell
def _(mo):
    mo.md(r"""
    # Modelling goes here
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    return DataLoader, Dataset, torch


@app.cell
def _(VOCAB):
    VOCAB_SIZE = len(VOCAB)
    EMB_DIM = 32
    HIDDEN_DIM = 64
    return


@app.cell
def _(Dataset, encode_input, encode_output, torch):
    class ArthemeticDataset(Dataset):
        def __init__(self, data):
            self.data = data
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            question, answer = self.data[idx]

            answer = str(answer)
            enc_ids = encode_input(question)
            out_ids = encode_output(answer)

            enc_inputs = torch.tensor(enc_ids, dtype=torch.long)
            dec_inputs = torch.tensor(out_ids[:-1], dtype=torch.long)
            dec_targets = torch.tensor(out_ids[1:], dtype=torch.long)

            return enc_inputs, dec_inputs, dec_targets
    return (ArthemeticDataset,)


@app.cell
def _(ArthemeticDataset, char_to_id, decode, df):
    all_data = list(df.itertuples(index=False, name=None))

    dataset = ArthemeticDataset(all_data)
    enc, dec_in, dec_tgt = dataset[0]

    print(enc.shape)
    print(dec_in.shape)  
    print(dec_tgt.shape) 

    print(decode(enc.tolist()))
    print(decode([char_to_id["<SOS>"]] + dec_tgt.tolist()))

    return (dataset,)


@app.cell
def _(DataLoader, dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for enc_input, dec_input, dec_target in loader:
        print(enc_input.shape)
        print(dec_input.shape)
        print(dec_target.shape)
        break

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
