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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
