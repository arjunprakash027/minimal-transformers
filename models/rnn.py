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
    df = pd.read_csv('../bodmas.csv')
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
    import wandb

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(device)
    return DataLoader, Dataset, device, nn, torch, wandb


@app.cell
def _(VOCAB):
    VOCAB_SIZE = len(VOCAB)
    EMB_DIM = 32
    HIDDEN_DIM = 64
    return EMB_DIM, HIDDEN_DIM, VOCAB_SIZE


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
    data_size = df.shape[0]
    train_size = int(data_size * 0.8)
    test_size = data_size - train_size

    train_df = df[:train_size]
    test_df = df[train_size:]

    train_data = list(train_df.itertuples(index=False, name=None))
    test_data = list(test_df.itertuples(index=False, name=None))

    dataset = ArthemeticDataset(train_data)

    enc, dec_in, dec_tgt = dataset[0]

    print(enc.shape)
    print(dec_in.shape)  
    print(dec_tgt.shape) 

    print(decode(enc.tolist()))
    print(decode([char_to_id["<SOS>"]] + dec_tgt.tolist()))
    return dataset, test_data


@app.cell
def _(DataLoader, dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for enc_input, dec_input, dec_target in loader:
        print(enc_input.shape)
        print(dec_input.shape)
        print(dec_target.shape)
        break
    return (loader,)


@app.cell
def _(char_to_id, nn, torch):
    class Encoder(nn.Module):
        def __init__(self, vocab_size, emb_dim, hidden_dim, pad_idx):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
            self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, x):
            emb = self.emb(x)
            _, h = self.rnn(emb)
            h = self.norm(h.squeeze(0))
            return h

    class Decoder(nn.Module):
        def __init__(self, vocab_size, emb_dim, hidden_dim, pad_idx):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
            self.cell = nn.RNNCell(emb_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)
            self.fc = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x, h0):
            B, S = x.shape

            emb = self.emb(x)
            h = h0
            outputs = []        

            for t in range(S):
                h = self.cell(emb[:, t], h)
                h = self.norm(h)
                logits = self.fc(h)
                outputs.append(logits)

            return torch.stack(outputs, dim=1), h

    class Seq2Seq(nn.Module):
        def __init__(self, vocab_size, emb_dim, hidden_dim, pad_idx):
            super().__init__()
            self.encoder = Encoder(vocab_size, emb_dim, hidden_dim, pad_idx)
            self.decoder = Decoder(vocab_size, emb_dim, hidden_dim, pad_idx)

        def forward(self, x, y):
            h = self.encoder(x)
            out, _ = self.decoder(y, h)
            return out

    loss_fn = nn.CrossEntropyLoss(ignore_index=char_to_id["<PAD>"])
    return Seq2Seq, loss_fn


@app.cell
def _(char_to_id, decode, encode_input, torch):
    PAD_ID = char_to_id["<PAD>"]
    SOS_ID = char_to_id["<SOS>"]
    EOS_ID = char_to_id["<EOS>"]

    def infer(model, questions, max_len=20, device="mps"):
        model.eval()
        batch_size = len(questions)

        enc_ids_infer = [encode_input(q) for q in questions]
        enc_inp_infer = torch.tensor(enc_ids_infer, dtype=torch.long).to(device)

        with torch.no_grad():
            h = model.encoder(enc_inp_infer)


        current_tokens = torch.full((batch_size, 1), SOS_ID, dtype=torch.long).to(device)
        all_preds = torch.zeros((batch_size, max_len), dtype=torch.long).to(device)

        for i in range(max_len):
            with torch.no_grad():
                out_logits, h = model.decoder(current_tokens, h)

            next_tokens = out_logits.argmax(dim=-1)
            all_preds[:, i] = next_tokens.squeeze(1).cpu()
            current_tokens = next_tokens

        return [decode(row.tolist()) for row in all_preds]
    return (infer,)


@app.cell
def _(EMB_DIM, HIDDEN_DIM, Seq2Seq, VOCAB_SIZE, char_to_id, device):
    model = Seq2Seq(
        vocab_size=VOCAB_SIZE,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        pad_idx=char_to_id["<PAD>"]
    ).to(device)
    return (model,)


@app.cell
def _(
    EMB_DIM,
    HIDDEN_DIM,
    VOCAB_SIZE,
    device,
    loader,
    loss_fn,
    model,
    torch,
    wandb,
):
    # single pass
    NUM_EPOCHS = 250

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    wandb.init(
        project="training-a-calculator",
        config={
            "epochs": NUM_EPOCHS,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "emb_dim": EMB_DIM,
            "hidden_dim": HIDDEN_DIM,
            "vocab_size": VOCAB_SIZE,
        }
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for enc_inp_train, dec_inp_train, dec_targ_train in loader:
            enc_inp_train = enc_inp_train.to(device)
            dec_inp_train = dec_inp_train.to(device)
            dec_targ_train = dec_targ_train.to(device)

            logits = model(enc_inp_train, dec_inp_train)

            B, S, V = logits.shape

            loss = loss_fn(
                logits.view(B * S, V),
                dec_targ_train.view(B * S)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        wandb.log({"epoch": epoch, "loss": avg_loss})

    wandb.finish()
    return


@app.cell
def _(infer, model, test_data):
    from sklearn.metrics import accuracy_score
    y_true = []
    y_pred = []
    batch_size = 64

    for items in range(0, len(test_data), batch_size):

        batch = test_data[items : items + batch_size]
        qs = [item[0] for item in batch]
        ans = [item[1] for item in batch]

        preds = infer(model, qs)

        y_true.extend(ans)
        y_pred.extend(preds)
    return accuracy_score, y_pred, y_true


@app.cell
def _(accuracy_score, y_pred, y_true):
    accuracy_score(y_true, y_pred)
    return


@app.cell
def _(y_pred):
    print(y_pred)
    return


@app.cell
def _(accuracy_score, y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
