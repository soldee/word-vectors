from torch import nn
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import json

from config import cfg
from vocabulary import Vocabulary

class CBOW(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, mode: int, window_size: int):
        super().__init__()
        self.mode = mode

        match mode:
            case 1:
                print("Selected CBOW mode 1: Weighted context words by proximity to the central word")
                window_left = window_size // 2
                weights_l = [i for i in range(1, window_left+1)]
                weights_r = [i for i in range(1, (window_size-window_left))]
                weights_r.reverse()

                # for a window of size 7: weights=[1,2,3,3,2,1]
                weights_l.extend(weights_r)
                print(f"Using fixed weights: {weights_l}")
                # the view is used to unsqueeze (add dimensions to left and right)
                #   the embeddings outputs will be (B,C,E) and we need to multiply the weights along the C dimension)
                weights: torch.Tensor = torch.tensor(weights_l).view(1, -1, 1)
                self.register_buffer('weights', weights)
            case 2:
                print("Selected CBOW mode 2: Learnable context word's weights (Scalar)")
                self.weights = nn.Parameter(torch.randn(window_size-1).view(1, -1, 1), requires_grad=True)
            case 3:
                print("Selected CBOW mode 3: Learnable context word's weights (Vector)")
                self.weights = nn.Parameter(torch.randn(1, window_size-1, embedding_dim), requires_grad=True)
            case _:
                print("Selected CBOW mode: Standard CBOW")

        # size of matrix = num_embeddings x embedding_dim
        self.embedding = nn.Embedding(
            # number of words. Each word is one embedding
            num_embeddings=num_embeddings,
            # number of dimensions of the vector to represent each word.
            # more dims means more expressive power at the risk of more cost and overfitting
            embedding_dim=embedding_dim, 
            padding_idx=padding_idx
        )
        self.linear = nn.Linear(embedding_dim, num_embeddings, bias=False)

    # B = Batch size
    # C = Window size = Number of context words (left + right)
    # E = Embedding dim
    def forward(self, x: torch.Tensor):
        # (B, C)
        emb: torch.Tensor = self.embedding(x)
        # (B, C, E)

        if self.mode in (1, 2, 3):
            emb = self.embedding(x)
            # (B, C, E)
            emb = emb * self.weights
            # (B, C, E)
        
        u = emb.sum(dim=1) # sum C vectors element-wise
        # (B, E)
        z = self.linear(u)
        # (B, C)
        return z
        

class CorpusDataset(Dataset):
    def __init__(self, x_file: Path, y_file: Path):
        x_np: np.ndarray = np.load(x_file, allow_pickle=True)
        y_np: np.ndarray = np.load(y_file, allow_pickle=True)

        self.x = torch.from_numpy(x_np).long()
        self.y = torch.from_numpy(y_np).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train(model: nn.Module, criterion, optimizer, loader: DataLoader, device: str):
    model.train()
    
    # stats
    total_loss = 0
    n_correct = 0
    n_tokens = 0
    i = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        model.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        # stats
        total_loss += loss.item()
        n_correct += (torch.max(output, 1)[1] == y_batch).sum().item()
        n_tokens += y_batch.numel()
        i += 1

        if i % 500 == 0:
            print(f"Train: i={i}, ntokens/i={n_tokens//i}, accuracy={(n_correct/n_tokens)*100:.2f}, loss={total_loss / n_tokens:.2f}")

    total_loss = total_loss / n_tokens
    accuracy = 100 * n_correct / n_tokens
    print(f"Train: i={i}, ntokens/i={n_tokens//i}, accuracy={accuracy:.2f}, loss={total_loss:.2f}")
    return accuracy, total_loss

@torch.no_grad()
def validate(model: nn.Module, criterion, loader: DataLoader, device: str):
    model.eval()

    total_loss = 0
    n_correct = 0
    n_tokens = 0
    i = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        output = model(x_batch)
        loss = criterion(output, y_batch)
        total_loss += loss.item()
        n_correct += (torch.max(output, 1)[1] == y_batch).sum().item()
        n_tokens += y_batch.numel()
        i += 1

    accuracy = (n_correct / n_tokens) * 100
    total_loss = (total_loss / n_tokens)
    return accuracy, total_loss


if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    print(f"Torch using device '{device}'")

    data_dir = Path(cfg.DATA_DIR)

    print("Loading Vocabulary...")
    vocabulary = Vocabulary()
    vocabulary.load(data_dir / 'vocab.pickle')

    model = CBOW(
        num_embeddings=len(vocabulary),
        embedding_dim=cfg.EMBEDDING_DIM,
        padding_idx=vocabulary.pad_idx,
        mode=cfg.CBOW_MODE,
        window_size=cfg.WINDOW_SIZE
    ).to(device)

    print(model)
    for name, param in model.named_parameters():
        print(f'{name:20}\t{param.numel()}\t{list(param.shape)}')
    print(f'TOTAL\t\t\t{sum(p.numel() for p in model.parameters())}\n')

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters())

    train_dataset = CorpusDataset(
        x_file=data_dir / (cfg.TRAIN_FILE + ".x.np"),
        y_file=data_dir / (cfg.TRAIN_FILE + ".y.np")
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True
    )

    val_dataset = CorpusDataset(
        x_file=data_dir / (cfg.VALID_FILE + ".x.np"),
        y_file=data_dir / (cfg.VALID_FILE + ".y.np")
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.BATCH_SIZE//4,
        shuffle=False
    )

    checkpoint_dir = Path(cfg.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = Path(cfg.METADATA_DIR)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    max_epoch = 0
    checkpoint_file: Path | None = None

    for file in checkpoint_dir.iterdir():
        if file.name.startswith(f"cbow_mode_{cfg.CBOW_MODE}_epoch"):
            epoch_match = re.search('epoch_(.+?).pt', file.name)
            if epoch_match:
                try:
                    epoch = int(epoch_match.group(1))
                    if checkpoint_file is None or epoch > max_epoch:
                        max_epoch = epoch
                        checkpoint_file = file
                except ValueError as e:
                    print(f"Invalid checkpoint file name. Unable to extract epoch: {file.name}")

    if checkpoint_file is None:
        print(f"No checkpoints to resume from found at '{cfg.CHECKPOINT_DIR}'. Starting training from scratch.")
    else:
        print(f"Found checkpoint to resume from: '{checkpoint_file}' at epoch {max_epoch}/{cfg.EPOCHS-1}")
        max_epoch += 1
        if max_epoch == (cfg.EPOCHS):
            print(f"Model already trained for {cfg.EPOCHS}. Either increase the number of epochs or delete the trained models.")
            exit(0)

        checkpoint = torch.load(checkpoint_file, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(max_epoch, cfg.EPOCHS):
        print(f"EPOCH {epoch}/{cfg.EPOCHS-1}")
        train_acc, train_loss = train(model, criterion, optimizer, train_loader, device)
        
        valid_acc, valid_loss = validate(model, criterion, val_loader, device)
        print(f"\nVALIDATE: accuracy={valid_acc:.2f}, loss={valid_loss:.2f}", end="\n\n\n")

        # checkpoint metadata
        history["val_acc"].append(valid_acc)
        history["val_loss"].append(valid_loss)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        checkpoint_name = f"cbow_mode_{cfg.CBOW_MODE}_epoch_{epoch}"
        metadata_file = metadata_dir / f"{checkpoint_name}.json"
        with open(metadata_file, "w") as f:
            json.dump(history, f, indent=2)

        # checkpoint model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'train_accuracy': train_acc,
            'valid_accuracy': valid_acc
        }
        checkpoint_file = checkpoint_dir / f"{checkpoint_name}.pt"
        torch.save(checkpoint, checkpoint_file)

    test_dataset = CorpusDataset(
        x_file=data_dir / (cfg.TEST_FILE + ".x.np"),
        y_file=data_dir / (cfg.TEST_FILE + ".y.np")
    )
    test_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False
    )
    acc, loss = validate(model, criterion, test_loader, device)
    print(f"\nTEST: accuracy={acc:.2f}, loss={loss:.2f}", end="\n\n\n")
    
