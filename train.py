from torch import nn
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np

from config import cfg
from vocabulary import Vocabulary

class CBOW(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        super().__init__()

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
        padding_idx=vocabulary.pad_idx
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
        x_file=data_dir / (cfg.TRAIN_FILE + ".x.np"),
        y_file=data_dir / (cfg.TRAIN_FILE + ".y.np")
    )
    val_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE//4,
        shuffle=False
    )

    checkpoint_dir = Path(cfg.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_accuracy = []
    val_accuracy = []

    for epoch in range(cfg.EPOCHS):
        print(f"EPOCH {epoch}/{cfg.EPOCHS-1}")
        train_acc, loss = train(model, criterion, optimizer, train_loader, device)
        train_accuracy.append(train_acc)

        valid_acc, loss = validate(model, criterion, val_loader, device)
        val_accuracy.append(valid_acc)
        print(f"\nVALIDATE: accuracy={valid_acc:.2f}, loss={loss:.2f}", end="\n\n\n")

        # checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'train_accuracy': train_acc,
            'valid_accuracy': valid_acc
        }
        torch.save(checkpoint, checkpoint_dir / f"cbow_epoch_{epoch}.pt")

    test_dataset = CorpusDataset(
        x_file=data_dir / (cfg.TRAIN_FILE + ".x.np"),
        y_file=data_dir / (cfg.TRAIN_FILE + ".y.np")
    )
    test_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False
    )
    acc, loss = validate(model, criterion, test_loader, device)
    print(f"\nTEST: accuracy={acc:.2f}, loss={loss:.2f}", end="\n\n\n")
    
