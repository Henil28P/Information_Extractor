import torch
from torchtext.data.metrics import bleu_score


def calculate_bleu(test_data, model, source_lang, target_lang, device):
    targets = []
    outputs = []

    model.eval()
    with torch.no_grad():
        for example in test_data:
            src = vars(example)["src"]
            trg = vars(example)["trg"]

            output = model(src.to(device), trg.to(device))
            output = output.argmax(dim=-1)

            # Cut off <eos> token
            output = output[1:]

            targets.append([trg.cpu().numpy()])
            outputs.append(output.cpu().numpy())

    return bleu_score(outputs, targets)


def saveCheckpoint(model, optimizer, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"    Checkpoint saved to '{filename}'")

# Usage
# save_checkpoint(model, optimizer, 'transformer_checkpoint.pth')


def loadCheckpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"    Checkpoint loaded from '{filename}'")

# Usage
# load_checkpoint(model, optimizer, 'transformer_checkpoint.pth')

