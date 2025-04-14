import torch
from tqdm import tqdm
from utils.metrics import compute_accuracy
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for batch_X, batch_Y in tqdm(dataloader, desc="Training", leave=False):
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_Y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)

        correct += compute_accuracy(logits, batch_Y)
        total += batch_Y.size(0)

        _, preds = torch.max(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch_Y.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    f1 = f1_score(all_targets, all_preds, average='macro')
    return avg_loss, accuracy, f1


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch_X, batch_Y in tqdm(dataloader, desc="Validation", leave=False):
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_Y)
            running_loss += loss.item() * batch_X.size(0)

            correct += compute_accuracy(logits, batch_Y)
            total += batch_Y.size(0)

            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_Y.cpu().numpy())

            # Apply softmax or sigmoid to get probabilities
            if logits.shape[1] == 1:
                # Binary classification
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.squeeze())
            elif logits.shape[1] == 2:
                # Binary classification with two logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                positive_class_probs = probs[:, 1]  # Probability for class 1
                all_probs.extend(positive_class_probs)
            else:
                # Multi-class classification
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs)

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    f1 = f1_score(all_targets, all_preds, average='macro')   # 如果是多分类问题使用 'macro'
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_targets, all_preds)

    return avg_loss, accuracy, f1, precision, recall, conf_matrix, all_preds, all_targets, all_probs

