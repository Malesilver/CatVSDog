from utils import *
import tensorboard_logger as tl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import Data
from model import *
import copy


def main():
    make_new_dir()
    tl.configure(logdir=SAVE_PATH + '/log', flush_secs=3)

    train_data, val_data = Data(TRAIN_IMG_PATH, "train"), Data(TRAIN_IMG_PATH, "val")
    train_dataloader, val_dataloader = DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=WORKERS), \
                                       DataLoader(val_data, BATCH_SIZE, shuffle=True, num_workers=WORKERS)

    model = Simplenet(1, 1).to(DEVICE)
    optimizer = t.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.001,
                                  min_lr=0)
    wait = 0  # for early stop
    lowest_val_loss = 1e3

    for epoch in range(1, MAX_EPOCH + 1):
        print('Epoch {}/{}'.format(epoch, MAX_EPOCH))
        print('-' * 10)

        train_loss, train_acc = train(model, train_dataloader, optimizer)
        tl.log_value("train_loss", train_loss, step=epoch)
        tl.log_value("train_acc", train_acc, step=epoch)

        val_loss, val_acc = val(model, val_dataloader)
        tl.log_value("val_loss", val_loss, step=epoch)
        tl.log_value("val_acc", val_acc, step=epoch)

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait > 20:
            print("Stop at epoch {}".format(epoch))
            break

        print()

        scheduler.step(val_loss)

    print("Best val Loss: {:.4f}", lowest_val_loss)
    model_name = "{}/model/000best_val_loss_{:.4f}_model.pt".format(SAVE_PATH, lowest_val_loss)
    t.save(best_model_wts, model_name)


def train(model, data_loader, optimizer):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for data in tqdm(data_loader):
        (imgs, labels) = data
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        pred = model(imgs)
        loss = F.binary_cross_entropy(t.sigmoid(pred), labels.float().unsqueeze(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        pred_classes = [1 if p > 0.5 else 0 for p in t.sigmoid(pred)]
        pred_classes = t.tensor(pred_classes).to(DEVICE)
        correct += (pred_classes == labels).sum().item()
        total += len(imgs)
    epoch_loss = epoch_loss / len(data_loader)
    acc = correct / total * 100
    print("Train_Loss:{:.4f}, Train_acc:{:.2f}%".format(epoch_loss, acc))
    return epoch_loss, acc


def val(model, data_loader):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with t.no_grad():
        for data in tqdm(data_loader):
            (imgs, labels) = data
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            pred = model(imgs)
            loss = F.binary_cross_entropy(t.sigmoid(pred), labels.float().unsqueeze(-1))
            epoch_loss += loss.item()
            pred_classes = [1 if p > 0.5 else 0 for p in t.sigmoid(pred)]
            pred_classes = t.tensor(pred_classes).to(DEVICE)
            correct += (pred_classes == labels).sum().item()
            total += len(imgs)
    epoch_loss = epoch_loss / len(data_loader)
    acc = correct / total * 100
    print("Val_Loss:{:.4f}, Val_acc:{:.2f}%".format(epoch_loss, acc))
    return epoch_loss, acc


def test():
    model = Simplenet(1, 1).to(DEVICE)
    model.eval()
    load_weights(model, WEIGHTS)
    test_data = Data(TEST_IMG_PATH, "test")
    test_dataloader = DataLoader(test_data, BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    result = []
    for data in tqdm(test_dataloader):
        (imgs, ids) = data
        imgs = imgs.to(DEVICE)
        pred = model(imgs)
        pred_classes = ["Cat" if p > 0.5 else "Dog" for p in t.sigmoid(pred)]
        batch_result = [(id, pred_class) for id, pred_class in zip(ids, pred_classes)]
        result += batch_result
    write_csv(result, TEST_SAVE_PATH)


if __name__ == '__main__':
    # main()
    test()