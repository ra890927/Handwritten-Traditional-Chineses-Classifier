import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Union

from model import ResNet
from evaluator import Evaluator
from dataset import HCCRDataset

from torch import nn
    

class Trainer:
    def __init__(
            self,
            expr: Union[str, Path],
            model: ResNet,
            evaluator: Evaluator,
            dataset: HCCRDataset,
            device: torch.device,
            lr: float,
            bs: int
    ) -> None:
        self.expr = expr
        self.model = model
        self.device = device
        self.evaluator = evaluator
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(model.parameters(), lr=lr)
        self.dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4)

    def __call__(self, epochs: int) -> None:
        best_acc = 0
        for epoch in range(epochs):
            self.model.train()
            acc, total_loss = 0, 0
            for data, label in (pbar := tqdm(self.dataloader)):
                inputs = data.to(self.device)
                labels = label.to(self.device).float()

                pred = self.model(inputs)
                loss = self.criterion(pred, labels)
                total_loss += loss.item()

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                pred_argmax = torch.argmax(pred, dim=1)
                label_argmax = torch.argmax(labels, dim=1)
                acc += (pred_argmax == label_argmax).sum().item()

                self.__print_pbar(pbar, epoch, float(loss.item()))

            test_acc = self.evaluator(epoch + 1)
            print(f'Eval: {round(test_acc, 4)}%')
            if test_acc > best_acc:
                print('Save model')
                best_acc = test_acc
                torch.save(self.model.state_dict(), str(self.expr / 'best.pth'))

    def __print_pbar(self, pbar: tqdm, epoch: int, loss: float) -> None:
        pbar.set_description(f'Epoch: {epoch}', refresh=False)
        pbar.set_postfix(loss=round(loss, 4), refresh=False)
        pbar.refresh()
