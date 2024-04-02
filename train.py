import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Union

from .model import ResNet
from .eval import Evaluator
from .dataset import HCCRDataset


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
        self.dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

    def __call__(self) -> None:
        best_acc = 0
        for epoch in (pbar := tqdm(range(self.epochs))):
            self.model.train()
            acc, total_loss = 0, 0
            for data, label in self.dataloader:
                inputs = data.to(self.device)
                labels = label.to(self.device).long()

                pred = self.model(inputs)
                loss = self.criterion(pred, labels)
                total_loss += loss.item()

                self.optim.zero_grad()
                loss.backard()
                self.optim.step()

                acc += (torch.argmax(pred, dim=1) == labels).sum().item()

                self.__print_pbar(pbar, epoch, float(loss.item()))

            test_acc = self.evaluator(epoch + 1)
            print(f'Eval: {round(test_acc, 4)}%')
            if test_acc > best_acc:
                print('Save model')
                torch.save(self.model.state_dict(), str(self.expr / 'best.pth'))

    def __print_pbar(self, pbar: tqdm, epoch: int, loss: float) -> None:
        pbar.set_description(f'Epoch: {epoch}', refresh=False)
        pbar.set_postfix(loss=round(loss, 4), refresh=False)
        pbar.refresh()
