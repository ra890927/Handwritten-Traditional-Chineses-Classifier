import torch
from torch import no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Union

from .model import ResNet
from .dataset import HCCRDataset


class Evaluator:
    def __init__(
            self,
            expr: Union[str, Path],
            alphabet: str,
            model: ResNet,
            dataset: HCCRDataset,
            device: torch.device,
            bs: int
    ) -> None:
        self.expr = expr
        self.model = model
        self.device = device
        self.alphabet = alphabet
        self.dataloader = DataLoader(dataset, batch_size=bs)

    @no_grad()
    def __call__(self, epoch: int) -> float:
        with open(self.expr / f'result_{epoch:02d}.txt', 'w+', encoding='utf-8') as f:
            self.model.eval()
            acc, total_id = 0, 1
            for data, label in tqdm(self.dataloader):
                inputs = data.to(self.device)
                labels = label.to(self.device).long()

                pred = self.model(inputs)
                pred_argmax = torch.argmax(pred, dim=1)
                label_argmax = torch.argmax(labels, dim=1)

                for pid, lid in zip(pred_argmax, label_argmax):
                    acc += pid == lid
                    f.write(f'{total_id} | {self.alphabet[pid]} | {self.alphabet[lid]}\n')
                    total_id += 1

        return acc / len(self.dataloader.dataset) * 100
