import torch
from torch import no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Union

from model import ResNet
from dataset import HCCRDataset


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
        self.dataloader = DataLoader(dataset, batch_size=bs, num_workers=4)

        self.all_records = open(self.expr / 'records.txt', 'w+', encoding='utf-8')

    def __del__(self) -> None:
        self.all_records.close()

    @no_grad()
    def __call__(self, epoch: int) -> float:
        with open(self.expr / f'result_{epoch:02d}.txt', 'w+', encoding='utf-8') as f:
            self.model.eval()
            acc, total_id = 0, 1
            for data, label in tqdm(self.dataloader):
                inputs = data.to(self.device)
                labels = label.to(self.device).float()

                pred = self.model(inputs)
                pred_argmax = torch.argmax(pred, dim=1)
                label_argmax = torch.argmax(labels, dim=1)

                for pid, lid in zip(pred_argmax, label_argmax):
                    pid, lid = pid.item(), lid.item()
                    acc += pid == lid
                    check = self.alphabet[pid] == self.alphabet[lid]
                    f.write(f'{total_id} | {self.alphabet[pid]} | {self.alphabet[lid]} | {check}\n')
                    total_id += 1

        self.all_records.write(f'{epoch:03d}: {acc / len(self.dataloader.dataset) * 100:.4f}\n')

        return acc / len(self.dataloader.dataset) * 100
