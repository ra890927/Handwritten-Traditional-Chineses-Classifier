import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn import preprocessing
from typing import Tuple, List, Union


class HCCRDataset(Dataset):
    def __init__(self, fp: Union[str, Path], alphabet: str) -> None:
        self.alphabet = alphabet
        self.data = self.__transform_df_to_data(fp)

        self.transforms = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, List[int]]:
        img_path, label = self.data[index]
        img = self.transforms(Image.open(img_path))
        return img, label

    def __transform_df_to_data(self, fp: Union[str, Path]) -> List[Tuple[str, np.array]]:
        df = pd.read_csv(str(fp))
        char_to_index = {c: i for i, c in enumerate(self.alphabet)}
        df['label_index'] = df['label'].map(char_to_index)

        lb = preprocessing.LabelBinarizer()
        lb.fit([i for i in range(len(self.alphabet))])

        ret = []
        for _, row in df.iterrows():
            ret.append((
                row['image'],
                np.array(lb.transform([row['label_index']]))
            ))

        return ret
