import json
import torch
from pathlib import Path
from shutil import rmtree
from torch.utils.data import ConcatDataset
from argparse import ArgumentParser, Namespace

from trainer import Trainer
from evaluator import Evaluator
from dataset import HCCRDataset
from model import ResNet18, ResNet50, ResNet152


def remove_and_create_dir(path: Path) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    if path.is_dir():
        rmtree(path)
    path.mkdir()


def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--expr', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--alphabet', type=str)
    parser.add_argument('--eval', action='append', type=str)
    parser.add_argument('--train', action='append', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bs', type=int, default=80)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--save', action='store_true', default=True)
    parser.add_argument('--load', type=str)
    return parser.parse_args()

def save_args(args: Namespace) -> None:
    with open(Path('./history') / args.expr / 'args.json', 'w+', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)

def laod_args(args: Namespace) -> Namespace:
    with open(args.load, 'r', encoding='utf-8') as f:
        loaded_args = json.load(f)
    for key, val in loaded_args.items():
        setattr(args, key, val)
    return args

def main() -> None:
    args = parse_argument()

    if args.load:
        args = laod_args(args)

    with open(args.alphabet, 'r', encoding='utf-8') as f:
        alphabet = f.read().strip()

    if args.model == 'ResNet18':
        model = ResNet18(len(alphabet))
    elif args.model == 'ResNet50':
        model = ResNet50(len(alphabet))
    else:
        model = ResNet152(len(alphabet))

    device = torch.device(args.device)
    model = model.to(device)
    expr = Path('./history') / args.expr
    remove_and_create_dir(expr)

    if args.save:
        save_args(args)
    
    eval_dataset = ConcatDataset([HCCRDataset('eval', eval_path, alphabet) for eval_path in args.eval])
    train_dataset = ConcatDataset([HCCRDataset('train', train_path, alphabet) for train_path in args.train])

    evaluator = Evaluator(expr, alphabet, model, eval_dataset, device, args.bs)
    trainer = Trainer(expr, model, evaluator, train_dataset, device, args.lr, args.bs)

    if args.resume:
        model.load_state_dict(torch.load(args.resume))

    if args.test_only:
        acc = evaluator(0)
        print(f'ACC: {round(acc, 4)}')
        return

    trainer(args.epochs)


if __name__ == '__main__':
    main()