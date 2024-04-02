import torch
from pathlib import Path
from shutil import rmtree
from argparse import ArgumentParser, Namespace

from .train import Trainer
from .eval import Evaluator
from .model import ResNet18, ResNet50, ResNet152


def remove_and_create_dir(path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    if path.is_dir():
        rmtree(path)
    path.mkdir()


def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--expr', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--alphabet', type=str, required=True)
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--valid', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_only', action='store_false')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--load', type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_argument()

    with open(args.alphabet, 'r', encoding='utf-8') as f:
        alphabet = f.read().strip()

    if args.model == 'ResNet18':
        model = ResNet18(len(alphabet))
    elif args.model == 'ResNet50':
        model = ResNet50(len(alphabet))
    else:
        model = ResNet152(len(alphabet))

    device = torch.device(args.device)
    expr = Path('./history') / args.expr
    remove_and_create_dir(expr)

    evaluator = Evaluator(expr, alphabet, model, args.valid, device, args.bs)
    trainer = Trainer(expr, model, evaluator, args.train, device, args.lr, args.bs)

    if args.load:
        model.load_state_dict(torch.load(args.load))

    if args.test_only:
        evaluator(0)
        return

    trainer()
