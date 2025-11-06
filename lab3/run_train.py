from argparse import ArgumentParser
from train_vit import train_ViT_DINO
from model import DinoViT, DinoHead
from train_dataset import train_dataset
import copy
import torch
import wandb

parser = ArgumentParser(
    prog="run_train",
)
parser.add_argument("--device", default="cuda")
parser.add_argument("--checkpoint")
parser.add_argument("--save", default="DINO_ViT.pth")
parser.add_argument("--batch_size", default=256)
parser.add_argument("--num_workers", default=16)
parser.add_argument("--epochs", default=100)
args = parser.parse_args()

device = torch.device(args.device)
print(f"[*] Running on {device}")

teacher = DinoViT()

if isinstance(args.checkpoint, str):
    print(f"[+] loading checkpoint from {args.checkpoint}")
    teacher.load_state_dict(torch.load(args.checkpoint))

student = copy.deepcopy(teacher)

teacher_head = DinoHead(teacher.embed_dim, 128)
student_head = DinoHead(teacher.embed_dim, 128)

train_ViT_DINO(
    dataset=train_dataset,
    teacher=teacher,
    teacher_head=teacher_head,
    student=student,
    student_head=student_head,
    device=device,
    save_path=args.save,
    batch_size=int(args.batch_size) if args.batch_size else None,
    num_workers=int(args.num_workers) if args.num_workers else None,
    epochs=int(args.epochs) if args.epochs else None,
)
