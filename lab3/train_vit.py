import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import DinoViT, DinoHead
from tqdm import tqdm
import torch.nn.functional as F
import wandb


def train_ViT_DINO(
    teacher: DinoViT,
    teacher_head: DinoHead,
    student: DinoViT,
    student_head: DinoHead,
    device: torch.types.Device,
    dataset: torch.utils.data.Dataset,
    save_path="DINO_ViT.pth",
    batch_size=256,
    num_workers=4,
    epochs=200,
    lr=5e-4,
    student_temp=0.1,
    teacher_temp=0.04,
    alpha=0.95,
    center_alpha=0.9,
    n_global_views=2,
    n_local_views=5,
):
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    teacher, teacher_head = teacher.to(device), teacher_head.to(device)
    student, student_head = student.to(device), student_head.to(device)

    optimizer = AdamW(
        list(student.parameters()) + list(student_head.parameters()), lr=lr
    )
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    student.train()
    student_head.train()
    teacher.eval()
    teacher_head.eval()

    for p in list(teacher.parameters()) + list(teacher_head.parameters()):
        p.requires_grad = False

    center = torch.zeros(teacher_head.out_dim).to(device)

    try:
        train_bar = tqdm(range(epochs), position=0)
        for epoch in train_bar:
            epoch_bar = tqdm(
                train_loader, desc=f"Epoch {epoch}", leave=False, position=1
            )

            running_loss = 0
            for batch, _ in epoch_bar:
                optimizer.zero_grad()

                global_views = torch.cat(batch[:2]).to(device)
                views = torch.cat(batch).to(device)

                student_feats = student(views)
                student_logits = student_head(student_feats)
                student_logits /= student_temp
                student_logits = student_logits.chunk(n_global_views + n_local_views)

                with torch.no_grad():
                    teacher_feats = teacher(global_views)
                    teacher_output = teacher_head(teacher_feats)
                    teacher_probs = F.softmax(
                        (teacher_output - center) / teacher_temp, dim=-1
                    )
                    teacher_probs = teacher_probs.detach().chunk(n_global_views)

                total_loss = 0
                loss_terms = 0
                for iq, q in enumerate(teacher_probs):
                    for v in range(len(student_logits)):
                        if iq == v:
                            continue
                        loss = torch.sum(
                            -q * F.log_softmax(student_logits[v], dim=-1), dim=-1
                        )
                        total_loss += loss.mean()
                        loss_terms += 1
                total_loss /= loss_terms

                running_loss += total_loss.item()

                total_loss.backward()

                grad_norm = nn.utils.clip_grad.clip_grad_norm_(
                    list(student.parameters()) + list(student_head.parameters()),
                    max_norm=5,
                )

                optimizer.step()
                lr_scheduler.step()

                epoch_bar.set_postfix(
                    {"loss": total_loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
                )

                with torch.no_grad():
                    for teacher_p, student_p in zip(
                        teacher.parameters(), student.parameters()
                    ):
                        teacher_p.mul_(alpha).add_(
                            (1 - alpha) * student_p.detach().data
                        )

                    for teacher_p, student_p in zip(
                        teacher_head.parameters(), student_head.parameters()
                    ):
                        teacher_p.mul_(alpha).add_(
                            (1 - alpha) * student_p.detach().data
                        )

                    batch_center = torch.mean(teacher_output, dim=0)
                    center = center * center_alpha + (1 - center_alpha) * batch_center

            epoch_bar.close()
            avg_loss = running_loss / len(train_loader)
            # print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
            train_bar.set_postfix(
                {
                    "loss": avg_loss,
                }
            )
    finally:
        torch.save(teacher.state_dict(), save_path)
