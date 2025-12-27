import torch
import torch.nn.functional as F
import os
import csv

class OutputDifferenceLogger:
    def __init__(self, log_path="output_diff_log.csv"):
        self.log_path = log_path

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "jsd", "top1_match_rate"])

    def js_divergence(self, p, q, eps=1e-8):
        m = 0.5 * (p + q)
        return 0.5 * (F.kl_div((p + eps).log(), m + eps, reduction='batchmean') +
                      F.kl_div((q + eps).log(), m + eps, reduction='batchmean'))

    def log_difference(self, model, p_tuning, p_projected, input_ids, attention_mask, device, epoch, custom_forward=None, global_semantic_center=None):
        model.eval()

        # Step 1: 用 P_tuning 做 forward
        model.soft_prompts.data = p_tuning.to(device)
        outputs_tuning = custom_forward(input_ids=input_ids, attention_mask=attention_mask, device=device, epoch=epoch, global_semantic_center=global_semantic_center)
        logits_tuning = outputs_tuning.logits[:, -1, :]

        # Step 2: 用 P_projected 做 forward
        model.soft_prompts.data = p_projected.to(device)
        outputs_projected = custom_forward(input_ids=input_ids, attention_mask=attention_mask, device=device, epoch=epoch, global_semantic_center=global_semantic_center)
        logits_projected = outputs_projected.logits[:, -1, :]

        # Step 3: softmax + 差異計算
        probs_tuning = torch.softmax(logits_tuning, dim=-1)
        probs_projected = torch.softmax(logits_projected, dim=-1)

        jsd = self.js_divergence(probs_tuning, probs_projected).item()
        top1_match = (torch.argmax(probs_tuning, dim=-1) == torch.argmax(probs_projected, dim=-1)).float().mean().item()

        # Step 4: 寫入 CSV
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, jsd, top1_match])

        print(f"[Epoch {epoch}] Output Diff — JSD: {jsd:.4f}, Top-1 Match: {top1_match:.4f}")
