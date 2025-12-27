# Helper/utils.py
import torch
import logging
import os
import numpy as np

logging.basicConfig(filename="Logs/training.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
def train_model(model, dataloader, optimizer, device, epoch, save_checkpoint=True, custom_forward=None, global_semantic_center=None):
    """
    通用訓練模型方法，支持自定義前向傳播。
    """
    model.train()
    total_loss = 0
    batch_count = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
                
        # 支持自定義前向傳播
        if custom_forward:
            outputs = custom_forward(input_ids=input_ids, labels=labels, attention_mask=batch['attention_mask'], epoch=epoch,  global_semantic_center=global_semantic_center, device=device)
        else:
            outputs = model(input_ids=input_ids, labels=labels)

        
        loss = outputs.loss
        total_loss += loss.item()

        # 反向傳播
        loss.backward()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        
        logging.info(f"Batch {batch_idx + 1}/{batch_count}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / batch_count
    
    logging.info(f"Epoch {epoch} - Training Loss: {avg_loss:.4f}")

    # if save_checkpoint:
    #     checkpoint_path = f"Checkpoints/model_epoch_{epoch}.pt"
    #     torch.save(model.state_dict(), checkpoint_path)
    #     logging.info(f"Model checkpoint saved at {checkpoint_path}")

    return avg_loss

def evaluate_model(model, dataloader, device, epoch, custom_forward=None, global_semantic_center=None):
    """
    通用評估模型方法，支持自定義前向傳播，並計算損失和準確率。
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # 支持自定義前向傳播
            if custom_forward:
                outputs = custom_forward(input_ids=input_ids, labels=labels, attention_mask=batch['attention_mask'], epoch=epoch,  global_semantic_center=global_semantic_center, device=device)
            else:
                outputs = model(input_ids=input_ids, labels=labels)

            loss = outputs.loss
            logits = outputs.logits

            # 累計損失
            total_loss += loss.item()

            # 計算準確率
            preds = torch.argmax(logits, dim=-1)  # 預測值
            mask = labels != -100  # 忽略 padding 的 token
            correct = (preds[mask] == labels[mask]).sum().item()  # 計算正確的預測數量
            total_correct += correct
            total_samples += mask.sum().item()  # 計算有效樣本數量

    # 平均損失和準確率
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    logging.info(f"Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    print(f"Epoch: {epoch}, Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
