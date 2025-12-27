import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from Config.config import *
from Model.PEZPromptTuning import PEZPromptTuning
from Model.t5_soft_prompt_tuning import SoftPromptTuning
from Dataset.custom_dataset import CustomDataset
from Helper.utils import train_model, evaluate_model
from Helper.early_stopping import EarlyStopping
from Helper.output_difference import OutputDifferenceLogger
import faiss

# from Helper.utils import train_model, evaluate_model
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def main():
    set_random_seed(20)
    # 加載數據集名稱
    task = "wnli"

    split = "train"
    
    # 初始化 Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    # 加載訓練集
    print("Loading dataset...")
    train_dataset = CustomDataset(tokenizer, task, split=split, max_length=512, sample_size=SAMPLE_SIZE)
    print(f"Train dataset loaded with {len(train_dataset)} samples.")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Training DataLoader has {len(train_dataloader)} batches with batch size {BATCH_SIZE}.")

    if task in ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli']:
        # GLUE 任務使用原始 validation 集
        validation_dataset = CustomDataset(tokenizer, task, split='validation', max_length=512)
        print(f"✅ Validation dataset loaded with {len(validation_dataset)} samples.")
        validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # GLUE 任務使用官方 `test` 集作為最終測試
        test_dataset = CustomDataset(tokenizer, task, split='test', max_length=512)
        print(f"✅ Test dataset loaded with {len(test_dataset)} samples.")
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    else:
        # 非 GLUE 任務：從 `train` 集中切割 20% 當作驗證集
        print("Non-GLUE task detected. Splitting 'train' dataset into training and validation sets...")
        split_datasets = train_dataset.data.train_test_split(test_size=0.2, seed=42)

        train_dataset = CustomDataset(tokenizer, task, split='train', max_length=512)
        train_dataset.data = split_datasets['train']
        print(f"✅ Train dataset loaded with {len(train_dataset)} samples.")

        validation_dataset = CustomDataset(tokenizer, task, split='train', max_length=512)
        validation_dataset.data = split_datasets['test']
        print(f"✅ Validation dataset loaded with {len(validation_dataset)} samples.")

        # 保留 `test` 作為最終測試資料 (完整的 Test Set)
        test_dataset = CustomDataset(tokenizer, task, split='test', max_length=512)
        print(f"✅ Test dataset loaded with {len(test_dataset)} samples.")

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    # 初始化純 T5 模型
    # print("Initializing T5 model...")
    # pure_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    # optimizer = torch.optim.Adam(pure_model.parameters(), lr=LEARNING_RATE)

    # # 訓練純 T5 模型（可選）
    # print("Training pure T5 model...")
    # for epoch in range(1, NUM_EPOCHS + 1):
    #     train_loss = train_model(pure_model, train_dataloader, optimizer, DEVICE, epoch)
    #     eval_loss = evaluate_model(pure_model, train_dataloader, DEVICE)
    #     print(f"Epoch {epoch} - Pure T5 Evaluation Loss: {eval_loss:.4f}")

    # 初始化 Soft Prompt Tuning 模型
    print("Training Soft Prompt Tuning model...")
    # soft_prompt_model = SoftPromptTuning(MODEL_NAME, N_TOKENS, PREFIX_LEN).to(DEVICE)

    # 初始化 PEZ Prompt Tuning 模型
    soft_prompt_model = PEZPromptTuning(MODEL_NAME, N_TOKENS, PREFIX_LEN, LEARNING_RATE, train_dataloader, DEVICE).to(DEVICE)
    early_stopping = EarlyStopping(patience=4)
    output_logger = OutputDifferenceLogger()
    
    # 儲存soft prompt更新歷史
    soft_prompt_history = []
    soft_prompt_history.append(soft_prompt_model.soft_prompts.clone().detach().cpu())
    soft_prompt_after_projection_history = []
    # soft_prompt_after_projection_history.append(soft_prompt_model.soft_prompts.clone().detach().cpu())
    # 儲存Acc（實驗用）
    acc_history = []
    
    optimizer = torch.optim.Adam(soft_prompt_model.parameters(), lr=LEARNING_RATE)
    checkpoint_path = os.path.join('Checkpoints', "soft_prompts.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading saved soft prompts from {checkpoint_path}...")
        soft_prompt_model.load_soft_prompts(checkpoint_path)
    
    embedding_space, raw_texts = soft_prompt_model.extract_and_cache_training_sentence_embeddings(soft_prompt_model.model, train_dataloader, DEVICE)
    global_semantic_center = soft_prompt_model.compute_global_semantic_center(embedding_space)
    embedding_space = embedding_space.float()  # 確保是 float32
    embedding_space = torch.nn.functional.normalize(embedding_space, p=2, dim=1)  # L2 norm
    # np.save("embedding_space_wnli.npy", embedding_space.cpu().numpy()) 
    embedding_space = embedding_space.numpy().astype("float32")
    # 建立 index，這裡用 cosine 相似度（其實是 normalized dot product）
    index = faiss.IndexFlatIP(embedding_space.shape[1])  # dim = D
    index.add(embedding_space)  # 加進所有向量
    
    # index = ""
    # global_semantic_center = ""
    
    # embedding_space = soft_prompt_model.model.shared.weight  # 詞嵌入矩陣
    # raw_texts = ""
    
    # cosine_dist = soft_prompt_model.compute_cosine_distance_to_embedding_space(soft_prompt_model.soft_prompts, embedding_space)
    # print(f"初始化的 Soft Prompt 與嵌入空間的平均最小 Cosine 距離: {cosine_dist:.4f}")
    
    static_batch = next(iter(validation_dataloader))
    input_ids_obs = static_batch["input_ids"].to(DEVICE)
    attention_mask_obs = static_batch["attention_mask"].to(DEVICE)
        
    # 訓練 Soft Prompt Tuning 模型
    
    for epoch in range(1, NUM_EPOCHS + 1):

        # 鎖定基礎模型參數
        for param in soft_prompt_model.model.parameters():
            param.requires_grad = False  # 確保基礎模型參數不參與更新

        # 檢查優化器參數
        # print("Optimizer parameters:")
        # for group in optimizer.param_groups:
        #     for param in group['params']:
        #         print(f"Shape: {param.shape}, Requires_grad: {param.requires_grad}")

        # 使用通用訓練方法，傳入自定義前向傳播
        train_loss = train_model( 
            model=soft_prompt_model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
            custom_forward=soft_prompt_model.custom_forward,
            global_semantic_center=global_semantic_center
        )
        
        # === 儲存 P_tuning（投影前的 soft prompt）===
        p_tuning = soft_prompt_model.soft_prompts.detach().cpu()
        
        # 在每個 epoch 後執行離散化投影 (PEZ) 
        soft_prompt_model.finalize_discrete_prompts(
            train_loader=train_dataloader, 
            device=DEVICE, 
            soft_prompt_after_projection_history=soft_prompt_after_projection_history, 
            embedding_matrix=embedding_space, 
            index=index, 
            raw_texts=raw_texts, 
            global_semantic_center=global_semantic_center, 
            epoch=epoch
        )
        # train_loss = train_model(soft_prompt_model, train_dataloader, optimizer, DEVICE, epoch, False)
        # === 儲存 P_projected（投影訓練後的 soft prompt）===
        p_projected = soft_prompt_model.soft_prompts.detach().cpu()
        
        # === 計算距離並記錄 ===
        soft_prompt_model.log_prompt_shift(epoch, p_tuning, p_projected)
        
        # === 計算輸出並記錄 ===
        output_logger.log_difference(
            model=soft_prompt_model,
            p_tuning=p_tuning,
            p_projected=p_projected,
            input_ids=input_ids_obs,
            attention_mask=attention_mask_obs,
            device=DEVICE,
            epoch=epoch,
            custom_forward=soft_prompt_model.custom_forward,
            global_semantic_center=global_semantic_center
        )
        
        # 驗證模型
        eval_loss, eval_accuracy = evaluate_model(
            model=soft_prompt_model,
            dataloader=validation_dataloader,
            device=DEVICE,
            epoch=epoch,
            custom_forward=soft_prompt_model.custom_forward,
            global_semantic_center=global_semantic_center
        )
    
        # 早停設置
        early_stop_triggered = early_stopping(eval_accuracy, soft_prompt_model)
        # semantic_stable_triggered = stability_check.check_stability(current_soft_prompts)

        # if combined_monitor.check_stop(early_stop_triggered, semantic_stable_triggered):
        #     print(f"✅ Training Stopped at Epoch {epoch+1}")
        #     break
        if early_stop_triggered:
            print(f"✅ Training Stopped at Epoch {epoch+1}")
            break

        acc_history.append(eval_accuracy)
    
        # # 基於 P' 的損失更新 P
        # soft_prompt_history.append(soft_prompt_model.soft_prompts.clone().detach().cpu())
        
        # # 保存每個 epoch 後的 soft_prompts
        # epoch_checkpoint_path = f"Checkpoints/soft_prompts_epoch_{epoch}.pt"
        # soft_prompt_model.save_soft_prompts(epoch_checkpoint_path)
        # print(f"Saved soft prompts checkpoint at {epoch_checkpoint_path}")
        
    best_model = soft_prompt_model
    best_model.load_state_dict(torch.load("best_model.pt"))
    best_model.eval()

    # 判斷是否為 GLUE 任務，避免使用沒標籤的 test set
    if task in ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli']:
        print("⚠️ Skipping final test evaluation for GLUE task (no labels in test set).")
        print("✅ Use validation set accuracy as final performance.")
    else:
        # 非 GLUE 任務才做 test set 評估
        final_loss, final_accuracy = evaluate_model(
            model=soft_prompt_model,
            dataloader=test_dataloader,
            device=DEVICE,
            epoch="Final Evaluation",
            custom_forward=soft_prompt_model.custom_forward
        )
        print(f"🔹 Final Accuracy (Best Model): {final_accuracy:.4f}")
        
    # 將嵌入儲存為.npy
    # soft_prompt_tuning = np.array([emb.cpu().numpy() if isinstance(emb, torch.Tensor) else emb for emb in soft_prompt_history])
    # np.save("soft_prompt_AG_news_embedding_layer_initialization.npy", soft_prompt_tuning)
    # soft_prompt_projection = np.array([emb.cpu().numpy() if isinstance(emb, torch.Tensor) else emb for emb in soft_prompt_after_projection_history])
    # np.save("soft_prompt_projection_wnli.npy", soft_prompt_projection)

    # print("Every epoch's eval accuracy:", acc_history)
if __name__ == "__main__":
    main()
