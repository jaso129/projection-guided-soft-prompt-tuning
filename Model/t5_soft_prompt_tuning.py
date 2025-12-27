import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn.functional as F
from Config.config import *
from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)
class SoftPromptTuning(nn.Module):
    """
    Soft Prompt Tuning 模型
    """
    def __init__(self, model_name: str, n_tokens: int, prefix_len: int, train_dataloader, device):
        """
        初始化 SoftPromptTuning 模型
        Args:
            model_name (str): 預訓練模型的名稱 (e.g., "t5-base")
            n_tokens (int): 引導的虛擬 token 數量
            prefix_len (int): 引導的前綴長度
        """
        super(SoftPromptTuning, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.n_tokens = n_tokens
        self.prefix_len = prefix_len
        
        # 從訓練資料嵌入空間初始化
        # embedding_space = self.extract_and_cache_training_embeddings_with_context(self.model, train_dataloader, device)
        # self.soft_prompts = nn.Parameter(self.init_prompts_from_training_embeddings(embedding_space, self.n_tokens))
        
        #改進的初始化，從詞嵌入矩陣中選取隨機嵌入進行初始化
        #🚀 這裡用 torch.Generator() 產生獨立的隨機狀態
        # gen = torch.Generator()
        # gen.manual_seed(int(time.time() * 1000000) % 2**32)
        # self.soft_prompts = nn.Parameter(self.init_prompts_from_vocab(generator=gen))
        self.soft_prompts = nn.Parameter(self.init_prompts_from_vocab())
        
        # 完全隨機初始化
        # self.soft_prompts = nn.Parameter(self.init_prompts_random())
        
        
    def init_prompts_from_vocab(self, generator=None):
        """
        初始化 Prompt 嵌入，從模型的詞嵌入矩陣中選取隨機嵌入。
        """
        vocab_size, embed_dim = self.model.shared.weight.shape
        # 使用指定的 Generator 來確保 soft prompt 初始化是隨機的
        # random_indices = torch.randint(0, vocab_size, (self.n_tokens,), generator=generator)
        random_indices = torch.randint(0, vocab_size, (self.n_tokens,))
        initial_embeds = self.model.shared.weight[random_indices].detach().clone()
        return initial_embeds
    
    def init_prompts_far_from_embedding(self, embedding_space, scale_factor=5):
        """
        基於 Cosine Similarity 初始化一個遠離訓練資料嵌入空間的 soft prompt。

        Args:
            embedding_space (torch.Tensor): 訓練資料的嵌入空間 (num_samples, embedding_dim)
            scale_factor (float): 控制遠離程度的係數 (預設為 5)

        Returns:
            torch.Tensor: 距離嵌入空間足夠遠的 soft prompt
        """
        embed_mean = embedding_space.mean(dim=0)  # 嵌入空間的均值
        embed_std = embedding_space.std(dim=0)    # 嵌入空間的標準差
        
        # **步驟 1：隨機初始化 soft prompt，並標準化為單位向量**
        random_vectors = torch.randn(self.n_tokens, embedding_space.shape[1])  # 隨機初始化
        random_vectors = F.normalize(random_vectors, p=2, dim=1)  # L2 正規化，確保是單位向量
        
        # **步驟 2：計算與嵌入空間的 Cosine Similarity**
        cosine_sim = F.cosine_similarity(random_vectors.unsqueeze(1), embedding_space.unsqueeze(0), dim=2)  # (n_tokens, num_samples)
        min_cosine_sim, _ = cosine_sim.max(dim=1)  # 取與嵌入空間中最相似的點
        
        # **步驟 3：調整遠離程度**
        while min_cosine_sim.max() > 0.2:  # 確保 Cosine Similarity 低於 0.2 (距離較遠)
            random_vectors += scale_factor * torch.randn_like(random_vectors)  # 添加擾動
            random_vectors = F.normalize(random_vectors, p=2, dim=1)  # 重新正規化
            cosine_sim = F.cosine_similarity(random_vectors.unsqueeze(1), embedding_space.unsqueeze(0), dim=2)
            min_cosine_sim, _ = cosine_sim.max(dim=1)

        return random_vectors
    
    def init_prompts_random(self):
        """
        初始化 Prompt 嵌入，使其為完全隨機的數值，無關於詞嵌入空間。
        """
        embed_dim = self.model.shared.weight.shape[1]
        random_embeddings = torch.randn(self.n_tokens, embed_dim)  # 標準正態分佈 N(0,1)
        return random_embeddings

    def forward_with_prompt(self, input_ids, labels=None, attention_mask=None, epoch=None, global_semantic_center=None, device=None):
        """
        帶有 Prompt 的前向傳播方法，用於 Soft Prompt Tuning。
        Args:
            input_ids (torch.Tensor): 輸入的 token ID。
            attention_mask (torch.Tensor, optional): 注意力遮罩，默認為 None。
            labels (torch.Tensor, optional): 標籤 token ID，默認為 None。
        Returns:
            torch.Tensor: 模型的輸出，包括損失和 logits。
        """
        self.current_epoch = epoch if epoch is not None else self.current_epoch
        # 創建 soft prompt 嵌入
        soft_prompts = self.soft_prompts.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        # 原始嵌入
        inputs_embeds = self.model.shared(input_ids)
        # 合併 soft prompts 和原始嵌入
        combined_embeds = torch.cat((soft_prompts, inputs_embeds), dim=1)

        # 更新 attention mask 以適應新加入的 soft prompts
        if attention_mask is not None:
            # extended_attention_mask = torch.cat(
            #     (torch.ones(soft_prompts.size()[:2], dtype=attention_mask.dtype).to(attention_mask.device),
            #      attention_mask),
            #     dim=1
            # )
            soft_prompt_mask = torch.ones(
                (soft_prompts.size(0), soft_prompts.size(1)),  # (batch, prompt_len)
                dtype=attention_mask.dtype,
                device=device
            )
            extended_attention_mask = torch.cat((soft_prompt_mask, attention_mask.to(device)), dim=1)  # (batch, prompt_len + seq_len)
        else:
            extended_attention_mask = None

        # fix: 確保長度一致
        assert combined_embeds.size(1) == extended_attention_mask.size(1), \
            f"Embeds len: {combined_embeds.size(1)}, Mask len: {extended_attention_mask.size(1)}"

        # 前向傳播
        if labels is not None:
            outputs = self.model(
                inputs_embeds=combined_embeds,
                attention_mask=extended_attention_mask,
                labels=labels
            )
        else:
            # 自動生成 batch_size 個 decoder_input_ids，都是 <pad>（T5 預設 decoder 開始 token）
            decoder_start_token_id = self.model.config.decoder_start_token_id or self.tokenizer.pad_token_id
            decoder_input_ids = torch.full(
                (input_ids.size(0), 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=input_ids.device
            )

            outputs = self.model(
                inputs_embeds=combined_embeds,
                attention_mask=extended_attention_mask,
                decoder_input_ids=decoder_input_ids
            )

        
        ce_loss = outputs.loss
        semantic_loss = torch.tensor(0.0).to(device)

        if self.training and getattr(self, "training_mode", "vanilla") in ["semantic_only", "two_stage"]:
            if self.training_mode == "semantic_only" or self.current_epoch < getattr(self, "projection_start_epoch", 5):
                semantic_loss = self.semantic_alignment_loss_cosine(
                    soft_prompts=soft_prompts,
                    global_semantic_center=global_semantic_center
                )
        
        lambda_sem = getattr(self, "semantic_loss_weight", 0.05)
        
        # 🛠️ 根據是否有 ce_loss，決定 total_loss 要不要合併
        if ce_loss is not None:
            total_loss = ce_loss + lambda_sem * semantic_loss
        else:
            total_loss = None
        # print(f"ce: {ce_loss.item()}, semantic: {semantic_loss.item()}, total: {total_loss.item()}")
        # 5. 包裝回傳格式
        outputs.loss = total_loss
        outputs.ce_loss = ce_loss
        outputs.semantic_loss = semantic_loss
        
        return outputs

    def semantic_alignment_loss_cosine(self, soft_prompts, global_semantic_center):
        """
        計算 soft prompt 向量平均值與語義中心之間的 cosine loss。
        Args:
            soft_prompts: (B, P, D)
            global_semantic_center: (D,)
        Returns:
            Cosine-based loss
        """
        device = soft_prompts.device
        prompt_mean = soft_prompts.mean(dim=1)  # (B, D)
        global_center = global_semantic_center.to(device).unsqueeze(0)  # (1, D)

        cosine_sim = F.cosine_similarity(prompt_mean, global_center.expand_as(prompt_mean), dim=-1)  # (B,)
        cosine_loss = 1 - cosine_sim  # 越接近 1 表示越對齊 → loss 越小

        return cosine_loss.mean()

    def new_semantic_alignment_loss_global_center(self, soft_prompts, global_semantic_center):
        """
        改寫版本：將 soft prompt 平均值對齊至整個訓練集的語義中心。
        
        Args:
            soft_prompts: shape (B, P, D)
            global_semantic_center: torch.Tensor, shape (D,)
        
        Returns:
            MSE loss between soft prompt mean and global semantic center
        """
        device = soft_prompts.device
        prompt_mean = soft_prompts.mean(dim=1)  # (B, D)
        global_center = global_semantic_center.to(device).unsqueeze(0)  # (1, D)

        return F.mse_loss(prompt_mean, global_center.expand_as(prompt_mean))

    def custom_forward(self, input_ids, labels=None):
        """
        自定義前向傳播，適用於外部訓練或驗證方法。
        """
        return self.forward_with_prompt(input_ids=input_ids, labels=labels)

    def save_soft_prompts(self, path="Checkpoints/soft_prompts.pt"):
        """
        保存 soft prompts 到指定路徑。
        Args:
            path (str): 保存文件的路徑，默認為 "Checkpoints/soft_prompts.pt"。
        """
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        torch.save(self.soft_prompts.data, path)
        print(f"Soft prompts saved to {path}")

    def load_soft_prompts(self, path="Checkpoints/soft_prompts.pt"):
        """
        加載 soft prompts 文件。
        Args:
            path (str): 加載文件的路徑，默認為 "Checkpoints/soft_prompts.pt"。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found at {path}")
        
        self.soft_prompts.data = torch.load(path).to(self.soft_prompts.device)
        print(f"Soft prompts loaded from {path}")


    def project_soft_prompt_to_discrete_space(self, embedding_matrix):
        """
        將連續 prompt 投影到離散詞嵌入空間。
        Args:
            embedding_matrix (torch.Tensor): 詞嵌入矩陣。
        Returns:
            torch.Tensor: 投影後的 prompt 嵌入。
        """
        prompt_norm = F.normalize(self.soft_prompts, p=2, dim=-1)
        embedding_norm = F.normalize(embedding_matrix, p=2, dim=-1)
        similarity = torch.matmul(prompt_norm, embedding_norm.T)
        indices = torch.argmax(similarity, dim=-1)
        discrete_prompt = embedding_matrix[indices]
        return discrete_prompt
    
    def project_soft_prompt(self, current_soft_prompt, faiss_index, embedding_space_tensor, device):
        """
        將 current_soft_prompt 投影到語義空間中，產生 projected_soft_prompt，
        同時保留梯度鏈接（計算圖不中斷）
        """

        projected_soft_prompt = current_soft_prompt.clone()

        with torch.no_grad():
            query = current_soft_prompt.detach().cpu().numpy()
            _, indices = faiss_index.search(query, k=1)  # [L, 1]
            nearest_vecs = embedding_space_tensor[indices.squeeze()]  # 可能為 [d] 或 [L, d]

        nearest_vecs = torch.from_numpy(nearest_vecs).to(device)

        # 保險：強制與原 prompt 同 shape
        if nearest_vecs.shape != current_soft_prompt.shape:
            nearest_vecs = nearest_vecs.view_as(current_soft_prompt)

        projected_soft_prompt.data = nearest_vecs

        return projected_soft_prompt
    

    def project_to_embedding_space(self):
        """
        使用最近鄰投影將連續 Prompt 嵌入映射到模型的詞嵌入空間。
        Returns:
            torch.Tensor: 投影後的嵌入和最近鄰索引。
        """
        with torch.no_grad():
            soft_prompts = self.soft_prompts.view(-1, self.soft_prompts.shape[-1])
            soft_prompts = normalize_embeddings(soft_prompts)  # Query

            embedding_matrix = self.model.shared.weight
            embedding_matrix = normalize_embeddings(embedding_matrix)  # Corpus

            hits = semantic_search(soft_prompts, embedding_matrix, 
                                   query_chunk_size=soft_prompts.shape[0], 
                                   top_k=3, score_function=dot_score)

            nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=soft_prompts.device)
            projected_embeds = self.model.shared(nn_indices)

            return projected_embeds, nn_indices

    def integrate_projected_embeds(self):
        """
        將投影後的嵌入整合到當前模型中，用於進一步訓練或評估。
        """
        projected_embeds, nn_indices = self.project_to_embedding_space()
        self.soft_prompts.data = projected_embeds.data
        return nn_indices
    
    def extract_and_cache_training_sentence_embeddings(self, model, dataloader, device):
        """
        提取並緩存整個訓練集的句子級別上下文相關嵌入向量，支持 2D 標籤。
        Args:
            model: 預訓練的模型，用於提取上下文相關詞嵌入。
            dataloader: 包含訓練集的 dataloader。
            device: 訓練設備（如 "cuda" 或 "cpu"）。

        Returns:
            sentence_embeddings (torch.Tensor): 訓練集的句子級別嵌入，形狀為 [num_sentences, embedding_dim]。
            labels (torch.Tensor): 訓練集的目標標籤，形狀為 [num_sentences, num_classes]。
        """
        model.eval().to(device)
        all_sentence_embeddings = []
        all_labels = []
        all_texts = []


        with torch.no_grad():
            for batch in dataloader:
                # 載入數據
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)  # 提取標籤（支持 2D）
                raw_texts_batch = batch["raw_text"]

                # 提取 token 級別嵌入
                encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
                token_embeddings = encoder_outputs.last_hidden_state  # [batch_size, seq_len, embedding_dim]

                # 聚合為句子級嵌入
                for i in range(token_embeddings.size(0)):  # 遍歷每個句子
                    valid_token_mask = attention_mask[i].bool()  # 過濾有效 token
                    valid_embeddings = token_embeddings[i][valid_token_mask]  # [num_valid_tokens, embedding_dim]

                    # 檢查是否有有效 token
                    if valid_embeddings.size(0) == 0:
                        print(f"Warning: No valid tokens for sentence {i}. Using zero vector.")
                        sentence_embedding = torch.zeros(token_embeddings.size(-1))  # 使用零向量
                    else:
                        sentence_embedding = valid_embeddings.mean(dim=0)  # 平均池化
                    
                    all_sentence_embeddings.append(sentence_embedding.cpu())  # 保存嵌入
                    all_texts.append(raw_texts_batch[i])

                    # 保存 2D 標籤（直接保留原格式）
                    all_labels.append(labels[i].cpu())  # 注意不再轉換為標量

        # 確保嵌入和標籤數量匹配
        assert len(all_sentence_embeddings) == len(all_labels), "Mismatch between embeddings and labels!"
        # 確保嵌入與文本數量匹配
        assert len(all_sentence_embeddings) == len(all_texts), "Mismatch between embeddings and texts!"

        # 將所有句子嵌入組成矩陣，形狀 [num_sentences, embedding_dim]
        sentence_embeddings = torch.stack(all_sentence_embeddings, dim=0)

        # 將標籤組成矩陣，形狀 [num_sentences, num_classes]
        all_labels = torch.stack(all_labels, dim=0)

        return sentence_embeddings, all_texts

    def compute_cosine_distance_to_embedding_space(self, soft_prompt, embedding_space):
        """
        計算 soft prompt 與嵌入空間的平均最小 Cosine 距離。

        Args:
            soft_prompt (torch.Tensor): 初始化的 Soft Prompt (n_tokens, embedding_dim)
            embedding_space (torch.Tensor): 訓練數據構建的嵌入空間 (num_samples, embedding_dim)

        Returns:
            float: Soft Prompt 與嵌入空間的平均最小 Cosine 距離
        """
        soft_prompt = soft_prompt.clone().detach().cpu()
        embedding_space = torch.from_numpy(embedding_space)
        # **計算 Cosine Similarity**
        cosine_sim = F.cosine_similarity(soft_prompt.unsqueeze(1), embedding_space.unsqueeze(0), dim=2)  # (n_tokens, num_samples)

        # **取每個 Soft Prompt 向量到嵌入空間最近的 Cosine Similarity**
        max_similarity, _ = cosine_sim.max(dim=1)  # 找出 soft prompt 與嵌入空間最接近的點 (值越接近 1，代表越相似)

        # **計算 Cosine 距離 = 1 - 最大相似度**
        cosine_distance = 1 - max_similarity  # Cosine Distance = 1 - Cosine Similarity

        return cosine_distance.mean().item()  # 回傳平均 Cosine 距離