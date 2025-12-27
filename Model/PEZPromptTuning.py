import torch
from Model.t5_soft_prompt_tuning import SoftPromptTuning
import torch.nn.functional as F
import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
import os
import json
import csv

class PEZPromptTuning(SoftPromptTuning):
    def __init__(self, model_name: str, n_tokens: int, prefix_len: int, learning_rate: float, train_dataloader, device):
        super(PEZPromptTuning, self).__init__(model_name, n_tokens, prefix_len, train_dataloader, device)
        self.optimizer = torch.optim.Adam([self.soft_prompts], lr=learning_rate)
        self.discrete_prompt_history = []  # 用於記錄每次離散化的詞彙索引
        
        self.training_mode = "vanilla"
        self.semantic_loss_weight = 0
        self.projection_start_epoch = 5
        
    def custom_forward(self, input_ids, labels=None, attention_mask=None, epoch=None, global_semantic_center=None, device=None):
        """
        自定義前向傳播，僅返回損失以便在 train_model 中進行梯度計算。
        """
        # embedding_matrix = self.model.shared.weight  # 詞嵌入矩陣
        # discrete_prompts = self.project_soft_prompt_to_discrete_space(embedding_matrix)
        # soft_prompts_backup = self.soft_prompts.data.clone()  # 暫存連續嵌入

        # self.soft_prompts.data = discrete_prompts.data  # 替換為離散嵌入
        outputs = self.forward_with_prompt(input_ids, labels, attention_mask, epoch, global_semantic_center, device)

        # self.soft_prompts.data = soft_prompts_backup  # 恢復連續嵌入
        return outputs

    def finalize_discrete_prompts(self, train_loader, device, soft_prompt_after_projection_history, embedding_matrix, index=None,raw_texts=None, global_semantic_center=None, epoch=None):
        """
        基於離散化的 P' 計算損失，並使用梯度更新連續嵌入 P，確保投影方向與先前更新方向一致。
        Args:
            train_loader: 訓練數據加載器
            device: 訓練設備
            soft_prompt_history: 保存的 soft prompt 歷史
            threshold: 方向一致性的閾值
        """
        self.train()  # 設置為訓練模式
        total_loss = 0
        # 儲存當前epoch所有的投影嵌入
        current_projection_embeddings = []
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # 1. 暫存連續嵌入 P
            soft_prompts_backup = self.soft_prompts.data.clone()
            
            # 2. 計算 P'（離散化投影）
            discrete_prompts = self.project_soft_prompt(self.soft_prompts, index, embedding_matrix, device)
            # embedding_matrix = embedding_matrix.to(device)
            # discrete_prompts = self.project_soft_prompt_to_discrete_space(embedding_matrix)
    
            # # **1.1 計算 batch 內 token 的 pairwise 相似度**
            # similarity_matrix = self.compute_pairwise_similarity(discrete_prompts.cpu().numpy())
            
            # # **1.2 找出過於分散的 token**
            # low_similarity_tokens = self.should_replace_token(similarity_matrix)
            
            # # **1.3 計算 batch 語義中心**
            # batch_center = self.compute_batch_center(discrete_prompts.cpu().numpy())

            # # **1.4 修正過於分散的 token**
            # if len(low_similarity_tokens) > 0:
            #     nearest_neighbors = self.find_nearest_neighbors(discrete_prompts.cpu().numpy(), embedding_matrix.cpu().numpy())
            #     for idx in low_similarity_tokens:
            #         discrete_prompts[idx] = torch.tensor(
            #             self.smooth_token_adjustment(discrete_prompts[idx].cpu().numpy(),
            #                                     embedding_matrix[nearest_neighbors[idx][0]].cpu().numpy(),
            #                                     batch_center)
            #         ).to(device)

            # 3. 記錄投影嵌入**
            current_projection_embeddings.append(discrete_prompts.clone().detach().cpu())

            # 4. 替換為離散嵌入 P' 進行前向傳播
            self.soft_prompts.data = discrete_prompts.data
            
            # 5. 計算損失
            self.optimizer.zero_grad()
            outputs = self.forward_with_prompt(input_ids, labels=labels, attention_mask=batch['attention_mask'], epoch=epoch, global_semantic_center=global_semantic_center, device=device)
            loss = outputs.loss

            # 6. 恢復原始連續嵌入 P
            self.soft_prompts.data = soft_prompts_backup

            # 7. 基於損失更新連續嵌入 P
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        # average_embedding = self.analyze_embedding_distribution(current_projection_embeddings)
        # soft_prompt_after_projection_history.append(torch.tensor(average_embedding))
        # chosen_index = self.get_top1_nearest_text(average_embedding, embedding_matrix, raw_texts)
        # top1_neighbors_texts = self.get_topk_neighbors_for_top1(chosen_index, embedding_matrix, raw_texts, k=5)
        # self.save_epoch_neighbors(top1_neighbors_texts)
              
        # # **🔹 印出 Top-1 最近鄰的 Top-5 近鄰文本**
        # print("\n🔹 當前 Epoch 的 Top-1 最近鄰的 Top-5 近鄰文本:")
        # for idx, text in enumerate(top1_neighbors_texts, 1):
        #     print(f"  {idx}. {text}")
        
        # print(f"Finalized with Loss: {total_loss / len(train_loader):.4f}")

        
    def save_epoch_neighbors(self, neighbors, file_path="epoch_texts/all_epochs_neighbors.json"):
        """
        儲存所有 Epoch 產生的 Top-5 近鄰文本，並自動累積新 Epoch 的結果。
        如果目錄不存在，則自動創建。

        Parameters:
            neighbors (list): 當前 Epoch 產生的 Top-5 鄰近文本 (list of strings)。
            file_path (str): 存儲鄰近文本的 JSON 檔案路徑。
        """
        # 確保目錄存在，否則創建
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)  # 自動創建目錄

        # 如果檔案已存在，則讀取現有內容
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []  # 如果讀取失敗，則初始化為空列表
        else:
            data = []  # 如果檔案不存在，則初始化為空列表

        # 追加當前 Epoch 的鄰近文本
        data.append({"top5_neighbors": neighbors})

        # 存回 JSON 檔案
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"✅ 已存儲 {len(data)} 個 Epoch 的鄰近文本至 {file_path}")

    def log_prompt_shift(self, epoch, p_before, p_after, path="epoch_prompt_shift.csv"):
        p_before = p_before.view(-1)
        p_after = p_after.view(-1)
        cos_sim = F.cosine_similarity(p_before.unsqueeze(0), p_after.unsqueeze(0), dim=1)
        cos_dist = 1 - cos_sim.item()

        file_exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["epoch", "cosine_distance_between_tuning_and_projected"])
            writer.writerow([epoch, cos_dist])
        print(f"[Epoch {epoch}] Cosine Distance (P_tuning vs P_projected): {cos_dist:.4f}")


    def get_top1_nearest_text(self, average_embedding, embedding_matrix, raw_texts):
        """
        取得 Soft Prompt 當前最接近的文本（Top-1 最近鄰）。
        
        Parameters:
            average_embedding (numpy.ndarray): Soft Prompt 的平均投影嵌入
            embedding_matrix (numpy.ndarray): 訓練數據嵌入矩陣
            raw_texts (list): 訓練數據的文本列表
            
        Returns:
            str: 最接近的文本
        """
        # 確保 `embedding_matrix` 轉換為 NumPy
        if isinstance(embedding_matrix, torch.Tensor):
            embedding_matrix = embedding_matrix.cpu().numpy()  # 移動到 CPU，轉換為 numpy
        
        if isinstance(average_embedding, torch.Tensor):
            average_embedding = average_embedding.cpu().numpy().reshape(1, -1)  # 確保為 NumPy 陣列
            
        similarities = sk_cosine_similarity(average_embedding, embedding_matrix)[0]
        top1_index = np.argmax(similarities)  # 取最接近的索引
        top1_similarity = similarities[top1_index]
        
        print(f"🔹 Top-1 最近鄰相似度: {top1_similarity:.4f}")
        return top1_index

    def get_topk_neighbors_for_top1(self, top1_index, embedding_matrix, raw_texts, k=5):
        """
        取得 Top-1 最近鄰的 Top-K 近鄰文本。
        
        Parameters:
            top1_index (int): Top-1 最近鄰的索引
            embedding_matrix (numpy.ndarray): 訓練數據的嵌入矩陣
            raw_texts (list): 訓練數據的文本列表
            k (int): 取前 K 個最近鄰
            
        Returns:
            list: K 個最接近的文本
        """
        if isinstance(embedding_matrix, torch.Tensor):
            embedding_matrix = embedding_matrix.cpu().numpy()  # 移動到 CPU，轉換為 numpy
            
        top1_embedding = embedding_matrix[top1_index].reshape(1, -1)
        similarities = sk_cosine_similarity(top1_embedding, embedding_matrix)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]  # 取最相似的 K 個索引
        return [raw_texts[i] for i in top_k_indices]

    def compute_pairwise_similarity(self, projection_embeddings):
        """
        計算 batch 內所有 token 之間的相似度。

        Args:
            projection_embeddings (numpy.ndarray): 當前 batch 內的投影嵌入 [n_tokens, embedding_dim]

        Returns:
            numpy.ndarray: [n_tokens, n_tokens] 形狀的 pairwise similarity matrix
        """
        return cosine_similarity(projection_embeddings)

    def should_replace_token(self, pairwise_similarities, min_threshold=0.5, max_threshold=0.9):
        """
        根據 token 之間的相似度，決定是否需要替換 token。
        
        Args:
            pairwise_similarities (numpy.ndarray): [n_tokens, n_tokens] 形狀的相似度矩陣
            min_threshold (float): 允許的最低相似度，低於此值的 token 可能需要調整
            max_threshold (float): 允許的最高相似度，超過此值的 token 不需要調整
        
        Returns:
            list: 需要進行替換的 token 索引
        """
        mean_similarities = np.mean(pairwise_similarities, axis=1)  # 計算每個 token 的平均相似度
        replace_tokens = np.where(mean_similarities < min_threshold)[0]  # 找出過於分散的 token
        return replace_tokens

    def compute_batch_center(self, projection_embeddings):
        """
        計算 batch 內所有 token 的語義中心點（平均嵌入）。
        
        Args:
            projection_embeddings (numpy.ndarray): 當前 batch 內的投影嵌入 [n_tokens, embedding_dim]

        Returns:
            numpy.ndarray: 語義中心點 [1, embedding_dim]
        """
        return np.mean(projection_embeddings, axis=0)

    def find_nearest_neighbors(self, projection_embeddings, embedding_space, k=5):
        """
        對 batch 內每個 token 找最近鄰的嵌入向量。

        Args:
            projection_embeddings (numpy.ndarray): 當前 batch 內的投影嵌入，形狀為 [n_tokens, embedding_dim]
            embedding_space (numpy.ndarray): 訓練樣本構成的嵌入空間 [n_samples, embedding_dim]
            k (int): 近鄰數量

        Returns:
            list: 每個 token 在嵌入空間找到的最近鄰索引
        """
        neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
        neigh.fit(embedding_space)  # 訓練樣本作為基準
        
        distances, indices = neigh.kneighbors(projection_embeddings)
        
        return indices  # 回傳每個 token 在嵌入空間的 k 近鄰索引

    def smooth_token_adjustment(self, projection_embedding, nearest_embedding, batch_center, alpha=0.3):
        """
        讓 token 向最近鄰詞嵌入靠近一點，而不是完全變成最近鄰，以保留語義多樣性。

        Args:
            projection_embedding (numpy.ndarray): 原 token 的嵌入 [1, embedding_dim]
            nearest_embedding (numpy.ndarray): 最近鄰詞嵌入 [1, embedding_dim]
            batch_center (numpy.ndarray): batch 內 token 的語義中心 [1, embedding_dim]
            alpha (float): 控制 token 變動程度 (0~1)，值越大代表 token 變動越大

        Returns:
            numpy.ndarray: 平滑調整後的 token 嵌入
        """
        return alpha * nearest_embedding + (1 - alpha) * batch_center

    def find_tokens_far_from_center(self, projection_embeddings, batch_center, threshold=0.5):
        """
        找出 batch 內距離語義中心過遠的 token。

        Args:
            projection_embeddings (numpy.ndarray): 當前 batch 內的投影嵌入 [n_tokens, embedding_dim]
            batch_center (numpy.ndarray): batch 內 token 的語義中心 [1, embedding_dim]
            threshold (float): 設定的距離門檻

        Returns:
            list: 需要重新替換的 token 索引
        """
        distances = np.linalg.norm(projection_embeddings - batch_center, axis=1)  # 計算每個 token 到語義中心的距離
        far_tokens = np.where(distances > threshold)[0]  # 找出距離過遠的 token
        return far_tokens

    def analyze_embedding_distribution(self, embeddings):
        """
        Analyze the distribution of embeddings to calculate:
        1. The average embedding.
        2. A single metric to determine whether the embeddings are sufficiently concentrated.

        Parameters:
            embeddings (numpy.ndarray): A 2D array of shape (n_samples, embedding_dim),
                                        where each row is an embedding vector.

        Returns:
            dict: A dictionary containing:
                - 'average_embedding': The mean embedding vector.
                - 'mean_distance': The average distance from each embedding to the average embedding.
        """
        # Ensure embeddings is a NumPy array
        embeddings = np.array(embeddings)
        
        # Calculate the average embedding
        average_embedding = np.mean(embeddings, axis=0)
    
        # Calculate distances from each embedding to the average embedding
        distances = np.linalg.norm(embeddings - average_embedding, axis=1)

        # Calculate mean distance
        mean_distance = np.mean(distances)
        # print("mean distance:", mean_distance)

        # Return results as a dictionary
        return average_embedding

    def get_nearest_neighbors(self, average_embedding, embedding_matrix, raw_texts, k=5):
        """
        根據當前 epoch 的平均投影嵌入，找到其在語義空間中的最近鄰樣本。
        
        Parameters:
            average_embedding (numpy.ndarray): 當前 epoch 計算出的平均投影嵌入，形狀為 (embedding_dim,).
            embedding_matrix (numpy.ndarray): 預先構建的嵌入空間矩陣，形狀為 (n_samples, embedding_dim)。
            raw_texts (list): 與 embedding_matrix 對應的原始文本列表，長度為 n_samples。
            k (int): 需要檢索的最近鄰數量。

        Returns:
            list: 包含 k 個最接近的原始文本，代表 soft prompt 當前的語義對應文本。
        """
        # 確保 `embedding_matrix` 轉換為 NumPy
        if isinstance(embedding_matrix, torch.Tensor):
            embedding_matrix = embedding_matrix.cpu().numpy()  # 移動到 CPU，轉換為 numpy
        
        if isinstance(average_embedding, torch.Tensor):
            average_embedding = average_embedding.cpu().numpy().reshape(1, -1)  # 確保為 NumPy 陣列

        # 計算 cosine 相似度
        similarities = sk_cosine_similarity(average_embedding, embedding_matrix)[0]

        # 取最相似的 K 個索引
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        # 獲取對應的文本內容
        top_k_texts = [raw_texts[i] for i in top_k_indices]
        top_k_similarities = similarities[top_k_indices]

        # 顯示結果
        print("\n🔹 當前 epoch 平均投影嵌入的最近鄰文本：")
        for i, (text, sim) in enumerate(zip(top_k_texts, top_k_similarities)):
            print(f"Top {i+1}: (Similarity: {sim:.4f}) {text}")

        return top_k_texts
    
    def compute_global_semantic_center(self, embedding_space: torch.Tensor) -> torch.Tensor:
        """
        計算整個訓練集的語義中心（全域語義平均嵌入）。

        Args:
            embedding_space (torch.Tensor): shape (N, D)，包含整個訓練集的句子級嵌入向量。

        Returns:
            torch.Tensor: 全域語義中心，shape (D,)
        """
        return embedding_space.mean(dim=0)  # 對所有句子平均