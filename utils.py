import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd

from data_utils import *


def bpr_loss(user_embs, item_embs, edge_index_t_batch, num_users, regular = True, lambda_reg = 0.01):
    
    pos_scores = (user_embs[edge_index_t_batch[0, :]] * 
                  item_embs[edge_index_t_batch[1, :] - num_users]).sum(dim=1)
    
    neg_scores = (user_embs[edge_index_t_batch[0, :]] * 
                  item_embs[edge_index_t_batch[2, :] - num_users]).sum(dim=1)
    
    if regular == True:
        reg = (user_embs.norm(2).pow(2) + item_embs.norm(2).pow(2)) / user_embs.size(0)
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores)) + lambda_reg * reg
    else:
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    

def multi_negative_bpr_loss(user_embs, item_embs, edge_index_batch, num_users, 
                            regular = True, temperature = 0.2, lambda_reg = 0.01):
    """
    edge_index_batch: (2 + K, B)
        [user, pos_item, neg_item1, neg_item2, ..., neg_itemK]
    """
    user_idx = edge_index_batch[0]                   # (B,)
    pos_item_idx = edge_index_batch[1] - num_users   # (B,)
    neg_item_idx = edge_index_batch[2:] - num_users  # (K, B)

    u_emb = user_embs[user_idx]                      # (B, D)
    pos_emb = item_embs[pos_item_idx]                # (B, D)
    neg_emb = item_embs[neg_item_idx]                # (K, B, D)

    # Positive score: (B,)
    pos_score = torch.sum(u_emb * pos_emb, dim=1)

    # Negative scores: (K, B)
    neg_score = torch.sum(neg_emb * u_emb.unsqueeze(0), dim=2)

    # 모든 score 모아서 softmax (positive first)
    all_scores = torch.cat([pos_score.unsqueeze(0), neg_score], dim=0)  # (1+K, B)
    log_prob = F.log_softmax(all_scores / temperature, dim=0)  # temperature=0.2

    if regular: 
        reg = (user_embs.norm(2).pow(2) + item_embs.norm(2).pow(2)) / user_embs.size(0)
        loss = -log_prob[0].mean() + lambda_reg * reg
    else:
        # positive는 항상 첫 줄이므로 log_prob[0]
        loss = -log_prob[0].mean()

    return loss


def sample_mini_batch(edge_index, batch_size, num_users, num_items, K):

    n_edges = edge_index.shape[1]
    device = edge_index.device
    replace_flag = batch_size > n_edges

    if not replace_flag:
        batch_index = torch.randperm(n_edges, device=device)[:batch_size]
    else:
        batch_index = torch.randint(0, n_edges, (batch_size,), device=device)

    # batch_index = np.random.choice(range(edge_index.shape[1]), size=batch_size, replace=False)
    
    pos_edge_index = edge_index[:, batch_index]

    if K == 1:
        edge_index_pn = negative_sampling(pos_edge_index, num_users, num_items)

        return edge_index_pn
    
    elif K > 1:
        edge_index_mpn = multi_negative_sampling(pos_edge_index, num_users, num_items, K)

        return edge_index_mpn


def negative_sampling(edge_index, num_users, num_items):
    user_indices = edge_index[0]
    item_indices = edge_index[1]
    L = edge_index.size(1)

    # 각 사용자마다 랜덤하게 negative 아이템 샘플링
    negative_samples = torch.randint(num_users, num_users + num_items, (L,), device=edge_index.device)

    # 긍정 샘플과 동일한 경우 다시 샘플링 (충돌 방지)
    mask = (negative_samples == item_indices)
    while mask.any():
        negative_samples[mask] = torch.randint(num_users, num_users + num_items, (mask.sum(),), device=edge_index.device)
        mask = (negative_samples == item_indices)

    # 원래 edge_index에 negative 샘플 추가 (3 x L 형태)
    edge_index_with_neg = torch.cat([edge_index, negative_samples.unsqueeze(0)], dim=0)
    # print('neg edge', edge_index_with_neg)
          
    return edge_index_with_neg


def multi_negative_sampling(edge_index, num_users, num_items, K):
    user_indices = edge_index[0]  # shape: (B,)
    pos_items = edge_index[1]
    B = edge_index.shape[1]

    # K개 negative item 샘플링 (shape: K x B)
    neg_items = torch.randint(
        num_users, num_users + num_items, (K, B), device=edge_index.device
    )

    # 충돌 방지: negative가 positive와 같지 않도록
    for k in range(K):
        mask = (neg_items[k] == pos_items)
        while mask.any():
            neg_items[k, mask] = torch.randint(
                num_users, num_users + num_items, (mask.sum(),), device=edge_index.device
            )
            mask = (neg_items[k] == pos_items)

    # 최종: shape (2 + K, B)
    edge_index_with_neg = torch.cat(
        [user_indices.unsqueeze(0), pos_items.unsqueeze(0), neg_items], dim=0
    )

    return edge_index_with_neg


def contrastive_loss(model, edge_index_series, device, tau=0.5):

    # 마지막 시점 그래프에서 두 개의 subgraph 생성
    edge_index_aug1, edge_index_aug2 = generate_sub_edge_list(edge_index_series, device)

    # 모델을 사용해 두 개의 augmentation된 view에서 임베딩 추출
    user_embs1, item_embs1 = model(edge_index_aug1)
    user_embs2, item_embs2 = model(edge_index_aug2)

    # Contrastive Loss 계산
    loss_users = info_nce_loss(user_embs1, user_embs2, tau)
    loss_items = info_nce_loss(item_embs1, item_embs2, tau)

    return loss_users + loss_items


def mini_batch_contrastive_loss_num(model, edge_index_series, device, 
                                 tau=0.5, num_sampled_users=512, num_sampled_items=512):
    # 두 개의 augmentation 생성
    edge_index_aug1, edge_index_aug2 = generate_sub_edge_list(edge_index_series, device)

    # 전체 embedding 계산
    user_embs1, item_embs1 = model(edge_index_aug1)
    user_embs2, item_embs2 = model(edge_index_aug2)

    # 랜덤 샘플링
    sampled_user_idx = torch.randperm(user_embs1.shape[0])[:num_sampled_users].to(device)
    sampled_item_idx = torch.randperm(item_embs1.shape[0])[:num_sampled_items].to(device)

    # 샘플링된 임베딩만 선택
    z1_users = user_embs1[sampled_user_idx]
    z2_users = user_embs2[sampled_user_idx]

    z1_items = item_embs1[sampled_item_idx]
    z2_items = item_embs2[sampled_item_idx]

    # InfoNCE loss 계산
    loss_users = info_nce_loss(z1_users, z2_users, tau)
    loss_items = info_nce_loss(z1_items, z2_items, tau)

    return loss_users + loss_items


def mini_batch_contrastive_loss_ratio(model, edge_index_series, device, 
                                 tau=0.5, user_sample_ratio=0.7, item_sample_ratio=0.7):
    # 두 개의 augmentation 생성
    edge_index_aug1, edge_index_aug2 = generate_sub_edge_list(edge_index_series, device)

    # 전체 embedding 계산
    user_embs1, item_embs1 = model(edge_index_aug1)
    user_embs2, item_embs2 = model(edge_index_aug2)

    # 샘플링 개수 계산s
    num_users = user_embs1.shape[0]
    num_items = item_embs1.shape[0]

    num_sampled_users = max(1, int(user_sample_ratio * num_users))
    num_sampled_items = max(1, int(item_sample_ratio * num_items))

    # 랜덤 샘플링 인덱스
    sampled_user_idx = torch.randperm(num_users)[:num_sampled_users].to(device)
    sampled_item_idx = torch.randperm(num_items)[:num_sampled_items].to(device)

    # 샘플링된 임베딩 선택
    z1_users = user_embs1[sampled_user_idx]
    z2_users = user_embs2[sampled_user_idx]
    z1_items = item_embs1[sampled_item_idx]
    z2_items = item_embs2[sampled_item_idx]

    # InfoNCE loss 계산
    loss_users = info_nce_loss(z1_users, z2_users, tau)
    loss_items = info_nce_loss(z1_items, z2_items, tau)

    return loss_users + loss_items



def info_nce_loss(z1, z2, tau):
    """
    z1: 첫 번째 view의 노드 임베딩
    z2: 두 번째 view의 노드 임베딩
    tau: Temperature hyperparameter
    """
    
    # L2 정규화 (Cosine Similarity 계산을 위해)
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)

    # Cosine Similarity Matrix
    sim_matrix = torch.mm(z1, z2.T) / tau
    exp_sim = torch.exp(sim_matrix)

    # Positive Pair Similarity
    pos_sim = torch.diag(exp_sim)

    # Contrastive Loss 계산
    contrastive_loss = -torch.mean(torch.log(pos_sim / exp_sim.sum(dim=1)))

    return contrastive_loss


def save_model(epoch, model, optimizer, loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    
    
def recall_at_k(edge_list, user_embeddings, item_embeddings, K, device):
    # edge_list: 2 x N 행렬 (유저-아이템 페어)
    # user_embeddings: (num_users, embedding_dim) 크기의 유저 임베딩 텐서
    # item_embeddings: (num_items, embedding_dim) 크기의 아이템 임베딩 텐서
    
    # edge_list에서 유저와 아이템을 추출
    edge_list = edge_list.to(device)
    user_embeddings = user_embeddings.to(device)
    item_embeddings = item_embeddings.to(device)
    
    users = edge_list[0]
    items = edge_list[1]
    
    num_users = user_embeddings.shape[0]
    
    # 모든 유저와 아이템 임베딩 간의 내적(dot product) 계산
    scores = torch.matmul(user_embeddings, item_embeddings.T)  # (num_users, num_items)
    
    # 각 유저별로 상위 K개의 아이템을 가져옴 (점수 기준으로 정렬)
    top_k_items = torch.topk(scores, K, dim=1)[1] + num_users  # 각 유저에 대해 상위 K개의 아이템 인덱스
    
    # Recall 값을 누적할 변수 초기화
    total_recall = torch.tensor(0.0, device=device)
    
    # 각 유저에 대해 반복
    for user_idx in range(num_users):
        # 현재 유저에 대한 실제 아이템을 가져옴 (edge_list에서)
        relevant_items = items[users == user_idx].unique()  # 유저의 실제 연결된 아이템을 고유하게 추출
        
        if len(relevant_items) == 0:
            continue  # 만약 유저가 실제로 연결된 아이템이 없다면 건너뜀
        
        # 현재 유저에 대해 추천된 상위 K개의 아이템
        predicted_items = top_k_items[user_idx]
        
        # 추천된 아이템 중 실제로 연결된 아이템의 개수를 계산
        correct_predictions = torch.sum(torch.isin(predicted_items, relevant_items)).float()
        
        # 현재 유저의 Recall 계산 후 누적
        recall = correct_predictions / min(len(relevant_items), K)
        total_recall += recall
    
    # 모든 유저에 대한 평균 Recall 계산
    avg_recall = total_recall / num_users
    
    return avg_recall.item()


def ndcg_at_k(edge_list, user_embeddings, item_embeddings, K, device):
    # edge_list: 2 x N 행렬 (유저-아이템 페어)
    # user_embeddings: (num_users, embedding_dim) 크기의 유저 임베딩 텐서
    # item_embeddings: (num_items, embedding_dim) 크기의 아이템 임베딩 텐서
    
    # 유저와 아이템을 edge_list에서 추출
    edge_list = edge_list.to(device)
    user_embeddings = user_embeddings.to(device)
    item_embeddings = item_embeddings.to(device)    
    
    users = edge_list[0]
    items = edge_list[1]
    
    num_users = user_embeddings.shape[0]
    
    # 모든 유저와 아이템 임베딩 간의 내적(dot product)을 계산하여 유저별로 모든 아이템에 대한 점수를 얻음
    scores = torch.matmul(user_embeddings, item_embeddings.T)  # (num_users, num_items)
    
    # 각 유저별로 상위 K개의 아이템을 추출 (점수 기준으로 정렬)
    top_k_items = torch.topk(scores, K, dim=1)[1] + num_users  # (num_users, K) 상위 K개 아이템의 인덱스
    
    # NDCG 값을 누적할 변수 초기화
    total_ndcg = torch.tensor(0.0, device=device)
    
    # 각 유저에 대해 반복
    for user_idx in range(num_users):
        # 현재 유저에 대한 실제 연결된 아이템을 edge_list에서 추출
        relevant_items = items[users == user_idx].unique()  # 실제로 연결된 아이템들
        
        if len(relevant_items) == 0:
            continue  # 유저에게 실제로 연결된 아이템이 없으면 건너뜀
        
        # 현재 유저에 대해 추천된 상위 K개의 아이템
        predicted_items = top_k_items[user_idx]
        
        # DCG 계산 (추천된 아이템 중 실제로 연결된 아이템이 있을 경우)
        dcg = torch.tensor(0.0, device=device)
        for rank, item_idx in enumerate(predicted_items):
            # 추천된 아이템이 실제로 연결된 아이템이면 분자는 1, 아니면 0
            relevance = torch.tensor(1.0, device=device) if item_idx in relevant_items else torch.tensor(0.0, device=device)
            dcg += relevance / torch.log2(torch.tensor(rank + 2.0, device=device))  # 랭크는 0부터 시작하므로 log2(rank+2)
        
        # IDCG (이론적으로 가능한 최대 DCG) 계산
        idcg = torch.tensor(0.0, device=device)
        ideal_relevant_count = min(len(relevant_items), K)  # 실제 연결된 아이템 중 K개 또는 그 이하가 이상적
        for rank in range(ideal_relevant_count):
            idcg += 1.0 / torch.log2(torch.tensor(rank + 2.0, device=device))  # 랭크는 0부터 시작
        
        # NDCG 계산 (DCG / IDCG)
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = torch.tensor(0.0, device=device)
        
        # NDCG 값 누적
        total_ndcg += ndcg
    
    # 평균 NDCG 계산
    avg_ndcg = total_ndcg / num_users
    
    return avg_ndcg


def tensor_to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


def tensor_to_float(x, digits=4):
    return round(x.item(), digits) if torch.is_tensor(x) else round(x, digits)


