import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from torch.nn.parameter import Parameter
from utils import *
from datetime import datetime
from tqdm import tqdm
import math
import os

class ELGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, num_layers, device, 
                 agg_mode = 'mean', first = 'GRU', EMA = False, ema_alpha = 0.7):
        super().__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.device = device
        self.agg_mode = agg_mode
        self.first = first
        self.EMA = EMA
        self.ema_alpha = ema_alpha

        # 유저와 아이템 임베딩 레이어 (Learnable Embedding)
        self.user_embs_init = nn.Embedding(self.num_users, emb_dim)
        self.item_embs_init = nn.Embedding(self.num_items, emb_dim)
        
        nn.init.xavier_uniform_(self.user_embs_init.weight) 
        nn.init.xavier_uniform_(self.item_embs_init.weight)

        # GRU와 LightGCN 레이어 정의
        self.evolve_embs = mat_GRU_cell(emb_dim, num_users, num_items).to(device)
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)]).to(device)
        
        # Temporal Attention Layer 추가
        self.temporal_attention = TemporalAttention(emb_dim).to(device)

    def forward(self, edge_index_series, 
                epoch=None, 
                k=5,
                warmup_epochs=5):
        
        embs_init = torch.cat([self.user_embs_init.weight, 
                               self.item_embs_init.weight],
                              dim=0).to(self.device)
        
        if self.first == 'GRU':
        
            embs_init = embs_init.T
            embs_init_seq = [embs_init.T]
            for _ in range(len(edge_index_series) - 1):
                embs_init = self.evolve_embs(embs_init)
                embs_init_seq.append(embs_init.T)

            embs_t_seq = []
            for time, edge_index in enumerate(edge_index_series):
                embs = embs_init_seq[time].to(self.device)
                
                if self.EMA is False:
                    embs_sum = embs.clone()
                    count = 1

                    for conv in self.convs:
                        embs = conv(x=embs, edge_index=edge_index)
                        embs_sum += embs
                        count += 1

                    embs_t = embs_sum / count
                    
                else:
                    alpha = self.ema_alpha
                    for conv in self.convs:
                        out = conv(x=embs, edge_index=edge_index)
                        embs = alpha * out + (1 - alpha) * embs
                    embs_t = embs
                    
                embs_t_seq.append(embs_t)

            embs_t_stack = torch.stack(embs_t_seq, dim=0)
            
        elif self.first == 'GCN':
            
            h = embs_init
            h_seq = []  # 각 시점의 임베딩 저장

            # 시간 순서대로 GCN → GRU 반복
            for t in range(len(edge_index_series)):
                edge_index_t = edge_index_series[t].to(self.device)

                x = h  # 이전 시점의 hidden state

                if self.EMA is False:
                    x_sum = x.clone()
                    count = 1

                    for conv in self.convs:
                        x = conv(x, edge_index_t)  # (N, D)
                        x_sum += x
                        count+=1
                    x = x_sum / count
                    
                else:
                    alpha = self.ema_alpha
                    for conv in self.convs:
                        out = conv(x, edge_index_t)
                        x=  alpha * out + (1-alpha)*x

                # GRU로 업데이트 (GRU는 입력 (D, N)을 받기 때문에 transpose)
                h = self.evolve_embs(x.T).T  # (D, N) → (D, N) → (N, D)
                h_seq.append(h)  # 시점별 결과 저장

            # 3. 시점별 임베딩 stack (T, N, D)
            embs_t_stack = torch.stack(h_seq, dim=0)
            
        
        if self.agg_mode == 'mean' or (self.agg_mode == 'attention' and epoch is not None and epoch < warmup_epochs):
            final_embs = torch.mean(embs_t_stack, dim=0)
        elif self.agg_mode == 'attention':
            attn_input = embs_t_stack  # [-k:]
            final_embs = self.temporal_attention(attn_input)
        else:
            raise ValueError("Invalid agg_mode. Choose from ['mean', 'attention']")
        
        final_user_embs = final_embs[:self.num_users]
        final_item_embs = final_embs[self.num_users:]

        return final_user_embs, final_item_embs
    
    def train_model(self, edge_index_series, edge_index_series_2, optimizer, 
                    sample_mini_batch, bpr_loss, multi_negative_bpr_loss, one_batch_sample, sgl, cont_loss, ssl_lambda,
                    num_users, num_items, batch_size, time_steps, 
                    tr_first_time, tr_last_time, val_first_time, val_last_time, 
                    epochs, patience, improvement_threshold, recent_loss,
                    model_ver, multi_sampling,
                    data_name, model_name):

        best_epoch = 0
        best_val_loss = float('inf')
        no_improve_count = 0
        best_model_state = None
        tr_total_loss = []
        val_total_loss = []

        for epoch in range(epochs):
            tr_epoch_start_time = datetime.now()
            self.train()
            
            tr_time_range = range(
                    tr_first_time,
                    tr_last_time - time_steps + 1 if not one_batch_sample else tr_last_time
                )
            
            print(f"\n[Epoch {epoch + 1}] Training...")
            
            if sgl:
                
                time_loss = 0.0
                for t in tqdm(tr_time_range, desc=f"Train"):
                    
                    current_window = edge_index_series_2[t-time_steps:t]
                    current_window = [cw.to(self.device) for cw in current_window]                            
                    
                    if one_batch_sample:
                        edge_index_out = edge_index_series[t].to(self.device)            
                    else:
                        edge_index_window = edge_index_series[t:t+time_steps]
                        edge_index_out = torch.cat([ei.to(self.device) for ei in edge_index_window], dim=1)
                        
                    n_batch = math.ceil(edge_index_out.shape[1] / batch_size) 
                    
                    total_bpr = 0.0
                    for _ in range(n_batch):
                        optimizer.zero_grad()
                        
                        user_embs, item_embs = self(current_window, epoch=epoch)
                        
                        if cont_loss == 'vanilla':
                            ssl_loss = contrastive_loss(self, current_window, self.device)
                        elif cont_loss == 'num_batch':
                            ssl_loss = mini_batch_contrastive_loss_num(self, current_window, self.device)
                        elif cont_loss == 'ratio_batch':
                            ssl_loss = mini_batch_contrastive_loss_ratio(self, current_window, self.device)  
                        
                        if multi_sampling:
                            edge_index_t_batch = sample_mini_batch(edge_index_out, batch_size, num_users, num_items, K=5).to(self.device)
                            batch_bpr = multi_negative_bpr_loss(user_embs, item_embs, edge_index_t_batch, num_users)
                        else:
                            edge_index_t_batch = sample_mini_batch(edge_index_out, batch_size, num_users, num_items, K=1).to(self.device)
                            batch_bpr = bpr_loss(user_embs, item_embs, edge_index_t_batch, num_users)
                        
                        loss = batch_bpr + ssl_lambda * ssl_loss
                        loss.backward()
                        optimizer.step()
                        
                        total_bpr += batch_bpr.item()
                        
                    avg_bpr = total_bpr / n_batch
                    
                    time_loss += (avg_bpr + ssl_lambda * ssl_loss.item())                    
                
            else:
                
                time_loss = 0.0
                for t in tqdm(tr_time_range, desc=f"Train"):
                    
                    current_window = edge_index_series_2[t-time_steps:t]
                    current_window = [cw.to(self.device) for cw in current_window]                            
                    
                    if one_batch_sample:
                        edge_index_out = edge_index_series[t].to(self.device)            
                    else:
                        edge_index_window = edge_index_series[t:t+time_steps]
                        edge_index_out = torch.cat([ei.to(self.device) for ei in edge_index_window], dim=1)
                        
                    n_batch = math.ceil(edge_index_out.shape[1] / batch_size)
                    
                    total_bpr = 0.0
                    for _ in range(n_batch):
                        optimizer.zero_grad()
                            
                        user_embs, item_embs = self(current_window, epoch=epoch)
                            
                        if multi_sampling:
                            edge_index_t_batch = sample_mini_batch(edge_index_out, batch_size, num_users, num_items, K=5).to(self.device)
                            batch_bpr = multi_negative_bpr_loss(user_embs, item_embs, edge_index_t_batch, num_users)
                        else:
                            edge_index_t_batch = sample_mini_batch(edge_index_out, batch_size, num_users, num_items, K=1).to(self.device)
                            batch_bpr = bpr_loss(user_embs, item_embs, edge_index_t_batch, num_users)
                            
                        batch_bpr.backward()            
                        optimizer.step()                         
                        total_bpr += batch_bpr.item()     

                    time_loss += total_bpr / n_batch   
                # total_loss = time_loss

            epoch_loss = time_loss / len(tr_time_range)
                
            tr_total_loss.append(epoch_loss)
            
            tr_epoch_end_time = datetime.now()
            tr_epoch_duration = tr_epoch_end_time - tr_epoch_start_time
            
            print(f"Train Loss: {epoch_loss:.4f} | complete at {tr_epoch_end_time}, Duration: {tr_epoch_duration}")

            # Validation
            self.eval()
            with torch.no_grad():
                val_epoch_start_time = datetime.now()
                print(f"[Epoch {epoch + 1}] Validation...")
                
                val_time_range = range(
                    val_first_time,
                    val_last_time - time_steps + 1 if not one_batch_sample else val_last_time
                )

                if sgl:
                    time_loss = 0.0
                    for t in tqdm(val_time_range, desc=f"Valid"):
                        
                        current_window = edge_index_series_2[t-time_steps:t]
                        current_window = [cw.to(self.device) for cw in current_window]
                        
                        if one_batch_sample:
                            edge_index_out = edge_index_series[t].to(self.device)
                        else:
                            edge_index_window = edge_index_series[t:t+time_steps]
                            edge_index_out = torch.cat([ei.to(self.device) for ei in edge_index_window], dim=1)
                        
                        n_batch = math.ceil(edge_index_out.shape[1] / batch_size)
                            
                        total_bpr = 0.0
                        for _ in range(n_batch):
                            
                            user_embs, item_embs = self(current_window)
                            
                            if cont_loss == 'vanilla':
                                ssl_loss = contrastive_loss(self, current_window, self.device)
                            elif cont_loss == 'num_batch':
                                ssl_loss = mini_batch_contrastive_loss_num(self, current_window, self.device)
                            elif cont_loss == 'ratio_batch':
                                ssl_loss = mini_batch_contrastive_loss_ratio(self, current_window, self.device)   
                                
                            if multi_sampling == True:
                                edge_index_t_batch = sample_mini_batch(edge_index_out, batch_size, num_users, num_items, K=5).to(self.device)
                                batch_bpr = multi_negative_bpr_loss(user_embs, item_embs, edge_index_t_batch, num_users)   
                            else:
                                edge_index_t_batch = sample_mini_batch(edge_index_out, batch_size, num_users, num_items, K=1).to(self.device)
                                batch_bpr = bpr_loss(user_embs, item_embs, edge_index_t_batch, num_users) 
                                
                            total_bpr += batch_bpr
                        avg_bpr = total_bpr / n_batch
                        time_loss += (float(avg_bpr) + ssl_lambda * float(ssl_loss))
                    # total_loss = time_loss
                    
                else:
                    time_loss = 0.0
                    for t in tqdm(val_time_range, desc=f"Valid"):
                        current_window = edge_index_series_2[t-time_steps:t]
                        current_window = [cw.to(self.device) for cw in current_window]
                        user_embs, item_embs = self(current_window)
                        
                        if one_batch_sample:
                            edge_index_out = edge_index_series[t].to(self.device)
                        else:
                            edge_index_window = edge_index_series[t:t+time_steps]
                            edge_index_out = torch.cat([ei.to(self.device) for ei in edge_index_window], dim=1)
                        
                        n_batch = math.ceil(edge_index_out.shape[1] / batch_size)
                        
                        total_bpr = 0.0
                        for _ in range(n_batch):
                            
                            if multi_sampling == True:
                                edge_index_t_batch = sample_mini_batch(edge_index_out, batch_size, num_users, num_items, K=5).to(self.device)
                                batch_bpr = multi_negative_bpr_loss(user_embs, item_embs, edge_index_t_batch, num_users)   
                            else:
                                edge_index_t_batch = sample_mini_batch(edge_index_out, batch_size, num_users, num_items, K=1).to(self.device)
                                batch_bpr = bpr_loss(user_embs, item_embs, edge_index_t_batch, num_users)   
                                
                            total_bpr += float(batch_bpr)
                        time_loss += total_bpr / n_batch
                    # total_loss = time_loss.item()

                avg_val_loss = time_loss / len(val_time_range)
                val_total_loss.append(avg_val_loss)
                
                val_epoch_end_time = datetime.now()
                val_epoch_duration = val_epoch_end_time - val_epoch_start_time
                total_epoch_duration = val_epoch_end_time - tr_epoch_start_time
                
                avg_val_loss_value = avg_val_loss.item() if torch.is_tensor(avg_val_loss) else avg_val_loss
                best_val_loss_value = best_val_loss.item() if torch.is_tensor(best_val_loss) else best_val_loss

                recent_losses = val_total_loss[-recent_loss:] if len(val_total_loss) >= recent_loss else val_total_loss
                avg_recent_losses = np.round(np.mean(recent_losses), 5)
                
                if avg_val_loss_value < avg_recent_losses - improvement_threshold:
                # if avg_val_loss_value < avg_recent_losses:
                    best_val_loss_value = avg_val_loss_value
                    best_epoch = epoch + 1
                    no_improve_count = 0
                    best_model_state = {
                        'epoch': best_epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss_value,
                    }
                else:
                    no_improve_count += 1
                    
                print(f"Validation Loss: {avg_val_loss:.4f} ({avg_recent_losses}) ({no_improve_count}) | complete at {val_epoch_end_time}, Duration: {val_epoch_duration}")

                if no_improve_count >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                
            print(f"Epoch {epoch + 1} total duration: {total_epoch_duration}")
            print()

        # Save results
        tr_total_loss_cpu = [tensor_to_float(loss) for loss in tr_total_loss]
        val_total_loss_cpu = [tensor_to_float(loss) for loss in val_total_loss]
        loss_df = pd.DataFrame({
            'Epoch': list(range(1, len(tr_total_loss_cpu) + 1)),
            'Train Loss': tr_total_loss_cpu,
            'Validation Loss': val_total_loss_cpu
        })
        
        save_result_dir = os.path.join(f"{model_name}_save_result", data_name)
        best_model_dir = os.path.join(f"{model_name}_best_model", data_name)      
        os.makedirs(save_result_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)
        
        loss_path = os.path.join(save_result_dir, f'{model_ver}_tr_val_loss.csv')
        loss_df.to_csv(loss_path, index=False)

        if best_model_state:
            best_model_path = os.path.join(best_model_dir, f'{model_ver}_ep{best_epoch}.pth')
            # model_save_path = f'{best_model_dir}/{data_name}_{model_name}_{model_ver}_ep{best_epoch}.pth' 
            torch.save(best_model_state, best_model_path)
            print(f"Best model saved at epoch {best_epoch}")
        
        return best_model_state, best_model_path
    

    def test_model(self, model_path, edge_index_series_2, edge_index_series,
                   recall_at_k, ndcg_at_k, time_steps, ts_first_time, ts_last_time, device, 
                   model_ver, model_name, data_name):

        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()

        recall_total = []
        ndcg_total = []
        k_range = [10, 20, 30, 50]

        with torch.no_grad():
            for k in k_range:
                recall_sum = 0.0
                ndcg_sum = 0.0
                for t in tqdm(range(ts_first_time, ts_last_time), desc=f"K={k}"): 
                    current_window = edge_index_series_2[t-time_steps:t]
                    current_window = [cw.to(device) for cw in current_window]
                    
                    current_edges = torch.cat(current_window, dim=1)
                    
                    user_embs, item_embs = self(current_window)
                    
                    recall = recall_at_k(current_edges, user_embs, item_embs, K=k, device=device)
                    ndcg = ndcg_at_k(current_edges, user_embs, item_embs, K=k, device=device)
                    
                    recall_sum += recall
                    ndcg_sum += ndcg

                average_recall = recall_sum / (ts_last_time - ts_first_time)
                average_ndcg = ndcg_sum / (ts_last_time - ts_first_time)
                recall_total.append(average_recall)
                ndcg_total.append(average_ndcg)

        recall_total_cpu = [tensor_to_float(r) for r in recall_total]
        ndcg_total_cpu = [tensor_to_float(n) for n in ndcg_total]
        results_df = pd.DataFrame({
            'K': k_range,
            'Recall': recall_total_cpu,
            'NDCG': ndcg_total_cpu
        })
        
        save_result_dir = os.path.join(f"{model_name}_save_result", data_name)
        prfm_path = os.path.join(save_result_dir, f'{model_ver}_recall_ndcg.csv')
        results_df.to_csv(prfm_path, index=False)
        
        return results_df
    
    
class mat_GRU_cell(nn.Module):
    def __init__(self, emb_dim, num_users, num_items):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_users = num_users
        self.num_items = num_items        
        
        self.update = mat_GRU_gate(emb_dim, num_users, num_items, nn.Sigmoid())
        self.reset = mat_GRU_gate(emb_dim, num_users, num_items, nn.Sigmoid())
        self.htilda = mat_GRU_gate(emb_dim, num_users, num_items, nn.Tanh())

    def forward(self, prev_hidden):
        x = prev_hidden
        
        update = self.update(x, prev_hidden)
        reset = self.reset(x, prev_hidden)
        
        h_cap = reset * prev_hidden
        h_cap = self.htilda(x, h_cap)
        
        new_hidden = (1 - update) * prev_hidden + update * h_cap

        return new_hidden
        
        
class mat_GRU_gate(nn.Module):
    def __init__(self, rows, num_users, num_items, activation):
        super().__init__()
        self.activation = activation
        self.num_users = num_users
        self.num_items = num_items

        self.W = Parameter(torch.Tensor(rows, rows))
        self.U = Parameter(torch.Tensor(rows, rows))
        self.user_b = Parameter(torch.zeros(rows, 1))
        self.item_b = Parameter(torch.zeros(rows, 1))
        
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.xavier_uniform_(self.U)
        torch.nn.init.xavier_uniform_(self.user_b)
        torch.nn.init.xavier_uniform_(self.item_b)

    def forward(self, x, hidden):
        user_part = self.activation(self.W.matmul(x[:, :self.num_users]) + 
                                    self.U.matmul(hidden[:, :self.num_users]) + 
                                    self.user_b.repeat(1, self.num_users))
        
        item_part = self.activation(self.W.matmul(x[:, self.num_users:]) + 
                                    self.U.matmul(hidden[:, self.num_users:]) + 
                                    self.item_b.repeat(1, self.num_items))

        return torch.cat([user_part, item_part], dim=1)
    
    
class TemporalAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=4):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.d_k = emb_dim // num_heads
        
        self.W_q = nn.Linear(emb_dim, emb_dim, bias=True)
        self.W_k = nn.Linear(emb_dim, emb_dim, bias=True)
        self.W_v = nn.Linear(emb_dim, emb_dim, bias=True)
        
        self.out_proj = nn.Linear(emb_dim, emb_dim)  # optional: restore dim
        self.residual_proj = nn.Linear(emb_dim, emb_dim)
    
    def forward(self, embs_t_seq, temperature = 2.0):   # (T, N, D)
        
        T, N, D = embs_t_seq.shape
        
        Q = self.W_q(embs_t_seq[-1])  # (N, D)
        K = self.W_k(embs_t_seq)      # (T, N, D)
        V = self.W_v(embs_t_seq)      # (T, N, D)
        
        # Reshape for multi-head: (N, H, d_k)
        Q = Q.view(N, self.num_heads, self.d_k).transpose(0, 1)          # (H, N, d_k)
        K = K.view(T, N, self.num_heads, self.d_k).permute(2, 1, 0, 3)   # (H, N, T, d_k)
        V = V.view(T, N, self.num_heads, self.d_k).permute(2, 1, 0, 3)   # (H, N, T, d_k)
        
        # Compute attention scores: (H, N, T)
        attn_scores = torch.einsum('hnd,hntd->hnt', Q, K) / (self.d_k ** 0.5)
        attn_scores = attn_scores / temperature
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention weights to values: (H, N, T, d_k) × (H, N, T) → (H, N, d_k)
        attn_weights = attn_weights.unsqueeze(-1)    
        weighted_V = (attn_weights * V).sum(dim=2)  # sum over time T → (H, N, d_k)
        
        # Concat heads: (N, H, d_k) → (N, D)
        concat = weighted_V.transpose(0, 1).reshape(N, D)   # (N, d_k)

        final_embs = self.out_proj(concat)   
        # Residual + LayerNorm
        residual = self.residual_proj(embs_t_seq[-1])  # (N, D)
        final_embs = F.layer_norm(final_embs + residual, final_embs.shape)
        
        return final_embs
