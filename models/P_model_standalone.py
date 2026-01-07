import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import get_performance, get_loss_fn
from models.WaveletRGATEncoder import WaveletRGATEncoder
from transformers import AutoModel, AutoTokenizer, BertModel
import torch.nn.functional as F
from models.HyperbolicKGPred import HyperbolicKGPred
from models.neighbor_semantic_sim import compute_neighbor_semantic_similarity
from typing import Dict, Tuple, Optional, Iterable
from tqdm import tqdm


def _safe_to(x, device):
    """tensordevice，None"""
    if x is None:
        return None
    if not isinstance(x, torch.Tensor):
        return x
    return x if x.device == device else x.to(device)


def batch_neg_ce_loss(
        Z_ent: torch.Tensor,  # [N, d]
        Z_rel: torch.Tensor,  # [R, d]
        ent_rel: torch.LongTensor,
        labels: torch.LongTensor,
        *,
        hr_vector: torch.Tensor,
        tail_vector: torch.Tensor,
        neg_tail_vector: torch.Tensor,
        neg_ids: torch.LongTensor,
        model: nn.Module,
        temperature: float = 1.1,
        device: Optional[torch.device] = None,
        score_margin_weight: float = 0.0,
        score_margin: float = 0.0,
) -> torch.Tensor:
    """
    " s  r  t*"。
    -  batch "/"， ent_rel[:,1] （ r+n_rel），。
    - gt_tail: ， (s,r) （）
    """
    if device is None:
        device = Z_ent.device

    if neg_tail_vector is None or neg_ids is None:
        raise ValueError(" (neg_tail_vector, neg_ids)")

    ent_rel = ent_rel.to(device)
    labels = labels.to(device)
    neg_tail_vector = _safe_to(neg_tail_vector, device)
    neg_ids = neg_ids.to(device)

    B = ent_rel.size(0)
    s = ent_rel[:, 0].to(device)  # [B]
    r = ent_rel[:, 1].to(device)  # [B]
    t_pos = labels.to(device)  # [B]


    h = Z_ent[s]  # [B, d]
    rel = Z_rel[r]  # [B, d]
    t_true = Z_ent[t_pos]  # [B, d]


    h_norm = F.normalize(h, p=2, dim=-1)  # [B, d]
    rel_norm = F.normalize(rel, p=2, dim=-1)  # [B, d]
    t_true_norm = F.normalize(t_true, p=2, dim=-1)  # [B, d]

    sim_h_t = F.cosine_similarity(h_norm, t_true_norm, dim=-1)
    sim_hr_t = F.cosine_similarity(F.normalize(h * rel, p=2, dim=-1), t_true_norm, dim=-1)  # [B] - Hadamard
    sim_add_t = F.cosine_similarity(F.normalize(h + rel, p=2, dim=-1), t_true_norm, dim=-1)

    
    pos_sim_2 = 0.05 * sim_hr_t + 0.65 * sim_add_t + 0.3 * sim_h_t  # [B]

    out_pos = model(h, rel, t_true, pos_sim_2)
    score_pos = out_pos["score"]


    B_neg, K, H = neg_tail_vector.shape  #
    assert B_neg == B, f"batch size ({B_neg})  ({B}) "
    d = Z_ent.size(1)

    t_neg_flat = Z_ent[neg_ids].reshape(B * K, d)  # [B*K, d]
    t_neg_flat_norm = F.normalize(t_neg_flat, p=2, dim=-1)  # [B*K, d]

    h_norm_exp = h_norm.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)  # [B*K, d]
    rel_norm_exp = rel_norm.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)  # [B*K, d]
    h_exp = h.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)  # [B*K, d]
    rel_exp = rel.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)  # [B*K, d]

    neg_sim_h_t = F.cosine_similarity(h_norm_exp, t_neg_flat_norm, dim=-1)  # [B*K]
    neg_sim_hr_t = F.cosine_similarity(F.normalize(h_exp * rel_exp, p=2, dim=-1), t_neg_flat_norm, dim=-1)  # [B*K]
    neg_sim_add_t = F.cosine_similarity(F.normalize(h_exp + rel_exp, p=2, dim=-1), t_neg_flat_norm, dim=-1)  # [B*K]


    neg_sim_flat_raw = 0.05 * neg_sim_hr_t + 0.65 * neg_sim_add_t + 0.3 * neg_sim_h_t  # [B*K]
    neg_sim_flat = neg_sim_flat_raw.unsqueeze(-1)  # [B*K, 1]

    h_rep_struct = h.unsqueeze(1).expand(B, K, d).reshape(B * K, d)  # [B*K,d]
    rel_rep_struct = rel.unsqueeze(1).expand(B, K, d).reshape(B * K, d)  # [B*K,d]

    out_neg = model(h_rep_struct, rel_rep_struct, t_neg_flat, neg_sim_flat)
    score_neg = out_neg["score"].view(B, K)

    logits = torch.cat([score_pos.view(-1, 1), score_neg], dim=1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    L_pred = F.cross_entropy(logits, labels)

    contrastive_temperature = 0.6  #0.4-0.6

    hr_query = F.normalize(h + rel, p=2, dim=-1)  # [B, d]
    pos_contrastive_sim = F.cosine_similarity(hr_query, t_true_norm, dim=-1)  # [B]
    pos_contrastive_logit = pos_contrastive_sim / contrastive_temperature

    t_neg = t_neg_flat.reshape(B, K, d)  # [B, K, d]
    t_neg_norm = F.normalize(t_neg, p=2, dim=-1)
    neg_contrastive_sim = F.cosine_similarity(
        hr_query.unsqueeze(1),  # [B, 1, d]
        t_neg_norm,  # [B, K, d]
        dim=-1
    )  # [B, K]
    neg_contrastive_logit = neg_contrastive_sim / contrastive_temperature

    # InfoNCE loss
    contrastive_logits = torch.cat([
        pos_contrastive_logit.unsqueeze(1),  # [B, 1]
        neg_contrastive_logit  # [B, K]
    ], dim=1)  # [B, 1+K]
    contrastive_labels = torch.zeros(B, dtype=torch.long, device=device)
    L_contrastive = F.cross_entropy(contrastive_logits, contrastive_labels)

    contrastive_weight = 0.10  #

    sim_inbatch = torch.matmul(hr_query, t_true_norm.t())
    inbatch_logits = sim_inbatch / contrastive_temperature
    inbatch_labels = torch.arange(B, device=device)
    L_contrastive_inbatch = F.cross_entropy(inbatch_logits, inbatch_labels)

    L_contrastive_total = 0.5 * (L_contrastive + L_contrastive_inbatch)

    L_struct = L_pred + contrastive_weight * L_contrastive_total

    loss = L_struct
    if score_margin_weight > 0.0:
        score_neg_batched = score_neg.view(B, K)
        max_neg_score, _ = score_neg_batched.max(dim=1)
        score_gap = score_pos - max_neg_score
        L_score_margin = F.relu(score_margin - score_gap).mean()
        loss = loss + score_margin_weight * L_score_margin

    return loss


def _to_int_scalar(x) -> int:
    """ x  Python int。 int/np /01torch.Tensor/listtuple。"""
    if isinstance(x, (int,)):
        return int(x)
    if isinstance(x, float):
        return int(x)
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("center_ent contains empty list/tuple.")
        return _to_int_scalar(x[0])
    if torch.is_tensor(x):
        if x.dim() == 0:
            return int(x.item())
        if x.numel() == 1:
            return int(x.view(-1)[0].item())
        raise TypeError(f"Tensor with more than 1 element: shape={tuple(x.shape)}")
    try:
        return int(x)
    except Exception as e:
        raise TypeError(f"Cannot cast {type(x)} -> int, value={x}") from e


def _row_to_int_list(row) -> List[int]:
    """ ids  List[int]。row  list/tuple  1D torch.Tensor。"""
    if torch.is_tensor(row):
        if row.dim() == 0:
            return [int(row.item())]
        if row.dim() == 1:
            return [int(v.item()) for v in row]
        raise TypeError(f"neighbor row tensor must be 0D/1D, got shape={tuple(row.shape)}")
    # list/tuple
    return [_to_int_scalar(v) for v in row]


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        masked = last_hidden_state.masked_fill(~input_mask_expanded, -1e4)
        output_vector = masked.max(dim=1).values
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector


class KGCPromptTuner(nn.Module):
    def __init__(self, configs, text_dict, gt):
        super().__init__()
        self.configs = configs
        self.ent_names = text_dict['ent_names']
        self.rel_names = text_dict['rel_names']
        self.ent_descs = text_dict['ent_descs']
        self.rho_hier_rel_ids = []
        self.rho_sim_rel_ids = []
        for idx, name in enumerate(self.rel_names):
            n = name.strip().lower()
            if n == 'hypernym' or n == 'instance hypernym':
                self.rho_hier_rel_ids.append(idx)
            if n == 'similar to':
                self.rho_sim_rel_ids.append(idx)
        self.text_neighbors_all_tail_gt = text_dict['text_neighbors_all_tail_gt']
        self.text_neighbors_all_head_gt = text_dict['text_neighbors_all_head_gt']
        self.all_tail_gt = gt['all_tail_gt']
        self.all_head_gt = gt['all_head_gt']
        self.neighbors_all_tail_gt = gt['neighbors_all_tail_gt']
        self.neighbors_all_head_gt = gt['neighbors_all_head_gt']
        self.neighbors_train_tail_gt = gt['neighbors_train_tail_gt']
        self.neighbors_train_head_gt = gt['neighbors_train_head_gt']

        self.train_edge_index_all = gt['train_edge_index_all']
        self.train_edge_type_all = gt['train_edge_type_all']
        self.train_edge_weight_all = gt['train_edge_weight_all']
        self.train_E_tail = gt['train_E_tail']
        self.train_edge_map_all = gt['train_edge_map_all']
        self.train_L_rescaled_all = gt['train_L_rescaled_all']
        self.max_neig = configs.max_neigh
        self.ent_embed = nn.Embedding(self.configs.n_ent, self.configs.embed_dim)

        self.hr_bert = AutoModel.from_pretrained(configs.pretrained_model)
        self.hr_bert.tokenizer = AutoTokenizer.from_pretrained(configs.pretrained_model)

        self.ent_text_embed = None
        self.text_to_struct = nn.Linear(configs.model_dim, configs.embed_dim)
        self.struct_to_text = nn.Linear(configs.embed_dim, configs.model_dim)

        self.text_q_proj = nn.Linear(configs.model_dim, configs.model_dim)
        self.text_k_proj = nn.Linear(configs.model_dim, configs.model_dim)

        self.history = {'perf': ..., 'loss': []}
        self._MASKING_VALUE = -1e4 if self.configs.use_fp16 else -1e9
        n_rel_total = configs.n_rel * (2 if configs.add_inverse_rel else 1)


        print("Initializing WaveletRGATEncoder...")
        self.waveletRGATEncoder = WaveletRGATEncoder(
            in_dim=configs.WaGATembed_dim,
            hidden_dim=configs.wavelet_hidden_dim,
            out_dim=configs.struct_out_dim,
            n_rel_total=n_rel_total,
            rel_dim=configs.rel_dim,
            cheb_K=configs.cheb_K,
            gat_heads=configs.gat_heads,
            dropout=configs.gnn_dropout,
            concat_heads=configs.concat_heads,
            beta_init=configs.beta_init,
        )
        print(f"WaveletRGATEncoder initialized: in_dim={configs.WaGATembed_dim}, out_dim={configs.struct_out_dim}")

        self.HyperbolicKGPred = HyperbolicKGPred(
            hind=configs.embed_dim,
            d=configs.Dir_s_dim,
            P=64
        )

        self._val_ranks = {}
        self._test_ranks = {}

        self.cached_all_ent_vectors = None

        self.cached_hr_sim = {}
        self.cached_hr_sim_test = {}

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output('cls', cls_output, mask, last_hidden_state)
        return cls_output

    def save_final_embeddings(self, Z_ent, Z_rel):
        """"""
        self.final_Z_ent = Z_ent.detach()
        self.final_Z_rel = Z_rel.detach()
        print(f"Final embeddings saved: Z_ent shape: {Z_ent.shape}, Z_rel shape: {Z_rel.shape}")

    def precompute_all_entity_vectors(self):
        """（/）"""
        if self.cached_all_ent_vectors is not None:
            return self.cached_all_ent_vectors

        print("Precomputing entity text vectors...")
        device = next(self.parameters()).device
        N_ent = self.configs.n_ent
        all_ent_vectors = []
        ent_batch_size = 512

        self.hr_bert.eval()
        with torch.no_grad():
            for start_idx in range(0, N_ent, ent_batch_size):
                end_idx = min(start_idx + ent_batch_size, N_ent)
                batch_ent_ids = list(range(start_idx, end_idx))

                batch_texts = []
                for eid in batch_ent_ids:
                    ent_name = self.ent_names[eid].strip() if eid < len(self.ent_names) else ""
                    ent_desc = self.ent_descs[eid].strip() if eid < len(self.ent_descs) else ""
                    text = f"{ent_name} : {ent_desc}" if ent_name and ent_desc else (ent_name or ent_desc)
                    batch_texts.append(text)

                encoded = self.hr_bert.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.configs.text_len,
                    return_tensors='pt',
                    return_overflowing_tokens=False
                )

                for key in encoded:
                    encoded[key] = encoded[key].to(device)

                batch_ent_vecs = self._encode(
                    self.hr_bert,
                    encoded['input_ids'],
                    encoded['attention_mask'],
                    encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids']))
                )  # [batch_size, H]
                all_ent_vectors.append(batch_ent_vecs.cpu())

        self.cached_all_ent_vectors = torch.cat(all_ent_vectors, dim=0)  # [N_ent, H]
        print(f"Entity text vectors computed: shape={self.cached_all_ent_vectors.shape}")
        return self.cached_all_ent_vectors

    def get_fused_entity_embeddings(self, Z_ent: torch.Tensor) -> torch.Tensor:
        device = Z_ent.device
        if getattr(self, 'ent_text_embed', None) is None or self.ent_text_embed.device != device:
            all_ent_vectors = self.precompute_all_entity_vectors()
            self.ent_text_embed = all_ent_vectors.to(device)
        alpha = float(getattr(self.configs, 'text_struct_mix_alpha', 0.0))
        if alpha <= 0.0:
            return Z_ent
        text_vec = self.ent_text_embed
        text_proj = self.text_to_struct(text_vec)
        return Z_ent + alpha * text_proj

    def precompute_hr_sim_matrix(self, dataloader, is_test=False):
        """
        (h,r)

        Args:
            dataloader: dataloader（list of dataloaders）
            is_test: 
        """
        cache_name = 'cached_hr_sim_test' if is_test else 'cached_hr_sim'
        cache_dict = getattr(self, cache_name)

        if len(cache_dict) > 0:
            mode = "Test" if is_test else "Validation"
            print(f"{mode} hr-t similarity already cached, skipping")
            return

        mode = "test" if is_test else "validation"
        print(f"Precomputing {mode} hr-t similarity matrix...")
        device = next(self.parameters()).device

        all_ent_vectors = self.precompute_all_entity_vectors()  # [N_ent, H]
        all_ent_vectors = all_ent_vectors.to(device)
        all_ent_vectors_norm = F.normalize(all_ent_vectors, p=2, dim=-1)  # [N_ent, H]

        unique_hr_pairs = set()

        dataloaders = dataloader if isinstance(dataloader, list) else [dataloader]

        for dl in dataloaders:
            for batch_data in dl:
                ent_rel = batch_data['ent_rel']  # [B, 2]
                for i in range(ent_rel.size(0)):
                    h = int(ent_rel[i, 0].item())
                    r = int(ent_rel[i, 1].item())
                    unique_hr_pairs.add((h, r))

        print(f"Total {len(unique_hr_pairs)} unique (h,r) pairs to compute")

        hr_pairs_list = list(unique_hr_pairs)
        hr_batch_size = 128

        self.hr_bert.eval()
        with torch.no_grad():
            for start_idx in range(0, len(hr_pairs_list), hr_batch_size):
                end_idx = min(start_idx + hr_batch_size, len(hr_pairs_list))
                batch_hr_pairs = hr_pairs_list[start_idx:end_idx]

                batch_hr_texts_1 = []
                batch_hr_texts_2 = []

                for h, r in batch_hr_pairs:
                    ent_name = self.ent_names[h].strip() if h < len(self.ent_names) else ""
                    ent_desc = self.ent_descs[h].strip() if h < len(self.ent_descs) else ""

                    if ent_name and ent_desc:
                        text1 = f"{ent_name} : {ent_desc}"
                    else:
                        text1 = ent_name or ent_desc or ""

                    if r >= self.configs.n_rel:
                        actual_r = r - self.configs.n_rel
                        rel_name = self.rel_names[actual_r].strip() if actual_r < len(self.rel_names) else ""
                        text2 = f"reversed: {rel_name}"
                    else:
                        rel_name = self.rel_names[r].strip() if r < len(self.rel_names) else ""
                        text2 = rel_name

                    batch_hr_texts_1.append(text1)
                    batch_hr_texts_2.append(text2)

                encoded = self.hr_bert.tokenizer(
                    batch_hr_texts_1,
                    text_pair=batch_hr_texts_2,
                    padding=True,
                    truncation=True,
                    max_length=self.configs.text_len,
                    return_tensors='pt',
                    return_overflowing_tokens=False
                )

                for key in encoded:
                    encoded[key] = encoded[key].to(device)

                batch_hr_vecs = self._encode(
                    self.hr_bert,
                    encoded['input_ids'],
                    encoded['attention_mask'],
                    encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids']))
                )  # [batch_size, H]

                batch_hr_vecs_norm = F.normalize(batch_hr_vecs, p=2, dim=-1)  # [batch_size, H]

                sim_matrix = torch.matmul(batch_hr_vecs_norm, all_ent_vectors_norm.T)  # [batch_size, N_ent]

                for i, (h, r) in enumerate(batch_hr_pairs):
                    cache_dict[(h, r)] = sim_matrix[i].cpu()

                if (start_idx // hr_batch_size + 1) % 10 == 0:
                    print(f"  Processed {end_idx}/{len(hr_pairs_list)} (h,r) pairs")

        print(f"hr-t similarity precomputation done! Cached {len(cache_dict)} (h,r) pairs")

        setattr(self, cache_name, cache_dict)

    def compute_global_sim1_edge(self, max_neighbors_per_entity):
        """"""
        device = next(self.parameters()).device
        
        return compute_neighbor_semantic_similarity(
            bert_model=self.hr_bert,
            neighbors_train_tail_gt=self.neighbors_train_tail_gt,
            neighbors_train_head_gt=self.neighbors_train_head_gt,
            ent_names=self.ent_names,
            ent_descs=self.ent_descs,
            train_edge_index=self.train_edge_index_all,
            train_edge_map=self.train_edge_map_all,
            max_neighbors_per_entity=max_neighbors_per_entity,
            device=device
        )

    def precompute_graph_embeddings(self):

        device = next(self.parameters()).device
        edge_index = self.train_edge_index_all.to(device)
        edge_type = self.train_edge_type_all.to(device)
        edge_weight = self.train_edge_weight_all.to(device) if self.train_edge_weight_all is not None else None
        L_rescaled = self.train_L_rescaled_all.to(device) if self.train_L_rescaled_all is not None else None

        print("Computing GAT semantic prior sim1_edge...")
        sim1_edge = self.compute_global_sim1_edge(max_neighbors_per_entity=self.max_neig)
        print(f"sim1_edge computed, non-zero edges: {(sim1_edge > 0).sum().item()}/{len(sim1_edge)}")

        return edge_index, edge_type, edge_weight, L_rescaled, sim1_edge


    def forward(self, src_ids, src_mask, src_token_type_ids, tgt_ent_ids, tgt_ent_mask, tgt_ent_token_type_ids,
                ):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=src_ids,
                                 mask=src_mask,
                                 token_type_ids=src_token_type_ids)

        tail_vector = self._encode(self.hr_bert,
                                   token_ids=tgt_ent_ids,
                                   mask=tgt_ent_mask,
                                   token_type_ids=tgt_ent_token_type_ids)

        return hr_vector, tail_vector

    def training_step(self, batched_data, Z_ent=None, Z_rel=None):
        """

        :
            batched_data: 
            Z_ent: （1self.ent_embed.weight，2+RGAT）
            Z_rel: （1self.rel_embed.weight，2+RGAT）
        """
        # if Z_ent is None:
        #     Z_ent = self.ent_embed.weight
        # if Z_rel is None:
        #     Z_rel = self.rel_embed.weight

        assert Z_ent is not None and Z_rel is not None, "+RGATZ_entZ_rel"

        Z_ent = self.get_fused_entity_embeddings(Z_ent)

        neg_entity_ids = batched_data['neg_entity_ids']

        ent_rel = batched_data['ent_rel']
        labels = batched_data['labels']


        device = Z_ent.device
        B_train = ent_rel.size(0)
        K_train = neg_entity_ids.size(1) if neg_entity_ids is not None else int(self.configs.neg_K)
        N_ent = Z_ent.size(0)

        h_idx = ent_rel[:, 0]  # [B]
        r_idx = ent_rel[:, 1]  # [B]
        h_emb = Z_ent[h_idx]  # [B, d]
        rel_emb = Z_rel[r_idx]  # [B, d]
        tail_idx = labels  # [B]

        use_hard_negative = not getattr(self.configs, 'disable_hard_negative', False)

        if not hasattr(self, '_neg_sampling_logged'):
            strategy = "self-adversarial" if use_hard_negative else "random"
            print(f"Negative sampling strategy: {strategy} (K={K_train})")
            self._neg_sampling_logged = True

        if use_hard_negative:
            with torch.no_grad():
                h_norm = F.normalize(h_emb, p=2, dim=-1)  # [B, d]
                rel_norm = F.normalize(rel_emb, p=2, dim=-1)  # [B, d]
                all_ent_norm = F.normalize(Z_ent, p=2, dim=-1)  # [N_ent, d]

                sim_h_all = torch.mm(h_norm, all_ent_norm.t())  # [B, N_ent]
                sim_hr_all = torch.mm(F.normalize(h_emb * rel_emb, p=2, dim=-1), all_ent_norm.t())
                sim_add_all = torch.mm(F.normalize(h_emb + rel_emb, p=2, dim=-1), all_ent_norm.t())

                sim_all = 0.05 * sim_hr_all + 0.65 * sim_add_all + 0.3 * sim_h_all  # [B, N_ent]

                hyp_output_all = self.HyperbolicKGPred(h_emb, rel_emb, Z_ent, sim_all)
                all_scores = hyp_output_all['score']  # [B, N_ent]

                all_scores_masked = all_scores.clone()
                all_scores_masked[torch.arange(B_train, device=device), tail_idx] = -1e9

                alpha = getattr(self.configs, 'self_adversarial_temperature', 0.5)
                neg_probs = F.softmax(alpha * all_scores_masked, dim=-1)  # [B, N_ent]

                try:
                    neg_entity_ids = torch.multinomial(neg_probs, K_train, replacement=False)  # [B, K]
                except RuntimeError:
                    print(f"Warning: Self-adversarial sampling failed (K={K_train} may be too large), falling back to random sampling")
                    neg_entity_ids = torch.randint(0, N_ent, (B_train, K_train), device=device)
                    for i in range(B_train):
                        mask = (neg_entity_ids[i] == tail_idx[i])
                        if mask.any():
                            num_conflicts = mask.sum().item()
                            for attempt in range(10):
                                new_indices = torch.randint(0, N_ent, (num_conflicts,), device=device)
                                if tail_idx[i] not in new_indices:
                                    neg_entity_ids[i][mask] = new_indices
                                    break
        else:
            neg_entity_ids = torch.randint(0, N_ent, (B_train, K_train), device=device)
            for i in range(B_train):
                mask = (neg_entity_ids[i] == tail_idx[i])
                if mask.any():
                    num_conflicts = mask.sum().item()
                    for attempt in range(10):
                        new_indices = torch.randint(0, N_ent, (num_conflicts,), device=device)
                        if tail_idx[i] not in new_indices:
                            neg_entity_ids[i][mask] = new_indices
                            break


        B_train = ent_rel.size(0)
        K_train = neg_entity_ids.size(1) if neg_entity_ids is not None else 0
        H_hidden = self.configs.bert_dim if hasattr(self.configs, 'bert_dim') else 768

        hr_vector = torch.zeros(B_train, H_hidden, device=device)
        tail_vector = torch.zeros(B_train, H_hidden, device=device)
        neg_vec = torch.zeros(B_train, K_train, H_hidden, device=device) if K_train > 0 else None

        temperature = getattr(self.configs, 'loss_temperature', 1.3)
        score_margin_weight = float(getattr(self.configs, 'score_margin_weight', 0.0))
        score_margin = float(getattr(self.configs, 'score_margin', 0.0))
        base_loss = batch_neg_ce_loss(
            Z_ent=Z_ent,
            Z_rel=Z_rel,
            ent_rel=ent_rel,
            labels=labels,
            hr_vector=hr_vector,
            tail_vector=tail_vector,
            neg_tail_vector=neg_vec,
            neg_ids=neg_entity_ids,
            model=self.HyperbolicKGPred,
            temperature=temperature,
            device=Z_ent.device,
            score_margin_weight=score_margin_weight,
            score_margin=score_margin,
        )

        align_weight = getattr(self.configs, 'text_align_weight', 0.0)
        if align_weight > 0.0:
            align_loss = self.compute_entity_text_align_loss(Z_ent, ent_rel, labels)
            loss = base_loss + align_weight * align_loss
        else:
            loss = base_loss

        text_sim_weight = float(getattr(self.configs, 'text_sim_weight', 0.0))
        if text_sim_weight > 0.0:
            text_sim_loss = self.compute_text_simkgc_loss(ent_rel, labels)
            loss = loss + text_sim_weight * text_sim_loss

        return loss

    def compute_entity_text_align_loss(self, Z_ent, ent_rel, labels):
        device = Z_ent.device
        if getattr(self, 'ent_text_embed', None) is None or self.ent_text_embed.device != device:
            all_ent_vectors = self.precompute_all_entity_vectors()
            self.ent_text_embed = all_ent_vectors.to(device)
        ent_text = self.ent_text_embed
        h_idx = ent_rel[:, 0]
        t_idx = labels
        batch_ent_ids = torch.cat([h_idx, t_idx]).unique()
        struct_vec = Z_ent[batch_ent_ids]
        text_vec = ent_text[batch_ent_ids]
        struct_proj = self.struct_to_text(struct_vec)
        struct_proj = F.normalize(struct_proj, p=2, dim=-1)
        text_norm = F.normalize(text_vec, p=2, dim=-1)
        loss = 1.0 - (struct_proj * text_norm).sum(dim=-1).mean()
        return loss

    def compute_text_simkgc_loss(self, ent_rel, labels):
        """ SimKGC  InfoNCE （ in-batch negative）。"""
        device = next(self.parameters()).device

        if getattr(self, 'ent_text_embed', None) is None or self.ent_text_embed.device != device:
            all_ent_vectors = self.precompute_all_entity_vectors()
            self.ent_text_embed = all_ent_vectors.to(device)

        h_idx = ent_rel[:, 0].to(device)
        r_idx = ent_rel[:, 1].to(device)
        t_idx = labels.to(device)

        B = h_idx.size(0)
        if B <= 1:
            return torch.zeros([], device=device)

        hr_texts_1 = []
        hr_texts_2 = []
        for i in range(B):
            h = int(h_idx[i].item())
            r = int(r_idx[i].item())

            ent_name = self.ent_names[h].strip() if h < len(self.ent_names) else ""
            ent_desc = self.ent_descs[h].strip() if h < len(self.ent_descs) else ""
            if ent_name and ent_desc:
                text1 = f"{ent_name} : {ent_desc}"
            else:
                text1 = ent_name or ent_desc or ""

            if r >= self.configs.n_rel:
                actual_r = r - self.configs.n_rel
                rel_name = self.rel_names[actual_r].strip() if actual_r < len(self.rel_names) else ""
                text2 = f"reversed: {rel_name}"
            else:
                rel_name = self.rel_names[r].strip() if r < len(self.rel_names) else ""
                text2 = rel_name

            hr_texts_1.append(text1)
            hr_texts_2.append(text2)

        encoded = self.hr_bert.tokenizer(
            hr_texts_1,
            text_pair=hr_texts_2,
            padding=True,
            truncation=True,
            max_length=self.configs.text_len,
            return_tensors='pt',
            return_overflowing_tokens=False,
        )

        for key in encoded:
            encoded[key] = encoded[key].to(device)

        hr_vec = self._encode(
            self.hr_bert,
            encoded['input_ids'],
            encoded['attention_mask'],
            encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids'])),
        )  # [B, H]

        q = self.text_q_proj(hr_vec)  # [B, D]
        k = self.text_k_proj(self.ent_text_embed[t_idx])  # [B, D]

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        sim = torch.matmul(q, k.t())  # [B, B]
        temperature = float(getattr(self.configs, 'contrastive_temperature', 0.4))
        logits = sim / temperature
        labels_ce = torch.arange(B, device=device)
        loss = F.cross_entropy(logits, labels_ce)
        return loss

    def validation_step(self, batched_data, Z_ent=None, Z_rel=None, dataloader_idx=0):
        """ - hr-t

        :
            batched_data: 
            Z_ent: （1self.ent_embed.weight，2+RGAT）
            Z_rel: （1self.rel_embed.weight，2+RGAT）
            dataloader_idx: （0=，1=）
        """
        assert Z_ent is not None and Z_rel is not None, "+RGATZ_entZ_rel"

        Z_ent = self.get_fused_entity_embeddings(Z_ent)

        val_triples = batched_data['triple']
        ent_rel = batched_data['ent_rel']  # [B, 2]
        src_ent, rel = ent_rel[:, 0], ent_rel[:, 1]
        tgt_ent = batched_data['tgt_ent']

        h_embed = Z_ent[src_ent]  # [B, d]
        rel_embed = Z_rel[rel]  # [B, d]

        h_norm = F.normalize(h_embed, p=2, dim=-1)  # [B, d]
        all_struct_emb = Z_ent  # [N_ent, d]
        all_struct_emb_norm = F.normalize(all_struct_emb, p=2, dim=-1)  # [N_ent, d]

        sim_h_all = F.cosine_similarity(
            h_norm.unsqueeze(1),  # [B, 1, d]
            all_struct_emb_norm.unsqueeze(0),  # [1, N_ent, d]
            dim=-1  # [B, N_ent]
        )

        hr_hadamard_norm = F.normalize(h_embed * rel_embed, p=2, dim=-1)  # [B, d]
        sim_hr_all = F.cosine_similarity(
            hr_hadamard_norm.unsqueeze(1),  # [B, 1, d]
            all_struct_emb_norm.unsqueeze(0),  # [1, N_ent, d]
            dim=-1  # [B, N_ent]
        )

        hr_add_norm = F.normalize(h_embed + rel_embed, p=2, dim=-1)  # [B, d]
        sim_add_all = F.cosine_similarity(
            hr_add_norm.unsqueeze(1),  # [B, 1, d]
            all_struct_emb_norm.unsqueeze(0),  # [1, N_ent, d]
            dim=-1  # [B, N_ent]
        )

        sim_matrix = 0.05* sim_hr_all + 0.65 * sim_add_all + 0.3 * sim_h_all  # [B, N_ent]

        out_pos = self.HyperbolicKGPred(h_embed, rel_embed, Z_ent, sim_matrix)
        struct_logits = out_pos["score"].detach()  # [B, N_ent]

        sem_score_weight = float(getattr(self.configs, 'sem_score_weight', 0.0))
        if sem_score_weight > 0.0:
            gate_a = self.HyperbolicKGPred.gate_rho_a
            gate_b = self.HyperbolicKGPred.gate_rho_b
            g_mat = torch.sigmoid(gate_a * sim_matrix + gate_b)
            sem_logits = sem_score_weight * sim_matrix
            struct_logits = (1.0 - g_mat) * struct_logits + g_mat * sem_logits

        text_score_weight = float(getattr(self.configs, 'text_score_weight', 12))
        if text_score_weight > 0.0:
            device = struct_logits.device

            if getattr(self, 'ent_text_embed', None) is None or self.ent_text_embed.device != device:
                all_ent_vectors = self.precompute_all_entity_vectors()  # [N_ent, H]
                self.ent_text_embed = all_ent_vectors.to(device)

            h_idx = src_ent
            r_idx = rel
            B = h_idx.size(0)
            hr_texts_1 = []
            hr_texts_2 = []
            for i in range(B):
                h = int(h_idx[i].item())
                r = int(r_idx[i].item())

                ent_name = self.ent_names[h].strip() if h < len(self.ent_names) else ""
                ent_desc = self.ent_descs[h].strip() if h < len(self.ent_descs) else ""
                if ent_name and ent_desc:
                    text1 = f"{ent_name} : {ent_desc}"
                else:
                    text1 = ent_name or ent_desc or ""

                if r >= self.configs.n_rel:
                    actual_r = r - self.configs.n_rel
                    rel_name = self.rel_names[actual_r].strip() if actual_r < len(self.rel_names) else ""
                    text2 = f"reversed: {rel_name}"
                else:
                    rel_name = self.rel_names[r].strip() if r < len(self.rel_names) else ""
                    text2 = rel_name

                hr_texts_1.append(text1)
                hr_texts_2.append(text2)

            encoded = self.hr_bert.tokenizer(
                hr_texts_1,
                text_pair=hr_texts_2,
                padding=True,
                truncation=True,
                max_length=self.configs.text_len,
                return_tensors='pt',
                return_overflowing_tokens=False,
            )

            for key in encoded:
                encoded[key] = encoded[key].to(device)

            with torch.no_grad():
                hr_vec = self._encode(
                    self.hr_bert,
                    encoded['input_ids'],
                    encoded['attention_mask'],
                    encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids'])),
                )  # [B, H]

            q = self.text_q_proj(hr_vec)                     # [B, D]
            k_all = self.text_k_proj(self.ent_text_embed)   # [N_ent, D]
            q = F.normalize(q, p=2, dim=-1)
            k_all = F.normalize(k_all, p=2, dim=-1)
            text_logits = torch.matmul(q, k_all.t())        # [B, N_ent]

            logits = struct_logits + text_score_weight * text_logits
        else:
            logits = struct_logits

        gt = self.all_tail_gt if dataloader_idx == 0 else self.all_head_gt

        for i in range(len(src_ent)):
            hi, ti, ri = src_ent[i].item(), tgt_ent[i], rel[i].item()
            if self.configs.is_temporal:
                tgt_filter = gt[(hi, ri, val_triples[i][3])]
            else:
                # tgt_filter .type: list()
                tgt_filter = gt.get((hi, ri), [])
            ## store target score
            tgt_score = logits[i, ti].item()
            ## remove the scores of the entities we don't care
            logits[i, tgt_filter] = self._MASKING_VALUE
            # recover the target values
            logits[i, ti] = tgt_score

        _, argsort = torch.sort(logits, dim=1, descending=True)
        argsort = argsort.cpu().numpy()

        ranks = []
        for i in range(len(src_ent)):
            hi, ti, ri = src_ent[i].item(), tgt_ent[i], rel[i].item()
            rank = np.where(argsort[i] == ti)[0][0] + 1
            ranks.append(rank)

        return ranks

    def test_step(self, batched_data, Z_ent=None, Z_rel=None, dataloader_idx=0):
        """ - hr-t


        """
        assert Z_ent is not None and Z_rel is not None, "+RGATZ_entZ_rel"
        Z_ent = self.get_fused_entity_embeddings(Z_ent)
        test_triples = batched_data['triple']
        ent_rel = batched_data['ent_rel']  # [B, 2]
        src_ent, rel = ent_rel[:, 0], ent_rel[:, 1]
        tgt_ent = batched_data['tgt_ent']
        B = src_ent.size(0)
        device = Z_ent.device

        h_embed = Z_ent[src_ent]  # [B, d]
        rel_embed = Z_rel[rel]  # [B, d]

        h_norm = F.normalize(h_embed, p=2, dim=-1)  # [B, d]
        rel_norm = F.normalize(rel_embed, p=2, dim=-1)  # [B, d]
        all_struct_emb = Z_ent  # [N_ent, d]
        all_struct_emb_norm = F.normalize(all_struct_emb, p=2, dim=-1)  # [N_ent, d]

        sim_h_all = F.cosine_similarity(
            h_norm.unsqueeze(1),  # [B, 1, d]
            all_struct_emb_norm.unsqueeze(0),  # [1, N_ent, d]
            dim=-1  # [B, N_ent]
        )

        hr_hadamard_norm = F.normalize(h_embed * rel_embed, p=2, dim=-1)  # [B, d]
        sim_hr_all = F.cosine_similarity(
            hr_hadamard_norm.unsqueeze(1),  # [B, 1, d]
            all_struct_emb_norm.unsqueeze(0),  # [1, N_ent, d]
            dim=-1  # [B, N_ent]
        )

        hr_add_norm = F.normalize(h_embed + rel_embed, p=2, dim=-1)  # [B, d]
        sim_add_all = F.cosine_similarity(
            hr_add_norm.unsqueeze(1),  # [B, 1, d]
            all_struct_emb_norm.unsqueeze(0),  # [1, N_ent, d]
            dim=-1  # [B, N_ent]
        )

        sim_matrix = 0.05 * sim_hr_all + 0.65 * sim_add_all + 0.3 * sim_h_all  # [B, N_ent]


        gt = self.all_tail_gt if dataloader_idx == 0 else self.all_head_gt
        out_pos = self.HyperbolicKGPred(h_embed, rel_embed, Z_ent, sim_matrix)
        struct_logits = out_pos["score"]  # [B, N_ent]



        text_score_weight = float(getattr(self.configs, 'text_score_weight',13))
        if text_score_weight > 0.0:
            device = struct_logits.device

            if getattr(self, 'ent_text_embed', None) is None or self.ent_text_embed.device != device:
                all_ent_vectors = self.precompute_all_entity_vectors()
                self.ent_text_embed = all_ent_vectors.to(device)

            h_idx = src_ent
            r_idx = rel
            B = h_idx.size(0)
            hr_texts_1 = []
            hr_texts_2 = []
            for i in range(B):
                h = int(h_idx[i].item())
                r = int(r_idx[i].item())

                ent_name = self.ent_names[h].strip() if h < len(self.ent_names) else ""
                ent_desc = self.ent_descs[h].strip() if h < len(self.ent_descs) else ""
                if ent_name and ent_desc:
                    text1 = f"{ent_name} : {ent_desc}"
                else:
                    text1 = ent_name or ent_desc or ""

                if r >= self.configs.n_rel:
                    actual_r = r - self.configs.n_rel
                    rel_name = self.rel_names[actual_r].strip() if actual_r < len(self.rel_names) else ""
                    text2 = f"reversed: {rel_name}"
                else:
                    rel_name = self.rel_names[r].strip() if r < len(self.rel_names) else ""
                    text2 = rel_name

                hr_texts_1.append(text1)
                hr_texts_2.append(text2)

            encoded = self.hr_bert.tokenizer(
                hr_texts_1,
                text_pair=hr_texts_2,
                padding=True,
                truncation=True,
                max_length=self.configs.text_len,
                return_tensors='pt',
                return_overflowing_tokens=False,
            )

            for key in encoded:
                encoded[key] = encoded[key].to(device)

            with torch.no_grad():
                hr_vec = self._encode(
                    self.hr_bert,
                    encoded['input_ids'],
                    encoded['attention_mask'],
                    encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids'])),
                )  # [B, H]

            q = self.text_q_proj(hr_vec)                   # [B, D]
            k_all = self.text_k_proj(self.ent_text_embed)  # [N_ent, D]
            q = F.normalize(q, p=2, dim=-1)
            k_all = F.normalize(k_all, p=2, dim=-1)
            text_logits = torch.matmul(q, k_all.t())       # [B, N_ent]
            logits = struct_logits + text_score_weight * text_logits
        else:
            logits = struct_logits
        for i in range(len(src_ent)):
            hi, ti, ri = src_ent[i].item(), tgt_ent[i], rel[i].item()
            if self.configs.is_temporal:
                tgt_filter = gt[(hi, ri, test_triples[i][3])]
            else:
                # tgt_filter .type: list()
                tgt_filter = gt.get((hi, ri), [])
            ## store target score
            tgt_score = logits[i, ti].item()
            ## remove the scores of the entities we don't care
            logits[i, tgt_filter] = self._MASKING_VALUE
            ## recover the target values
            logits[i, ti] = tgt_score
        _, argsort = torch.sort(logits, dim=1, descending=True)
        argsort = argsort.cpu().numpy()

        ranks = []
        for i in range(len(src_ent)):
            hi = src_ent[i].item() if hasattr(src_ent[i], 'item') else int(src_ent[i])
            ti_raw = tgt_ent[i]
            ti = int(ti_raw.item()) if hasattr(ti_raw, 'item') else int(ti_raw)
            ri = rel[i].item() if hasattr(rel[i], 'item') else int(rel[i])
            ranking = argsort[i]
            rank = np.where(ranking == ti)[0][0] + 1
            ranks.append(rank)

        return ranks


class Trainer:
    def __init__(self, model, configs, train_dataloader, val_dataloader=None, test_dataloader=None):
        self.model = model
        self.configs = configs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        bert_params = list(self.model.hr_bert.parameters())

        # embed_params = list(self.model.ent_embed.parameters()) + list(self.model.rel_embed.parameters())

        text_proj_params = (list(self.model.text_to_struct.parameters()) +
                            list(self.model.struct_to_text.parameters()) +
                            list(self.model.text_q_proj.parameters()) +
                            list(self.model.text_k_proj.parameters()))
        embed_params = (list(self.model.ent_embed.parameters()) +
                        list(self.model.waveletRGATEncoder.parameters()) +
                        text_proj_params)
        print(
            f"Optimizer param groups: BERT={len(bert_params)}, Embed+RGAT+TextProj={len(embed_params)}, Pred={len(list(self.model.HyperbolicKGPred.parameters()))}")

        pred_params = list(self.model.HyperbolicKGPred.parameters())

        self.optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': 2e-5, 'weight_decay': 0.0001},
            {'params': embed_params, 'lr': 1.0e-3, 'weight_decay': 0.0001},
            {'params': pred_params, 'lr': 1.0e-3, 'weight_decay': 0.0001},
        ])

        if hasattr(configs, 'scheduler_step_size'):
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=configs.scheduler_step_size,
                gamma=configs.scheduler_gamma
            )
        else:
            self.scheduler = None

        self.train_losses = []
        self.val_performances = []

    def train_epoch(self, epoch):

        self.model.train()
        total_loss = 0
        num_batches = 0


        print(f"Epoch {epoch}: Precomputing graph structure...")
        edge_index, edge_type, edge_weight, L_rescaled, sim1_edge = self.model.precompute_graph_embeddings()

        update_interval = getattr(self.configs, 'graph_update_interval', 1)
        print(f"Epoch {epoch}: Using periodic update (every {update_interval} batches)")

        Z_ent_detached = None
        Z_rel_detached = None

        with tqdm(self.train_dataloader, desc=f"Epoch {epoch}",
                  dynamic_ncols=True, leave=True, ncols=120) as pbar:
            for batch_idx, batched_data in enumerate(pbar):
                for key in batched_data:
                    if torch.is_tensor(batched_data[key]):
                        batched_data[key] = batched_data[key].to(self.device)

                if batch_idx % update_interval == 0:
                    Z_ent, Z_rel = self.model.waveletRGATEncoder(
                        X=self.model.ent_embed.weight,
                        edge_index=edge_index,
                        edge_type=edge_type,
                        sim1_edge=sim1_edge,
                        edge_weight=edge_weight,
                        L_rescaled=L_rescaled
                    )

                    loss = self.model.training_step(batched_data, Z_ent, Z_rel)
                self.optimizer.zero_grad()
                loss.backward()
                if hasattr(self.configs, 'grad_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.configs.grad_clip_norm)
                    self.optimizer.step()

                    Z_ent_detached = Z_ent.detach()
                    Z_rel_detached = Z_rel.detach()
                else:
                    loss = self.model.training_step(batched_data, Z_ent_detached, Z_rel_detached)
                    self.optimizer.zero_grad()
                    loss.backward()
                    if hasattr(self.configs, 'grad_clip_norm'):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.configs.grad_clip_norm)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        print(f'Epoch {epoch} - Average Loss: {avg_loss:.4f}')

        if self.scheduler is not None:
            self.scheduler.step()

            if len(self.optimizer.param_groups) >= 1:
                self.optimizer.param_groups[0]['lr'] = 2e-5

            lr_bert = self.optimizer.param_groups[0]['lr'] if len(self.optimizer.param_groups) >= 1 else 0.0
            lr_embed = self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) >= 2 else lr_bert
            print(f'Learning Rate: BERT={lr_bert:.6f}, Embed/Pred={lr_embed:.6f}')


    # return Z_ent, Z_rel

    def validate(self, epoch):
        """"""
        if self.val_dataloader is None:
            return None

        if hasattr(self.model, 'cached_all_ent_vectors'):
            self.model.cached_all_ent_vectors = None
        if hasattr(self.model, 'ent_text_embed'):
            self.model.ent_text_embed = None

        self.model.eval()

        all_ranks = {0: [], 1: []}  # tail_ranks, head_ranks
        with torch.no_grad():

            print("Validation: Computing graph embeddings...")
            edge_index, edge_type, edge_weight, L_rescaled, sim1_edge = self.model.precompute_graph_embeddings()
            Z_ent, Z_rel = self.model.waveletRGATEncoder(
                X=self.model.ent_embed.weight,
                edge_index=edge_index,
                edge_type=edge_type,
                sim1_edge=sim1_edge,
                edge_weight=edge_weight,
                L_rescaled=L_rescaled
            )

            for dataloader_idx, dataloader in enumerate(self.val_dataloader):
                ranks = []
                for batched_data in tqdm(dataloader, desc=f'Validation {dataloader_idx}'):
                    for key in batched_data:
                        if torch.is_tensor(batched_data[key]):
                            batched_data[key] = batched_data[key].to(self.device)

                    # batch_ranks = self.model.validation_step(batched_data, dataloader_idx=dataloader_idx)

                    batch_ranks = self.model.validation_step(batched_data, Z_ent, Z_rel, dataloader_idx)

                    ranks.extend(batch_ranks)

                all_ranks[dataloader_idx] = ranks

        tail_ranks = np.array(all_ranks.get(0, []))
        head_ranks = np.array(all_ranks.get(1, []))

        if tail_ranks.size == 0 and head_ranks.size == 0:
            return None

        perf = get_performance(self.model, tail_ranks, head_ranks)
        self.val_performances.append(perf)

        print(f'Epoch {epoch} - Validation Results:')
        print(perf)

        return perf

    def test(self):
        """"""
        if self.test_dataloader is None:
            return None

        if hasattr(self.model, 'cached_all_ent_vectors'):
            self.model.cached_all_ent_vectors = None
        if hasattr(self.model, 'ent_text_embed'):
            self.model.ent_text_embed = None

        self.model.eval()

        all_ranks = {0: [], 1: []}  # tail_ranks, head_rank
        with torch.no_grad():

            print("Test: Computing graph embeddings...")
            edge_index, edge_type, edge_weight, L_rescaled, sim1_edge = self.model.precompute_graph_embeddings()
            Z_ent, Z_rel = self.model.waveletRGATEncoder(
                X=self.model.ent_embed.weight,
                edge_index=edge_index,
                edge_type=edge_type,
                sim1_edge=sim1_edge,
                edge_weight=edge_weight,
                L_rescaled=L_rescaled
            )

            for dataloader_idx, dataloader in enumerate(self.test_dataloader):
                ranks = []
                for batched_data in tqdm(dataloader, desc=f'Testing {dataloader_idx}'):
                    for key in batched_data:
                        if torch.is_tensor(batched_data[key]):
                            batched_data[key] = batched_data[key].to(self.device)

                    # batch_ranks = self.model.test_step(batched_data, dataloader_idx=dataloader_idx)

                    batch_ranks = self.model.test_step(batched_data, Z_ent, Z_rel, dataloader_idx)

                    ranks.extend(batch_ranks)

                all_ranks[dataloader_idx] = ranks

        tail_ranks = np.array(all_ranks.get(0, []))
        head_ranks = np.array(all_ranks.get(1, []))

        if tail_ranks.size == 0 and head_ranks.size == 0:
            return None

        perf = get_performance(self.model, tail_ranks, head_ranks)

        print('Test Results:')
        print(perf)

        return perf

    def train(self, num_epochs, validate_every, save_path=None, start_epoch=1, best_val_mrr=0.0):
        """Training loop.

        Args:
            best_val_mrr: Best MRR from checkpoint (for continuing training)
        """

        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\n=== Epoch {epoch}/{num_epochs} ===")

            self.train_epoch(epoch)

            if epoch % validate_every == 0 and self.val_dataloader is not None:
                val_perf = self.validate(epoch)

                if val_perf is not None:
                    val_mrr = float(val_perf.loc['mean ranking', 'mrr'])

                    if val_mrr > best_val_mrr:
                        best_val_mrr = val_mrr
                        print(f"New best validation MRR: {best_val_mrr:.4f}")

                        if save_path is not None:
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'val_mrr': val_mrr,
                            }, save_path)
                            print(f"Model saved to: {save_path}")

        return self.train_losses, self.val_performances

