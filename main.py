import os
import argparse
from datetime import datetime
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import AutoConfig, BertTokenizer
from models.P_model_standalone import KGCPromptTuner, Trainer
from kgc_data import KGCDataModule
from helper import get_num, read, read_name, read_file, get_gt, get_performance
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
from tqdm import tqdm
import time


def build_training_graph_all(
        triples: List[Tuple[int, int, int]],  # [(h,t,r), ...]
        n_ent: int,
        n_rel: int,
        add_inverse_rel: bool = True,
        use_rel_weights: bool = True,
        weight_mode: str = "log",  # 'log'|'pow'|'count'|'norm'|'inv'
        pow_alpha: float = 0.5
):
    rel_cnt = Counter([r for _, _, r in triples])
    max_cnt = max(rel_cnt.values()) if rel_cnt else 1

    def rel_weight(cnt: int) -> float:
        if not use_rel_weights: return 1.0
        if weight_mode == "log":  return math.log1p(float(cnt))
        if weight_mode == "pow":  return float(cnt) ** float(pow_alpha)
        if weight_mode == "count": return float(cnt)
        if weight_mode == "norm": return float(cnt) / float(max_cnt)
        if weight_mode == "inv":  return 1.0 / (float(cnt) + 1.0)
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    src, dst, et, ew = [], [], [], []
    edge_map_all: Dict[Tuple[int, int], int] = {}

    for h, t, r in triples:
        w = rel_weight(rel_cnt[r])
        eid = len(src)
        src.append(h);
        dst.append(t);
        et.append(r);
        ew.append(w)
        edge_map_all[(h, t)] = eid

    E_tail = len(src)

    if add_inverse_rel:
        for h, t, r in triples:
            w = rel_weight(rel_cnt[r])
            eid = len(src)
            src.append(t);
            dst.append(h);
            et.append(r + n_rel);
            ew.append(w)
            edge_map_all[(t, h)] = eid

    edge_index_all = torch.tensor([src, dst], dtype=torch.long)  # [2,E_all]
    edge_type_all = torch.tensor(et, dtype=torch.long)  # [E_all]
    edge_weight_all = torch.tensor(ew, dtype=torch.float) if use_rel_weights else None

    return edge_index_all, edge_type_all, edge_weight_all, edge_map_all, E_tail


def build_rescaled_laplacian(
        num_nodes: int,
        edge_index: torch.Tensor,  # [2,E]
        edge_weight: Optional[torch.Tensor] = None,
        add_self_loops: bool = True,
        symmetrize: bool = False,  # True: A = (A + A^T)/2
        eps: float = 1e-9
) -> torch.sparse.FloatTensor:
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)

    if add_self_loops:
        idx = torch.arange(num_nodes, dtype=torch.long)
        self_loops = torch.stack([idx, idx], dim=0)  # [2,N]
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        edge_weight = torch.cat([edge_weight, torch.ones(num_nodes, dtype=torch.float)], dim=0)

    A = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).coalesce()

    if symmetrize:
        i = A.indices();
        v = A.values()
        At = torch.sparse_coo_tensor(torch.stack([i[1], i[0]], dim=0), v, A.size()).coalesce()
        A = (A + At).coalesce()
        A = torch.sparse_coo_tensor(A.indices(), 0.5 * A.values(), A.size()).coalesce()

    rows, cols = A.indices()
    vals = A.values()
    deg = torch.zeros(num_nodes, dtype=torch.float).scatter_add_(0, rows, vals)
    deg_inv_sqrt = (deg + eps).pow(-0.5)
    norm_vals = deg_inv_sqrt[rows] * vals * deg_inv_sqrt[cols]
    A_norm = torch.sparse_coo_tensor(A.indices(), norm_vals, A.size()).coalesce()

    I_idx = torch.arange(num_nodes, dtype=torch.long)
    I = torch.sparse_coo_tensor(torch.stack([I_idx, I_idx], dim=0),
                                torch.ones(num_nodes, dtype=torch.float),
                                A.size()).coalesce()
    L = (I - A_norm).coalesce()
    L_rescaled = (L - I).coalesce()
    return L_rescaled


def build_out_neighbors_complete(
        all_tail_gt: Dict[Tuple[int, int], List[int]],
        ent_names: Optional[List[str]] = None,
        ent_descs: Optional[List[str]] = None,
        sort_neighbors: bool = True
) -> Dict[int, List[int]]:

    neigh = defaultdict(set)
    all_entities = set()

    for (h, r), tails in all_tail_gt.items():
        all_entities.add(h)
        all_entities.update(tails)
        neigh[h].update(tails)

    if ent_names is not None:
        all_entities.update(range(len(ent_names)))
    if ent_descs is not None:
        all_entities.update(range(len(ent_descs)))

    for e in all_entities:
        _ = neigh[e]

    return {e: (sorted(list(ns)) if sort_neighbors else list(ns)) for e, ns in neigh.items()}


def build_neighbor_name_des_table(
        neighbors_all_tail_gt: Dict[int, List[int]],
        ent_names: List[str],
        ent_descs: List[str],
        keep_all_entities: bool = False
) -> Dict[int, List[Dict[int, Dict[str, str]]]]:
    """
    Build neighbor name and description table.
    
    Returns:
        {
            entity_id: [
                { neighbor1_id: {'name': <name>, 'des': <description>} },
                { neighbor2_id: {'name': <name>, 'des': <description>} },
                ...
            ],
            ...
        }
    """
    result: Dict[int, List[Dict[int, Dict[str, str]]]] = {}

    for e, neighs in neighbors_all_tail_gt.items():
        rows: List[Dict[int, Dict[str, str]]] = []
        for nid in neighs:
            name = ent_names[nid].strip() if 0 <= nid < len(ent_names) else ""
            des = ent_descs[nid].strip() if 0 <= nid < len(ent_descs) else ""
            rows.append({nid: {"name": name, "des": des}})
        result[e] = rows

    if keep_all_entities:
        max_len = max(len(ent_names), len(ent_descs))
        for eid in range(max_len):
            if eid not in result:
                result[eid] = []

    return result


def main():
    if configs.save_dir == '':
        if configs.jobid == 'XXXXXXXX':
            configs.save_dir = os.path.join('./checkpoint',
                                            configs.dataset + '-' + datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
        else:
            configs.save_dir = os.path.join('./checkpoint', configs.jobid)
    os.makedirs(configs.save_dir, exist_ok=True)

    ## read triples
    train = read(configs, configs.dataset_path, configs.dataset, 'train2id.txt')
    valid = read(configs, configs.dataset_path, configs.dataset, 'valid2id.txt')  # 3034
    test = read(configs, configs.dataset_path, configs.dataset, 'test2id.txt')
    all_triples = train + valid + test

    ## construct ground truth dictionary
    # ground truth .shape: dict, example: {hr_str_key1: [t_id11, t_id12, ...], (hr_str_key2: [t_id21, t_id22, ...], ...}
    train_tail_gt, train_head_gt = get_gt(configs, train)
    # val_tail_gt, val_head_gt = get_gt(configs, valid)
    # test_tail_gt, test_head_gt = get_gt(configs, test)
    all_tail_gt, all_head_gt = get_gt(configs, all_triples)
    neighbors_train_tail_gt = build_out_neighbors_complete(train_tail_gt)
    neighbors_train_head_gt = build_out_neighbors_complete(train_head_gt)

    neighbors_all_tail_gt = build_out_neighbors_complete(all_tail_gt)
    neighbors_all_head_gt = build_out_neighbors_complete(all_head_gt)

    print("Building graph structure...")
    edge_index_all, edge_type_all, edge_weight_all, edge_map_all, E_tail = build_training_graph_all(
        triples=train,
        n_ent=n_ent,
        n_rel=n_rel,
        add_inverse_rel=True,
        use_rel_weights=True,
        weight_mode="log",
        pow_alpha=0.5
    )

    L_rescaled_all = build_rescaled_laplacian(
        num_nodes=n_ent,
        edge_index=edge_index_all,
        edge_weight=edge_weight_all,
        add_self_loops=True,
        symmetrize=False
    )
    print(f"Graph structure built: {edge_index_all.shape[1]} edges")

    gt = {
        'train_tail_gt': train_tail_gt,
        'train_head_gt': train_head_gt,

        'all_tail_gt': all_tail_gt,
        'all_head_gt': all_head_gt,

        'neighbors_train_tail_gt': neighbors_train_tail_gt,
        'neighbors_train_head_gt': neighbors_train_head_gt,
        'neighbors_all_tail_gt': neighbors_all_tail_gt,
        'neighbors_all_head_gt': neighbors_all_head_gt,
        'train_edge_index_all': edge_index_all,
        'train_edge_type_all': edge_type_all,
        'train_edge_weight_all': edge_weight_all,
        'train_E_tail': E_tail,
        'train_edge_map_all': edge_map_all,
        'train_L_rescaled_all': L_rescaled_all,
    }

    ent_names, rel_names = read_name(configs, configs.dataset_path, configs.dataset)
    ent_descs = read_file(configs, configs.dataset_path, configs.dataset, 'entityid2description.txt',
                          'desc')

    text_neighbors_train_tail_gt = build_neighbor_name_des_table(neighbors_train_tail_gt, ent_names,
                                                                 ent_descs)
    text_neighbors_train_head_gt = build_neighbor_name_des_table(neighbors_train_head_gt, ent_names,
                                                                 ent_descs)
    text_neighbors_all_tail_gt = build_neighbor_name_des_table(neighbors_all_tail_gt, ent_names,
                                                               ent_descs)
    text_neighbors_all_head_gt = build_neighbor_name_des_table(neighbors_all_head_gt, ent_names,
                                                               ent_descs)
    tok = BertTokenizer.from_pretrained(configs.pretrained_model,
                                        add_prefix_space=False)

    text_dict = {
        'ent_names': ent_names,
        'rel_names': rel_names,
        'ent_descs': ent_descs,
        'text_neighbors_train_tail_gt': text_neighbors_train_tail_gt,
        'text_neighbors_train_head_gt': text_neighbors_train_head_gt,
        'text_neighbors_all_tail_gt': text_neighbors_all_tail_gt,
        'text_neighbors_all_head_gt': text_neighbors_all_head_gt,
    }

    ## construct datamodule
    datamodule = KGCDataModule(configs, train, valid, test, text_dict, tok,
                               gt)
    print('datamodule construction done.', flush=True)

    ## construct model
    model = KGCPromptTuner(configs, text_dict, gt)
    print('model construction done.', flush=True)

    trainable_params, non_trainable_params = 0, 0
    for name, params in model.named_parameters():
        if params.requires_grad:
            print('name:', name, 'shape:', params.shape, 'numel:', params.numel())
            trainable_params += params.numel()
        else:
            non_trainable_params += params.numel()
    print('trainable params:', trainable_params, 'non trainable params:', non_trainable_params)

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    trainer = Trainer(
        model=model,
        configs=configs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    )
    torch.autograd.set_detect_anomaly(True)
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    if configs.model_path == '':
        if configs.continue_path == '':
            checkpoint_path = os.path.join(configs.save_dir, f'{configs.dataset}-best.pth')
            if os.path.exists(checkpoint_path):
                print(f"Warning: Found existing checkpoint: {checkpoint_path}")
                print(f"Warning: Training from scratch will overwrite this checkpoint!")
                print(f"Warning: To continue training, use: -continue_path {checkpoint_path}")
                response = input("Continue training from scratch? (yes/no): ")
                if response.lower() != 'yes':
                    print("Training cancelled.")
                    exit(0)
            train_losses, val_performances = trainer.train(
                num_epochs=configs.epochs,
                validate_every=configs.check_val_every_n_epoch,
                save_path=os.path.join(configs.save_dir, f'{configs.dataset}-best.pth')
            )
            model_path = os.path.join(configs.save_dir, f'{configs.dataset}-best.pth')
        else:
            checkpoint = torch.load(configs.continue_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Continuing training from epoch {start_epoch}")

            print("  Phase 2C: Resetting Gate parameters...")
            print(
                f"    Before: gate_rho_a={model.HyperbolicKGPred.gate_rho_a.item():.1f}, gate_rho_b={model.HyperbolicKGPred.gate_rho_b.item():.1f}")

            for idx, param_group in enumerate(trainer.optimizer.param_groups):
                if idx == 0:
                    param_group['lr'] = 2e-5
                    print(f"  BERT lr kept at: {param_group['lr']}")
                else:
                    old_lr = param_group['lr']
                    param_group['lr'] =1.0e-4
                    print(f"  Increased lr: {old_lr} -> {param_group['lr']}")

            best_mrr_from_checkpoint = checkpoint.get('val_mrr', 0.0)
            print(f"  Loaded best MRR: {best_mrr_from_checkpoint:.4f}")

            train_losses, val_performances = trainer.train(
                num_epochs=configs.epochs,
                validate_every=configs.check_val_every_n_epoch,
                save_path=os.path.join(configs.save_dir, f'{configs.dataset}-best.pth'),
                start_epoch=start_epoch,
                best_val_mrr=best_mrr_from_checkpoint
            )
            model_path = os.path.join(configs.save_dir, f'{configs.dataset}-best.pth')
        checkpoint = torch.load(model_path, map_location=trainer.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model: {model_path}')

    else:
        model_path = configs.model_path
        checkpoint = torch.load(model_path, map_location=trainer.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model: {model_path}')

    print('model_path:', model_path, flush=True)

    trainer.test()


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    transformers.logging.set_verbosity_error()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset_path', type=str, default='./data/processed')
    parser.add_argument('-jobid', type=str, default='XXXXXXXX')
    parser.add_argument('-dataset', dest='dataset', default='WN18RR',
                        help='Dataset to use, default: InferWiki16k')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-seed', dest='seed', default=2025, type=int, help='Seed for randomization')
    parser.add_argument('-num_workers', type=int, default=0, help='Number of processes to construct batches')
    parser.add_argument('-save_dir', type=str, default='checkpoint/WN18RR', help='Directory to save checkpoints')
    parser.add_argument('-pretrained_model', type=str, default='bert-base-uncased', help='Pretrained BERT model')
    parser.add_argument('-batch_size', default=1024, type=int, help='Batch size')
    parser.add_argument('-val_batch_size', default=256, type=int, help='Validation batch size')
    parser.add_argument('-src_max_length', default=512, type=int, help='Max source sequence length')
    parser.add_argument('-epoch', dest='epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')

    parser.add_argument('-model_path', dest='model_path', default='', help='The path for reloading models')
    parser.add_argument('-desc_max_length', default=50, type=int, help='Max description length')
    parser.add_argument('-embed_dim', default=128, type=int, help='Embedding dimension')

    parser.add_argument('-bias', dest='bias', action='store_true', help='Whether to use bias in the model')
    parser.add_argument('-label_smoothing', default=0.1, type=float, help='Label smoothing')
    parser.add_argument('-continue_path', dest='continue_path', default='', help='The path for continuing training')
    parser.add_argument('-check_val_every_n_epoch', default=, type=int,
                        help='Validation frequency (epochs)')
    parser.add_argument('-text_len', default=100, type=int, help='Max text length for BERT input')

    parser.add_argument('-use_fp16', action='store_true', default=False,
                        help='Use FP16 mixed precision training')

    parser.add_argument('-alpha_step', default=0.00001, type=float, help='Alpha step for loss weighting')

    parser.add_argument('-max_neigh', default=10, type=float, help='Max neighbors per entity')
    parser.add_argument('-neg_K', default=300, type=float, help='Number of negative samples (larger values like 256 may increase train-eval gap)')
    parser.add_argument('--add_inverse_rel', action='store_true', default=True, help='Add inverse relations')

    parser.add_argument('--disable_hard_negative', action='store_true', default=False,
                        help='Disable hard negative sampling (use random sampling)')


    parser.add_argument('-WaGATembed_dim', default=128, type=int, help='Wavelet GAT input embedding dimension')
    parser.add_argument('-wavelet_hidden_dim', default=128, type=int, help='Wavelet hidden dimension')
    parser.add_argument('-struct_out_dim', default=128, type=int, help='Structure output dimension')
    parser.add_argument('-gat_heads', default=4, type=int, help='Number of GAT attention heads')
    parser.add_argument('-gnn_dropout', default=0.1, type=float, help='GAT dropout rate')
    parser.add_argument('-rel_dim', default=None, type=int, help='Relation dimension; None uses out_dim')
    parser.add_argument('-cheb_K', default=4, type=int, help='Chebyshev polynomial order K (wavelet scale)')
    parser.add_argument('-beta_init', default=1.0, type=float, help='Initial beta value')
    parser.add_argument('-concat_heads', action='store_true', default=True, help='Concat GAT heads (True=concat, False=average)')
    parser.add_argument('-Dir_s_dim', default=128, type=int, help='Directional score dimension')

    parser.add_argument('-scheduler_step_size', default=20, type=int, help='LR scheduler step size (epochs)')
    parser.add_argument('-scheduler_gamma', default=0.9, type=float, help='LR scheduler gamma (decay factor)')
    parser.add_argument('-grad_clip_norm', default=2.0, type=float, help='Gradient clipping norm')
    parser.add_argument('-patience', default=10, type=int, help='Early stopping patience')

    parser.add_argument('-loss_temperature', default=1.1, type=float, help='Loss temperature for softmax')
    parser.add_argument('-text_align_weight', default=0.0, type=float, help='Text-structure alignment loss weight')
    parser.add_argument('-text_struct_mix_alpha', default=0.0, type=float, help='Text-structure mixing alpha')

    parser.add_argument('-score_margin_weight', default=0.05, type=float, help='Score margin loss weight')
    parser.add_argument('-score_margin', default=1.0, type=float, help='Score margin value')

    parser.add_argument('-graph_update_interval', default=1, type=int,
                        help='Graph embedding update interval (N batches). 1=every batch, 50=periodic, 9999=once per epoch')


    configs = parser.parse_args()
    n_ent = get_num(configs.dataset_path, configs.dataset, 'entity')
    n_rel = get_num(configs.dataset_path, configs.dataset, 'relation')
    configs.n_ent = n_ent  # 40943
    configs.n_rel = n_rel  # 11
    configs.vocab_size = AutoConfig.from_pretrained(configs.pretrained_model).vocab_size  # 30522
    configs.model_dim = AutoConfig.from_pretrained(configs.pretrained_model).hidden_size  # 1024
    configs.is_temporal = 'ICEWS' in configs.dataset
    print(configs, flush=True)

    torch.manual_seed(configs.seed)
    np.random.seed(configs.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(configs.seed)
        torch.cuda.manual_seed_all(configs.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(profile='full')
    main()



