import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset

from helper import dataloader_output_to_tensor, get_lar_sample_bank
from typing import Dict, Iterable, Optional, Tuple
from typing import Dict, List, Tuple, Any
import torch, random,math
from torch.utils.data import DataLoader

def dataloader_output_to_tensor(batch_list: List[Dict[str, Any]], key: str,
                                padding_value: int = None, return_list: bool = False):
    vals = [x[key] for x in batch_list if key in x]
    if return_list:
        return vals
    if padding_value is None:
        return torch.tensor(vals)
    maxlen = max((len(v) for v in vals), default=0)
    out = []
    for v in vals:
        out.append(v + [padding_value] * (maxlen - len(v)))
    return torch.tensor(out, dtype=torch.long)

def pad_3d_int(bkl: List[List[List[int]]], pad: int = 0):
    Kmax = max((len(neis) for neis in bkl), default=0)
    Lmax = 0
    for neis in bkl:
        for seq in neis:
            if len(seq) > Lmax:
                Lmax = len(seq)
    ids_bkl, mask_tok_bkl, exist_bk = [], [], []
    for neis in bkl:
        row_ids, row_mask, row_exist = [], [], []
        for seq in neis:
            pad_len = Lmax - len(seq)
            row_ids.append(seq + [pad] * pad_len)
            row_mask.append([1] * len(seq) + [0] * pad_len)
            row_exist.append(1)
        for _ in range(Kmax - len(neis)):
            row_ids.append([pad] * Lmax)
            row_mask.append([0] * Lmax)
            row_exist.append(0)
        ids_bkl.append(row_ids)
        mask_tok_bkl.append(row_mask)
        exist_bk.append(row_exist)
    return ids_bkl, mask_tok_bkl, exist_bk

class BaseDataset(Dataset):
    def __init__(self, configs, tok, triples, text_dict, mode='train',mode1='val', gt=None):
        super().__init__()
        self.configs = configs
        self.tok = tok
        self.ent_names = text_dict['ent_names']
        self.rel_names = text_dict['rel_names']
        self.ent_descs = text_dict['ent_descs']
        self.triples = triples
        self.mode = mode
        self.mode1 = mode1
        if gt is not None:
            self.train_tail_gt = gt['train_tail_gt']
            self.train_head_gt = gt['train_head_gt']

    def parse_ent_name(self, name):
        if self.configs.dataset == 'WN18RR':
            name = ' '.join(name.split(' , ')[:-2])
            return name
        return name or ''

    def construct_input_text(self, src_ent=None, rel=None, timestamp=None, predict='predict_tail'):
        src_name = self.ent_names[src_ent]
        rel_name = self.rel_names[rel]
        src_desc = ':' + self.ent_descs[src_ent] if self.configs.desc_max_length > 0 else ''

        timestamp = ' | ' + timestamp if timestamp else ''
        if predict == 'predict_tail':
            return src_name + ' ' + src_desc, rel_name + timestamp
        elif predict == 'predict_head':
            return src_name + ' ' + src_desc, 'reversed: ' + rel_name + timestamp
        else:
            raise ValueError('Mode is not correct!')

    def construct_input_text_tgt_ent_src(self, src_ent=None, predict='predict_tail'):
        src_name = self.ent_names[src_ent]
        src_desc = ':' + self.ent_descs[src_ent] if self.configs.desc_max_length > 0 else ''

        if predict == 'predict_tail':
            return src_name + ' ' + src_desc,
        elif predict == 'predict_head':
            return src_name + ' ' + src_desc,
        else:
            raise ValueError('Mode is not correct!')

    def construct_input_text_head_src(self, src_ent=None, predict='predict_tail'):
        src_name = self.ent_names[src_ent]
        src_desc = ':' + self.ent_descs[src_ent] if self.configs.desc_max_length > 0 else ''

        if predict == 'predict_tail':
            return src_name + ' ' + src_desc,
        elif predict == 'predict_head':
            return src_name + ' ' + src_desc,
        else:
            raise ValueError('Mode is not correct!')

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['ent_rel'] = dataloader_output_to_tensor(data, 'ent_rel')
        agg_data['tgt_ent'] = dataloader_output_to_tensor(data, 'tgt_ent', return_list=True)
        agg_data['triple'] = dataloader_output_to_tensor(data, 'triple', return_list=True)
        
        if self.mode == 'train':
            agg_data['labels'] = dataloader_output_to_tensor(data, 'labels').squeeze(-1)
        else:
            agg_data['labels'] = torch.tensor([])
        
        # =======================
        if self.mode == 'train':
            neg_entity_ids_ll = [d['neg_entity_ids'] for d in data]  # List[List[int]]
            maxK = max((len(x) for x in neg_entity_ids_ll), default=0)
            neg_entity_ids_bk = []
            for row in neg_entity_ids_ll:
                if len(row) < maxK:
                    row = row + [0] * (maxK - len(row))
                neg_entity_ids_bk.append(row)
            agg_data['neg_entity_ids'] = torch.tensor(neg_entity_ids_bk, dtype=torch.long)  # [B,K]

        return agg_data

class CELossDataset(BaseDataset):
    def __init__(self, configs, tok, triples, text_dict, mode='train',mode1=None,gt=None):
        super().__init__(configs, tok, triples, text_dict, mode,mode1, gt)
        self.all_ent = set(range(configs.n_ent))
        self.mode1 = mode1
        if self.mode == "train":
            self.neigh_out = gt.get('neighbors_train_tail_gt', {})
            self.neigh_in = gt.get('neighbors_train_head_gt', {})
            self.text_neigh_out = text_dict.get('text_neighbors_train_tail_gt', {})
            self.text_neigh_in = text_dict.get('text_neighbors_train_head_gt', {})

            self.train_tail_gt = gt.get('train_tail_gt', {})
            self.train_head_gt = gt.get('train_head_gt', {})
            self.all_tail_gt = gt.get('all_tail_gt', {})
            self.all_head_gt = gt.get('all_head_gt', {})

        self.ent_names = text_dict.get('ent_names', None)  # list
        self.ent_descs = text_dict.get('ent_descs', None)  # list

        self.max_neigh = configs.max_neigh
        self.neg_K = configs.neg_K

        self._tok_cache: Dict[Tuple[str, int], Dict[str, List[int]]] = {}


    def __len__(self):
        return len(self.triples) * 2 if self.mode == 'train' else len(self.triples)


    def _concat_name_desc(self, eid: int) -> str:
        name, desc = "", ""
        if self.ent_names is not None and 0 <= eid < len(self.ent_names):
            name = (self.ent_names[eid] or "").strip()
        if self.ent_descs is not None and 0 <= eid < len(self.ent_descs):
            desc = (self.ent_descs[eid] or "").strip()
        if name and desc:
            return f"{name} : {desc}"
        return name or desc

    def _get_center_entity(self, triple, mode):
        if not self.configs.is_temporal:
            head, tail, rel = triple
        else:
            head, tail, rel, _ = triple
        return head if mode == 'predict_tail' else tail

    def _get_neighbors_ids_and_texts(self, center_ent: int,predict='predict_tail'):
        if predict=='predict_tail':
            neigh_ids = self.neigh_out.get(center_ent, [])
        else:
            neigh_ids = self.neigh_in.get(center_ent, [])
        if self.max_neigh is not None and len(neigh_ids) > self.max_neigh:
            neigh_ids = neigh_ids[:self.max_neigh]
        if predict == 'predict_tail':
            text_map = self.text_neigh_out.get(center_ent, None)
        else:
            text_map = self.text_neigh_in.get(center_ent, None)
        neigh_texts: List[str] = []
        if isinstance(text_map, dict) and len(text_map) > 0:
            for nid in neigh_ids:
                rec = text_map.get(nid, None)
                if rec is None:
                    neigh_texts.append(self._concat_name_desc(nid))
                else:
                    t = rec.get("text") or (
                                rec.get("name", "") + (":" if rec.get("name") and rec.get("desc") else "") + rec.get(
                            "desc", ""))
                    neigh_texts.append((t or "").strip())
        else:
            for nid in neigh_ids:
                neigh_texts.append(self._concat_name_desc(nid))

        return neigh_ids, neigh_texts


    from typing import Dict, List

    def _tok_text_cached_1d(self, text: str) -> Dict[str, List[int]]:
        """
         tokenizer list：
        {'input_ids':[L], 'attention_mask':[L], 'token_type_ids':[L](0)}
        ，。
        """
        key = (text, self.configs.text_len)
        if key in self._tok_cache:
            return self._tok_cache[key]

        enc = self.tok(text=text, max_length=self.configs.text_len, truncation=True, return_overflowing_tokens=False)

        if hasattr(enc, "data"):  # HF BatchEncoding
            enc = enc.data  # -> dict

        single: Dict[str, List[int]] = {}
        for k, v in enc.items():
            if torch.is_tensor(v):
                if v.dim() == 2 and v.size(0) == 1:
                    single[k] = v.squeeze(0).tolist()
                elif v.dim() == 1:
                    single[k] = v.tolist()
                else:
                    raise ValueError(f"Unexpected tensor shape for {k}: {tuple(v.shape)}")
            elif isinstance(v, list):
                if len(v) > 0 and isinstance(v[0], list):
                    single[k] = v[0]
                else:
                    single[k] = v
            else:
                raise TypeError(f"Unexpected type for {k}: {type(v)}")

        if 'token_type_ids' not in single:
            single['token_type_ids'] = [0] * len(single['input_ids'])

        self._tok_cache[key] = single
        return single

    def _get_neg_ids_and_texts(self,
                               center_ent: int,
                               rel_id: int,
                               true_target: Optional[int] = None,
                               K: Optional[int] = None,
                               predict: str = "predict_tail",
                               use_all_gt: bool = False,
                               max_tries: int = 20) -> Tuple[List[int], List[str]]:
        """
        forbid = {true_target} ∪ gt[(center_ent, rel_id)]
         [0, n_ent)  K  forbid 。
        """
        if K is None:
            K = self.neg_K

        if predict == "predict_tail":
            gt_dict = (self.all_tail_gt if use_all_gt else self.train_tail_gt) or {}
        else:
            gt_dict = (self.all_head_gt if use_all_gt else self.train_head_gt) or {}

        n_ent = self.configs.n_ent
        forbid = set()
        if true_target is not None:
            forbid.add(int(true_target))
        if isinstance(gt_dict, dict):
            cand = gt_dict.get((int(center_ent), int(rel_id)), ())
            try:
                cand = list(cand)
            except TypeError:
                cand = []
            forbid.update(int(x) for x in cand)

        chosen: List[int] = []
        tries = 0
        while len(chosen) < K and tries < max_tries:
            need = K - len(chosen)
            cand = torch.randint(low=0, high=n_ent, size=(need,))
            for e in cand.tolist():
                if e not in forbid:
                    chosen.append(e)
            tries += 1
        #     pad = torch.randint(low=0, high=n_ent, size=(K - len(chosen),)).tolist()
        #     chosen.extend(pad)

        if len(chosen) < K:
            allow = list(set(range(n_ent)) - forbid)
            if len(allow) >= (K - len(chosen)):
                extra = random.sample(allow, K - len(chosen))
            else:
                extra = (allow * math.ceil((K - len(chosen)) / max(1, len(allow))))[:(K - len(chosen))]
            chosen.extend(extra)


        neg_ids = chosen[:K]
        neg_texts = [self._concat_name_desc(eid) for eid in neg_ids]
        return neg_ids, neg_texts

    def __getitem__(self, index):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.mode == 'train':
            mode = 'predict_tail' if index % 2 == 0 else 'predict_head'
            triple = self.triples[index // 2]
        else:
            mode = self.mode
            triple = self.triples[index]

        if not self.configs.is_temporal:
            head, tail, rel = triple
            timestamp = None
        else:
            head, tail, rel, timestamp = triple
        if mode == 'predict_tail':
            src = self.construct_input_text(src_ent=head,
                                            rel=rel,
                                            timestamp=timestamp,
                                            predict='predict_tail')
            # head_src = self.construct_input_text_head_src(src_ent=head,
            tgt_ent = tail
            tgt_ent_src = self.construct_input_text_tgt_ent_src(src_ent=tgt_ent,
                                                    predict='predict_tail')

            center_ent = head
            triple = triple
            if self.mode == 'train':
                rel_used = rel
                neg_ids, _ = self._get_neg_ids_and_texts(
                    center_ent=center_ent,
                    rel_id=rel_used,
                    true_target=tgt_ent,
                    K=self.neg_K,
                    predict='predict_tail',
                    use_all_gt=False
                )

        elif mode == 'predict_head':
            src = self.construct_input_text(src_ent=tail,
                                            rel=rel,
                                            timestamp=timestamp,
                                            predict='predict_head')
            # head_src = self.construct_input_text_head_src(src_ent=head,
            tgt_ent = head
            tgt_ent_src = self.construct_input_text_tgt_ent_src(src_ent=tgt_ent,
                                            predict='predict_head')
            center_ent = tail
            rel_riv = rel + self.configs.n_rel
            triple = [head, tail, rel_riv]


            if self.mode == 'train':
                rel_used = rel + self.configs.n_rel
                neg_ids, _ = self._get_neg_ids_and_texts(
                    center_ent=center_ent,
                    rel_id=rel_used,
                    true_target=tgt_ent,
                    K=self.neg_K,
                    predict='predict_head',
                    use_all_gt=False
                )
                # neg_input_ids, neg_attn_mask, neg_token_type_ids = [], [], []
                # for txt in neg_texts:
                #     enc = self._tok_text_cached_1d(txt)
                #     neg_input_ids.append(enc['input_ids'])
                #     neg_attn_mask.append(enc['attention_mask'])
                #     neg_token_type_ids.append(enc.get('token_type_ids', [0] * len(enc['input_ids'])))

        else:
            raise ValueError('Mode is not correct!')

        ent_rel = (head, rel) if mode == 'predict_tail' else (tail, rel + self.configs.n_rel)
        

        out = {
            'triple': triple,
            'ent_rel': ent_rel,#（，）
            'tgt_ent': tgt_ent,
            'center_ent': center_ent}

        if self.mode == 'train':
            out.update({
                'neg_entity_ids': neg_ids,
            })

        if self.mode == 'train':
            out['labels'] = [tgt_ent]
        return out


class KGCDataModule(pl.LightningDataModule):
    def __init__(self, configs, train, valid, test, text_dict, tok, gt):
        super().__init__()
        self.configs = configs
        self.train = train
        self.valid = valid
        self.test = test
        self.text_dict = text_dict
        self.tok = tok
        self.gt = gt

        self.train_both  = CELossDataset(configs, tok, train, text_dict, 'train', ' ',self.gt )
        self.valid_tail  = CELossDataset(configs, tok, valid, text_dict, 'predict_tail', 'val',self.gt)
        self.valid_head  = CELossDataset(configs, tok, valid, text_dict, 'predict_head','val', self.gt)
        self.test_tail   = CELossDataset(configs, tok, test,  text_dict, 'predict_tail', 'test',self.gt)
        self.test_head   = CELossDataset(configs, tok, test,  text_dict, 'predict_head','test', self.gt)



    def train_dataloader(self):
        train_loader = DataLoader(self.train_both,
                                  batch_size=self.configs.batch_size,
                                  shuffle=True,
                                  collate_fn=self.train_both.collate_fn,
                                  pin_memory=True,
                                  num_workers=self.configs.num_workers)
        return train_loader

    def val_dataloader(self):
        valid_tail_loader = DataLoader(self.valid_tail,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_tail.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)
        valid_head_loader = DataLoader(self.valid_head,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_head.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)
        return [valid_tail_loader, valid_head_loader]

    def test_dataloader(self):
        test_tail_loader = DataLoader(self.test_tail,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_tail.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)
        test_head_loader = DataLoader(self.test_head,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_head.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)
        return [test_tail_loader, test_head_loader]
