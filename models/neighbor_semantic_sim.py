

import random
import torch
import torch.nn.functional as F
from typing import Dict, List
from transformers import PreTrainedModel


def compute_neighbor_semantic_similarity(
    bert_model: PreTrainedModel,
    neighbors_train_tail_gt: Dict[int, List[int]],
    neighbors_train_head_gt: Dict[int, List[int]],
    ent_names: List[str],
    ent_descs: List[str],
    train_edge_index: torch.Tensor,
    train_edge_map: Dict[tuple, int],
    max_neighbors_per_entity: int,
    device: torch.device,
    batch_size: int = 1024,
    max_text_length: int = 512
) -> torch.Tensor:
    """
    Compute semantic similarity between neighbors for all entities in the training graph.
    
    For each center entity, compute pairwise text semantic similarity among all its neighbors,
    then assign the sum of similarities as the semantic prior weight for each edge.
    
    Args:
        bert_model: Pre-trained BERT model for encoding entity text
        neighbors_train_tail_gt: Neighbor dict for tail prediction {entity_id: [neighbor_ids]}
        neighbors_train_head_gt: Neighbor dict for head prediction {entity_id: [neighbor_ids]}
        ent_names: List of entity names
        ent_descs: List of entity descriptions
        train_edge_index: Edge index of training graph [2, E]
        train_edge_map: Edge to index mapping {(src, dst): edge_idx}
        max_neighbors_per_entity: Maximum number of neighbors to consider per entity
        device: Computation device
        batch_size: Batch size for processing entities
        max_text_length: Maximum text length for BERT encoding
        
    Returns:
        sim1_edge: Semantic similarity weight for each edge [E]
    """
    E = train_edge_index.size(1)
    sim1_edge = torch.zeros(E, device=device, dtype=torch.float32)

    print(f"Computing global semantic similarity (max {max_neighbors_per_entity} neighbors per entity)...")

    all_entities = set(neighbors_train_tail_gt.keys()) | set(neighbors_train_head_gt.keys())
    entity_ids = list(all_entities)

    print(f"Total entities to process: {len(entity_ids)}")

    for batch_start in range(0, len(entity_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(entity_ids))
        batch_entity_ids = entity_ids[batch_start:batch_end]

        for center_ent_id in batch_entity_ids:
            tail_neighbors = neighbors_train_tail_gt.get(center_ent_id, [])
            head_neighbors = neighbors_train_head_gt.get(center_ent_id, [])

            all_neighbors = list(set(tail_neighbors + head_neighbors))

            if len(all_neighbors) > max_neighbors_per_entity:
                all_neighbors = random.sample(all_neighbors, max_neighbors_per_entity)

            if len(all_neighbors) < 2:
                continue

            neighbor_texts = []
            valid_neighbor_ids = []

            for neighbor_id in all_neighbors:
                if neighbor_id < len(ent_names) and neighbor_id < len(ent_descs):
                    name = ent_names[neighbor_id].strip()
                    desc = ent_descs[neighbor_id].strip()
                    text = f"{name} {desc}".strip()
                    if text:
                        neighbor_texts.append(text)
                        valid_neighbor_ids.append(neighbor_id)

            if len(neighbor_texts) < 2:
                continue

            try:
                encoded = bert_model.tokenizer(
                    neighbor_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_text_length,
                    return_tensors='pt',
                    return_overflowing_tokens=False
                )

                for key in encoded:
                    encoded[key] = encoded[key].to(device)

                with torch.no_grad():
                    outputs = bert_model(**encoded)
                    neighbor_vectors = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                    neighbor_vectors = F.normalize(neighbor_vectors, p=2, dim=-1)

                K = len(neighbor_vectors)
                if K > 1:
                    sim_matrix = torch.matmul(neighbor_vectors, neighbor_vectors.T)  # [K, K]
                    sim1_per_neighbor = sim_matrix.sum(dim=1) - torch.diag(sim_matrix)

                    for i, neighbor_id in enumerate(valid_neighbor_ids):
                        edge_key_forward = (center_ent_id, neighbor_id)
                        edge_idx_forward = train_edge_map.get(edge_key_forward)

                        if edge_idx_forward is not None and edge_idx_forward < E:
                            sim1_value = sim1_per_neighbor[i].item()
                            if sim1_value > sim1_edge[edge_idx_forward].item():
                                sim1_edge[edge_idx_forward] = sim1_value

                        edge_key_backward = (neighbor_id, center_ent_id)
                        edge_idx_backward = train_edge_map.get(edge_key_backward)

                        if edge_idx_backward is not None and edge_idx_backward < E:
                            sim1_value = sim1_per_neighbor[i].item()
                            if sim1_value > sim1_edge[edge_idx_backward].item():
                                sim1_edge[edge_idx_backward] = sim1_value

            except Exception as e:
                print(f"Error computing similarity for entity {center_ent_id}: {e}")
                continue

    print(f"Semantic similarity computation done, non-zero edges: {(sim1_edge > 0).sum().item()}")
    return sim1_edge
