import torch
import numpy as np

def calculate_list_iou(list1, list2):
        """
        Calculate Intersection over Union (IoU) for two lists of frame indices.
        
        Args:
            list1: List of frame indices (retrieved frames)
            list2: List of frame indices (relevant frames)
            
        Returns:
            float: IoU value between 0 and 1
        """
        set1 = set(list1)
        set2 = set(list2)
        # print(f"set1: {set1}")
        # print(f"set2: {set2}")
        # print(f"intersection: {len(set1.intersection(set2))}")
        
        if len(set1) == 0 and len(set2) == 0:
            return 1.0  # Both sets are empty, perfect match
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
def cosine_similarity(obs_embedding, query_embedding):
    """Calculate cosine similarity between observation and query embeddings using numpy.
    
    Args:
        obs_embedding: numpy array or torch tensor of shape (batch_size, embedding_dim)
        query_embedding: numpy array or torch tensor of shape (batch_size, embedding_dim)
    
    Returns:
        similarity: numpy array of shape (batch_size,)
    """
    from numpy.linalg import norm
    
    # Convert torch tensors to numpy if needed
    if isinstance(obs_embedding, torch.Tensor):
        obs_embedding = obs_embedding.cpu().numpy()
    if isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.cpu().numpy()
    
    # Ensure inputs are 2D
    if obs_embedding.ndim == 1:
        obs_embedding = obs_embedding.reshape(1, -1)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
        
    similarity = np.sum(obs_embedding * query_embedding, axis=1) / (
        norm(obs_embedding, axis=1) * norm(query_embedding, axis=1)
    )
    return similarity



# Copy from Video R1 accuracy_reward
# Video-R1/src/r1-v/grpo.py
def reward_fn(sample, output_ans, question_type, step_id=None, tool_using_num=0):
    try:
        #output_ans = model_output
        gt_ans = sample.get("solution", "")
        if "<answer>" in gt_ans and "</answer>" in gt_ans:
            gt_ans = gt_ans.split("<answer>")[1].split("</answer>")[0]

        if question_type == "multiple choice":
            #print(f"[Debug] multiple choice, output_ans: {output_ans}, gt_ans: {gt_ans}")
            base_score = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0

            if step_id is not None:
                if base_score == 1.0:
                    base_score = base_score - 0.1 * (step_id - 1)

            return base_score

        # free-form reward: Video-R1/src/r1-v/grpo.py
        elif question_type == "free-form":

            assert False, "free-form reward is not implemented"

        elif question_type == "numerical":
            gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
            out_has_decimal = ("." in output_ans) or ("," in output_ans)

            if gt_has_decimal != out_has_decimal:
                return 0.0

            gt_number = float(gt_ans)
            out_number = float(output_ans)

            if gt_number is None or out_number is None:
                return 0.0

            #print(f"[Debug-0712] - round(gt_number, 2): {round(gt_number, 2)}, round(out_number, 2): {round(out_number, 2)}")

            #if gt_number == out_number:
            return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
        # elif question_type == "regression":
        #     gt_number = normalize_number(gt_ans)
        #     out_number = normalize_number(output_ans)
        #     if gt_number is None or out_number is None:
        #         return 0.0
        #     mra = mean_relative_accuracy(out_number, gt_number)
        #     return mra
        else:
            return 0.0
    except Exception as e:
        return 0.0
