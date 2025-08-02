from tqdm import tqdm
import torch
import clip
import gc


def get_gpu_memory_usage():
    """
    Lấy thông tin sử dụng bộ nhớ GPU hiện tại.
    Trả về: (allocated_memory_GB, max_memory_GB)
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
        return allocated, max_allocated
    return 0, 0

def reset_peak_memory():
    """
    Reset peak memory stats để đo chính xác cho từng giai đoạn.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def print_memory_usage(stage_name):
    """
    In thông tin sử dụng bộ nhớ cho một giai đoạn cụ thể.
    """
    current, peak = get_gpu_memory_usage()
    print(f"[{stage_name}] Current VRAM: {current:.2f} GB, Peak VRAM: {peak:.2f} GB")
    return peak

def clear_cache():
    """
    Dọn dẹp cache GPU để có thông số chính xác hơn.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    
    return acc


def clip_classifier(classnames, templates, clip_model):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            prompts = [template.format(classname.replace('_', ' ')) for template in templates]
            tokenized_prompts = clip.tokenize(prompts).cuda()
            text_embeddings = clip_model.encode_text(tokenized_prompts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            mean_text_embedding = text_embeddings.mean(dim=0)
            mean_text_embedding /= mean_text_embedding.norm()
            clip_weights.append(mean_text_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
        
    return clip_weights



def pre_load_features(clip_model, loader):
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu())
            labels.append(target.cpu())
        features, labels = torch.cat(features), torch.cat(labels)
    
    return features, labels

