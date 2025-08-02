# @title lora.py

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import clip  

from utils import *

from loralib.utils import apply_lora, load_adapter, save_lora, load_lora, apply_adapter, mark_only_lora_as_trainable, get_lora_parameters, save_adapter
# ==============================================================================


def evaluate(args, clip_model, loader, dataset):
    """
    Đánh giá hiệu suất của mô hình trên một tập dữ liệu cụ thể.
    Hàm này chung cho cả mô hình gốc và mô hình đã được tinh chỉnh.
    """
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0]
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)

    acc /= tot_samples
    return acc


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    """
    Hàm chính điều phối toàn bộ quy trình, đã tích hợp Prompt Ensembling và Context-Aware Adapter.
    """
    print("\nMoving model and data to GPU for evaluation...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        clip_model.cuda()
    
    # === [THAY ĐỔI 1]: Tích hợp Prompt Ensembling và Tính toán Task Vector ===
    templates = dataset.template
    print(f"\nSử dụng {len(templates)} templates cho Prompt Ensembling.")

    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, templates, clip_model)

    print("Calculating task-level context vector for Context-Aware Adapter.")
    with torch.no_grad():
        task_vector = textual_features.mean(dim=1).cuda()
    print(f"    Task vector created with shape: {task_vector.shape}")
    # =====================================================================

    print("\nLoading visual features and labels from test set for zero-shot evaluation.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_loading = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\n--- Resource Metrics (Loading Phase) ---")
        print(f"    Peak VRAM for zero-shot eval setup: {peak_vram_loading:.3f} GB")
        print(f"----------------------------------------\n")
        
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    del test_features, test_labels
    torch.cuda.empty_cache()

    list_adapter_layers = []
    if args.adapter == 'lora':
        list_adapter_layers = apply_lora(args, clip_model)
    elif args.adapter in ['singlora', 'gmhsinglora']:
        # Giả định apply_adapter đã được cập nhật để truyền embed_dim
        list_adapter_layers = apply_adapter(args, clip_model)
    
    clip_model = clip_model.cuda()

    if args.eval_only:
        print(f"\nEvaluation-only mode for {args.adapter.upper()} adapter.")
        if args.adapter == 'lora':
            load_lora(args, list_adapter_layers)
        elif args.adapter in ['singlora', 'gmhsinglora']:
            load_adapter(args, list_adapter_layers)

        acc_test = evaluate(args, clip_model, test_loader, dataset)
        print(f"**** Test accuracy: {acc_test:.2f}. ****\n")
        return

    mark_only_lora_as_trainable(clip_model)
    optimizer_params = get_lora_parameters(clip_model)
    print(f"Number of trainable parameters: {sum(p.numel() for p in optimizer_params)}")
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)

    total_iters = args.n_iters * args.shots
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    print(f"\nStarting {args.adapter.upper()} fine-tuning for {total_iters} iterations...")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    vram_checked_after_first_step = False
    scaler = torch.amp.GradScaler(device='cuda')
    count_iters = 0

    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.

        if args.encoder == 'vision':
            text_features_train = textual_features.clone().half()

        for i, (images, target) in enumerate(tqdm(train_loader, desc=f"Iter {count_iters}/{total_iters}")):
            images, target = images.cuda(), target.cuda()
            
            if args.encoder == 'text' or args.encoder == 'both':
                # Tính lại text_features với Prompt Ensembling để phản ánh adapter đã được huấn luyện
                text_features_train = clip_classifier(dataset.classnames, templates, clip_model)

            # === [THAY ĐỔI 2]: Truyền task_vector vào Image Encoder ===
            # Giả định encode_image đã được sửa để nhận 'task_vector'
            adapter_kwargs = {}
            if args.adapter == 'gmhsinglora': # Chỉ truyền khi dùng adapter context-aware
                 adapter_kwargs['task_vector'] = task_vector

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images, **adapter_kwargs)
            else: # encoder == 'text'
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        # Vẫn truyền task_vector ngay cả khi không huấn luyện vision encoder
                        # để đảm bảo logic nhất quán nếu mạng gating được áp dụng.
                        image_features = clip_model.encode_image(images, **adapter_kwargs)
            # =========================================================

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ text_features_train.t()
            
            # === [THAY ĐỔI 3]: Tích hợp Label Smoothing (Cải tiến đề xuất) ===
            loss = F.cross_entropy(cosine_similarity, target, label_smoothing=0.1) # Thêm label_smoothing
            # ===============================================================

            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if not vram_checked_after_first_step:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    peak_vram_first_step = torch.cuda.max_memory_allocated() / (1024**3)
                    print(f"\n--- VRAM Checkpoint (During Training) ---")
                    print(f"    Peak VRAM after first optimization step: {peak_vram_first_step:.3f} GB")
                    print(f"-----------------------------------------\n")
                vram_checked_after_first_step = True

            count_iters += 1
            if count_iters >= total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_finetuning_time = end_time - start_time
    
    peak_vram_total_finetuning = 0
    if torch.cuda.is_available():
        peak_vram_total_finetuning = torch.cuda.max_memory_allocated() / (1024**3)

    print("\n--- Final Resource Metrics (Finetuning Phase) ---")
    print(f"Total Finetuning Time: {total_finetuning_time:.2f} seconds")
    print(f"Peak VRAM during entire Finetuning: {peak_vram_total_finetuning:.3f} GB")
    print("---------------------------------------------------\n")

    print("\nFine-tuning finished.")

    acc_test = evaluate(args, clip_model, test_loader, dataset)
    print(f"**** Final test accuracy with {args.adapter.upper()}: {acc_test:.2f}. ****\n")
    if args.save_path is not None:
        print(f"Saving {args.adapter.upper()} weights to {args.save_path}...")
        if args.adapter == 'lora':
            save_lora(args, list_adapter_layers)
        elif args.adapter in ['singlora', 'gmhsinglora']:
            save_adapter(args, clip_model)

    return acc_test