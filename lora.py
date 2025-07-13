# @title lora.py
# %%writefile /content/CLIP-LoRA/lora.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import clip  # Giả sử thư mục clip đã có trong dự án

from utils import *

# ==============================================================================
# Import các hàm tiện ích cho cả LoRA và SingLoRA
# ==============================================================================

# Các hàm này hoạt động cho cả hai vì chúng tìm kiếm tiền tố 'lora_'
from loralib.utils import mark_only_lora_as_trainable, get_lora_parameters

# Import các hàm cụ thể cho từng loại adapter
from loralib.utils import apply_lora, save_lora, load_lora
from loralib.utils import apply_singlora, save_singlora, load_singlora
# ==============================================================================


def evaluate(args, clip_model, loader, dataset):
    """
    Đánh giá hiệu suất của mô hình trên một tập dữ liệu cụ thể.
    Hàm này chung cho cả mô hình gốc và mô hình đã được tinh chỉnh.
    """
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0]
        # Tokenize classnames
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        # Chuẩn hóa các đặc trưng văn bản
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)

            # Chuẩn hóa các đặc trưng hình ảnh
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Tính toán độ tương đồng cosine
            cosine_similarity = image_features @ text_features.t()

            # Tính độ chính xác
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)

    acc /= tot_samples
    return acc


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    """
    Hàm chính điều phối toàn bộ quy trình tinh chỉnh và đánh giá,
    hỗ trợ cả LoRA và SingLoRA.
    """

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load test features để đánh giá zero-shot
    print("\nLoading visual features and labels from test set for zero-shot evaluation.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Đánh giá Zero-shot CLIP làm baseline
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    # Giải phóng bộ nhớ GPU
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    # ==============================================================================
    # Logic điều kiện để áp dụng Adapter
    # ==============================================================================
    list_adapter_layers = []
    if args.adapter == 'lora':
        print(f"Applying LoRA adapter with r={args.r}, alpha={args.alpha}...")
        list_adapter_layers = apply_lora(args, clip_model)
    elif args.adapter == 'singlora':
        print(f"Applying SingLoRA adapter with r={args.r}, alpha={args.alpha}, T={args.ramp_up_steps}...")
        list_adapter_layers = apply_singlora(args, clip_model)

    clip_model = clip_model.cuda()


    import copy
    clip_model_original = copy.deepcopy(clip_model)
    clip_model_original.cuda().eval()

    # Áp dụng adapter
    list_adapter_layers = []
    if args.adapter == 'singlora':
        print(f"Applying SingLoRA adapter...")
        list_adapter_layers = apply_singlora(args, clip_model)

    clip_model = clip_model.cuda()
    clip_model.eval() # Chuyển sang chế độ eval để so sánh

    # ==================== DEBUG LOGS START ====================
    print("\n--- DEBUG SANITY CHECK: Comparing MHA outputs ---")
    # Lấy một khối attention từ vision encoder để kiểm tra
    # original_block = clip_model_original.visual.transformer.resblocks[0]
    # singlora_block = clip_model.visual.transformer.resblocks[0]

    # # Tạo một tensor đầu vào ngẫu nhiên
    # dummy_input = torch.randn(197, 1, 768).cuda() # (seq_len, batch_size, embed_dim) for ViT-B/16

    # # Lấy đầu ra từ MHA gốc
    # with torch.no_grad():
    #     output_original, _ = original_block.attn(dummy_input, dummy_input, dummy_input, need_weights=False)

    # # Lấy đầu ra từ MHA của chúng ta (chưa có cập nhật SingLoRA vì u(0)=0)
    # with torch.no_grad():
    #     output_singlora, _ = singlora_block.attn(dummy_input, dummy_input, dummy_input, need_weights=False)

    # # So sánh sự khác biệt
    # difference = torch.dist(output_original, output_singlora, p=2).item()
    # print(f"Difference between original MHA and SingLoRA MHA (at init): {difference:.8f}")
    # if difference < 1e-5:
    #     print("SANITY CHECK PASSED: Initial outputs are identical.")
    # else:
    #     print("SANITY CHECK FAILED: There is a logic error in MHA layer replacement.")
    # print("--------------------------------------------------\n")
    # ===================== DEBUG LOGS END =====================

    # Chế độ chỉ đánh giá (không huấn luyện)
    if args.eval_only:
        print(f"\nEvaluation-only mode for {args.adapter.upper()} adapter.")
        if args.adapter == 'lora':
            load_lora(args, list_adapter_layers)
        elif args.adapter == 'singlora':
            load_singlora(args, list_adapter_layers)

        acc_test = evaluate(args, clip_model, test_loader, dataset)
        print(f"**** Test accuracy: {acc_test:.2f}. ****\n")
        return

    # Đóng băng các tham số không phải của adapter
    mark_only_lora_as_trainable(clip_model)

    # Thiết lập optimizer chỉ với các tham số của adapter
    optimizer_params = get_lora_parameters(clip_model)
    print(f"Number of trainable parameters: {sum(p.numel() for p in optimizer_params)}")
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)

    # Thiết lập scheduler
    total_iters = args.n_iters * args.shots
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    # Bắt đầu quá trình huấn luyện
    print(f"\nStarting {args.adapter.upper()} fine-tuning for {total_iters} iterations...")
    scaler = torch.amp.GradScaler(device='cuda')
    count_iters = 0

    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.

        # Thiết lập text_features một lần nếu chỉ fine-tune vision encoder
        if args.encoder == 'vision':
            text_features_train = textual_features.clone().half()

        # Vòng lặp qua từng batch dữ liệu huấn luyện
        for i, (images, target) in enumerate(tqdm(train_loader, desc=f"Iter {count_iters}/{total_iters}")):

            images, target = images.cuda(), target.cuda()

            # Nếu text encoder được fine-tune, cần tính lại text_features mỗi bước
            if args.encoder == 'text' or args.encoder == 'both':
                template = dataset.template[0]
                texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features_train = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            # Tính image_features
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else: # Nếu chỉ fine-tune text encoder, không cần tính lại gradient cho vision
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Tính loss
            cosine_similarity = logit_scale * image_features @ text_features_train.t()
            loss = F.cross_entropy(cosine_similarity, target)

            # Cập nhật các chỉ số
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]

            # Backward pass và cập nhật trọng số
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=1.0)
            # ==================== DEBUG LOGS START ====================
            # if count_iters < 5: # Chỉ kiểm tra ở vài bước đầu
            #     print(f"\n--- DEBUG GRADIENTS (Iter {count_iters}) ---")
            #     found_adapter = False
            #     for name, param in clip_model.named_parameters():
            #         # Chỉ kiểm tra gradient của các tham số adapter
            #         if 'lora_A' in name and param.requires_grad:
            #             if param.grad is not None:
            #                 grad_norm = torch.linalg.norm(param.grad.float()).item()
            #                 print(f"Gradient norm for {name}: {grad_norm:.6f}")
            #             else:
            #                 print(f"Gradient for {name} is None.")
            #             found_adapter = True
            #     if not found_adapter:
            #         print("No adapter parameters found to check gradients.")
            #     print(f"------------------------------------\n")
            # ===================== DEBUG LOGS END =====================

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            count_iters += 1
            if count_iters >= total_iters:
                break

        # In log sau mỗi epoch (hoặc sau một số lần lặp nhất định)
        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))

    print("\nFine-tuning finished.")

    # Đánh giá cuối cùng trên tập test
    acc_test = evaluate(args, clip_model, test_loader, dataset)
    print(f"**** Final test accuracy with {args.adapter.upper()}: {acc_test:.2f}. ****\n")

    # ==============================================================================
    # Logic điều kiện để lưu Adapter
    # ==============================================================================
    if args.save_path is not None:
        print(f"Saving {args.adapter.upper()} weights to {args.save_path}...")
        if args.adapter == 'lora':
            save_lora(args, list_adapter_layers)
        elif args.adapter == 'singlora':
            save_singlora(args, list_adapter_layers)

    return acc_test