# @title lora.py
import torch
import torch.nn.functional as F

from losses.center_loss import CenterLoss
from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers
from losses.arcface import ArcFaceLoss
from losses.cosface import CosFaceLoss
from losses.max_entropy import MaximumEntropyLoss


def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0]
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):

    VALIDATION = False

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)

    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    test_features = test_features.cpu()
    test_labels = test_labels.cpu()


    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()

    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    lora_params = get_lora_parameters(clip_model)
    optimizer = torch.optim.AdamW(lora_params, weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0

    center_loss_func = None
    optimizer_center_loss = None
    arcface_loss_func = None
    cosface_loss_func = None
    maxent_loss_func = None

    if args.loss_type == 'ce_center':
        print("Using Cross-Entropy + Center Loss.")
        feature_dim = textual_features.shape[0]
        num_classes = len(dataset.classnames)
        center_loss_func = CenterLoss(num_classes=num_classes, feat_dim=feature_dim, use_gpu=True)
        optimizer_center_loss = torch.optim.SGD(center_loss_func.parameters(), lr=0.5)
    elif args.loss_type == 'arcface':
        print(f"Using ArcFace Loss with s={args.arcface_s} and m={args.arcface_m}.")
        arcface_loss_func = ArcFaceLoss(s=args.arcface_s, m=args.arcface_m)
    elif args.loss_type == 'cosface': # Thêm nhánh này
        print(f"Using CosFace Loss with s={args.arcface_s} and m={args.arcface_m}.")
        cosface_loss_func = CosFaceLoss(s=args.arcface_s, m=args.arcface_m)
    elif args.loss_type == 'ce_maxent': # Thêm nhánh này
        print(f"Using Cross-Entropy + Maximum Entropy Loss with weight={args.maxent_loss_weight}.")
        maxent_loss_func = MaximumEntropyLoss()


    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:
        clip_model.train()
        if center_loss_func:
            center_loss_func.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(tqdm(train_loader)):

            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)

            # Tính toán logits/similarity
            cosine_similarity = image_features @ text_features.t()
            logits = logit_scale * cosine_similarity

            loss = 0.0
            if args.loss_type == 'ce':
                loss = F.cross_entropy(logits, target)
            elif args.loss_type == 'ce_ls':
                loss = F.cross_entropy(logits, target, label_smoothing=args.label_smoothing)
            elif args.loss_type == 'ce_center':
                loss_ce = F.cross_entropy(logits, target)
                loss_center = center_loss_func(image_features, target)
                loss = loss_ce + args.center_loss_weight * loss_center
            elif args.loss_type == 'arcface':
                loss = arcface_loss_func(cosine_similarity, target)
            elif args.loss_type == 'cosface':
                loss = cosface_loss_func(cosine_similarity, target)
            elif args.loss_type == 'ce_maxent': # Thêm nhánh này
                loss_ce = F.cross_entropy(logits, target)
                loss_entropy = maxent_loss_func(logits)
                loss = loss_ce - args.maxent_loss_weight * loss_entropy
            else:
                raise ValueError(f"Unknown loss type: {args.loss_type}")

            acc_train += cls_acc(logits, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]

            optimizer.zero_grad()
            if optimizer_center_loss:
                optimizer_center_loss.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            if optimizer_center_loss:
                for param in center_loss_func.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(1. / scaler.get_scale())
                optimizer_center_loss.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))


        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))


    acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))

    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return