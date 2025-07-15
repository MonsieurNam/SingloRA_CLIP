```
apt install python3 python3-pip git zip unzip

```
```
%cd /content/CLIP-LoRA/
# 'dtd', 'eurosat', 'caltech101', 'food101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 'sun397', 'ucf101', 'imagenet', 'fgvc'
!python3 main.py \
--dataset caltech101 \
--root_path /content/DATA \
--shots 1 \
--adapter singlora \
--ramp_up_steps 100 \
--params q k v \
--lr 2e-4 \
--position all \
--save_path ./checkpoints
```

or
```
chmod +x run_all.sh
./run_all.sh
```
```
chmod +x scan.sh
./scan.sh
```