# inc-all
python main.py --dataset inc --dataroot data --lr 0.1 --ft --pretrain_dir params --epochs 50 >> fact-cdan-inc-all.txt

# all-all
python main.py --dataset all --dataroot data --lr 0.1 --ft --pretrain_dir params --epochs 50 >> fact-cdan-all-all.txt

# inc-fc
# python main.py --dataset inc --dataroot data --lr 0.1 --pretrain_dir params --epochs 50 >> fact-cdan-inc-fc.txt

# all-fc
python main.py --dataset all --dataroot data --lr 0.1 --pretrain_dir params --epochs 50 >> fact-cdan-all-fc.txt