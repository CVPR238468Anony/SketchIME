#!/bin/bash
python main.py --epochs 50 --batch_size_source 48 --batch_size_target 48 --name_source sketch/source --name_tgttrain sketch/test1/56/train --name_tgttest sketch/test1/56/test --result_path logs/CDAN/sketch/ --entropy False
python main.py --epochs 50 --batch_size_source 48 --batch_size_target 48 --name_source sketch/source --name_tgttrain sketch/test1/56/train --name_tgttest sketch/test1/56/test --result_path logs/CDAN/sketch/ --entropy False --target_supervised
python main.py --epochs 50 --batch_size_source 48 --batch_size_target 48 --name_source sketch/source --name_tgttrain sketch/test1/56/train --name_tgttest sketch/test1/56/test --result_path logs/CDAN+E/sketch/ --entropy True
python main.py --epochs 50 --batch_size_source 48 --batch_size_target 48 --name_source sketch/source --name_tgttrain sketch/test1/56/train --name_tgttest sketch/test1/56/test --result_path logs/CDAN+E/sketch/ --entropy True --target_supervised


