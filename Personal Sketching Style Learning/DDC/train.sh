#!/bin/bash
python main.py --epochs 50 --batch_size_source 48 --batch_size_target 48 --name_source sketch/source --name_tgttrain sketch/test1/56/train --name_tgttest sketch/test1/56/test --result_path logs/sketch/ --adapt_domain
python main.py --epochs 50 --batch_size_source 48 --batch_size_target 48 --name_source sketch/source --name_tgttrain sketch/test1/56/train --name_tgttest sketch/test1/56/test --result_path logs/sketch/ --adapt_domain --target_supervised
