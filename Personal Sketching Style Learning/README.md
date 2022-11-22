# Personal Sketching Style Learning
---

Pytorch implementation of four neural network based domain adaptation techniques, i.e., DeepCORAL, DDC, CDAN and CDAN+E, for personal sketching style learning. 

The code repository is constructed based on the released code repository [Deep-Unsupervised-Domain-Adaptation](https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation). If you use any content of this repo for your work, please cite the following bib entry:

	@article{preciado2021evaluation,
		title={Evaluation of Deep Neural Network Domain Adaptation Techniques for Image Recognition},
		author={Preciado-Grijalva, Alan and Muthireddy, Venkata Santosh Sai Ramireddy},
		journal={arXiv preprint arXiv:2109.13420},
		year={2021}
	}
	
**Training and inference**
---

To train the model in your computer you must download the [**SketchIME-PSSL**](https://drive.google.com/file/d/1T9CaF02Tt3hf6MdM3Mqm3sg9G06s60E-/view?usp=share_link) dataset and put it in your DataSet folder. 

Execute training of a method by going to its folder (e.g. DeepCORAL):

```
cd DeepCORAL/
python main.py --epochs 50 --batch_size_source 48 --batch_size_target 48 --name_source sketch/source --name_tgttrain sketch/ALL5/66/train --name_tgttest sketch/ALL5/66/test --result_path logs/sketch/ --adapt_domain
python main.py --epochs 50 --batch_size_source 48 --batch_size_target 48 --name_source sketch/source --name_tgttrain sketch/ALL5/66/train --name_tgttest sketch/ALL5/66/test --result_path logs/sketch/ --adapt_domain --target_supervised
```

The following is a list of the arguments the user can provide:

* ```--epochs``` number of training epochs
* ```--batch_size_source``` batch size of source data
* ```--batch_size_target``` batch size of target data
* ```--name_source``` name of source dataset
* ```--name_tgttrain``` name of target dataset for domain adaptation training
* ```--name_tgttest``` name of target dataset for domain adaptation testing
* ```--num_classes``` no. classes in dataset
* ```--load_model``` flag to load pretrained model
* ```--adapt_domain``` bool argument to train with or without specific transfer loss

**Requirements**
---
* tqdm
* PyTorch
* matplotlib
* numpy
* pickle
* scikit-image
* torchvision

**References**
---
- [Evaluation of Deep Neural Network Domain Adaptation Techniques for Image Recognition](https://arxiv.org/abs/2109.13420) paper
- [DeepCORAL](https://arxiv.org/abs/1607.01719) paper
- [DDC](https://arxiv.org/abs/1412.3474) paper
- [CDAN](https://arxiv.org/abs/1705.10667) paper
