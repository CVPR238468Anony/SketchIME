## FACT

#### Requirement

* [PyTorch-1.12 and torchvision](https://pytorch.org/)

* numpy-1.21.5
* tqdm

#### Dataset

Download dataset from [Here](https://drive.google.com/file/d/1zU_exwxQTI3V2hoTQHYL0BH2xD19SjE4/view?usp=share_link).

Unzip the file and put "sketch" into "FACT/data/".

#### Training

```sh
cd FACT
sh train_sketch.sh
```



## LIMIT

#### Requirement

* [PyTorch-1.12 and torchvision](https://pytorch.org/)
* numpy-1.21.5
* tqdm

#### Dataset

Download dataset from [Here](https://drive.google.com/file/d/1zU_exwxQTI3V2hoTQHYL0BH2xD19SjE4/view?usp=share_link).

Unzip the file and put "sketch" into "LIMIT/data/".

#### Pretrain Model

Put pretrain model into "LIMIT/params". There are two ways to get pretrain model:

1. Download the model from [Here](https://drive.google.com/file/d/1vx5obUnBXmdpuXjxNKDT1CTc7KE-QjIz/view?usp=share_link).

2. Run:

   ```sh
   cd LIMIT
   sh pretrain_sketch.sh
   ```

   Then, you can find pretrain model named  "session0__max_acc.pth" from "checkpoint" directory.

#### Training

```sh
cd LIMIT
sh train_sketch.sh
```



## SPPR

#### Requirement

* [PyTorch-1.12 and torchvision](https://pytorch.org/)
* numpy-1.21.5
* tqdm

#### Dataset

Download dataset from [Here](https://drive.google.com/file/d/1zU_exwxQTI3V2hoTQHYL0BH2xD19SjE4/view?usp=share_link).

Unzip the file and put "sketch" into "SPPR/datasets/sketch/".

#### Training

```sh
cd SPPR
sh train_sketch.sh
```



## CFSCIL

#### Requirement

* [PyTorch-1.12 and torchvision](https://pytorch.org/)
* numpy-1.21.5
* dotmap
* tqdm

#### Dataset

Download dataset from [Here](https://drive.google.com/file/d/1zU_exwxQTI3V2hoTQHYL0BH2xD19SjE4/view?usp=share_link).

Unzip the file and put "sketch" into "CFSCIL/code/data/".

#### Training

```sh
cd CFSCIL/code
sh train_sketch.sh
```



## CDAN

#### Requirement

* [PyTorch-1.12 and torchvision](https://pytorch.org/)
* numpy-1.21.5
* tqdm

#### Dataset

Download dataset from [Here](https://drive.google.com/file/d/1zU_exwxQTI3V2hoTQHYL0BH2xD19SjE4/view?usp=share_link).

Unzip the file and put "sketch" into "CDAN/data/".

#### Pretrain Model

Put all pretrain models into "CDAN/params".There are two ways to get pretrain models.

1. Download the model from [Here](https://drive.google.com/file/d/1vx5obUnBXmdpuXjxNKDT1CTc7KE-QjIz/view?usp=share_link).
2. 
2. Train **FACT** to get models. You can find 17 models named "session*__max_acc.pth" for each sessions respectively.

#### Training

```sh
cd CDAN
sh train_sketch.sh
```

