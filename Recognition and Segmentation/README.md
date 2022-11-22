
# SketchRecSeg and SketchIME-SRS dataset

## SketchRecSeg
### 1. SketchIME-SRS Data
***

 Click the [download](https://drive.google.com/file/d/14ErMD-Uo39blVQKzUC5QGEvvsV7nukJz/view?usp=share_link) button to download the normalized data from the SketchIME-SRS dataset, in which the number of points in each sketch is normalized to 200, the password is etvp. There are some items may be used:
| item                       | content                                                                                                                                                                                                                                        | format |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| SketchIME-SRS_train.ndjson | The file is the train set file, including all training sketch information, which are key_id, drawings, recog_label, png_path and component_path.                                                                                               | ndjson |
| SketchIME-SRS_test.ndjson  | The file is the test set file, including all test sketch information, which are key_id, drawings, recog_label, png_path and component_path.                                                                                                    | ndjson |
| SketchIME-SRS_valid.ndjson | The file is the validation set file, including all validated sketch information, which are key_id, drawings, recog_label, png_path and component_path.                                                                                         | ndjson |
| pngs                       | The directory includes all pngs in the training set, verification set, test set.                                                                                                                                                               | png    |
| components                 | The directory has conponent-point relevance files, each file in this dir is corresponding to a sketch, the content in the file is a S*N matrix, where S means component label number(95) and N means the number of the normalized points(200). | npy    |
| class_component_matrix     | The content in the file is a R*S matrix, which reveals the relation between recognation category and segmentaion labels, where R is the recognation number(197) and S is segmentation label number(95).                                        | npy    |
Download the data to the following directory:
* SketchRecSeg
	* data 
		* SRS200
			* train

[Download](https://drive.google.com/file/d/1n6JQeuWGNpr1yWPUXkqYWOLeOKEcMKGt/view?usp=share_link) the pretrained model.
### 2. Environment
***
python 3.8
torch 1.10.2
torch-geometric 2.0.4
### 3. Using
Enter the SketchRecSeg directory
  **train**
 python `train.py`  --shuffle --stochastic
 **test**
 python `evaluate.py` --timestamp train_time --which-epoch bestloss
