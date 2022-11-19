
# SketchRecSeg and SketchIME-SRS dataset
## 1. SketchIME-SRS dataset
***
SketchIME-SRS dataset includes197 categories and 95 component labels for recogniation and segmentation tasks. Click the [download](https://pan.baidu.com/s/1y_GrOvMpvLhpRx07V4veeg) button to download the dataset, the password is etvp. The content of the dataset is shown in the following table.
| item       | content                                                                                                                                                                                                                       | format |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| svgs       | This directory includes all original collected svgs.                                                                                                                                                                          | svg    |
| pngs       | This directory includes all pngs are converted by svgs, each png is 3-channels.                                                                                                                                               | png    |
| label_data | This directory includes all point-level information of each sketch, each point is seen as a Quaternion(x,y,st,se), where x, y is absolute coordinates, st is the stroke order num and se is the segmentation label order num. | txt    |
## 2. SketchRecSeg
### 1. Data
***

 Click the [download](https://pan.baidu.com/s/1y_GrOvMpvLhpRx07V4veeg) button to download the normalized data from the SketchIME-SRS dataset, in which the number of points in each sketch is normalized to 200. There are some items may be used:
| item                       | content                                                                                                                                                                                                                                        | format | download                          |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | --------------------------------- |
| SketchIME-SRS_train.ndjson | The file is the train set file, including all training sketch information, which are key_id, drawings, recog_label, png_path and component_path.                                                                                               | ndjson | [download](https://www.baidu.com) |
| SketchIME-SRS_test.ndjson  | The file is the test set file, including all test sketch information, which are key_id, drawings, recog_label, png_path and component_path.                                                                                                    | ndjson | [download](https://www.baidu.com) |
| SketchIME-SRS_valid.ndjson | The file is the validation set file, including all validated sketch information, which are key_id, drawings, recog_label, png_path and component_path.                                                                                         | ndjson | [download](https://www.baidu.com) |
| pngs                       | The directory includes all pngs in the training set, verification set, test set.                                                                                                                                                               | png    | [download](https://www.baidu.com) |
| components                 | The directory has conponent-point relevance files, each file in this dir is corresponding to a sketch, the content in the file is a S*N matrix, where S means component label number(95) and N means the number of the normalized points(200). | npy    | [download](https://www.baidu.com) |
| class_component_matrix     | The content in the file is a R*S matrix, which reveals the relation between recognation category and segmentaion labels, where R is the recognation number(197) and S is segmentation label number(95).                                        | npy    | [download](https://www.baidu.com) |
Download the data to the following directory:
* SketchRecSeg
	* data 
		* SRS200
			* train
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