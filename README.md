# Financial-sentiment-analysis-NLP-Transformers

## Introduction
This project is the implement of "SEEKING BETTER NLP TRANSFORMERS FOR FINANCIAL SENTIMENT ANALYSIS". We released our datasets and code. One of the two datasets is labeled by our experts, consisting of Reddit-News. We evaluate different NLP Transformers on the two financial datasets. Based on our experimental results, traders and investors can choose better model for financial sentiment analysis (FSA). 
This project is the implement of "SEEKING BETTER NLP TRANSFORMERS FOR FINANCIAL SENTIMENT ANALYSIS". We evaluate different NLP Transformers on two financial datasets. Based on our experimental results, traders and investors can choose better model for financial sentiment analysis (FSA). 

## Set up
Tip: You need to install pytorch before starting this project.
## Dataset
One of the two datasets is labeled by our experts, consisting of Reddit-News that may have influence on financial markets. Another dataset is Financial Phrase-Bank dataset. You can find the two datasets under folder `dataset`.  

## Environment
Our code is implemented in `python=3.8.3,pytorch=1.6.0(GPU version)`. 

This project is implemented by transfromers library of Huggingface(https://github.com/huggingface/transformers). You need to install transformers. The process is as following:
## Set up
This project is implemented by transfromers library of [Huggingface](https://github.com/huggingface/transformers). You need to install transformers. The process is as following:

```bash
git clone https://github.com/cczhou-nju/Financial-sentiment-analysis-NLP-Transformers.git
cd Financial-sentiment-analysis-NLP-Transformers
pip install -e .
```

Or you can directly download transformers library or the whole project(following the installation process in https://github.com/huggingface/transformers). And you need to download the dataset folder in our project and move it to the path of transformers project. 
Or you can directly download transformers library or the whole project by following the installation process [here](https://github.com/huggingface/transformers). Then you need to download the dataset folder in our project and move it to the path of your transformers project. 

## Usage
### Transforerms
When training Transformers for FSA, you can just run the file `run_classification.sh`. After you complete the parameters in bash file, enter the following command at the terminal：
When training Transformers for FSA, you need to download the pre-trained model you need [here](https://huggingface.co/models). 

```bash
bash run_classification.sh
export TASK_NAME=SST-3
export MODEL=[pre-trained model in huggingface ]
export GLUE_DIR=[path to your dataset]
export W_DIR=[the path to your model]
export OUT_DIR=[path to your output-dir]
CUDA_VISIBLE_DEVICES=[GPU used] python three_classification.py\
    --model_name_or_path $W_DIR \
    --task_name $TASK_NAME \
    --do_eval \
    --do_train \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_device_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --learning_rate 2e-5 \
    --num_train_epochs 5.0 \
    --seed 42 \
    --overwrite_output_dir \
    --output_dir $OUT_DIR/
```

### SVM
When training SVM for FSA, you can run the file `svm.py`. After you complete the parameters in python file, enter the following command at the terminal：
When training SVM for FSA, you can run the file `svm.py`. You can change the parameters in this python file.

```python
python svm.py
```


## Benchmark
The results in Reddit-news dataset:

| Model                 | Precison(%) | Recall(%)   | F1(%)   |  
| --------------------- | ----------- | ----------- | ------- |  
| SVM                   | 72.79       | 71.78       | 71.73   |  
| bert-base-cased       | 96.39       | 96.36       | 96.36   |  
| bert-large-cased      | 97.02       | 97.00       | 97.00   |  
| xlnet-base-cased      | 90.30       | 89.83       | 89.80   |  
| xlnet-large-cased     | 89.87       | 89.11       | 89.04   |  
| distilbert-base-cased | 95.83       | 95.76       | 95.77   |  
| albert-base-v2        | 95.45       | 95.38       | 95.38   |  
| longformer-base-4096  | 92.76       | 92.30       | 92.32   |  
| roberta-base          | 93.55       | 93.39       | 93.39   |  
| roberta-large         | 96.03       | 95.98       | 95.99   |  
| bart-base             | 94.97       | 94.93       | 94.94   |  
| bart-large            | 95.85       | 95.80       | 95.80   |  
 
The results in Financial Phrase-Bank dataset:

| Model                 | Precison(%) | Recall(%)   | F1(%)   |  
| --------------------- | ----------- | ----------- | ------- |  
| SVM                   | 61.80       | 58.40       | 56.81   |  
| bert-base-cased       | 80.26       | 79.34       | 79.13   |  
| bert-large-cased      | 85.46       | 85.40       | 85.42   |  
| xlnet-base-cased      | 83.15       | 82.92       | 82.84   |  
| xlnet-large-cased     | 74.30       | 74.40       | 74.30   |  
| distilbert-base-cased | 80.32       | 79.89       | 79.79   |  
| albert-base-v2        | 79.52       | 78.79       | 78.51   |  
| longformer-base-4096  | 86.50       | 86.50       | 86.49   |  
| roberta-base          | 83.41       | 83.20       | 83.23   |  
| roberta-large         | 87.69       | 87.60       | 87.59   |  
| bart-base             | 86.15       | 85.67       | 85.73   |  
| bart-large            | 88.31       | 88.15       | 88.18   |  

The speed of Transformers is shown in speed.png

![image](https://github.com/cczhou-nju/Financial-sentiment-analysis-NLP-Transformers/blob/main/speed.png)
