# Financial-sentiment-analysis-NLP-Transformers

## Introduction
This project is the implement of "SEEKING BETTER NLP TRANSFORMERS FOR FINANCIAL SENTIMENT ANALYSIS". We released our datasets and code. One of the two datasets is labeled by our experts, consisting of Reddit-News. We evaluate different NLP Transformers on the two financial datasets. Based on our experimental results, traders and investors can choose better model for financial sentiment analysis (FSA). 

## Set up
Tip: You need to install pytorch before starting this project.

This project is implemented by transfromers library of Huggingface(https://github.com/huggingface/transformers). You need to install transformers. The process is as following:

```bash
git clone https://github.com/cczhou-nju/Financial-sentiment-analysis-NLP-Transformers.git
cd Financial-sentiment-analysis-NLP-Transformers
pip install -e .
```

Or you can directly download transformers library or the whole project(following the installation process in https://github.com/huggingface/transformers). And you need to download the dataset folder in our project and move it to the path of transformers project. 

## Usage
### Transforerms
When training Transformers for FSA, you can just run the file `run_classification.sh`. After you complete the parameters in bash file, enter the following command at the terminal：

```bash
bash run_classification.sh
```

### SVM
When training Transformers for FSA, you can run the file `svm.py`. After you complete the parameters in python file, enter the following command at the terminal：

```python
python svm.py
```

## Benchmark
The results in Reddit-news dataset:
 Model  | Precison  | Recall  | F1  |  
 ---- | ----- | ------  
 SVM  | 72.79  | 71.78  | 71.73
 bert-base-cased  | 96.39  |96.36  |96.36
 bert-large-cased  | 97.02  | 97.00  | 97.00
 xlnet-base-cased  | 90.30  | 89.83  | 89.80
 xlnet-large-cased  | 89.87  | 89.11  | 89.04
 distilbert-base-cased  | 95.83  | 95.76  | 95.77
 albert-base-v2  | 95.45  | 95.38  | 95.38
 longformer-base-4096  | 92.76  | 92.30  | 92.32
 roberta-base  | 93.55  | 93.39  | 93.39
 roberta-large  | 96.03  | 95.98  | 95.99
 bart-base  | 94.97  | 94.93  | 94.94
 bart-large  | 95.85  | 95.80  | 95.80
 
