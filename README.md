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
When training SVM for FSA, you can run the file `svm.py`. After you complete the parameters in python file, enter the following command at the terminal：

```python
python svm.py
```

Or you can add `nohup` to specifu the location of the output file.

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

The speed of Transformers:

![image](https://github.com/cczhou-nju/Financial-sentiment-analysis-NLP-Transformers/image/speed.png)