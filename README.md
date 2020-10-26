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
When training Transformers for FSA, you can just run the file three_classification.sh. Enter the following command at the terminal：

```bash
bash 
```
