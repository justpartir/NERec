# NERec
Code for paper "Neighbor-based Enhanced GNN for Social Recommendation via more Informative Neighbor Aggregation"

github: 
- [Code for NERec](https://github.com/justpartir/NERec)

(due to space limitations, we upload the newest code and whole datasets with our experiments in this link)


# Requirements:

requirements.txt


# How to use:

## First Step
(if you want to change dataset, please use the step)

```
python preprocess_data.py
```
or if you use jupyter notebook, you can use "preprocess_data.ipynb";

you will find a new "pkl" file in "data" path (such as "ciao_dense.pkl", we use it in our discussions of EXPERIMENT part)

## Second Step

(param_parser.py : each parameter's setting, change the parameter to conduct experiments.)
```
python Run_NERec_examples.py
```
or if you use jupyter notebook, you can use "Run_NERec_examples.ipynb".

## Tip

Raw Datasets (Ciao and Epinions)  can be downloaded at [http://www.cse.msu.edu/~tangjili/trust.html](http://www.cse.msu.edu/~tangjili/trust.html)


# Useful Resources

- [Code for some traditional and social recommendation methods](https://github.com/hongleizhang/RSAlgorithms)
- [Code for GraphRec](https://github.com/wenqifan03/GraphRec-WWW19) 
- [Paper summary for social recommendation](https://github.com/Weizhi-Ying/Social-Recommendation)
- [Code for GonsisRec](https://github.com/YangLiangwei/ConsisRec)

If you use this code, please cite:


paper link: https://arxiv.org/pdf/2105.02254  
@inproceedings{yang2021consisrec,  
	title={ConsisRec: Enhancing GNN for Social Recommendation viaConsistent Neighbor Aggregation},  
	author={Yang, Liangwei and Liu, Zhiwei and Dou, Yingtong and Ma, Jing and Philip S. Yu},  
	journal={Proceedings of the 44th international ACM SIGIR conference on Research and development in information retrieval},  
	year={2021},  
	publisher={ACM}  
}

paper link: https://arxiv.org/pdf/1902.07243.pdf  
@inproceedings{fan2019graph,  
  title={Graph neural networks for social recommendation},  
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},  
  booktitle={The World Wide Web Conference},  
  pages={417--426},  
  year={2019}  
}
