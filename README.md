# ETR: An Efficient Transformer for Re-ranking in Visual Place Recognition


# Feature Extraction 

Please follow the [instruction](https://github.com/tensorflow/models/blob/master/research/delf/INSTALL_INSTRUCTIONS.md) to install the DELG library.

All the instructions below assume the DELG package is installed in [DELG_ROOT](https://github.com/tensorflow/models/tree/master/research/delf).

please first read the [instrcution](https://github.com/tensorflow/models/blob/master/research/delf/delf/python/delg/DELG_INSTRUCTIONS.md), which is writed for the feature extraction of Revisited Oxford/Paris. And make sure that you fully understand the instrcution. 

Next, we take the feature extraction of Tokyo247 as an example to explain the process of the DELG feature extraction.

### Extract the features of Tokyo247
Download the test set of Tokyo247 first, then modify the path of the images in the file `dataset/Tokyo247/tokyo247_db_c.txt` and `dataset/Tokyo247/tokyo247_query_c.txt`.

Then we need copy-paste the file `delg_feature_extract.py` and `delg_feature_extract.sh` to the **DELG_ROOT/delf/python/delg**. use the shell script `delg_feature_extract.sh` to extract features. Note that the scriptes may not work out-of-the-box, you may still need to set the paths of the input/output directories properly.

# Evaluation

### ETR Reranking
Before reranking, we need to generate a initial rank result. Take Tokyo247 as an example, the initial rank file is `dataset/Tokyo247/delg_feats/delg_tokyo247_rank_index.npy`. you can generate a initial rank file by yourself use any other global retrieval methods. Then modify the python script `evaluate_etr.py`, change the dataset_name to dataset that you want to evaluate and run this:

```
python evaluate_etr.py
```


