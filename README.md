# Simple Quantum-GCN pytroch implementation

## Example

1. download dataset from [google-drive](https://drive.google.com/drive/u/0/folders/1xRTs9nTDK_d87NDQIt5hIzr4cYAiy8EW) and move those data to data directory as follows.

```
├── config/
├── dataset/
├── trainer/
├── util/
├── data/
    └── csv/
        ├── Amazon_Books.csv
        ├── gowalla.csv
        ├── ml-100k.csv
        ├── ml-1m.csv
        └── yelp.csv
```

2. execute main.py for each dataset and model.

### datasets:

- ml-100k
- ml-1m
- yelp
- gowalla
- Amazon_Books


### models:

- SimpleQGCN (SQ-GCN)
- SimpleQGCN_p (SQ-GCN-p)
- SimpleQGCN_c (SQ-GCN-c)
- kNNSimpleQGCN (kNN-SQ-GCN)
- AttentionkNNSimpleQGCN (Attention-kNN-SQ-GCN)

```
python main.py --dataset ml-100k --model SimpleQGCN
python main.py --dataset ml-1m --model SimpleQGCN
python main.py --dataset ylep --model SimpleQGCN
python main.py --dataset gowalla --model SimpleQGCN_c
python main.py --dataset ml-100k --model SimpleQGCN
python main.py --dataset ml-100k --model kNNSimpleQGCN
python main.py --dataset ml-100k --model AttentionkNNSimpleQGCN
```

## Environment

- python == 3.7.11
- pytorch == 1.10.0
- numpy == 1.19.2
- scikit-learn == 1.0.1
