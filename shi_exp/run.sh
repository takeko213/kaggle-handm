#!/bin/bash
python prepare_dataset.py
python recall_itemcf.py
python recall_hot.py
python recall.py
python rank_feature.py
python rank_lgb