# Recall
just like run.sh
```shell
python prepare_dataset.py
python cal_itemcf_sim.py
python recall_itemcf.py
python recall_hot_and_cold.py
python recall.py
python rank_feature.py
python rank_lgb
```

```python
user_multi_recall_dict =  {'itemcf_sim_itemcf_recall': {},
                           'embedding_sim_item_recall': {},
                           'youtubednn_recall': {},
                           'youtubednn_usercf_recall': {}, 
                           'cold_start_recall': {}}
```


itemcf: hitreate5:0.0947438206249629, hitreate10:0.11780689081949694, hitreate20:0.14912429993469356, hitreate40:0.18974490906572203, hitreate50:0.2054184559973086