import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os
import math
import warnings
import math
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import faiss

save_path = 'result/'


def embdding_sim(item, item_emb_np, topk=10):

    # item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    # item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
    # item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
    item_index.add(item_emb_np)
    sim, idx = item_index.search(item_emb_np, topk)

    # item_sim_dict = defaultdict(dict)
    # for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np)), sim, idx)):
    #     target_raw_id = item_idx_2_rawid_dict[target_idx]
    #     for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
    #         rele_raw_id = item_idx_2_rawid_dict[rele_idx]
    #         item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value

    # pickle.dump(item_sim_dict, open(save_path + 'emb_i2i_sim.pkl', 'wb'))

    # return item_sim_dict


if __name__ == '__main__':

    INPUT_DIR = 'dataset/'
    article_embedding = np.load(INPUT_DIR+'articles.npy')
    artiles = pd.read_parquet(INPUT_DIR + 'articles.parquet')[['article_id']]

    print(article_embedding.shape[1])


    emb_i2i_sim = embdding_sim(
        artiles, article_embedding, topk=10)
