import numpy as np
import pandas as pd
import os
import random
from itertools import chain
from sklearn.preprocessing import StandardScaler
from delaynay2D import *

class preprocess():
    def __init__(self, file : str, fix_seed : bool = False):
        super(preprocess, self).__init__()
        cwd = os.get_cwd()
        self.df = pd.read_excel(os.path.join(cwd, "data", file))
        self.coords = self.standardize(self.df)
        self.vertices = self.generate_graph(self.coords)

        ind_list = self.df.iloc[:, 6:8].drop_duplicates(keep='last').index.values
        df_drop = self.df.iloc[ind_list,:].copy()
        df_drop.reset_index(inplace=True)
        df_drop.drop(['Unnamed: 0'], axis=1, inplace=True)

        if fix_seed:
            random.seed(27)

        self.features = self.preset_features(df_drop)
        self.targets = self.preset_targets(df_drop)
        self.ids = self.preset_ids(df_drop)

        self.JSON_FILE = [MakeDictionary_GR(all_vertice, i) for i in range(len(id_array))]
        self.JSON_SHUFFLE = sorted(JSON_FILE, key=lambda k: random.random())
        self.JSON_TRAIN = [JSON_SHUFFLE[i] for i in range(0, 450)]
        self.JSON_VAL = [JSON_SHUFFLE[i] for i in range(450, 500)]
        self.JSON_TEST = [JSON_SHUFFLE[i] for i in range(500, len(JSON_SHUFFLE))]

        return self.JSON_TRAIN, self.JSON_VAL, self.JSON_TEST

    def standardize(self, df):
        scaler = StandardScaler()
        coords = df.iloc[:, 6:8].drop_duplicates(keep="last".to_numpy().astype(np.float64))
        standardized_coords = scaler.fit_transform(coords)

        return standardized_coords

    def generate_graph(self, coords):
        dt = Delaunay2D()
        for c in coords:
            dt.addPoint(c)

        all_vertice = np.array(dt.exportTriangles_list())
        return all_vertice

    def seperate_graph(self, dt_array, v_index):
        r = np.where(dt_array == v_index)
        g = dt_array[r[0], :]
        return g

    def preset_features(self, df):
        df_features = df.copy().drop(['index', 'No', '프로젝트코드', '시추공코드', '시추공명', '지하수위', '지층코드', '지층시작심도', '지층종료심도', '지층두께', '학술용 지층명(USCS)'], axis=1)
        df_features.loc[:,['X','Y']] = scale_coords
        features_array = df_features.to_numpy()
        return features_array

    def preset_targets(self, df):
        df_targets = df.copy()['지층종료심도']
        targets_array = df_targets.to_numpy().reshape(-1,1)
        return targets_array

    def preset_ids(self, df):
        df_id = df.copy()['시추공코드']
        id_array = df_id.to_numpy().reshape(-1,1)
        return id_array

    def rotate(self, l, n=1):
        return l[n:] + l[:n]

    def v_to_first(self, order, key):
        for i in range(order):
            key = np.append(key[1:], key[:1])
        return key

    def v_to_first_array(self, order, array):
        for i in range(order):
            array = np.concatenate((array[1:],array[:1]), axis=0)
        return array

    def d_weight(x):
        return 1 / (1 + np.exp(x))

    def MakeDictionary_GR(self, dt_array_all, v_index, targets_array=targets_array,
                   id_array=id_array, node_features_array=node_features_array):
        dt_array = self.seperate_graph(dt_array_all, v_index)
        key = np.arange(len(np.unique(dt_array)))
        palette, index = np.unique(dt_array, return_inverse=True)
        dt_tris = key[index].reshape(dt_array.shape)
        where_v = np.where(np.unique(dt_array) == v_index)
        v_order = where_v[0][0]
        key_arrange = self.v_to_first(v_order, key)

        t_values = targets_array[np.unique(dt_array), :]
        t_arrange = self.v_to_first(v_order, t_values)
        target_value = t_arrange.item(0)
        t_features = t_arrange.copy()
        t_features[0] = 0
        n_features = self.v_to_first_array(v_order, node_features_array[np.unique(dt_array), :])

        a_values = list(chain(*[(lambda l: [(e1, e2) for (e1, e2) in zip(l, np.roll(l, 1))])(l) for l in dt_tris]))
        a_matrix = np.identity(len(np.unique(dt_array)))
        for t in a_values:
            node_distance = ((n_features[t[0]][1] - n_features[t[1]][1])**2 + (n_features[t[0]][2] - n_features[t[1]][2])**2)**0.5
            if node_distance == 0:
                a_matrix[t] = 0.0
                a_matrix[t[-1::-1]] = 0.0
            else:
                a_matrix[t] = self.d_weight(node_distance)
                a_matrix[t[-1::-1]] = self.d_weight(node_distance)

        id_value = id_array.item(v_index)
        n_values = np.concatenate((n_features, t_features.reshape(len(t_features),1)), axis=1)
        pad_width = 16 - len(key)
        #print(pad_width)

        a_matrix_pd = np.pad(a_matrix, [(0,pad_width),(0,pad_width)], 'constant', constant_values=0).tolist()
        n_values_pd = np.pad(n_values, [(0,pad_width),(0,0)], 'constant', constant_values=0).tolist()

        return dict(node_features = n_values_pd, adjacency_matrix = a_matrix_pd, target = target_value, id = id_value)
