import numpy as np

import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):

    def __init__(self,
                 graph,
                 data,
                 intv,
                 regime,
                 split,
                 obs_size=100,
                 node_size=100,
                 only_obs=False):
        super().__init__()

        self.data = [data[i] for i in split]
        self.intv = [intv[i] for i in split]
        self.regime = [regime[i] for i in split]
        self.label = [graph[i].fill_diagonal_(-100) for i in split]

        self.n_graph = len(self.label)
        self.graph_size = self.label[0].shape[0]
        self.total_obs = sum([len(d) for d in self.data])

        self.obs_size = obs_size
        self.node_size = node_size
        self.only_obs = only_obs

    def __getitem__(self, idx):
        idx_graph = np.random.randint(0, self.n_graph)
        data, intv, label = self.data[idx_graph], self.intv[idx_graph], self.label[idx_graph]

        if self.node_size < data.shape[-1]:
            regime = self.regime[idx_graph]
            if self.only_obs:
                data, intv, label = self.node_choice_random(data, intv, label, regime)
            else:
                data, intv, label = self.node_choice_train(data, intv, label, regime)
        else:
            idx_obs = np.random.choice(len(data), size=self.obs_size, replace=False)
            data, intv = data[idx_obs], intv[idx_obs]

        return {"data": data, "intv": intv, "label": label}

    def __len__(self):
        return (self.total_obs // self.obs_size) * (self.graph_size // self.node_size)

    def node_choice_random(self, data, intv, label, regime):
        idx_obs = np.random.choice(len(data), size=self.obs_size, replace=False)
        data, intv = data[idx_obs], intv[idx_obs]

        idx_node = np.random.choice(len(data[0]), size=self.node_size, replace=False)
        data = data[:, idx_node]
        intv = intv[:, idx_node]
        label = label[idx_node][:, idx_node]

        return data, intv, label

    def node_choice_train(self, data, intv, label, regime, debug=False):
        n_node = data.shape[-1]

        # select regimes
        intv_len = len(regime["len_to_regime"]) - 1
        n_regime_per_sample = int(self.node_size // (intv_len * (intv_len + 1) / 2))

        regime_slct = [regime["len_to_regime"][0][0]]  # regime idx for no_intv setting
        for k in range(1, intv_len + 1):
            regime_k = np.random.choice(regime["len_to_regime"][k],
                                        size=n_regime_per_sample,
                                        replace=False)
            regime_slct.extend(list(regime_k))

        # select obs and node candidates
        idx_obs = []
        idx_node = set([])
        for r in regime_slct:
            idx_obs.extend(regime["regime_obs"][r])
            idx_node = idx_node | regime["regime_to_intv"][r]
        idx_node = list(idx_node)

        # fill node_idx
        need_to_fill = self.node_size - len(idx_node)
        if need_to_fill > 0:
            node_left = np.ones(n_node, dtype=bool)
            node_left[idx_node] = False
            node_left = np.arange(n_node)[node_left]
            node_left = np.random.choice(node_left, size=need_to_fill, replace=False)
            idx_node.extend(list(node_left))
        elif need_to_fill < 0:
            if debug:
                print("Warning: node_size is larger than the number of nodes in the graph.")
            idx_node = idx_node[:self.node_size]
        idx_node = torch.tensor(idx_node)

        idx_obs = np.random.choice(idx_obs, size=self.obs_size, replace=False)
        data, intv = data[idx_obs], intv[idx_obs]

        if debug:
            assert len(set(idx_node)) == self.node_size
            assert intv.sum() == intv[:, idx_node].sum()

        data = data[:, idx_node]
        intv = intv[:, idx_node]
        label = label[idx_node][:, idx_node]

        return data, intv, label


class TestDataset(TrainDataset):

    def __init__(self,
                 graph,
                 data,
                 intv,
                 regime,
                 split,
                 obs_size=100,
                 node_size=100,
                 only_obs=False):
        super().__init__(graph,
                         data,
                         intv,
                         regime,
                         split=split,
                         obs_size=obs_size,
                         node_size=node_size,
                         only_obs=only_obs)

    def __getitem__(self, idx):
        idx_graph = idx % self.n_graph
        data, intv, label = self.data[idx_graph], self.intv[idx_graph], self.label[idx_graph]

        offset = idx // self.n_graph
        if self.node_size < data.shape[-1]:
            regime = self.regime[idx_graph]

            if self.only_obs:
                data, intv, label = self.node_choice_obs(data, intv, label, regime, offset)
            else:
                data, intv, label = self.node_choice_test(data, intv, label, regime, offset)
        else:
            hop = len(data) // self.obs_size
            idx_obs = torch.arange(self.obs_size) * hop + offset
            idx_obs = idx_obs % len(data)
            data, intv = data[idx_obs], intv[idx_obs]

        return {"data": data, "intv": intv, "label": label}

    def node_choice_test(self, data, intv, label, regime, offset, debug=False):
        n_node = data.shape[-1]

        # select regimes
        intv_len = len(regime["len_to_regime"]) - 1
        n_regime_per_sample = int(self.node_size // (intv_len * (intv_len + 1) / 2))

        l = max([len(regime["len_to_regime"][k]) for k in range(1, intv_len + 1)])
        hop = l // n_regime_per_sample
        offset_node = offset % hop
        offset_obs = offset // hop

        regime_slct = [regime["len_to_regime"][0][0]]  # regime idx for no_intv setting
        for k in range(1, intv_len + 1):
            regime_k = regime["len_to_regime"][k]
            regime_slct.extend(regime_k[offset_node::hop])

        # select obs and node candidates
        idx_obs_pool = []
        idx_node = set([])
        for r in regime_slct:
            idx_obs_pool.extend(regime["regime_obs"][r])
            idx_node = idx_node | regime["regime_to_intv"][r]
        idx_node = list(idx_node)

        # fill nodes to fit node_size
        need_to_fill = self.node_size - len(idx_node)
        if need_to_fill > 0:
            node_left = np.ones(n_node, dtype=bool)
            node_left[idx_node] = False
            node_left = np.arange(n_node)[node_left]
            node_left = node_left[offset_node * need_to_fill:(offset_node + 1) * need_to_fill]
            idx_node.extend(list(node_left))
        elif need_to_fill < 0:
            if debug:
                print("Warning: node_size is larger than the number of nodes in the graph.",
                      len(idx_node))
            idx_node = idx_node[:self.node_size]
        idx_node = torch.tensor(idx_node)

        # sample observation
        hop = len(idx_obs_pool) // self.obs_size
        idx_obs = np.arange(self.obs_size) * hop + offset_obs
        idx_obs = idx_obs % len(idx_obs_pool)
        idx_obs = np.array(idx_obs_pool)[idx_obs]
        data, intv = data[idx_obs], intv[idx_obs]

        if debug:
            assert len(idx_node) == self.node_size
            assert intv.sum() == intv[:, idx_node].sum()

        data = data[:, idx_node]
        intv = intv[:, idx_node]
        label = label[idx_node][:, idx_node]

        return data, intv, label

    def node_choice_obs(self, data, intv, label, regime, offset):
        n_node = data.shape[-1]

        hop_node = n_node // self.node_size
        offset_node = offset % hop_node
        offset_obs = offset // hop_node

        # sample observation
        hop_obs = len(data) // self.obs_size
        idx_obs = np.arange(self.obs_size) * hop_obs + offset_obs
        idx_obs = idx_obs % len(data)
        data, intv = data[idx_obs], intv[idx_obs]

        # sample node
        idx_node = np.arange(self.node_size) * hop_node + offset_node
        idx_node = idx_node % n_node

        data = data[:, idx_node]
        intv = intv[:, idx_node]
        label = label[idx_node][:, idx_node]

        return data, intv, label
