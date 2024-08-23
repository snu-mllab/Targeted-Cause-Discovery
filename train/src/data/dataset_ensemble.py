import numpy as np
import torch
from .dataset import TrainDataset
from math import ceil


class TestDatasetEnsemble(TrainDataset):

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

    def __getitem__(self, indices):
        idx_graph, offset_node, offset_obs = indices

        data, intv, label = self.data[idx_graph], self.intv[idx_graph], self.label[idx_graph]

        if self.node_size < data.shape[-1]:
            regime = self.regime[idx_graph]

            if self.only_obs:
                data, intv, label, idx_node = self.node_choice_obs(data, intv, label, regime,
                                                                   offset_node, offset_obs)
            else:
                data, intv, label, idx_node = self.node_choice_test(data, intv, label, regime,
                                                                    offset_node, offset_obs)
        else:
            hop = len(data) // self.obs_size
            idx_obs = np.arange(self.obs_size) * hop + offset_obs
            idx_obs = idx_obs % len(data)
            data, intv = data[idx_obs], intv[idx_obs]

            idx_node = np.arange(self.node_size)

        return {"data": data, "intv": intv, "label": label, "idx_node": idx_node}

    def node_choice_test(self, data, intv, label, regime, offset_node, offset_obs, debug=False):
        n_node = data.shape[-1]

        # select regimes
        chunk_size = self.node_size // 2
        n_chunk = ceil(n_node / chunk_size)
        loop_node = n_chunk * (n_chunk - 1) // 2

        regime_slct = [regime["len_to_regime"][0][0]]
        assert len(regime["len_to_regime"]) == 2, "not implemented for multi intv setting"

        regime_1 = []
        hop_node = offset_node // loop_node + 1
        offset_node = offset_node % loop_node
        for i in range(hop_node):
            regime_1.extend(regime["len_to_regime"][1][i::hop_node])

        idx_s = offset_node % n_chunk
        idx_t = (idx_s + offset_node // n_chunk + 1) % n_chunk
        assert idx_s != idx_t

        for idx in [idx_s, idx_t]:
            offset = max(0, chunk_size * (idx + 1) - len(regime_1))
            regime_slct.extend(regime_1[chunk_size * idx - offset:chunk_size * (idx + 1) - offset])

        # select obs and node candidates
        idx_obs_pool = []
        idx_node = set([])
        # regime_slct = regime_slct[:1] + sorted(regime_slct[1:])
        for r in regime_slct:
            idx_obs_pool.extend(regime["regime_obs"][r])
            idx_node = idx_node | regime["regime_to_intv"][r]
        idx_node = sorted(list(idx_node))

        # fill nodes to fit node_size
        need_to_fill = self.node_size - len(idx_node)
        if need_to_fill > 0:
            node_left = np.ones(n_node, dtype=bool)
            node_left[idx_node] = False
            node_left = np.arange(n_node)[node_left]
            node_left = node_left[idx_s * need_to_fill:(idx_s + 1) * need_to_fill]
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
            assert len(idx_node) == self.node_size, print(len(idx_node), flush=True)
            assert intv.sum() == intv[:, idx_node].sum()

        data = data[:, idx_node]
        intv = intv[:, idx_node]
        label = label[idx_node][:, idx_node]

        return data, intv, label, idx_node

    def node_choice_obs(self, data, intv, label, regime, offset_node, offset_obs):
        n_node = data.shape[-1]

        # select node
        chunk_size = self.node_size // 2
        n_chunk = ceil(n_node / chunk_size)
        loop_node = n_chunk * (n_chunk - 1) // 2

        hop_node = offset_node // loop_node + 1
        offset_node = offset_node % loop_node

        node_list = []
        node_full = np.arange(n_node).tolist()
        for i in range(hop_node):
            node_list.extend(node_full[i::hop_node])

        idx_s = offset_node % n_chunk
        idx_t = (idx_s + offset_node // n_chunk + 1) % n_chunk

        idx_node = []
        for idx in [idx_s, idx_t]:
            offset = max(0, chunk_size * (idx + 1) - n_node)
            idx_node.extend(node_list[chunk_size * idx - offset:chunk_size * (idx + 1) - offset])

        idx_node = torch.tensor(idx_node)

        # sample observation
        hop_obs = len(data) // self.obs_size
        idx_obs = np.arange(self.obs_size) * hop_obs + offset_obs
        idx_obs = idx_obs % len(data)
        data, intv = data[idx_obs], intv[idx_obs]

        # sample node
        data = data[:, idx_node]
        intv = intv[:, idx_node]
        label = label[idx_node][:, idx_node]

        return data, intv, label, idx_node
