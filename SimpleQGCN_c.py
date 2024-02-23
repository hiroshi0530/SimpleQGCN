import torch
import itertools
import numpy as np
import scipy.sparse as sp

from torch import nn
from torch import optim

import networkx as nx

from util.loss import BPRLoss
from util.loss import EmbLoss
from util.util import xavier_uniform_initialization

from tqdm import tqdm
from loguru import logger

from trainer.Trainer import Trainer


class SimpleQGCN_c(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()

        self.config = config

        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        self.user_id_name = config["user_id_name"]
        self.item_id_name = config["item_id_name"]

        self.user_idx_name = config["user_idx_name"]
        self.item_idx_name = config["item_idx_name"]
        self.neg_item_idx_name = config["neg_item_idx_name"]

        self.early_stop_num = config["early_stop_num"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataset = dataset
        self.epochs = config["epochs"]
        self.topk = config["topk"]

        self.alpha_list = config["alpha_list"]
        self.alpha_list = list(map(lambda x: x / sum(self.alpha_list), self.alpha_list))  # 正規化

        # load dataset info
        self.interaction_matrix = self.dataset.get_train_coo_matrix()

        self.embedding_size = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.learning_rate = config["learning_rate"]
        self.reg_weight = config["reg_weight"]

        self.qw = config["qw"]

        assert len(self.alpha_list) == self.n_layers + 1

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.user_num, embedding_dim=self.embedding_size).to(
            self.device
        )
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.item_num, embedding_dim=self.embedding_size).to(
            self.device
        )

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.norm_adj_matrix = self.get_normalized_adjacency_matrix().to(self.device)

        # eigenvalue decomposition
        l, p = torch.linalg.eigh(self.norm_adj_matrix.to_dense().to(self.device))

        pp = torch.pow(p, 2)

        l = l.to(self.device)
        p = p.to(self.device)
        pp = pp.to(self.device)

        self.Q0 = self.get_Q0(l, p, pp)

        ######################################################
        # cut
        input = self.Q0.reshape(-1).type(torch.FloatTensor)

        # inputのlengthが10e6を超えると，quantile too largeでエラーなので重複無しのサンプリングをおこなう
        if input.shape[0] > 1e6:
            input = torch.tensor(np.random.choice(input, int(1e6), replace=False))

        self.quantile_th = config["quantile_th"]
        q = torch.tensor([self.quantile_th], dtype=torch.float)

        cut_th = torch.quantile(input=input, q=q).to(self.device)

        logger.info(f"cut_th : {cut_th}")

        self.Q0 = torch.where(self.Q0 < cut_th, 0, self.Q0)

        ##################################################
        # weight
        self.norm_adj_matrix = self.qw * self.Q0 + (1 - self.qw) * self.norm_adj_matrix.to_dense().to(self.device)

        ##################################################
        # normalization
        self.norm_adj_matrix = self.norm_adj_matrix.to("cpu")  # to CPUへ
        D = torch.diag(torch.sum(self.norm_adj_matrix, dim=1)).type(torch.DoubleTensor)
        D_0_5_pow = torch.pow(torch.where(D == 0, 1, D), -1 / 2) - torch.where(D == 0, 1, 0)
        self.norm_adj_matrix = D_0_5_pow.to(float) @ self.norm_adj_matrix.to(float) @ D_0_5_pow.to(float)

        self.norm_adj_matrix = self.norm_adj_matrix.type(torch.FloatTensor).to(self.device)  # to GPU

        self.apply(xavier_uniform_initialization)

        # optimizer
        self.set_optimizer()

    def get_normalized_adjacency_matrix(self):
        B = nx.Graph()

        user_nodes = [i for i in range(self.dataset.user_num)]
        item_nodes = [(i + self.dataset.user_num) for i in range(self.dataset.item_num)]

        B.add_nodes_from(user_nodes, bipartite=0)
        B.add_nodes_from(item_nodes, bipartite=1)
        B.add_edges_from(
            [
                (row, self.dataset.user_num + col)
                for row, col in zip(self.interaction_matrix.row, self.interaction_matrix.col)
            ]
        )

        A = nx.adjacency_matrix(B).todense()
        A = torch.tensor(A, dtype=float)

        D = torch.diag(torch.sum(A, dim=1)).type(torch.LongTensor)
        D_0_5_pow = torch.pow(torch.where(D == 0, 1, D), -1 / 2) - torch.where(D == 0, 1, 0)

        A = A.to(self.device)
        D = D.to(self.device)
        D_0_5_pow = D_0_5_pow.to(self.device)
        A_hat = D_0_5_pow.to(float) @ A.to(float) @ D_0_5_pow.to(float)
        A_hat = A_hat.float()

        return A_hat.to_sparse()

    def get_Q0(self, l, p, pp):
        logger.info("start get Q0")
        Q0 = torch.zeros(len(p), len(p)).to(self.device)
        for i in tqdm(range(len(p))):
            Q0 += torch.multiply(pp[:, i : i + 1], pp[:, i : i + 1].T)

        logger.info("end get Q0")
        return Q0

    def get_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return embeddings

    def forward(self):

        all_embeddings = self.get_embeddings()
        embeddings_list = [self.alpha_list[0] * all_embeddings]

        for layer_idx, alpha in zip(range(self.n_layers), self.alpha_list[1:]):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(alpha * all_embeddings)

        temp = torch.stack(embeddings_list, dim=0)

        lightgcn_all_embeddings = torch.sum(temp, dim=0)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.user_num, self.item_num])
        return user_all_embeddings, item_all_embeddings

    def get_loss(self, interaction):
        user = interaction[self.user_idx_name]
        pos_item = interaction[self.item_idx_name]
        neg_item = interaction[self.neg_item_idx_name]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_embeddings = self.user_embedding(user)
        pos_embeddings = self.item_embedding(pos_item)
        neg_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.user_idx_name]
        item = interaction[self.item_idx_name]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def all_user_predict(self, interaction):
        user = interaction[self.user_idx_name]

        self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]

        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0)

    def fit(self):
        trainer = Trainer(self.config, self, self.dataset)
        self.best_valid_result, self.test_result = trainer.train()

    def calculate_gcn_diversity(self):
        user_embedding = self.user_embedding.weight
        item_embedding = self.item_embedding.weight

        # 解析
        score = torch.matmul(user_embedding, item_embedding.T)
        sorted_list, idx = torch.sort(score, dim=1)

        idx = torch.fliplr(idx)
        sorted_list = torch.fliplr(sorted_list)

        cdist_top_5_list = []
        cdist_middle_5_list = []
        cdist_middle_1_5_list = []
        cdist_middle_2_5_list = []
        cdist_middle_3_5_list = []
        cdist_middle_4_5_list = []
        cdist_middle_5_5_list = []

        combinations_top_5_list = []
        combinations_middle_5_list = []
        combinations_middle_1_5_list = []
        combinations_middle_2_5_list = []
        combinations_middle_3_5_list = []
        combinations_middle_4_5_list = []
        combinations_middle_5_5_list = []

        for i, e in enumerate(user_embedding):
            top_5 = item_embedding[idx[i, 0:5]]
            middle_5 = item_embedding[idx[i, 25:30]]
            middle_1_5 = item_embedding[idx[i, 50:55]]
            middle_2_5 = item_embedding[idx[i, 75:80]]
            middle_3_5 = item_embedding[idx[i, 100:105]]
            middle_4_5 = item_embedding[idx[i, 150:155]]
            middle_5_5 = item_embedding[idx[i, 250:255]]

            top_5_mean = torch.mean(top_5, dim=0).reshape(1, -1).to(torch.float)
            middle_5_mean = torch.mean(middle_5, dim=0).reshape(1, -1).to(torch.float)
            middle_1_5_mean = torch.mean(middle_1_5, dim=0).reshape(1, -1).to(torch.float)
            middle_2_5_mean = torch.mean(middle_2_5, dim=0).reshape(1, -1).to(torch.float)
            middle_3_5_mean = torch.mean(middle_3_5, dim=0).reshape(1, -1).to(torch.float)
            middle_4_5_mean = torch.mean(middle_4_5, dim=0).reshape(1, -1).to(torch.float)
            middle_5_5_mean = torch.mean(middle_5_5, dim=0).reshape(1, -1).to(torch.float)

            _cdist_top_5 = torch.cdist(top_5, top_5_mean).mean()
            _cdist_middle_5 = torch.cdist(middle_5, middle_5_mean).mean()
            _cdist_middle_1_5 = torch.cdist(middle_1_5, middle_1_5_mean).mean()
            _cdist_middle_2_5 = torch.cdist(middle_2_5, middle_2_5_mean).mean()
            _cdist_middle_3_5 = torch.cdist(middle_3_5, middle_3_5_mean).mean()
            _cdist_middle_4_5 = torch.cdist(middle_4_5, middle_4_5_mean).mean()
            _cdist_middle_5_5 = torch.cdist(middle_5_5, middle_5_5_mean).mean()

            cdist_top_5_list.append(_cdist_top_5.item())
            cdist_middle_5_list.append(_cdist_middle_5.item())
            cdist_middle_1_5_list.append(_cdist_middle_1_5.item())
            cdist_middle_2_5_list.append(_cdist_middle_2_5.item())
            cdist_middle_3_5_list.append(_cdist_middle_3_5.item())
            cdist_middle_4_5_list.append(_cdist_middle_4_5.item())
            cdist_middle_5_5_list.append(_cdist_middle_5_5.item())

            ## combination
            def get_combinations_norm(mat):
                temp = list(itertools.combinations(mat, 2))
                temp_list = []
                for i in temp:
                    temp_list.append(torch.dist(i[0], i[1]).item())
                return np.mean(temp_list)

            combinations_top_5_list.append(get_combinations_norm(top_5))
            combinations_middle_5_list.append(get_combinations_norm(middle_5))
            combinations_middle_1_5_list.append(get_combinations_norm(middle_1_5))
            combinations_middle_2_5_list.append(get_combinations_norm(middle_2_5))
            combinations_middle_3_5_list.append(get_combinations_norm(middle_3_5))
            combinations_middle_4_5_list.append(get_combinations_norm(middle_4_5))
            combinations_middle_5_5_list.append(get_combinations_norm(middle_5_5))

        self.cdist_top_5_list = float(np.mean(cdist_top_5_list))
        self.cdist_middle_5_list = float(np.mean(cdist_middle_5_list))
        self.cdist_middle_1_5_list = float(np.mean(cdist_middle_1_5_list))
        self.cdist_middle_2_5_list = float(np.mean(cdist_middle_2_5_list))
        self.cdist_middle_3_5_list = float(np.mean(cdist_middle_3_5_list))
        self.cdist_middle_4_5_list = float(np.mean(cdist_middle_4_5_list))
        self.cdist_middle_5_5_list = float(np.mean(cdist_middle_5_5_list))
        self.combinations_top_5_list = float(np.mean(combinations_top_5_list))
        self.combinations_middle_5_list = float(np.mean(combinations_middle_5_list))
        self.combinations_middle_1_5_list = float(np.mean(combinations_middle_1_5_list))
        self.combinations_middle_2_5_list = float(np.mean(combinations_middle_2_5_list))
        self.combinations_middle_3_5_list = float(np.mean(combinations_middle_3_5_list))
        self.combinations_middle_4_5_list = float(np.mean(combinations_middle_4_5_list))
        self.combinations_middle_5_5_list = float(np.mean(combinations_middle_5_5_list))


# import os
# import datetime
# import torch
# import pickle
# import models
#
# import numpy as np
# import scipy.sparse as sp
#
# from tqdm import tqdm
# from loguru import logger
# from torch import optim
# from trainer.Trainer import Trainer
#
# from util.loss import BPRLoss
# from util.loss import EmbLoss
# from util.util import xavier_uniform_initialization
#
#
# class LightQGCN_ver02(models.Model):
#     def __init__(self, config, dataset):
#         super().__init__(config, dataset)
#
#         self.config = config
#         self.dataset = dataset
#         self.epochs = config["epochs"]
#         self.topk = config["topk"]
#
#         self.alpha_list = config["alpha_list"]
#         self.alpha_list = list(map(lambda x: x / sum(self.alpha_list), self.alpha_list))  # 正規化
#
#         # load dataset info
#         self.interaction_matrix = self.dataset.get_train_coo_matrix()
#
#         self.embedding_size = config["embedding_size"]
#         self.n_layers = config["n_layers"]
#         self.learning_rate = config["learning_rate"]
#         self.reg_weight = config["reg_weight"]
#
#         assert len(self.alpha_list) == self.n_layers + 1
#
#         # define layers and loss
#         self.user_embedding = torch.nn.Embedding(num_embeddings=self.user_num, embedding_dim=self.embedding_size)
#         self.item_embedding = torch.nn.Embedding(num_embeddings=self.item_num, embedding_dim=self.embedding_size)
#         self.mf_loss = BPRLoss()
#         self.reg_loss = EmbLoss()
#
#         # storage variables for full sort evaluation acceleration
#         self.restore_user_e = None
#         self.restore_item_e = None
#
#         # generate intermediate data
#         self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
#
#         # Qの追加
#         self.model_name = config["model"]
#         self.dataset_name = config["dataset"]
#         self.qw = config["qw"]
#         self.is_q0 = config["is_q0"]
#         self.is_diag_corr = config["is_diag_corr"]
#         self.activation = config["activation"]
#         self.is_nonzero_filter = config["is_nonzero_filter"]
#         self.is_save_pkl = config["is_save_pkl"]
#
#         if self.is_save_pkl:
#             self.result_dir = "./result/QGCN_ver02/"
#             os.makedirs(self.result_dir, exist_ok=True)
#             self.yyyymmdd = datetime.datetime.now().strftime("%Y%m%d")
#
#         # エルミート行列の固有値分解の場合、eig()だと誤差で複素数が出るので、エルミート行列専用のeigh()を利用する！
#         l, p = torch.linalg.eigh(self.norm_adj_matrix.to_dense().to(self.device))
#
#         pp = torch.pow(p, 2)
#
#         l = l.to(self.device)
#         p = p.to(self.device)
#         pp = pp.to(self.device)
#
#         aaa = self.norm_adj_matrix.to_dense().to("cpu").detach().numpy().copy()
#         th_ratio = np.count_nonzero(aaa) / self.norm_adj_matrix.numel()
#
#         # Q0
#         if self.is_q0:
#             self.Q0 = self.get_Q0(l, p, pp)
#
#             if self.is_diag_corr:
#                 # Q0の対角成分だけを0にして正規化する
#                 Q0_diag = torch.diag(torch.diag(self.Q0))
#                 self.Q0 = self.Q0 - Q0_diag
#                 self.Q0 = self.Q0 + 1e-9 * torch.eye(len(p)).to(self.device)
#
#                 self.Q0 = self.Q0 / torch.sum(self.Q0, dim=1).reshape(-1, 1)
#
#             if self.is_nonzero_filter:
#                 for th in [10 ** (-0.001 * i) for i in range(10000)]:
#                     mask = self.Q0.ge(th)
#                     _temp_Q0 = torch.multiply(self.Q0, mask)
#                     if torch.count_nonzero(_temp_Q0) / self.Q0.numel() > th_ratio:
#                         self.Q0 = _temp_Q0
#                         break
#
#                 logger.info("=" * 30)
#                 logger.info("q0 count_non_zeros : {:.10f}".format(torch.count_nonzero(self.Q0) / self.Q0.numel()))
#                 logger.info(
#                     "norm m count_non_zeros : {:.10f}".format(np.count_nonzero(aaa) / self.norm_adj_matrix.numel())
#                 )
#                 logger.info("=" * 30)
#
#                 # 最正規化
#                 self.Q0 += 1e-9
#                 self.Q0 = self.Q0 / torch.sum(self.Q0, dim=1).reshape(-1, 1)
#
#             if self.is_save_pkl:
#                 # TODO: 230510: Q0は保存しない => 保存しているがloadしていない
#                 logger.info("save O0")
#                 q0_file_name = self.yyyymmdd + "_Q0_" + self.model_name + "_" + self.dataset_name + ".pkl"
#                 with open(self.result_dir + q0_file_name, mode="wb") as f:
#                     pickle.dump(self.Q0, f)
#                 logger.info("end get Q0")
#
#         ##################################################
#         # 重み調整
#         self.norm_adj_matrix = self.qw * self.Q0 + (1 - self.qw) * self.norm_adj_matrix.to_dense().to(self.device)
#
#         ##################################################
#         # 正規化
#         self.norm_adj_matrix = self.norm_adj_matrix.to("cpu")  # 一度CPUへ
#         D = torch.diag(torch.sum(self.norm_adj_matrix, dim=1)).type(torch.DoubleTensor)
#         D_0_5_pow = torch.pow(torch.where(D == 0, 1, D), -1 / 2) - torch.where(D == 0, 1, 0)
#         self.norm_adj_matrix = D_0_5_pow.to(float) @ self.norm_adj_matrix.to(float) @ D_0_5_pow.to(float)
#
#         self.norm_adj_matrix = self.norm_adj_matrix.type(torch.FloatTensor).to(self.device)  # GPUへ
#
#         ##################################################
#         # parameters initialization
#         self.apply(xavier_uniform_initialization)
#         self.other_parameter_name = ["restore_user_e", "restore_item_e"]
#
#         # optimizer
#         # TODO: parameter init の後に持ってくる
#         self.set_optimizer()
#
#     def get_norm_adj_mat(self):
#         A = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
#         inter_M = self.interaction_matrix
#         inter_M_t = self.interaction_matrix.transpose()
#         data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.user_num), [1] * inter_M.nnz))
#         data_dict.update(dict(zip(zip(inter_M_t.row + self.user_num, inter_M_t.col), [1] * inter_M_t.nnz)))
#         A._update(data_dict)
#         # norm adj matrix
#         sumArr = (A > 0).sum(axis=1)
#         # add epsilon to avoid divide by zero Warning
#         diag = np.array(sumArr.flatten())[0] + 1e-7
#         diag = np.power(diag, -0.5)
#         D = sp.diags(diag)
#         L = D * A * D
#         # covert norm_adj matrix to tensor
#         L = sp.coo_matrix(L)
#         row = L.row
#         col = L.col
#         i = torch.LongTensor([row, col])
#         data = torch.FloatTensor(L.data)
#         SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
#
#         return SparseL
#
#     def get_Q0(self, l, p, pp):
#         logger.info("start get Q0")
#         Q0 = torch.zeros(len(p), len(p)).to(self.device)
#         for i in tqdm(range(len(p))):
#             Q0 += torch.multiply(pp[:, i : i + 1], pp[:, i : i + 1].T)
#
#         return Q0
#
#     def get_ego_embeddings(self):
#         user_embeddings = self.user_embedding.weight
#         item_embeddings = self.item_embedding.weight
#         ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
#         return ego_embeddings
#
#     def forward(self):
#         all_embeddings = self.get_ego_embeddings()
#         embeddings_list = [self.alpha_list[0] * all_embeddings]
#
#         for layer_idx, alpha in zip(range(self.n_layers), self.alpha_list[1:]):
#             ################################################################
#             # https://pytorch.org/docs/stable/generated/torch.sparse.mm.html
#             # torch.sparse.mm は
#             # dense x dense => dense の計算ができる
#             # sparse x dense => dense
#             # sparse x sparse => sparse
#             ################################################################
#             all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
#             embeddings_list.append(alpha * all_embeddings)
#
#         ################################################################
#         # 20230416: ここで一度stackにしているのは，torch.tensor(embedding_list)にするとエラーになるから
#         # lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
#         # lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
#
#         ################################################################
#         # 20230416: dim=0にしてtorch.stackを利用してtensorにする
#         temp = torch.stack(embeddings_list, dim=0)
#
#         # alphaで加重平均を取っているので，ここでは和を取る
#         lightgcn_all_embeddings = torch.sum(temp, dim=0)
#
#         user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.user_num, self.item_num])
#         return user_all_embeddings, item_all_embeddings
#
#     def calculate_loss(self, interaction):
#         # clear the storage variable when training
#         if self.restore_user_e is not None or self.restore_item_e is not None:
#             self.restore_user_e, self.restore_item_e = None, None
#
#         user = interaction[self.user_idx_name]
#         pos_item = interaction[self.item_idx_name]
#         neg_item = interaction[self.neg_item_idx_name]
#
#         user_all_embeddings, item_all_embeddings = self.forward()
#         u_embeddings = user_all_embeddings[user]
#         pos_embeddings = item_all_embeddings[pos_item]
#         neg_embeddings = item_all_embeddings[neg_item]
#
#         # calculate BPR Loss
#         pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
#         neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
#         mf_loss = self.mf_loss(pos_scores, neg_scores)
#
#         # calculate BPR Loss
#         u_ego_embeddings = self.user_embedding(user)
#         pos_ego_embeddings = self.item_embedding(pos_item)
#         neg_ego_embeddings = self.item_embedding(neg_item)
#
#         reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
#         loss = mf_loss + self.reg_weight * reg_loss
#
#         return loss
#
#     def predict(self, interaction):
#         user = interaction[self.user_idx_name]
#         item = interaction[self.item_idx_name]
#
#         user_all_embeddings, item_all_embeddings = self.forward()
#
#         u_embeddings = user_all_embeddings[user]
#         i_embeddings = item_all_embeddings[item]
#         scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
#         return scores
#
#     def full_sort_predict(self, interaction):
#         user = interaction[self.user_idx_name]
#         if self.restore_user_e is None or self.restore_item_e is None:
#             self.restore_user_e, self.restore_item_e = self.forward()
#         # get user embedding from storage variable
#         u_embeddings = self.restore_user_e[user]
#
#         # dot with all item embedding to accelerate
#         scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
#
#         return scores.view(-1)
#
#     def set_optimizer(self):
#         self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0)
#         # self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg_weight)
#
#     def fit(self):
#         trainer = Trainer(self.config, self, self.dataset)
#         self.best_valid_result, self.test_result = trainer.train()
#
#         # 多様性の計算
#         trainer.get_diversity()
#
