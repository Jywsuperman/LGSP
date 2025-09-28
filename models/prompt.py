import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from models.Prompt_FFN import KeyFFN

class Prompt(nn.Module):
    def __init__(self, prompt_pool=False, embed_dim=768, prompt_key_init='uniform',
                 pool_size=None, top_k=None, batchwise_prompt=False, embedding_key='cls',):
        super().__init__()

        self.prompt_pool = prompt_pool # bool类型
        # self.prompt_key = prompt_key # bool类型
        # self.length = length # 这个和top_k的作用：选中几个小pool的个数
        self.embed_dim = embed_dim  # vit的最后一个维度
        self.embedding_key = embedding_key # x映射到和k比较的维度的方法
        # self.prompt_init = prompt_init
        self.pool_size = pool_size # k的数量
        self.top_k = top_k # 选择多少个分支
        self.batchwise_prompt = batchwise_prompt

        if prompt_pool:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
            elif prompt_key_init == 'proto_fc':
                load_path = "/home/jyw/SuperMan/PriViLege_Clear/run_script/vpt_Baseline_fc_weights.pth"
                fc_weights = torch.load(load_path) # [200,768]
                # print("哈哈")
                # print(fc_weights.shape)
                # 使用K-means聚类获取权重
                self.prompt_key = nn.Parameter(self.reduce_fc_weights(fc_weights, pool_size)) # [Pool_size,C]
                # 把key给冻结
                self.prompt_key.requires_grad = False        

                # 定义两个FFN网络
                self.key_ffn = KeyFFN(embed_dim=embed_dim, hidden_dim=embed_dim * 4)  
                self.x_embed_ffn = KeyFFN(embed_dim=embed_dim, hidden_dim=embed_dim * 4)

    # K-means 聚类
    def reduce_fc_weights(self, fc_weights, pool_size):
        # 查看 OrderedDict 的所有键
        # print(fc_weights.keys())
        # print(type(fc_weights))
        # 只取前 100 行
        fc_weights_subset = fc_weights['weight'][:100]  # shape: [100, 768]

        fc_weights_np = fc_weights_subset.cpu().numpy()
        kmeans = KMeans(n_clusters=pool_size, random_state=42)
        kmeans.fit(fc_weights_np)
        centroids = kmeans.cluster_centers_
        return torch.tensor(centroids, dtype=fc_weights['weight'].dtype, device=fc_weights['weight'].device)
    
    def reduce_fc_weights_pca(self, fc_weights, pool_size):

        # 设置随机种子
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 将 PyTorch 张量转换为 NumPy 数组
        fc_weights_np = fc_weights.cpu().numpy()  # [200, C]

        # 使用 PCA 进行降维
        pca = PCA(n_components=pool_size, svd_solver='full')  # 这里 pool_size = 20
        reduced_weights_np = pca.fit_transform(fc_weights_np.T)  # 转置为 [C, 200] 进行 PCA

        # 转置回 [20, C]
        reduced_weights_np = reduced_weights_np.T  # [20, C]

        # 将 NumPy 数组转换回 PyTorch 张量
        reduced_weights = torch.tensor(reduced_weights_np, dtype=fc_weights.dtype, device=fc_weights.device)
        return reduced_weights
    
    def reduce_fc_weights_cosine(self, fc_weights, pool_size):
        # 转换为 NumPy 数组
        fc_weights_np = fc_weights.cpu().numpy()  # [200, C]
        
        # 计算余弦相似度矩阵
        similarity_matrix = cosine_similarity(fc_weights_np)
        
        # 基于余弦距离进行聚类
        clustering = AgglomerativeClustering(n_clusters=pool_size, metric='precomputed', linkage='average')
        labels = clustering.fit_predict(1 - similarity_matrix)  # 聚类标签
        
        # 计算每个簇的均值作为质心
        centroids = []
        for i in range(pool_size):
            cluster_data = fc_weights_np[labels == i]
            centroids.append(cluster_data.mean(axis=0))
        
        return torch.tensor(centroids, dtype=fc_weights.dtype, device=fc_weights.device)
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def custom_gumbel_softmax(self, logits, tau=1.0, dim=-1, hard=False):
        """
        Compute Gumbel-Softmax samples.

        Args:
            logits (Tensor): Input logits.
            tau (float): Temperature parameter.
            dim (int): Dimension along which softmax will be computed.
            hard (bool): If True, return hard one-hot encoded sample using the straight-through estimator.

        Returns:
            Tensor: Gumbel-Softmax samples.
        """
        # Generate Gumbel noise
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0, 1)
        gumbels = (logits + gumbels) * tau  # ~Gumbel(logits, tau)

        # Compute softmax probabilities
        y_soft = F.softmax(gumbels, dim=dim)

        if hard:
            # Hard sampling: Straight-through estimator
            index = y_soft.max(dim=dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Soft sampling: Reparameterization trick
            ret = y_soft

        return ret

    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")
               
            # key通过FFN网络
            processed_key = self.key_ffn(self.prompt_key)
            self.prompt_key = nn.Parameter(processed_key)

            # x_embed_mea通过FFN网络
            processed_x_embed_mean = self.x_embed_ffn(x_embed_mean)
            x_embed_mean = processed_x_embed_mean

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B, Pool_size
            temperature = 5.0

            # 使用Gumbel softmax
            weights = self.custom_gumbel_softmax(similarity, tau=temperature, dim=1, hard=False)  # [batch_size, Pool_size]
            
            # # 使用常规的softmax
            # similarity = similarity * temperature
            # weights = F.softmax(similarity, dim=1)  # [batch_size, Pool_size]

            # 将权重存储到输出中
            out['weights'] = weights

            
            # 返回Index，需要额外使用损失函数
            # similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            # if prompt_mask is None:
            #     _, idx = torch.topk(similarity, k=self.top_k, dim=1) # index维度是B, top_k
            #     if self.batchwise_prompt:
            #         # 把这个batch里面的样本看作一个整体，选择出现次数最多的topk个
            #         # 因为batch里面每个样本都会选择属于自己的topk个prompt，如果分别进行操作会很难
            #         # 所以直接综合考虑这个小batch里出现次数最多的topk个prompt个
            #         prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
            #         # 将 idx 展平为一维张量，大小为 (B * top_k,)
            #         # prompt_id：唯一的 prompt 索引，维度为 (unique_prompts,)
            #         # id_counts：对应每个 prompt 索引的出现次数，维度为 (unique_prompts,)

            #         # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
            #         # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
            #         # Unless dimension is specified, this will be flattend if it is not already 1D.
            #         if prompt_id.shape[0] < self.pool_size:
            #             prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
            #             id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
            #         _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
            #         major_prompt_id = prompt_id[major_idx] # top_k
            #         # expand to batch
            #         idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            # else:
            #     idx = prompt_mask # B, top_k
            # out['prompt_idx'] = idx
            # # 这是我想要获取的东西
            
            # # batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
            # # batch_size, top_k, length, c = batched_prompt_raw.shape
            # # batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            # # Debugging, return sim as well
            # out['prompt_norm'] = prompt_norm
            # out['x_embed_norm'] = x_embed_norm
            # out['similarity'] = similarity

            # # Put pull_constraint loss calculation inside
            # batched_key_norm = prompt_norm[idx] # B, top_k, C
            # out['selected_key'] = batched_key_norm
            # x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            # sim = batched_key_norm * x_embed_norm # B, top_k, C
            # reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

            # out['reduce_sim'] = reduce_sim


        # else:
        #     if self.prompt_init == 'zero':
        #         self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
        #     elif self.prompt_init == 'uniform':
        #         self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
        #         nn.init.uniform_(self.prompt)
        #     batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        # out['total_prompt_len'] = batched_prompt.shape[1]
        # out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out
