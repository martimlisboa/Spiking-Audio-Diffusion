import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
import random

# This code is taken from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
# See the github for an example on how to put than into the encoder.

verbose = False

batch_norm = True

max_freq = True
min_freq = True
noise_neighbor = True

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.sigma = 2
        self.batch_norm = nn.BatchNorm1d(self.dim,affine = True)
        self.jitter_bool = True

        embed = torch.randn(dim, n_embed)
        self.register_buffer("count",torch.Tensor([0.]),persistent = True)
        self.register_buffer("embed", embed,persistent=True)
        self.register_buffer("cluster_size", torch.zeros(n_embed),persistent=True)
        self.register_buffer("cluster_freq", torch.zeros(n_embed),persistent=True)
        self.register_buffer("embed_avg", embed.clone(),persistent=True)
        print(f"Quantize: dim:{self.dim} , n_embed:{self.n_embed}")
        if batch_norm:
            print("Batch Norm")
        if max_freq:
            print("max_freq")
        if min_freq:
            print("min_freq")
        if noise_neighbor:
            print("noise_neighbor")

        
    def forward(self, input):
        #print(f"Quantize: dim:{self.dim} , n_embed:{self.n_embed}")
        flatten = input.reshape(-1, self.dim)

        if batch_norm:
            #batch norm the flattened input and make sure it is bigger than the encoding vectors to help in codebook colapse
            flatten = self.batch_norm(flatten)
            #input batchnormed
            input = flatten.view(*input.shape)


        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)


        if self.training: #Training
            self.count += 1;
            
            if max_freq:
                mask = self.cluster_freq/self.cluster_freq.sum() > 10/self.n_embed
                if torch.any(mask):
                    self.jitter_bool = True

                #when the frequency gets very high just randomly mess up the distances to the furthest away vector
                for k in range(self.n_embed):
                    if mask[k]:
                        dist[:,k] = dist.max(1)[0] + torch.rand_like(dist.max(1)[0])

                _, embed_ind = (-dist).max(1) # Quantization
            
            if noise_neighbor:
                #noise injection of variance sigma
                noise = torch.randn_like(embed_ind,dtype = torch.float16)*self.sigma
                embed_ind = (embed_ind + noise.int())%self.n_embed
            self.jitter_bool = False


            if verbose:
                aux = torch.zeros(self.n_embed)
                for ind in embed_ind:
                    aux[ind]+=1;
                print(f"non-zero indices: {torch.count_nonzero(aux)}, {torch.nonzero(aux).squeeze(1)} ")

            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)

            embed_ind = embed_ind.view(*input.shape[:-1])

            embed_onehot_sum = embed_onehot.sum(0) #n_i

            embed_sum = flatten.transpose(0, 1) @ embed_onehot # sum_j z_{i,j}


            cluster_on = self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            avg_on = self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            #Update the cluster size only if it is bigger than zero 
            self.embed_avg = torch.where(self.cluster_size>0, avg_on,embed_sum)
            self.cluster_size = torch.where(self.cluster_size>0, cluster_on,embed_onehot_sum)

            #print(f"cluster_size: {self.cluster_size}")
        

            embed_normalized = torch.where(self.cluster_size.data > 0., self.embed_avg.data / (self.cluster_size.data.unsqueeze(0)+self.eps), self.embed.data)
            self.embed.data.copy_(embed_normalized)
            self.cluster_freq.add_(embed_onehot_sum)
            if min_freq:
                cond = self.cluster_freq/self.cluster_freq.sum() < 0.01/self.n_embed
                if torch.any(cond):
                    self.jitter_bool = True
                cond = cond[:,None].expand(-1,self.dim).T
                inds = torch.randint(0,flatten.shape[0]-1,(self.n_embed,1)).squeeze()
                self.embed = torch.where(cond,flatten[inds].T,self.embed)
            
        else: #not Training
            print(f"max cluster size: {max(self.cluster_size)}, ind {torch.argmax(self.cluster_size)}")
            print(f"min cluster size: {min(self.cluster_size)}, ind {torch.argmin(self.cluster_size)}")
            s = self.cluster_freq/self.cluster_freq.sum()
            print(f"max freq: {max(s)}, ind {torch.argmax(s)}")
            print(f"min freq: {min(s)}, ind {torch.argmin(s)}")

            aux = torch.zeros(self.n_embed)
            for ind in embed_ind:
                aux[ind]+=1;
            print(f"non-zero indices: {torch.count_nonzero(aux)}, {torch.nonzero(aux).squeeze(1)} ")
            embed_ind = embed_ind.view(*input.shape[:-1])
            print(f"embed_ind:{embed_ind}")


        quantize = self.embed_code(embed_ind)
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        #print(f"embed_ind:{embed_ind}")
        return quantize, diff


        
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))



