import torch

def clist_bps(N,T,S):
    lN = torch.ceil(torch.log2(N + (1-torch.sign(N))))    
    lT = torch.ceil(torch.log2(T + (1-torch.sign(T))))
    return S*(lT+lN)
def compN_bps(N,T,S):
    #lN = torch.ceil(torch.log2(N))
    lT = torch.ceil(torch.log2(T + (1-torch.sign(T))))
    lS = torch.ceil(torch.log2(S + (1-torch.sign(S))))
    return S*lT + (N+1)*lS
def compT_bps(N,T,S):
    return compN_bps(T,N,S)
