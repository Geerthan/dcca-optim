import torch

def cca_loss(H1, H2, proj_dim, device, dtype):
    
    r1 = 1e-4
    r2 = 1e-4
    epsilon = 1e-9
    
    m, o = H1.size()
    H1, H2 = H1.t(), H2.t()
    
    m1 = H1.mean(dim=1).unsqueeze(1)
    m2 = H2.mean(dim=1).unsqueeze(1)
    
    H1_zm = H1 - m1
    H2_zm = H2 - m2
    
    c = 1.0 / (m-1)        
    
    sigma_11 = c*H1_zm.mm(H1_zm.t()) + r1*torch.eye(o, device=device)
    sigma_12 = c*H1_zm.mm(H2_zm.t())
    sigma_22 = c*H2_zm.mm(H2_zm.t()) + r2*torch.eye(o, device=device)
    
    D1, V1 = torch.linalg.eigh(sigma_11)
    D2, V2 = torch.linalg.eigh(sigma_22)
    
    sigma_11_root_inv = V1.mm((D1 ** -0.5).diag()).mm(V1.t())
    sigma_22_root_inv = V2.mm((D2 ** -0.5).diag()).mm(V2.t())
    
    Tval = sigma_11_root_inv.mm(sigma_12).mm(sigma_22_root_inv)
    
    trace_TT = Tval.t().mm(Tval)
    U, V = torch.linalg.eigh(trace_TT)
    
    # Prevent nan results
    U = torch.where(U > epsilon, U, epsilon * torch.ones(U.shape, dtype=dtype, device=device))
    
    U = U.topk(proj_dim)[0]
    corr = torch.sum(U.sqrt())
    
    return -corr