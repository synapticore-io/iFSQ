import torch


def hsic_unbiased(K, L):
    """
    Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC) as per Equation 5 in the paper.
    > Reference: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    """
    m = K.shape[0]

    # Zero out the diagonal elements of K and L
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    # Compute HSIC using the formula in Equation 5
    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value


def hsic_biased(K, L):
    """Compute the biased HSIC (the original CKA)"""
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)


def cknna(feats_A, feats_B, topk=None, distance_agnostic=False, unbiased=True):
    """similarity only cka variant"""
    n = feats_A.shape[0]

    if topk < 2:
        raise ValueError("CKNNA requires topk >= 2")

    if topk is None:
        topk = feats_A.shape[0] - 1

    K = feats_A @ feats_A.T
    L = feats_B @ feats_B.T
    device = feats_A.device

    def similarity(K, L, topk):
        if unbiased:
            K_hat = K.clone().fill_diagonal_(float("-inf"))
            L_hat = L.clone().fill_diagonal_(float("-inf"))
        else:
            K_hat, L_hat = K, L

        # get topk indices for each row
        # if unbiased we cannot attend to the diagonal unless full topk
        # else we can attend to the diagonal
        _, topk_K_indices = torch.topk(K_hat, topk, dim=1)
        _, topk_L_indices = torch.topk(L_hat, topk, dim=1)

        # create masks for nearest neighbors
        mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_indices, 1)
        mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_indices, 1)

        # intersection of nearest neighbors
        mask = mask_K * mask_L

        if distance_agnostic:
            sim = mask * 1.0
        else:
            if unbiased:
                sim = hsic_unbiased(mask * K, mask * L)
            else:
                sim = hsic_biased(mask * K, mask * L)
        return sim

    sim_kl = similarity(K, L, topk)
    sim_kk = similarity(K, K, topk)
    sim_ll = similarity(L, L, topk)

    return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()
