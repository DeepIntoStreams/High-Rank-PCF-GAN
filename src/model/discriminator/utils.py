import torch


def matrix_power_two_batch(A, k):
    orig_size = A.size()
    A, k = A.flatten(0, -3), k.flatten()
    ksorted, idx = torch.sort(k)
    # Abusing bincount...
    count = torch.bincount(ksorted)
    nonzero = torch.nonzero(count, as_tuple=False)
    A = torch.matrix_power(A, 2 ** ksorted[0])
    last = ksorted[0]
    processed = count[nonzero[0]]
    for exp in nonzero[1:]:
        new, last = exp - last, exp
        A[idx[processed:]] = torch.matrix_power(
            A[idx[processed:]], 2 ** new.item())
        processed += count[exp]
    return A.reshape(orig_size)


def rescaled_matrix_exp(f, A):
    """
    An efficient way of computing the tensor exponential
    :param f:
    :param A:
    :return:
    """
    normA = torch.max(torch.sum(torch.abs(A), axis=-2), axis=-1).values
    more = normA > 1
    s = torch.ceil(torch.log2(normA)).long()
    s = normA.new_zeros(normA.size(), dtype=torch.long)
    s[more] = torch.ceil(torch.log2(normA[more])).long()
    A_1 = torch.pow(
        0.5, s.float()).unsqueeze_(-1).unsqueeze_(-1).expand_as(A) * A
    # print(A_1.shape)
    return matrix_power_two_batch(f(A_1), s)
