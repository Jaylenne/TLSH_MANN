import numpy as np
from lib_lsh import *
import scipy.stats as sc
from dpe_tcam import *
from simArrayPy import simArrayPy


def cossim(v1, v2):
    """
    Cosine similarity
    """
    v1 = v1 / np.linalg.norm(v1, keepdims=True)
    v2 = v2 / np.linalg.norm(v1, keepdims=True)

    return np.dot(v1, v2.T)


def cossim_mat(v1, v2, axis):
    v1 = v1 / np.linalg.norm(v1, axis=axis, keepdims=True)
    v2 = v2 / np.linalg.norm(v2, axis=axis, keepdims=True)

    return np.dot(v1, v2.T)


def cosup(v1, v2):
    """
    Cosine update rule for two keys
    """
    v = v1 + v2

    return v / np.linalg.norm(v, keepdims=True)


def memoryupdate_cos(mem, memkey):
    """
    Cosine update
    """
    memory = []
    memorykey = []
    for i in range(mem.shape[0] // 5):
        memory.append(mem[i * 5])
        memorykey.append(memkey[i * 5])

    for i in range(4):
        Query = []
        query_key = []
        for j in range(mem.shape[0] // 5):
            Query.append(mem[i + 1 + j * 5])
            query_key.append(memkey[i + 1 + j * 5])
        for l, v in enumerate(Query):
            dist = []
            for k, m in enumerate(memory):
                dist.append(cossim(v, m))
            index = np.argmax(np.array(dist))
            if memorykey[index] == query_key[l]:
                memory[np.argmax(dist)] = cosup(v, memory[np.argmax(dist)])
            else:
                memory.append(v)
                memorykey.append(query_key[l])

    return np.array(memory), np.array(memorykey)


def memoryupdate_binary(mem_lsh, memkey):
    """
    core of binary memory update scheme
    """

    # counter
    def numcount(vec):
        a = np.zeros(vec.shape)
        for i, v in enumerate(vec):
            if v == 1:
                a[i] = 1

            elif v == 0:
                a[i] = -1

        return a

    # update

    def denumcount(vec):
        a = np.zeros(vec.shape)
        for i, v in enumerate(vec):
            if v > 0:
                a[i] = 1
            elif v < 0:
                a[i] = 0
            elif v == 0:
                a[i] = 3

        return a

    memory = []
    key = []
    count = []
    for i in range(mem_lsh.shape[0] // 5):
        memory.append(mem_lsh[i * 5])
        key.append(memkey[i * 5])
        count.append(numcount(mem_lsh[i * 5]))

    for i in range(4):
        Query = []
        query_key = []
        for j in range(mem_lsh.shape[0] // 5):
            Query.append(mem_lsh[i + 1 + j * 5])
            query_key.append(memkey[i + 1 + j * 5])
        for k, v in enumerate(Query):
            dist = []
            for l, m in enumerate(memory):
                dist.append(np.sum(tcam_logicalxor(v, m)))
            if key[np.argmin(dist)] == query_key[k]:
                count[np.argmin(dist)] += numcount(v)
                memory[np.argmin(dist)] = denumcount(count[np.argmin(dist)])
            else:
                memory.append(v)
                key.append(query_key[k])
                count.append(numcount(v))

    return np.array(memory), np.array(key)


def binaryupdate(m_lsh, q_lsh, cnn_memkey, cnn_querykey):
    # binary update all
    mem_lsh = []
    query_lsh = []
    memkey = []
    querykey = []
    k = 0
    mem_lsh.append(m_lsh[k])
    query_lsh.append(q_lsh[k])
    memkey.append(cnn_memkey[k])
    querykey.append(cnn_querykey[k])
    k = 1
    memory, key = memoryupdate_binary(m_lsh[k], cnn_memkey[k])
    mem_lsh.append(memory)
    memkey.append(key)
    query_lsh.append(q_lsh[k])
    querykey.append(cnn_querykey[k])
    k = 2
    mem_lsh.append(m_lsh[k])
    query_lsh.append(q_lsh[k])
    memkey.append(cnn_memkey[k])
    querykey.append(cnn_querykey[k])
    k = 3
    memory, key = memoryupdate_binary(m_lsh[k], cnn_memkey[k])
    mem_lsh.append(memory)
    memkey.append(key)
    query_lsh.append(q_lsh[k])
    querykey.append(cnn_querykey[k])

    return np.array(mem_lsh), np.array(query_lsh), np.array(memkey), np.array(querykey)


def lshinfer(mem, memkey, query, querykey, hashbits=[8, 16, 32, 64, 128], num=10):
    # lsh inference without update
    Accuracy = []
    for n in hashbits:
        Accur = []
        for r in range(num):
            a = np.random.randn(64, n)
            mem_lsh = (np.dot(mem, a) > 0).astype(int)
            query_lsh = (np.dot(query, a) > 0).astype(int)
            y_pred = []
            for i in query_lsh:
                h_dist = []
                for j in mem_lsh:
                    h_dist.append(np.sum(tcam_logicalxor(i, j)))
                h_dist = np.array(h_dist)
                y_pred.append(memkey[np.argmin(h_dist)])
            acc = np.sum((y_pred == querykey).astype(int)) / len(querykey)
            Accur.append(acc)
        Accuracy.append(Accur)

    return np.array(Accuracy)


def lshinfer_update(mem, query, memkey, querykey, hashbits=128, num=10):
    """
    lsh inference with binary update
    """
    Acc_total = []
    for i in range(num):
        hash_plane = np.random.randn(64, hashbits)
        lsh_mem = []
        lsh_query = []
        lsh_memkey = []
        lsh_querykey = []
        # hashing
        for j in range(len(mem)):
            lsh_mem.append((np.dot(mem[j], hash_plane) > 0).astype(int))
            lsh_query.append((np.dot(query[j], hash_plane) > 0).astype(int))
            lsh_memkey.append(memkey[j])
            lsh_querykey.append(querykey[j])
        # binary update
        mem_lsh, query_lsh, mkey, qkey = binaryupdate(
            lsh_mem, lsh_query, lsh_memkey, lsh_querykey)
        # hashing inference
        Accur = []
        for j in range(len(mem)):
            y_pred = []
            for q in query_lsh[j]:
                h_dist = []
                for m in mem_lsh[j]:
                    h_dist.append(np.sum(tcam_logicalxor(q, m)))
                h_dist = np.array(h_dist)
                y_pred.append(mkey[j][np.argmin(h_dist)])
            acc = np.sum((y_pred == qkey[j]).astype(int)) / len(qkey[j])
            Accur.append(acc)

        Acc_total.append(Accur)

    return np.array(Acc_total)


def programerr(G, std=5):
    # program erorr sim
    return np.abs(G + np.random.randn(G.shape[0], G.shape[1]) * std)


def Ierror1(I, slope, intercept, loc, scale):
    # readout error simulation
    error = np.zeros(I.shape)
    error = I * slope + intercept
    error = np.exp(np.log(
        error) + np.random.randn(I.shape[0], I.shape[1]) * (np.random.randn() * scale + loc))

    return error


def getcurrent(vec, dpe, slope):
    # differential current with experimental correction
    vec_neg, vec_pos = vec_pn(vec)
    I_neg = dpe.read_current(vec_neg.T * 0.2) * slope
    I_neg = I_neg + \
            np.random.randn(I_neg.shape[0], I_neg.shape[1]) * Ierror1(I_neg)
    I_pos = dpe.read_current(vec_pos.T * 0.2) * slope
    I_pos = I_pos + \
            np.random.randn(I_pos.shape[0], I_pos.shape[1]) * Ierror1(I_pos)
    I = I_pos - I_neg

    return I.T


def Gdrift_large(G, slope, intercept, scale):
    # Conductance drift
    loc = np.zeros(G.shape)
    for rr in range(G.shape[0]):
        for cc in range(G.shape[1]):
            if G[rr, cc] < 50:
                loc[rr, cc] = slope * np.log(G[rr, cc]) + intercept
            else:
                loc[rr, cc] = slope * np.log(50) + intercept

    drift = np.exp(loc + scale * np.random.randn(G.shape[0], G.shape[1]))

    return G + np.random.randn(G.shape[0], G.shape[1]) * drift


def Gdrift(G, slope, intercept, scale):
    # Conductance fluctuation for <50uS
    loc = np.zeros(G.shape)
    loc = slope * np.log(G) + intercept

    drift = np.exp(loc + scale * np.random.randn(G.shape[0], G.shape[1]))

    return np.abs(G + np.random.randn(G.shape[0], G.shape[1]) * drift)


def linearfit(qs, qs_lsh, hashbits=np.arange(8, 129, 4)):
    # correlation of cosine distance and
    def cosdis(v1, v2):
        cossim = (v1 / np.linalg.norm(v1, keepdims=True)
                  ) @ (v2 / np.linalg.norm(v2, keepdims=True)).T
        return 1 - cossim

    def hamming(b1, b2):
        return np.sum(tcam_logicalxor(b1, b2))

    totalHamming = []
    totalCosine = []
    totalr_value = []
    totalslope = []
    totalintercept = []

    Cosine = []
    for i in range(qs.shape[0]):
        for j in range(qs.shape[0]):
            Cosine.append(cosdis(qs[i], qs[j]))

    Cosine = np.array(Cosine)
    totalCosine.append(Cosine)

    for h in hashbits:
        Hamming = []
        for i in range(qs.shape[0]):
            for j in range(qs.shape[0]):
                Hamming.append(hamming(qs_lsh[i][:h], qs_lsh[j][:h]))

        Hamming = np.array(Hamming)
        totalHamming.append(Hamming)

        slope, intercept, r_value, p_value, std_err = sc.linregress(
            Hamming, Cosine)
        totalr_value.append(r_value)
        totalslope.append(slope)
        totalintercept.append(intercept)

    return totalHamming, totalCosine, totalr_value


def crossbarlsh(lsh_mem, lsh_query, lsh_memkey, lsh_querykey, G, hashbits=128, slope=0.782, intercept=-2.168, scale=1,
                num=1, bias=1, method='ACM'):
    # crossbar lsh simulation
    """
    slope, intercept, scale: parameters determined conductance fluctuation
    """
    Accuracy = []
    m_lsh = []
    q_lsh = []
    for i in range(len(lsh_mem)):
        Acc = []
        mem = lsh_mem[i]
        query = lsh_query[i]
        memory_lsh = np.zeros((mem.shape[0], hashbits))
        query_lsh = np.zeros((query.shape[0], hashbits))
        memkey = lsh_memkey[i]
        querykey = lsh_querykey[i]
        for nn in range(num):
            for j, mm in enumerate(mem):
                mm_neg, mm_pos = vec_pn(mm)
                g = np.abs(Gdrift(G, slope, intercept, scale))
                I_pos = np.dot(mm_pos, g) * 0.2
                g = np.abs(Gdrift(G, slope, intercept, scale))
                I_neg = np.dot(mm_neg, g) * 0.2
                I = I_pos - I_neg
                I = np.squeeze(I)
                if method == 'ACM':
                    memory_lsh[j] = ((I[:-1] - I[1:]) > 0).astype(int)
                elif method == 'TLSH':
                    memory_lsh[j] = tlsh((I[:-1] - I[1:]), bias)
            for j, qq in enumerate(query):
                qq_neg, qq_pos = vec_pn(qq)
                g = np.abs(Gdrift(G, slope, intercept, scale))
                I_pos = np.dot(qq_pos, g) * 0.2
                g = np.abs(Gdrift(G, slope, intercept, scale))
                I_neg = np.dot(qq_neg, g) * 0.2
                I = I_pos - I_neg
                I = np.squeeze(I)
                if method == 'ACM':
                    query_lsh[j] = ((I[:-1] - I[1:]) > 0).astype(int)
                elif method == 'TLSH':
                    query_lsh[j] = tlsh((I[:-1] - I[1:]), bias)

            y_pred = []
            for i in query_lsh:
                h_dist = []
                for j in memory_lsh:
                    h_dist.append(np.sum(tcam_logicalxor(i, j)))
                h_dist = np.array(h_dist)
                y_pred.append(memkey[np.argmin(h_dist)])
            y_pred = np.array(y_pred)
            accuracy = np.sum((y_pred == querykey).astype(int)) / len(querykey)
            Acc.append(accuracy)
        Accuracy.append(Acc)
        m_lsh.append(memory_lsh)
        q_lsh.append(query_lsh)
    if num == 1:
        return np.array(Accuracy), np.array(m_lsh), np.array(q_lsh)
    else:
        return np.array(Accuracy)


def crossbartcam(m_lsh, q_lsh, lsh_memkey, lsh_querykey, std=5):
    # crossbar 2dpe-TCAM
    Accuracy = []
    for i in range(m_lsh.shape[0]):
        mem_lsh = m_lsh[i]
        query_lsh = q_lsh[i]
        m_key = lsh_memkey[i]
        q_key = lsh_querykey[i]
        tcam_stor = tcam_storage(mem_lsh, 150, 0, method='2dpe')
        tcam_stor = programerr(tcam_stor, std)
        search_input = np.zeros((2 * query_lsh.shape[1], query_lsh.shape[0]))
        for i in range(query_lsh.shape[0]):
            search_input[:, i] = tcam_input(
                query_lsh[i], 1, 0, '2dpe').reshape(-1)

        Accuracy.append(np.sum(
            (m_key[np.argmin((search_input.T * 0.2) @ tcam_stor, axis=1)] == q_key).astype(int)) / query_lsh.shape[0])

    return Accuracy


def g_reconstruct(g, r_size=128, c_size=128, **kwargs):
    """
    input: Weight conductance before adding wire resistance /(uS)
    output: Effecitive conductance matrix after adding wire resistance
    """
    rw = kwargs['rw'] if 'rw' in kwargs.keys() else 1e-6

    n_r = np.ceil(g.shape[0] / r_size).astype(int)
    n_c = np.ceil(g.shape[1] / c_size).astype(int)
    g_new = np.zeros((r_size * n_r, c_size * n_c))
    g_new[:g.shape[0], :g.shape[1]] = g
    g_eff = np.zeros((n_r * r_size, g.shape[1]))
    c_last = g.shape[1] - (n_c - 1) * c_size
    for i, g_i in enumerate(np.hsplit(g_new, n_c)):
        if i == n_c - 1:
            g_i = g_i[:, :c_last]
        for j, g_j in enumerate(np.vsplit(g_i, n_r)):
            g_eff[j * r_size:(j + 1) * r_size, i * c_size:i * c_size +
                                                          g_j.shape[1]] = simArrayPy(g_j, rw).geff

    return g_eff[:g.shape[0], :]
