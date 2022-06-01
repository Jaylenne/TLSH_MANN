import numpy as np


def hamming(vec, matrix):
    vec = np.expand_dims(vec, 0)
    vec_matrix = np.tile(vec, [np.shape(matrix)[0], 1])

    res = np.logical_xor(vec_matrix, matrix)
    hamming = np.sum(res, axis=1)

    return hamming


def norm(Vec):
    if len(Vec.shape) == 1:
        Vec = np.expand_dims(Vec, 0)

    V = np.zeros(Vec.shape)
    for i, vec in enumerate(Vec):
        vec = vec.reshape(-1)
        vec_pos = vec.copy()
        vec_pos[vec_pos < 0] = 0

        vec_neg = vec.copy()
        vec_neg[vec_neg > 0] = 0

        scaling = max(vec_pos.max(), -vec_neg.min())

        V[i] = vec/scaling

    return V


def vec_pn(Vec):
    # Make sure the vector is normalized
    if len(Vec.shape) == 1:
        Vec = np.expand_dims(Vec, 0)

    Vec_neg = np.zeros(Vec.shape)
    Vec_pos = np.zeros(Vec.shape)
    for i, vec in enumerate(Vec):
        vec = vec.reshape(-1)

        vec_pos = vec.copy()
        vec_pos[vec_pos < 0] = 0

        vec_neg = vec.copy()
        vec_neg[vec_neg > 0] = 0

        scaling = max(vec_pos.max(), -vec_neg.min())
        vec_neg = -vec_neg/scaling
        vec_pos = vec_pos/scaling

        Vec_neg[i] = vec_neg
        Vec_pos[i] = vec_pos
    return Vec_neg, Vec_pos


def tlsh(vec, bias):
    index = np.where(np.absolute(vec) < bias)
    vec[np.where(vec > 0)] = 1
    vec[np.where(vec <= 0)] = 0
    vec[index] = 3

    return vec


def tcam_logicalxor(vec1, vec2):
    a = np.zeros(vec1.shape)
    b = abs(vec1-vec2)
    for i, j in enumerate(b):
        if j > 1:
            b[i] = 0
    a = b
    return a


def post_curr_pn_acm(Ipos, Ineg):
    Ires = Ipos-Ineg
    h = (Ires[:, :-1] > Ires[:, 1:]).astype(int)
    h = np.squeeze(h)

    return h


def post_curr_pn_de(Ipos, Ineg):
    Ires = Ipos-Ineg
    h = (Ires[:, ::2] > Ires[:, 1::2]).astype(int)
    h = np.squeeze(h)

    return h


def post_curr_pn_tlsh(Ipos, Ineg, Ibias):
    Ires = Ipos-Ineg
    h = Ires[:, :-1]-Ires[:, 1:]
    h = tlsh(h, Ibias)
    h = np.squeeze(h)

    return h


def get_lsh_dpe(Vec, dpe, array, method='ACM', c_sel=[0, 64], **kwargs):
    '''Get hashing bits from a normalized vector'''
    tdly = kwargs['tdly'] if 'tdly' in kwargs.keys() else 1000
    Vread = kwargs['Vread'] if 'Vread' in kwargs.keys() else 0.2
    Ibias = kwargs['Ibias'] if 'Ibias' in kwargs.keys() else 2e-6
    Vec_neg, Vec_pos = vec_pn(Vec)

    Ipos = dpe.multiply(array,
                        Vec_pos.T,
                        c_sel=c_sel,
                        r_start=0, mode=0, Tdly=tdly)

    Ineg = dpe.multiply(array,
                        Vec_neg.T,
                        c_sel=c_sel,
                        r_start=0, mode=0, Tdly=tdly)

    if method == 'DE':
        return post_curr_pn_de(Ipos, Ineg)

    elif method == 'ACM':
        return post_curr_pn_acm(Ipos, Ineg)

    elif method == 'ROUND':
        return post_curr_pn_round(Ipos, Ineg)

    elif method == 'TLSH':
        return post_curr_pn_round_tlsh(Ipos, Ineg, Ibias=Ibias)


def dpe_pm(Vec, dpe, array, c_sel=[0, 64], **kwargs):
    tdly = kwargs['tdly'] if 'tdly' in kwargs.keys() else 1000
    Vec_neg, Vec_pos = vec_pn(Vec)

    Ipos = dpe.multiply(array,
                        Vec_pos.T,
                        c_sel=c_sel,
                        r_start=0, mode=0, Tdly=tdly)

    Ineg = dpe.multiply(array,
                        Vec_neg.T,
                        c_sel=c_sel,
                        r_start=0, mode=0, Tdly=tdly)

    Ires = Ipos-Ineg

    return Ires


def get_lsh_dpe_128(Vec, dpe, array1, array2, array3, method='ACM', c_sel=[0, 64], **kwargs):
    '''Get hashing bits from a normalized vector'''
    get_current = kwargs['get_current'] if 'get_current' in kwargs.keys(
    ) else False
    tdly = kwargs['tdly'] if 'tdly' in kwargs.keys() else 1000
    Vread = kwargs['Vread'] if 'Vread' in kwargs.keys() else 0.2
    Ibias = kwargs['Ibias'] if 'Ibias' in kwargs.keys() else 2e-6
    Vec_neg, Vec_pos = vec_pn(Vec)

    Ipos1 = dpe.multiply(array1,
                         Vec_pos.T,
                         c_sel=c_sel,
                         r_start=0, mode=0, Tdly=tdly)

    Ipos2 = dpe.multiply(array2,
                         Vec_pos.T,
                         c_sel=c_sel,
                         r_start=0, mode=0, Tdly=tdly)

    Ipos3 = dpe.multiply(array3,
                         Vec_pos.T,
                         c_sel=[0, 1],
                         r_start=0, mode=0, Tdly=tdly)

    Ineg1 = dpe.multiply(array1,
                         Vec_neg.T,
                         c_sel=c_sel,
                         r_start=0, mode=0, Tdly=tdly)

    Ineg2 = dpe.multiply(array2,
                         Vec_neg.T,
                         c_sel=c_sel,
                         r_start=0, mode=0, Tdly=tdly)

    Ineg3 = dpe.multiply(array3,
                         Vec_neg.T,
                         c_sel=[0, 1],
                         r_start=0, mode=0, Tdly=tdly)

    Ipos = np.concatenate((Ipos1, Ipos2, Ipos3), axis=1)
    Ineg = np.concatenate((Ineg1, Ineg2, Ineg3), axis=1)

    current = {'Ipos': Ipos, 'Ineg': Ineg}
    if method == 'DE':
        if get_current == False:
            return post_curr_pn_de(Ipos, Ineg)
        else:
            return post_curr_pn_de(Ipos, Ineg), current

    elif method == 'ACM':
        if get_current == False:
            return post_curr_pn_acm(Ipos, Ineg)
        else:
            return post_curr_pn_acm(Ipos, Ineg), current

    elif method == 'TLSH':
        if get_current == False:
            return post_curr_pn_tlsh(Ipos, Ineg, Ibias=Ibias)
        else:
            return post_curr_pn_tlsh(Ipos, Ineg, Ibias=Ibias), current

    elif method == 'both':
        if get_current == False:
            return post_curr_pn_acm(Ipos, Ineg), post_curr_pn_tlsh(Ipos, Ineg, Ibias=Ibias)
        else:
            return post_curr_pn_acm(Ipos, Ineg), post_curr_pn_tlsh(Ipos, Ineg, Ibias=Ibias), current
