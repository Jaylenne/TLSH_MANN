import numpy as np


def tcam_input(vec, vsearch=1, r_start=0, method='2dpe'):
    """transfer search vector into voltage input"""

    vec = np.expand_dims(vec, 1)
    if method == '3dpe':
        vec = np.tile(vec, 3)
        for i in range(np.shape(vec)[0]):
            if vec[i][0] == 1:
                vec[i] = [1, -1, 0]
            elif vec[i][0] == 0:
                vec[i] = [0, -1, 1]
            else:
                vec[i] = [0, 0, 0]

    elif method == '2dpe':
        vec = np.tile(vec, 2)
        for i in range(np.shape(vec)[0]):
            if vec[i][0] == 1:
                vec[i] = [1, 0]
            elif vec[i][0] == 0:
                vec[i] = [0, 1]
            else:
                vec[i] = [0, 0]
    vec = np.reshape(vec, -1)
    vec = np.expand_dims(vec, 1)

    if vec.shape[0]<64:
        input_vec = np.zeros((64, 1))
        input_vec[r_start:r_start+np.shape(vec)[0], :] = vec*vsearch
        return input_vec
    elif vec.shape[0]>=64:
        return vec*vsearch


def tcam_storage(vec, G_on, G_off, method='2dpe'):
    """store info in memristor array """
    # input:
    #  array(int): which array
    #  vec(np.ndarray): info stored in TCAM [N, num_bits]
    stor_vec = []
    numRows = np.shape(vec)[0]
    numCols = np.shape(vec)[1]

    if method == '3dpe':
        for r in range(numRows):
            vec_in = []
            for c in range(numCols):
                if vec[r, c] == 1:
                    vec_in.extend([G_off, G_off, G_on])
                elif vec[r, c] == 0:
                    vec_in.extend([G_on, G_off, G_off])
                else:
                    vec_in.extend([G_off, G_off, G_off])
            stor_vec.append(vec_in)

    elif method == '2dpe':
        for r in range(numRows):
            vec_in = []
            for c in range(numCols):
                if vec[r, c] == 1:
                    vec_in.extend([G_off, G_on])
                elif vec[r, c] == 0:
                    vec_in.extend([G_on, G_off])
                else:
                    vec_in.extend([G_off, G_off])
            stor_vec.append(vec_in)

    stor_vec = np.array(stor_vec).T

    return stor_vec


def tcam_search(array, vec, dpe, c_sel=[0, 10]):
    """core part of tcam"""
    # Args:
    # array(int):which array to search
    # vec:the input

    # Return
    # the current of the output

    Iresult = dpe.multiply(array,
                           vec,
                           c_sel=c_sel,
                           r_start=0, mode=0, Tdly=1000)

    return Iresult
