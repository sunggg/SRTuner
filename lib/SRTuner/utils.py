import numpy as np
import fast_histogram
from anytree import PostOrderIter
import gc

# Define constants
FLOAT_MIN = (-1)*float('inf')
FLOAT_MAX = float('inf')


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def convert_encoding_to_dict(search_space, opt_stage_mapping, encoding):
    # opt-stage mapping is the same order w/ encoding
    opt_setting = dict()
    for opt_name, config in zip(opt_stage_mapping, encoding.split(',')):
        opt_setting[opt_name] = search_space[opt_name].configs[int(config)]
    return opt_setting

        
def convert_dict_to_encoding(self):
    pass

def delete_tree(root):
    for node in PostOrderIter(root):
        del node
    gc.collect()


def default_reward_func(perf, history, batch_size):
    # hyperparameters for reward calc
    # [TODO] Design reward policy that has less hyperparams
    C = 40
    window=700
    max_reward=100
    reward_margin = -0.03

    num_trials = len(history)
    ratio = history[-1]/perf

    # [TODO] Simplify.
    reward = 0
    if num_trials > max(batch_size, num_trials-window):
        if ratio > 1+reward_margin:
            if ratio<1:
                factor = 10*(ratio-1)+1
            else:
                factor = 30*(ratio-1)+1

            factor = max(0, factor)
            reward = min(C*factor, max_reward)
    return reward


def getUCT(N, n, r):
    c = np.sqrt(2)
    p_exploit = r/n
    p_explore = c*(np.sqrt(np.log(N)/n))
    return p_exploit + p_explore


def post_order_traversal(node, statTable = None):
    children = node.children
    if len(children) == 0: # leaf node
        return node.history
    else:
        aggr = []
        for child in node.children:
            data = post_order_traversal(child, statTable)
            if statTable != None:
                config = int(get_subencoding(child.encoding, -1))
                if node.depth >= len(statTable):
                    print(node)
                assert config < len(statTable[node.depth])
                statTable[node.depth][config].extend(data)
            aggr.extend(data)

        return aggr

def getNormHist(bins, data):
    count = fast_histogram.histogram1d(data, len(bins), range=(min(bins), max(bins)))
    tot = sum(count)
    num = len(data)
    assert(num >= tot)
    count = np.append(count, [num-tot])
    return count/num



def checksum(statTable, numAvailFlagKinds):
    chk = -1
    for i in range(numAvailFlagKinds):
        s = 0
        for j in range(len(statTable[i])):
            s += len(statTable[i][j])

        if chk == -1: 
            chk = s
        else:
            assert(chk == s)


def smooth(p):
    eps = 0.00000001
    numZeros = 0
    for v in p:
        if v == 0.0:
            numZeros += 1
    numNZ = len(p) - numZeros
    new_p = []
    for v in p:
        if v == 0.0:
            new_p.append(eps/numZeros)
        else:
            new_p.append(v-eps/numNZ)
    return new_p


# Kullback–Leibler divergence
def getKLD(p, q):
    # smoothing
    sp = smooth(p)
    sq = smooth(q)
    kl = 0
    for i in range(len(sp)):
        kl += sp[i]*np.log(sp[i]/sq[i])

    if(kl < 0):
        print(kl)
    assert(kl>=0)
    return kl


def shuffle_encoding(encoding, shuffle_mask):
    toks = encoding.split(',')
    new_encoding = ""
    for idx in shuffle_mask:
        if len(new_encoding):
            new_encoding += "," + toks[idx]
        else:
            new_encoding += toks[idx]
    return new_encoding


def get_subencoding(encoding, l):
    if l == -1:
        return encoding.split(',')[-1]
    else:
        toks = encoding.split(',')[0:l+1]
        return ",".join(toks)