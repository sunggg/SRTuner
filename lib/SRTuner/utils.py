import numpy as np
import fast_histogram


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
    for opt_name, config in zip(opt_stage_mapping, encoding):
        opt_setting[opt_name] = search_space[opt_name].configs[int(config)]
    return opt_setting

        
def convert_dict_to_encoding(self):
    pass

def default_reward_func(perf, best_perf):
    # hyperparameters for reward calc
    # [TODO] Design reward policy that has less hyperparams
    reward_factor = 40
    window=500
    max_reward=100
    reward_margin = -0.03

    ratio = best_perf/perf
    reward = 0
    if ratio > 1+reward_margin:
        reward = min(reward_factor*(1+ratio), max_reward)
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
                assert(int(child.encoding[-1]) < len(statTable[node.depth]))
                statTable[node.depth][int(child.encoding[-1])].extend(data)
            aggr.extend(data)

        return aggr

def getNormHist(bins, data):
    count = fast_histogram.histogram1d(data, len(bins), range=(min(bins), max(bins)))
    #count, _, _ = plt.hist(data, numBins)
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
    new_encoding = ""
    for idx in shuffle_mask:
        new_encoding += encoding[idx]
    return new_encoding