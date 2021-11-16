from anytree import Node, RenderTree, AsciiStyle, LevelOrderGroupIter, PostOrderIter
from anytree.dotexport import RenderTreeGraph

from .utils import (
                    FLOAT_MAX, FLOAT_MIN, isfloat, getNormHist,
                    getKLD, getUCT, post_order_traversal, checksum, convert_encoding_to_dict, shuffle_encoding, delete_tree,
                    default_reward_func
                   )
import random
import pandas as pd
import numpy as np
import gc

class FlagInfo:
    def __init__(self, name, configs):
        self.name = name
        self.configs = configs

class Evaluator:
    def __init__(self, path, num_repeats):
        self.path = path
        self.num_repeats = num_repeats
    
    def build(self):
        assert 0, "Undefined"

    def run(self):
        assert 0, "Undefined"

    def evaluate(self):
        assert 0, "Undefined"

    def clean(self):
        assert 0, "Undefined"


# This module is designed to be called by tuning framework
# Tuning framework will use the following functions. 
#    - batch_candidate_generation()
#    - reflect feedback

class SRTunerModule():
    def __init__(
            self, 
            search_space,
            evaluator = None,
            reward_func = None,
            opt_stage_mapping = None,
            default_perf = FLOAT_MAX,
        ):
        self.search_space = search_space

        if reward_func is None:
            self.reward_func = default_reward_func
        else:
            self.reward_func = reward_func

         # We need default ordering
        if opt_stage_mapping is None:
            self.opt_stage_mapping = list(search_space.keys())
        else:
            self.opt_stage_mapping = opt_stage_mapping

        self.num_optimizations = len(self.opt_stage_mapping)
        self.default_perf = default_perf

        # Create root node for multi-stage structure
        if default_perf is None:
            self.root = Node(self.opt_stage_mapping[0], encoding="", num=0, reward=0, isDone=False, history=[])
        elif default_perf != FLOAT_MAX:
            self.root = Node(self.opt_stage_mapping[0], encoding="", num=0, reward=0, isDone=False, history=[default_perf])

        self.best_perf, self.worst_perf = FLOAT_MAX, FLOAT_MIN
        self.visited = set()
        self.trials = []
        self.shuffle_mask = []

        # Maintain state across invocations. 
        # [TODO] Can we remove this?
        self.current_candidate_nodes = []
        self.batch_size = 1


    # This will give you candidate and leaf node
    def traverse(self):
        cur_node = self.root
        while True:
            assert not self.root.isDone, "Search space is completely explored."
            numChildren = len(cur_node.children)
            numConfigs = len(self.search_space[cur_node.name].configs)

            if numChildren < numConfigs:
                # Need to explore. Random sampling
                while True:
                    chosenConfig = random.randint(0,numConfigs-1)
                    if not any([child.encoding[-1] == str(chosenConfig) for child in cur_node.children]):
                        break

                # Attach new config
                if len(cur_node.encoding):
                    new_encoding = cur_node.encoding + "," + str(chosenConfig)
                else:
                    new_encoding = str(chosenConfig)

                if cur_node.depth+1 == self.num_optimizations:
                     # leaf node
                    return Node("leaf", parent=cur_node, encoding=new_encoding, num=0, reward=0, isDone=True, history=[])
                else:
                    # other nodes
                    cur_node = Node(self.opt_stage_mapping[cur_node.depth+1], parent=cur_node, 
                                encoding=new_encoding, num=0, reward=0, isDone=False, history=[])   
            else:
                # Balance 
                UCT_N = cur_node.num 
                max_id, max_uct = -1, -1
                zero_nodes = []
                for i in range(numChildren):
                    childNode = cur_node.children[i]
                    if childNode.isDone == False:
                        if childNode.num == 0:
                            zero_nodes.append(childNode)
                        else:
                            uct = getUCT(cur_node.num, childNode.num, childNode.reward)
                            if uct > max_uct:
                                max_id, max_uct = i, uct

                if max_id == -1:
                    if len(zero_nodes):
                        cur_node = cur_node.children[random.randint(0, len(zero_nodes)-1)]
                    else:   
                        # All done
                        cur_node.isDone = True
                        cur_node = cur_node.parent
                else:
                    cur_node = cur_node.children[max_id]
        assert 0, "Should not reach here"

    # generate
    def generate_candidates(self, batch_size = 1):
        self.batch_size = batch_size # [TODO] Can we remove this?
        candidates, self.current_candidate_nodes = [], []
        for _ in range(batch_size):
            leaf_node = self.traverse()
            assert(leaf_node.encoding not in self.visited)
            opt_setting = convert_encoding_to_dict(self.search_space, self.opt_stage_mapping, leaf_node.encoding)
            candidates.append(opt_setting)
            self.current_candidate_nodes.append(leaf_node)
            self.visited.add(leaf_node.encoding)
  
        return candidates


    def remap(self, numBins = 100, minSamples=15):
        # Collect stats
        bins = np.linspace(self.best_perf, self.worst_perf, numBins)
        statTable = []
        for i, opt_name in enumerate(self.opt_stage_mapping):
            num_configs = len(self.search_space[opt_name].configs)
            statTable.append([[] for i in range(num_configs)])
        
        post_order_traversal(self.root, statTable)

        # Verify
        checksum(statTable, self.num_optimizations)

        # Distribution-based impact estimation w/ KL divergence
        t_hist, t_kls = 0, 0
        klData = []
        for optId in range(self.num_optimizations):
            perfs = statTable[optId]
            num_configs = len(perfs)
            flag = self.opt_stage_mapping[optId]

            if num_configs > 2:
                min_avg = FLOAT_MAX
                max_avg = FLOAT_MIN
                min_idx = -1
                max_idx = -1

                for j in range(num_configs):
                    if len(perfs[j]) == 0:
                        continue

                    avg = np.mean([perf for perf in perfs[j] if perf != FLOAT_MAX])

                    if avg > max_avg:
                        max_avg, max_idx = avg, j

                    if avg < min_avg:
                        min_avg, min_idx = avg, j

                if min_idx > max_idx:
                    tmp = min_idx
                    min_idx = max_idx
                    max_idx = tmp

                if( (min_idx < max_idx) and (min_idx>=0) ):
                    p = perfs[min_idx]
                    q = perfs[max_idx]
                    labels = [str(min_idx), str(max_idx)]
                else:
                    p = []
                    q = []

            else:
                p = perfs[0]
                q = perfs[1]
                labels = ["0", "1"]

            # Impact
            if (len(p) >= minSamples) and (len(q) >= minSamples):
                # without smoothing
                dist0 = getNormHist(bins, p)
                dist1 = getNormHist(bins, q)
                # smooth dists internally
                kl = getKLD(dist0, dist1)
            else:
                kl = 0

            klData.append([kl, flag])

            del perfs, p, q

        del bins, statTable 

        # Order optimization in the order of its impact
        dfKlData = pd.DataFrame(klData, columns=["KL", "optimization"])
        dfKlData.sort_values(by=["KL"], inplace=True, ascending=False)
        self.shuffle_mask = dfKlData["optimization"].index.values.tolist()

        # Update the impact
        self.opt_stage_mapping = dfKlData["optimization"].values.tolist()

        del klData, dfKlData
        gc.collect()

        root_num = self.root.num
        delete_tree(self.root)

        # Create root node for multi-stage structure
        if default_perf is None:
            self.root = Node(self.opt_stage_mapping[0], encoding="", num=0, reward=0, isDone=False, history=[])
        elif default_perf != FLOAT_MAX:
            self.root = Node(self.opt_stage_mapping[0], encoding="", num=0, reward=0, isDone=False, history=[default_perf])
        df_trials = pd.DataFrame(self.trials, columns=["encoding", "performance"])
        
        encodings = df_trials["encoding"].values
        perfs = df_trials["performance"].values
        numTrials = len(perfs)
    
        for i in range(numTrials):
            perf = perfs[i]
            encoding = encodings[i]
            encoding = shuffle_encoding(encoding, self.shuffle_mask)
            self.trials[i][0] = encoding
            
            depth = 0
            cur_node = self.root
            while depth < len(encoding):
                sub_encoding = encoding[0:depth+1]
                found = False
                for child in cur_node.children:
                    if child.encoding == sub_encoding:
                        found = True
                        break
                
                if found:
                    cur_node = child
                else:
                    if depth+1 < self.num_optimizations:
                        new_name = self.opt_stage_mapping[depth+1]
                        new_node = Node(new_name, encoding=sub_encoding, num=0, reward=0, isDone=False, history=[], parent=cur_node)
                    else:
                        new_node = Node("leaf", encoding=sub_encoding, num=0, reward=0, isDone=True, history=[], parent=cur_node)
                    cur_node = new_node
                depth = cur_node.depth
            self.backpropagate(cur_node, perf)

        assert(root_num == self.root.num)
        del perfs, encodings, df_trials
        gc.collect()
        

    # update 
    def backpropagate(self, leaf_node, perf):
        assert leaf_node is not None
        assert isfloat(perf)

        node_list = [ leaf_node ]
        node_list.extend(reversed(leaf_node.ancestors))
        root = node_list[-1]

        for node in node_list:
            reward = self.reward_func(perf, self.root.history, self.batch_size)
            node.num += 1
            node.reward += reward
            node.history.append(perf)
            node.history.sort(reverse=True)
        

    def reflect_feedback(self, perfs, remap_freq = 100):
        for leaf_node, perf in zip(self.current_candidate_nodes, perfs):
            self.backpropagate(leaf_node, perf)

            self.best_perf = min(self.best_perf, perf)
            if perf != FLOAT_MAX:
                self.worst_perf = max(self.worst_perf, perf)
            self.trials.append([leaf_node.encoding, perf])
        
        self.current_candidate_nodes = []

        if self.root.num % remap_freq == 0:
            self.remap()
    

    def tune(self, budget, batch_size=1):
        best_opt_setting, best_perf = None, FLOAT_MAX
        i = 0
        while i<budget:
            candidates = self.generate_candidates(batch_size=batch_size)
            perfs = self.evaluate_candidates(candidates)
            self.reflect_feedback(perfs)

            i += len(candidates)
            for opt_setting, perf in zip(candidates, perfs):
                if perf < best_perf:
                    best_perf = perf
                    best_opt_setting = opt_setting
            
            if best_perf == FLOAT_MAX:
                print(f"[{i}] FLOAT MAX")
            else:
                print(f"[{i}] {best_perf:.3f}")
        return best_opt_setting, best_perf


    def extract_synergy(self):
        assert 0, "[TODO]"