# SRTuner v.0.0.1
  SRTuner is a tuning strategy that searches for the best possible optimization setting for the given run-time environment. Within the tuning budget, SRTuner endeavors to expose important inter-relatonship between optimizations and leverage them to focus on the promising search subspace.
  To allow fast integration, SRTuner is built in the form of python library that provides tuning primitives. Users can build a standalone tuning framework with these primitives or adopt them into the existing tuning framework as a new tuning method. 

# Structure highlight
```
|- lib/
   |- SRTuner/
      |- core.py                  # SRTuner primitives
      |- utils.py                 # utility functions
|- demo/
   |- gcc
      |- tuner/
         |- common.py              # basic structure of the tuning framework
         |- srtuner.py             # a standalone tuning framework built w/ SRTuner
         |- baseline_tuners.py     # baseline tuning frameworks w/ prior approaches
      |- cBench/                   # representative benchmark applications
      |- gcc_opts.txt              # optimizations and their possible configurations
      |- tune_gcc.py               # script that tunes GCC optimizations for cBench
```


# Installation
1. Install Dependencies:
```
sudo apt-get install python3
pip3 install fast_histogram numpy anytree pandas
```

2. Set environmental variable:
```
export SRTUNER_HOME=/path/to/repo
export PYTHONPATH=${SRTUNER_HOME}/lib:${PYTHONPATH}
```

# Demo: build a standaline tuning framework w/ SRTuner
`demo/gcc` shows an example of building an independent tuning framework by using SRTuner primitives. 

**How to install**

1. Install SRTuner
2. Install gcc

**How to run**

`python3 tune_gcc.py`


# Demo: integrate SRTuner into AutoTVM
https://github.com/sunggg/tvm/tree/tvm-srtuner shows example of SRTuner integration. 

**How to install**
1. Install SRTuner
2. `git clone --recursive -b tvm-srtuner https://github.com/sunggg/tvm.git`
3. Install tvm with SRTuner by following instruction: https://tvm.apache.org/docs/install/from_source.html

**How to run**
1. `cd tutorial/srtuner/`
2. `python3 tune_relay_cuda.py`



