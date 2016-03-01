#!/usr/bin/env python
# run as:
# ipython -i cptm/crunch/test_sample_from.py data/config.json
# from base directory of cptm. Make sure that directory is in PYTHONPATH, e.g.:
# export PYTHONPATH=./
import numpy as np
import scipy as sc
import crunch
import cptm

import argparse

from cptm.utils.experiment import load_config, get_corpus, get_sampler

parser = argparse.ArgumentParser()
parser.add_argument('json', help='json file containing experiment '
                    'configuration.')
args = parser.parse_args()

config = load_config(args.json)
corpus = get_corpus(config)

nTopics = config.get('expNumTopics')

sampler = get_sampler(config, corpus, nTopics[0])

# actual test
p = sc.stats.norm.pdf(range(100), 50, 10)
rng = cptm.crunch.Sampler()
rng2 = cptm.crunch.Sampler2()

cppsamp = [cptm.crunch.sample_from(p, rng) for i in range(1000000)]
cppsamp2 = [cptm.crunch.sample_from2(p, rng) for i in range(1000000)]
cppsamp3 = [cptm.crunch.sample_from3(p, rng2) for i in range(1000000)]
pysamp = [sampler.sample_from(p) for i in range(1000000)]

# ipython timing, total (useful for getting average time, which timeit doesn't)
"""
%time _ = [cptm.crunch.sample_from(p, rng) for i in range(1000000)]
%time _ = [cptm.crunch.sample_from2(p, rng) for i in range(1000000)]
%time _ = [cptm.crunch.sample_from3(p, rng2) for i in range(1000000)]
%time _ = [sampler.sample_from(p) for i in range(1000000)]

results:
CPU times: user 1.84 s, sys: 62.1 ms, total: 1.9 s
Wall time: 1.93 s
CPU times: user 702 ms, sys: 0 ns, total: 702 ms
Wall time: 701 ms
CPU times: user 702 ms, sys: 0 ns, total: 702 ms
Wall time: 702 ms
CPU times: user 5.9 s, sys: 0 ns, total: 5.9 s
Wall time: 5.9 s
"""

plt.hist(cppsamp, bins=100, alpha=0.5, range=(0,100))
plt.hist(cppsamp2, bins=100, alpha=0.5, range=(0,100))
plt.hist(cppsamp3, bins=100, alpha=0.5, range=(0,100))
plt.hist(pysamp, bins=100, alpha=0.5, range=(0,100))
plt.plot(np.arange(100)+0.5, p*1000000)
plt.show()

# ipython timing
"""
%timeit cptm.crunch.sample_from(p, rng)
%timeit cptm.crunch.sample_from2(p, rng)
%timeit cptm.crunch.sample_from3(p, rng2)
%timeit sampler.sample_from(p)

results:
1000000 loops, best of 3: 1.63 µs per loop
The slowest run took 12.95 times longer than the fastest. This could mean that an intermediate result is being cached 
1000000 loops, best of 3: 626 ns per loop
The slowest run took 11.28 times longer than the fastest. This could mean that an intermediate result is being cached 
1000000 loops, best of 3: 613 ns per loop
The slowest run took 6.26 times longer than the fastest. This could mean that an intermediate result is being cached 
100000 loops, best of 3: 5.75 µs per loop
"""
