[![Build Status](https://travis-ci.org/jvdzwaan/cptm.svg?branch=develop)](https://travis-ci.org/jvdzwaan/cptm)

# Cross-Perspective Topic Modeling

A Gibbs sampler to do Cross-Perspective Topic Modeling, as described in

> Fang, Si, Somasundaram, & Yu (2012). Mining Contrastive Opinions on Political Texts using Cross-Perspective Topic Model. In proceedings of the fifth ACM international conference on Web Search and Data Mining. http://dl.acm.org/citation.cfm?id=2124306

## Installation

Install prerequisites.

    sudo apt-get install gfortran libopenblas-dev liblapack-dev

Clone the repository.

    git clone https://github.com/jvdzwaan/cptm.git
    cd cptm

Install the requirements (in virtual environment if desired).

    pip install -r requirements.txt

Compile Cython code.

    python setup.py build_ext --inplace

Add the cptm directory to the `PYTHONPATH` (otherwise the scripts don't work).

    export PYTHONPATH=$PYTHONPATH:.

Tests can be run with `nosetests` (don't forget to `pip install nose` if you're using a virtual environment).

## Saving CPTCorpus to disk

    from CPTCorpus import CPTCorpus
    
    corpus = CPTCorpus(files, testSplit=20)
    corpus.save('/path/to/corpus.json')

## Loading CPTCorpus from disk

    from CPTCorpus import CPTCorpus

    corpus2 = CPTCorpus.load('/path/to/corpus.json')

---
Copyright Netherlands eScience Center.

Distributed under the terms of the Apache2 license. See LICENSE for details.
