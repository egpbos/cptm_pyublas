# Cross-Perspective Topic Modeling

A Gibbs sampler to do Cross-Perspective Topic Modeling, as described in

> Fang, Si, Somasundaram, & Yu (2012). Mining Contrastive Opinions on Political Texts using Cross-Perspective Topic Model. In proceedings of the fifth ACM international conference on Web Search and Data Mining. http://dl.acm.org/citation.cfm?id=2124306

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
