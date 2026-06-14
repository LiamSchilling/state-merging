# State-Merging Inference Algorithms

A parameterized implementation of state-merging inference algorithms for finite-state automata and transducers, which specializes to known such algorithms including RPNI, ALERGIA, OSTIA, and APTI2.

## Project Structure

- [`src/`](https://github.com/LiamSchilling/state-merging/tree/master/src)
    - [`state_merging/`](https://github.com/LiamSchilling/state-merging/tree/master/src/state_merging): A package exposing parameterized implementations of state-merging inference algorithms for finite-state automata and transducers.
        - [`automata/SFST.py`](https://github.com/LiamSchilling/state-merging/blob/master/src/state_merging/automata/SFST.py): Defines subsequential finite-state transducers (SFSTs).
        - [`operations/state_merging.py`](https://github.com/LiamSchilling/state-merging/blob/master/src/state_merging/operations/state_merging.py): Implements the iterative red/blue state-merging strategy.
        - [`operations/learner.py`](https://github.com/LiamSchilling/state-merging/blob/master/src/state_merging/operations/learner.py): Assembles the generalized learning algorithm.
        - [`algorithms/`](https://github.com/LiamSchilling/state-merging/tree/master/src/state_merging/algorithms): Provides the specialization to RPNI and OSTIA.
    - [`decision_tree_inference/`](https://github.com/LiamSchilling/state-merging/tree/master/src/decision_tree_inference): A package exposing parameterized implementations of decision tree inference algorithms, including a generalized version of the greedy splitting strategy à la ID3.
    - [`data/`](https://github.com/LiamSchilling/state-merging/tree/master/src/data): A package exposing data utilities, including access to the [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict).
- [`notebooks/`](https://github.com/LiamSchilling/state-merging/tree/master/notebooks): Jupyter notebooks with demonstrations and experiments using the packages exported by this project.

## References ([BibTeX](https://github.com/LiamSchilling/state-merging/blob/master/REFERENCES.bib))

- Akram, H. I., & de la Higuera, C. (2013). *Learning Probabilistic Subsequential Transducers from Positive Data*. In *ICAART 2013* (pp. 479–486).

- Angluin, D. (1987). *Learning Regular Sets from Queries and Counterexamples*. *Information and Computation*, 75(2), 87–106. https://doi.org/10.1016/0890-5401(87)90052-6

- de la Higuera, C. (2010). *Grammatical Inference: Learning Automata and Grammars*, Chapter 18. Cambridge University Press. https://doi.org/10.1017/CBO9781139194655

- Gildea, D., & Jurafsky, D. (1996). *Learning Bias and Phonological-Rule Induction*. *Computational Linguistics*, 22(4), 497–530.

- Mohri, M. (1997). *Finite-State Transducers in Language and Speech Processing*. *Computational Linguistics*, 23(2), 269–311.

- Oncina, J., García, P., & Vidal, E. (1993). *Learning Subsequential Transducers for Pattern Recognition Interpretation Tasks*. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 15(5), 448–458. https://doi.org/10.1109/34.211465
