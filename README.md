# What is SEOBNRv4HM_ROM?

[SEOBNRv4HM_ROM](https://inspirehep.net/literature/1788551) is a frequency-domain regression model that computes waveforms emitted by binary black holes. It is the reduced-order model (ROM) of a time-domain regression model [SEOBNRv4HM](https://inspirehep.net/literature/1664599). The ROM ensures a speedup of the code by a factor $200$ when compared to the original time-domain model. This speedup is achieved using various techniques such as the split of the training dataset in several domains that are fitted separately, and the use of the singular value decomposition (SVD) to compress the dataset.

The Python code in this repository is for prototyping. I wrote the production code to include it in the oper-source software library [LALSUITE](https://git.ligo.org/lscsoft/lalsuite). The source code of the production code is in C and can be found [here](https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRSEOBNRv4HMROM.c).


