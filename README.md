# psamnn: PyTorch Neural Net classification on the PSAM spam database
Bart Massey

This repository contains a couple of things:

* `personal.csv` contains instances from the
  [PSAM](http://www.cs.pdx.edu/~bart/papers/spam.pdf) spam
  corpus of some years ago. The instances consist of a name,
  a class (1 for spam, 0 for ham), and a vector of features
  obtained via big-bag-of-words and
  [SpamAssassin](https://spamassassin.apache.org/) analyses.

* Python code for a simple PyTorch Neural Net to classify
  the instances.

## Running

You will need PyTorch installed. Easiest is to say

    pip3 install -r requirements.txt

To run on the "personal" corpus with 10-way
cross-validation and a neighbor distance of 5, say

    python3 knn.py personal.csv 10 5

The output will consist of the accuracy for each
cross-validation split.
