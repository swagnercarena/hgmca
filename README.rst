===============================================================================================
hgmca - A hierarchical component separation algorithm based on sparsity in the wavelet basis.
===============================================================================================
.. image:: https://travis-ci.com/swagnercarena/hgmca.png?branch=master
	:target: https://travis-ci.org/swagnercarena/hgmca

.. image:: https://readthedocs.org/projects/hgmca/badge/?version=latest
	:target: https://hgmca.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status

.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/swagnercarena/hgmca/blob/s2let/LICENSE

.. image:: https://img.shields.io/badge/arXiv-1910.08077%20-yellowgreen.svg
    :target: https://arxiv.org/abs/1910.08077

``HGMCA`` is a source separation package primarily aimed towards use on the Cosmic Microwave Background. The software package includes implementations of gmca and hgmca as described in `Wagner-Carena et al 2019 <https://arxiv.org/abs/1910.08077>`_. For ease of use, we have included a number of `demos <https://github.com/swagnercarena/hgmca/tree/master/demos>`_ with the code.

Installation
------------

1. Clone the repo for hgmca:

.. code-block:: bash

	$ git clone https://github.com/swagnercarena/hgmca

2.	For ease of use and installation, HGMCA includes an implementation of the `s2let library <http://astro-informatics.github.io/s2let/scratch_install.html>`_ in python. To improve the fidelity of the alm transform, healpy requires us to download some weights. Go into the hgmca directory and run the git clone command.

 .. code-block:: bash

	$ cd <HGMCA_path>
	$ git clone https://github.com/healpy/healpy-data.git

3. Now all we need to do is install our package! The -e option allows you to pull any updates to hgmca and have them automatically take effect.

.. code-block:: bash

	$ pip install -e . -r requirements.txt
