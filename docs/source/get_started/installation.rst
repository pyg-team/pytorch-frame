Installation
============

:pyf:`PyTorch Frame` is available for `Python 3.10` to `Python 3.14` on Linux, Windows and macOS.

Installation via PyPI
---------------------

.. code-block:: bash

   pip install pytorch-frame

   # Install with optional dependencies
   pip install pytorch-frame[full]


Installation from master
------------------------

.. code-block:: bash

    pip install git+https://github.com/pyg-team/pytorch-frame.git


Installation for development
----------------------------

.. code-block:: bash

    git clone https://github.com/pyg-team/pytorch-frame.git
    cd pytorch-frame
    pip install -e .[dev]

    # Install with optional dependencies
    pip install -e .[dev,full]


Optional cuDF pandas acceleration
---------------------------------

:pyf:`PyTorch Frame` can opt into RAPIDS ``cudf.pandas`` to accelerate
internal pandas operations on GPU while preserving pandas fallback behavior for
unsupported operations. Install cuDF separately following RAPIDS instructions,
then enable the accelerator before importing :pyf:`PyTorch Frame`:

.. code-block:: bash

   PYTORCH_FRAME_USE_CUDF_PANDAS=1 python train.py

For notebooks or scripts where pandas may already have been imported, restart
the Python process and enable ``cudf.pandas`` before any pandas import. You can
also run scripts directly through RAPIDS:

.. code-block:: bash

   python -m cudf.pandas train.py
