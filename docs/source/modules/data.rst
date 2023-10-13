torch_frame.data
================

.. contents:: Contents
    :local:

.. currentmodule:: torch_frame.data

Data Objects
------------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   {% for name in torch_frame.data.data_classes %}
     {{ name }}
   {% endfor %}

Stats
-----

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   {% for name in torch_frame.data.stats_classes %}
     {{ name }}
   {% endfor %}

Data Loaders
------------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   {% for name in torch_frame.data.loader_classes %}
     {{ name }}
   {% endfor %}

Helper Functions
----------------

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for name in torch_frame.data.helper_functions %}
     {{ name }}
   {% endfor %}
