# coding: utf-8
"""Text processing frontend

All frontend module should have the following functions:

- text_to_sequence(text, p)
- sequence_to_text(sequence)

and the property:

- n_vocab

"""
from deepvoice3_paddle.frontend import en

# optinoal Japanese frontend
try:
    from deepvoice3_paddle.frontend import jp
except ImportError:
    jp = None

try:
    from deepvoice3_paddle.frontend import ko
except ImportError:
    ko = None

# if you are going to use the frontend, you need to modify _characters in
# symbol.py:
# _characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? ' + '¡¿ñáéíóúÁÉÍÓÚÑ'
try:
    from deepvoice3_paddle.frontend import es
except ImportError:
    es = None
