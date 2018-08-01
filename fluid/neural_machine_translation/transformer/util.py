import sys
import re
import six
import unicodedata

# Regular expression for unescaping token strings.
# '\u' is converted to '_'
# '\\' is converted to '\'
# '\213;' is converted to unichr(213)
# Inverse of escaping.
_UNESCAPE_REGEX = re.compile(r"\\u|\\\\|\\([0-9]+);")

# This set contains all letter and number characters.
_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
        unicodedata.category(six.unichr(i)).startswith("N")))


def unescape_token(escaped_token):
    """
    Inverse of encoding escaping.
    """

    def match(m):
        if m.group(1) is None:
            return u"_" if m.group(0) == u"\\u" else u"\\"

        try:
            return six.unichr(int(m.group(1)))
        except (ValueError, OverflowError) as _:
            return u"\u3013"  # Unicode for undefined character.

    trimmed = escaped_token[:-1] if escaped_token.endswith(
        "_") else escaped_token
    return _UNESCAPE_REGEX.sub(match, trimmed)


def subtoken_ids_to_str(subtoken_ids, vocabs):
    """
    Convert a list of subtoken(word piece) ids to a native string.
    Refer to SubwordTextEncoder in Tensor2Tensor. 
    """
    subtokens = [vocabs.get(subtoken_id, u"") for subtoken_id in subtoken_ids]

    # Convert a list of subtokens to a list of tokens.
    concatenated = "".join([
        t if isinstance(t, unicode) else t.decode("utf-8") for t in subtokens
    ])
    split = concatenated.split("_")
    tokens = []
    for t in split:
        if t:
            unescaped = unescape_token(t + "_")
            if unescaped:
                tokens.append(unescaped)

    # Convert a list of tokens to a unicode string (by inserting spaces bewteen
    # word tokens).
    token_is_alnum = [t[0] in _ALPHANUMERIC_CHAR_SET for t in tokens]
    ret = []
    for i, token in enumerate(tokens):
        if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
            ret.append(u" ")
        ret.append(token)
    seq = "".join(ret)

    return seq.encode("utf-8")
