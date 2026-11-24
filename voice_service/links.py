"""Prepare FAQ web links for speech and browser navigation."""

import re
from urllib.parse import urlsplit


# Use URL characters rather than non-whitespace so a URL directly followed by
# Chinese text (for example, /点击网址) leaves the instructions intact.
WEB_URL = re.compile(
    r"(?<!\w)(?:https?://|www\.)[a-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+",
    re.ASCII | re.IGNORECASE,
)

SPOKEN_SYMBOLS = {
    ".": ("点", "dot"), ":": ("冒号", "colon"), "/": ("斜杠", "slash"),
    "-": ("短横线", "hyphen"), "_": ("下划线", "underscore"), "~": ("波浪号", "tilde"),
    "?": ("问号", "question mark"), "#": ("井号", "hash"), "=": ("等号", "equals sign"),
    "&": ("和号", "ampersand"), "%": ("百分号", "percent sign"), "@": ("艾特符号", "at sign"),
    "+": ("加号", "plus sign"), "!": ("感叹号", "exclamation mark"), "$": ("美元符号", "dollar sign"),
    "'": ("单引号", "apostrophe"), "*": ("星号", "asterisk"), ",": ("逗号", "comma"),
    ";": ("分号", "semicolon"), "(": ("左圆括号", "open parenthesis"), ")": ("右圆括号", "close parenthesis"),
    "[": ("左方括号", "open square bracket"), "]": ("右方括号", "close square bracket"),
}
SPOKEN_DIGITS = ("零一二三四五六七八九", "zero one two three four five six seven eight nine".split())


def _address(token):
    end = len(token.rstrip(".,;:!?"))
    excess = {closing: token.count(closing) - token.count(opening) for opening, closing in ("()", "[]")}
    while end and excess.get(token[end - 1], 0) > 0:
        excess[token[end - 1]] -= 1
        end -= 1
    return token[:end]


def spoken_link_references(text, language):
    chinese = language == "zh"
    language_index = 0 if chinese else 1

    def spell_link(match):
        address = _address(match[0])
        words = []
        for char in address:
            if char in SPOKEN_SYMBOLS:
                words.append(SPOKEN_SYMBOLS[char][language_index])
            elif char.isdigit():
                words.append(SPOKEN_DIGITS[language_index][int(char)])
            elif char.isupper():
                words.append(f"大写 {char}" if chinese else f"capital {char}")
            else:
                words.append(char.upper())
        if chinese:
            reading = "【网址读法：" + "，".join(words) + "】"
        else:
            reading = "[URL reading: " + ", ".join(words) + "]"
        return reading + match[0][len(address):]

    return WEB_URL.sub(spell_link, text)


def selected_faq_links(retrieval):
    matches = retrieval.get(retrieval.get("selected_metric"), [])
    links = {}
    for hit in matches:
        for match in WEB_URL.finditer(hit["passage"]):
            address = _address(match[0])
            if address.lower().startswith("www."):
                address = "https://" + address
            try:
                url = urlsplit(address)
            except ValueError:
                continue
            if url.hostname and url.username is None:
                links[address] = None
    return list(links)
