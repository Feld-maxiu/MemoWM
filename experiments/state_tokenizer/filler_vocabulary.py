"""Lorem-ipsum vocabulary for the DOM long-span filler filter.

Why a hand-audited constant rather than a fitted lexicon: two fitted rules were
tried and both failed in opposite directions. Keying protection on the DOM tag
put ``true`` (which carries checkbox state, as ``value="true"``), ``close``,
``search`` and ``results`` into the filler set. Keying it on span length instead
protected every real lorem word, because each one appears at least once in some
short attribute, leaving only truncation fragments.

The reason no global word list can work is that the same word is filler inside a
paragraph and content inside a label. That is a property of the *occurrence*, not
the word, so the filter pairs this vocabulary with a run-length requirement (see
``filler_runs``): a word is only stripped when it sits inside a run of
consecutive vocabulary words. A standalone ``Close`` label is never in such a
run; a 60-token lorem tail always is.

Entries were read one by one against the corpus. ``integer``, ``at``, ``et``,
``in``, ``non``, ``est``, ``sit``, ``cum``, ``id`` and ``sem`` are genuine lorem
words that are also ordinary English or Latin; they stay in because the run rule
makes them safe. ``true``/``close``/``search``/``results`` were removed after
tracing them to real UI text.
"""
from __future__ import annotations

# The classic lorem corpus as it appears in this dataset's generated filler.
LOREM_WORDS: frozenset[str] = frozenset("""
ac accumsan adipiscing aenean aliquam aliquet amet ante arcu at auctor augue
bibendum blandit commodo condimentum congue consectetur consequat convallis cras
cum curabitur cursus dapibus diam dictum dictumst dignissim dis dolor donec dui
duis egestas eget eleifend elementum elit enim eros erat est et etiam euismod
facilisi facilisis fames faucibus felis fermentum feugiat fringilla fusce
gravida hac habitant habitasse hendrerit iaculis id imperdiet in integer interdum
ipsum justo lacinia lacus laoreet lectus leo libero ligula lobortis lorem luctus
maecenas magna magnis malesuada massa mattis mauris metus mi molestie mollis
montes morbi mus nascetur natoque nec neque netus nibh nisi nisl non nulla nullam
nunc odio orci ornare parturient pellentesque penatibus pharetra phasellus placerat
platea porta porttitor posuere potenti praesent pretium proin pulvinar purus
quam quis quisque rhoncus ridiculus risus rutrum sagittis sapien scelerisque
sed sem semper senectus sit sociis sodales sollicitudin suscipit suspendisse
tellus tempor tempus tincidunt tortor tristique turpis ullamcorper ultrices
ultricies urna ut varius vehicula vel velit venenatis vestibulum vitae vivamus
viverra volutpat vulputate
""".split())

# The DOM truncates text/value attributes mid-word, so the tail of a long span is
# routinely a prefix of a lorem word. Those prefixes are filler by construction --
# but only from four characters up. Below that the rule starts eating real words:
# ``no`` is a prefix of ``non`` and is also a live button label in this corpus
# (the overlap_words probe scores it), and ``se``/``di``/``do`` are similarly
# ambiguous. The short fragments it therefore misses (``ege``, ``nun``, ``urn``)
# occur a handful of times each, which is the cheaper error to make.
_TRUNCATION_MIN = 4


def is_filler(word: str) -> bool:
    """True for a lorem word or a prefix of one left behind by truncation."""
    lowered = word.lower()
    if lowered in LOREM_WORDS:
        return True
    if len(lowered) < _TRUNCATION_MIN:
        return False
    # a bare prefix is only filler if it cannot stand alone as a lorem word
    return any(
        entry.startswith(lowered) and entry != lowered for entry in LOREM_WORDS
    )


def filler_runs(words: list[str], min_run: int) -> list[bool]:
    """Mark words that sit inside a run of ``min_run`` consecutive filler words.

    The run requirement is what makes the vocabulary safe to apply. ``sed`` in a
    lorem paragraph is surrounded by more lorem; ``no`` on a button stands alone.
    Only the former is stripped, so a word never has to be classified globally as
    content or filler -- each occurrence is judged by its neighbours.
    """
    if min_run < 1:
        raise ValueError("min_run must be >= 1")
    flags = [is_filler(word) for word in words]
    keep = [False] * len(words)
    start = 0
    while start < len(words):
        if not flags[start]:
            start += 1
            continue
        stop = start
        while stop < len(words) and flags[stop]:
            stop += 1
        if stop - start >= min_run:
            for index in range(start, stop):
                keep[index] = True
        start = stop
    return keep


def filler_token_mask(text: str, offsets, min_run: int) -> list[bool]:
    """Per-token filler mask for one span, given each token's char offsets.

    Dropping only the filler *words* is not enough: the punctuation they sat
    between survives as its own tokens, and that residue is what keeps a span
    over the raw-slot threshold. On the copy-paste case the value attribute goes
    27 -> 12 tokens by word removal alone, still long; clearing the trailing
    separators too brings it to 7, and the value earns a raw slot.

    A separator is dropped only when the token it follows was dropped. That is
    one-sided on purpose: the hyphen in ``state-88347`` follows ``state``, which
    is kept, so it survives -- while ``diam.`` loses its period along with the
    word.
    """
    import re as _re

    words = [(m.start(), m.end(), m.group()) for m in _re.finditer(r"[A-Za-z]+", text)]
    stripped = filler_runs([w for _, _, w in words], min_run)
    dead = [span for span, drop in zip(words, stripped) if drop]

    mask, previous_dropped = [], False
    for start, stop in offsets:
        piece = text[start:stop]
        if not piece.strip():                       # pure whitespace rides along
            mask.append(previous_dropped)
            continue
        if any(start < end and stop > begin for begin, end, _ in dead):
            mask.append(True)
            previous_dropped = True
            continue
        if not any(char.isalnum() for char in piece):
            mask.append(previous_dropped)           # separator left by a dead word
            continue
        mask.append(False)
        previous_dropped = False
    return mask
