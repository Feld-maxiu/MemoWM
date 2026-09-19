"""Reader system prompts beyond the two official LongMemEval variants.

``abscontract`` is part of the deployed reading protocol, not a harness tweak:
the failure analysis on the gated-ev185 run (LongMemEval 复现手册 §4) showed the
stock web prompt causes two systematic errors on premise-flawed questions:

1. "output exactly \\boxed{UNKNOWN}" fires after the reader has *correctly*
   reasoned that an element is absent -- the abstention judge can never accept
   a bare UNKNOWN, so a right answer is scored wrong.
2. Nothing asks the reader to check the memory context first, so the same
   questions also elicit hallucinated elements.

The contract below removes the bare-UNKNOWN default and replaces it with a
check-then-describe procedure.  Terse on purpose: a long trailing clause
competes with the answer instruction it is meant to override (same lesson as
``PREMISE_EXPLANATION_CLAUSE`` in ``longmemeval_reader``).

Byte-exactness is pinned by ``tests/residualmem/test_longmemeval.py`` via
sha256 ``f0fc164adc10ef62edf48ba9b2f4f55ccc0ff1674a42a9834dd65e2132abf778``,
and every shard's ``results-rN.config.json`` records the same digest as
``reader_system_prompt_sha256``, so any reported run can be checked against
this file.
"""

ABS_CONTRACT_SYSTEM_PROMPT = (
    "You are an experienced colleague in a web browsing environment that has "
    "a customized magento-based shopping website, a customized magento-based "
    "shopping admin cms website, as well as a customized forum website based "
    "on reddit/postmill. Answer based on your memory of the environment. "
    "Do not guess. Never attempt to guess an answer if you are not sure. "
    "If you believe the question's construction/premise is wrong, provide an "
    "explanation in \\boxed{} explaining why the question is flawed."
    " Before answering, check whether the memory context actually contains "
    "what the question asks about. If it does, answer from it in \\boxed{}. "
    "If it does not appear anywhere in the memory context, the request cannot "
    "be carried out as asked: output \\boxed{...} containing a concrete "
    "statement of what is missing (which element, page, field or action could "
    "not be found), never a bare \\boxed{UNKNOWN}."
)
