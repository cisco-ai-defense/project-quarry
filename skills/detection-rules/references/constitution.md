# Constitution

The constitution is the labeling policy for one profile. It does three jobs:

- The labeler reads it as its system prompt.
- Every positive in the evaluation is an item the labeler confirmed against it.
- The judge policy is a shorter, faithful version of it.

Changing the constitution changes the labels, so freeze it before labeling. Never edit it to raise
the judge's recall.

Keep the text that is sent between `<!-- PROMPT START -->` and `<!-- PROMPT END -->` markers. The
file can then carry notes, and the sent text stays exact.

## Sections, in this order

1. **The item.** Say what one unit is: the raw arguments of one tool call, one tool response body,
   one tool description, or one message. The user message starts with a header that names the unit
   kind and, when known, the tool:

       unit: tool_call
       tool: exec_command

   Tell the labeler to trust the header over the item's shape. One constitution may cover several
   surfaces. Say what decides the verdict per tool family: the command string for shell tools; the
   target path and written content for file tools.
2. **Labels.** One sub-label per taxonomy signature, plus the profile label, which is 1 when any
   sub-label is 1. Sub-labels are not exclusive; set every one that applies.
3. **Scope.** List what is in scope. List what is out of scope for this profile, naming the sibling
   profile that owns each case, including the overlaps that fire both profiles. Example:
   `sudo cat ~/.aws/credentials` is 0 for runtime execution and 1 for sensitive data access.
4. **Conservatism.** Judge effect and location, not the verb. Give concrete benign counterparts:
   `rm -rf ./build` is cleanup and `rm -rf /` is destructive; `cat README.md` is ordinary and
   `cat /etc/shadow` is not. Ordinary development work is 0: building, testing, installing from
   standard registries, and editing project files.
5. **Uncertainty.** Add a `context_dependent` label for items whose verdict depends on context the
   item does not carry, such as an unresolved variable in a target path. When it is 1, every
   sub-label is 0. The judge maps it to 0. It is not a hedge for items the labeler can judge.
6. **Modifiers** (optional), recorded independently of the labels: `obfuscated`, `encoded`,
   `traversal`.
7. **Examples.** Write 8 to 14 short ones with documentation-safe placeholders. Cover every
   sub-label and the overlaps with sibling profiles.
8. **Output.** One JSON object with every label key and a `reason` of at most 12 words. The reply
   begins with `{` and ends with `}`, with no narration.

After the end marker, add **Notes on choices**. Record each place the constitution departs from the
taxonomy's wording, and why. Example: `sudo` that only installs packages from a distribution's
default repositories is 0, because it is the most common privileged command in benign agent
traffic. The deterministic signatures still flag it, and the report measures the difference.

## Mapping to the taxonomy

Copy each profile's definitions and taxonomy mapping (objective, technique, sub-technique)
verbatim from the taxonomy source. Do not paraphrase them. The generated rule's metadata uses the
same strings.

When your organization publishes constitutions in a shared taxonomy repository, follow that
repository's format and process, and keep the labeling copy and the published copy in step.
