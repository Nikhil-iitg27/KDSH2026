"""
PromptTemplates for centralized prompt templates for all extraction tasks.
"""


class PromptTemplates:
    """Centralized prompt templates for all extraction tasks."""
    
    @staticmethod
    def character_extraction(chunk: str) -> str:
        return f"""Extract ONLY the proper character names mentioned in this text. List one name per line. Do not add names not present.

Text: {chunk}

Character names:"""
    
    @staticmethod
    def constraint_extraction(chunk: str) -> str:
        return f"""Extract character constraints. A constraint belongs to the character WHO HAS IT. Distinguish backstory from main story:

Time tags:
- [past]: ONLY for backstory with markers like "was once", "used to be", "had been", "years ago"
- [present]: Main story events (even if written in past tense like "Alice met Bob")
- [future]: Plans or intentions ("will", "going to")
- [habitual]: Ongoing traits/abilities ("can", "is", "always")

CORRECT:
"Bob was once a police officer" → Bob | state | was once a police officer | past
"Alice is dedicated" → Alice | trait | is dedicated | habitual
"Alice met Bob" → (skip - this is an event, not a constraint)

WRONG:
"Carol trusts Alice" → Alice | trait | ... (Alice doesn't have this, Carol does)

Format: CHARACTER_NAME | TYPE | DESCRIPTION | TIME
Types: ability, prohibition, trait, state

Text: {chunk}

Output:"""
    
    @staticmethod
    def interaction_extraction(chunk: str) -> str:
        return f"""Extract character interactions between TWO DIFFERENT characters. ACTOR does action TO RECEIVER.

Time tags:
- [past]: ONLY backstory with markers like "was once", "years ago", "had met"
- [present]: Main story events (even if narrated in past tense)
- [future]: Planned actions ("will", "going to")

CORRECT:
"Bob told Alice about conspiracy" → Bob | Alice | speaks | told Alice about conspiracy | present
"Carol assigned the case" → Carol | Alice | assigns | assigned the case | present
"Bob was once a police officer" → (skip - not an interaction)
"She is dedicated" → (skip - trait description, not interaction)

WRONG:
"Bob warned Alice" → Alice | Bob | warns | ... (reverses direction)
"Alice is dedicated" → Alice | Alice | meets | ... (trait, not interaction; also self-loop)

Format: ACTOR | RECEIVER | TYPE | FULL_DESCRIPTION | TIME
Types: meets, fights, helps, speaks, warns, assigns, promises, trusts

Text: {chunk}

Output:"""
