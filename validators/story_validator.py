"""
StoryValidator for validating proposed stories against baseline constraints using structured LLM outputs.
"""

from typing import List, TYPE_CHECKING

from models import Constraint, Interaction, Violation, ViolationCheck

if TYPE_CHECKING:
    from core import QwenLLM


class StoryValidator:
    """Validates story constraints and interactions against baseline using LLM."""
    
    def __init__(self, baseline_constraints: List[Constraint], baseline_interactions: List[Interaction], llm: 'QwenLLM' = None):
        self.baseline_constraints = baseline_constraints
        self.baseline_interactions = baseline_interactions
        self.llm = llm
    
    def validate_constraints_and_interactions(
        self, 
        proposed_constraints: List[Constraint],
        proposed_interactions: List[Interaction]
    ) -> List[Violation]:
        """Validate proposed story events against baseline. Returns violations."""
        if not self.llm:
            return []
        
        violations = []
        
        # Validate each proposed constraint
        print(f"Checking {len(proposed_constraints)} constraints...")
        for i, constraint in enumerate(proposed_constraints, 1):
            print(f"  [{i}/{len(proposed_constraints)}] Validating: {constraint.character} - '{constraint.value[:60]}...'")
            violation = self._validate_single_constraint(constraint)
            if violation:
                print(f"      ❌ VIOLATION DETECTED!")
                violations.append(violation)
            else:
                print(f"      ✓ OK")
        
        # Validate each proposed interaction
        print(f"\nChecking {len(proposed_interactions)} interactions...")
        for i, interaction in enumerate(proposed_interactions, 1):
            print(f"  [{i}/{len(proposed_interactions)}] Validating: {interaction.character1} → {interaction.character2} - '{interaction.description[:50]}...'")
            violation = self._validate_single_interaction(interaction)
            if violation:
                print(f"      ❌ VIOLATION DETECTED!")
                violations.append(violation)
            else:
                print(f"      ✓ OK")
        
        return violations
    
    def _validate_single_constraint(self, constraint: Constraint) -> Violation | None:
        """Validate a single constraint against backstory using structured output."""
        character = constraint.character
        
        # Gather relevant backstory for this character
        character_backstory = [c for c in self.baseline_constraints if c.character == character]
        character_past_interactions = [i for i in self.baseline_interactions 
                                       if i.character1 == character or i.character2 == character]
        
        if not character_backstory and not character_past_interactions:
            print(f"      → No backstory found for {character}, skipping")
            return None
        
        # Prioritize prohibitions and critical constraints
        prohibitions = [c for c in character_backstory if c.constraint_type == "prohibition"]
        other_constraints = [c for c in character_backstory if c.constraint_type != "prohibition"]
        
        print(f"\n      → Found {len(prohibitions)} prohibitions, {len(other_constraints)} other constraints, {len(character_past_interactions)} past interactions")
        print(f"      → Backstory context:")
        
        # Build context with prohibitions first
        context = f"CHARACTER: {character}\n\n"
        
        if prohibitions:
            context += "PROHIBITIONS (CRITICAL):\n"
            for c in prohibitions:
                context += f"- {c.value} (type: {c.constraint_type})\n"
                print(f"         [PROHIBITION] {c.value[:80]}")
        
        if other_constraints:
            context += "\nOTHER BACKSTORY:\n"
            for c in other_constraints[:5]:  # Limit to prevent context overflow
                context += f"- {c.value} (type: {c.constraint_type})\n"
                print(f"         [BACKSTORY] {c.value[:80]}")
        
        if character_past_interactions:
            context += "\nPAST INTERACTIONS:\n"
            for i in character_past_interactions[:3]:  # Limit to 3
                context += f"- {i.character1} → {i.character2}: {i.description}\n"
                print(f"         [PAST] {i.character1} → {i.character2}: {i.description[:60]}")
        
        context += f"\nNEW EVENT/CONSTRAINT:\n- {constraint.value} (type: {constraint.constraint_type})\n"
        
        # LLM prompt for structured output
        prompt = f"""Analyze if this new constraint creates a LOGICAL CONTRADICTION in the story itself.

{context}

CHARACTER AGENCY RULE - MOST CRITICAL:
- Characters CAN and SHOULD violate prohibitions/agreements if that's their story choice
- "Carol prohibited from selecting Alice" + "Carol selects Alice anyway" → NOT a violation (Carol chose to break the rule)
- "Alice signed agreement not to return" + "Alice returns to police work" → NOT a violation (Alice chose to break agreement)
- Only flag if: Story says "Carol never selected Alice" THEN "Carol selected Alice" (factual contradiction)
- Prohibition + Character violates it = Valid storytelling (character agency/moral conflict)
- Prohibition + Story contradicts itself about facts = TRUE violation

CAPABILITY vs ACTION RULE:
- Prohibitions apply to ACTIONS, NOT capabilities/skills from past experience
- "Alice has skills to investigate" ≠ "Alice returned to police work"
- Only flag if character PERFORMS the prohibited action AND story contradicts itself

CHARACTER KNOWLEDGE RULE:
- Character A trusts Character B with hidden flaws → VALID (A doesn't know)
- Character works with someone without knowing restrictions → VALID

TEMPORAL AWARENESS:
- Past trait + Present opposite = Character growth (severity MAX 2)
- Present + Present contradiction = TRUE violation (severity 8-10)
- Prohibition + Character chooses to violate = Character choice (severity MAX 3)

Examples:
- Backstory: "Carol cannot select Alice", Story: "Carol selects Alice" → NOT violation (Carol's choice to break rule)
- Backstory: "Alice was corrupt", Story: "Alice is honest" → NOT violation (redemption)
- Backstory: "Bob is dead", Story: "Bob speaks" → VIOLATION severity 10 (impossible)
- Backstory: "Carol never selected Alice", Story: "Carol selected Alice" → VIOLATION severity 9 (factual contradiction)

SEVERITY SCALE:
10: Physically impossible
9: Factual contradiction in story ("never happened" then "happened")
3: Character deliberately violates prohibition/agreement (valid character choice)
2: Character growth over time
1: Minor inconsistency

Respond with ONLY valid JSON:
- has_violation: boolean
- character: string or null  
- severity: integer 1-10 or null
- explanation: string or null
- baseline_constraint: string or null

JSON:"""
        
        print(f"\n      → Sending structured LLM prompt...")
        
        try:
            structured_llm = self.llm.with_structured_output(ViolationCheck)
            result = structured_llm.invoke(prompt)
            
            print(f"      → Structured result: violation={result.has_violation}")
            
            if result.has_violation and result.character and result.severity and result.explanation:
                violation_type = "constraint_violation" if "prohib" in result.explanation.lower() else "state_contradiction"
                
                print(f"      ⚠ VIOLATION FOUND (severity {result.severity}/10): {result.explanation}")
                
                return Violation(
                    character=result.character,
                    violation_type=violation_type,
                    severity=result.severity,
                    baseline_constraint=result.baseline_constraint,
                    proposed_event=constraint.value,
                    explanation=result.explanation
                )
            else:
                print(f"      ✓ No violation detected\n")
                return None
                
        except Exception as e:
            print(f"      ⚠ Error in structured extraction: {e}")
            print(f"      Treating as no violation\n")
            return None
    
    def _validate_single_interaction(self, interaction: Interaction) -> Violation | None:
        """Validate a single interaction against backstory using structured output."""
        actor = interaction.character1
        receiver = interaction.character2
        
        # Gather backstory for both actor and receiver
        actor_backstory = [c for c in self.baseline_constraints if c.character == actor]
        receiver_backstory = [c for c in self.baseline_constraints if c.character == receiver]
        past_interactions = [i for i in self.baseline_interactions 
                            if (i.character1 == actor and i.character2 == receiver) or
                               (i.character1 == receiver and i.character2 == actor)]
        
        if not actor_backstory and not receiver_backstory and not past_interactions:
            print(f"      → No backstory found for {actor} or {receiver}, skipping")
            return None
        
        print(f"\n      → Found {len(actor_backstory)} backstory for {actor}, {len(receiver_backstory)} for {receiver}, {len(past_interactions)} past interactions")
        print(f"      → Backstory context:")
        for c in actor_backstory:
            print(f"         [BACKSTORY-{actor}] {c.value[:80]}")
        for c in receiver_backstory:
            print(f"         [BACKSTORY-{receiver}] {c.value[:80]}")
        for i in past_interactions:
            print(f"         [BACKSTORY] {i.character1} → {i.character2}: {i.description[:60]}")
        
        # Build context
        context = f"INTERACTION: {actor} → {receiver}\n"
        context += f"ACTION: {interaction.description} (type: {interaction.interaction_type})\n\n"
        
        if actor_backstory:
            context += f"BACKSTORY - {actor}:\n"
            for c in actor_backstory:
                context += f"- {c.value} (type: {c.constraint_type})\n"
        
        if receiver_backstory:
            context += f"\nBACKSTORY - {receiver}:\n"
            for c in receiver_backstory:
                context += f"- {c.value} (type: {c.constraint_type})\n"
        
        if past_interactions:
            context += f"\nPAST INTERACTIONS:\n"
            for i in past_interactions:
                context += f"- {i.character1} → {i.character2}: {i.description}\n"
        
        # LLM prompt for structured output
        prompt = f"""Analyze if this interaction creates a LOGICAL CONTRADICTION in the story itself.

{context}

CHARACTER AGENCY RULE - MOST CRITICAL:
- Characters CAN violate prohibitions if that's their choice (valid storytelling)
- "Carol prohibited from working with Alice" + "Carol works with Alice" → NOT violation (Carol's choice)
- Only flag factual contradictions ("never worked together" THEN "worked together")

CAPABILITY vs ACTION RULE:
- Prohibitions apply to ACTIONS, NOT capabilities/skills
- Having ability ≠ performing action

CHARACTER KNOWLEDGE RULE:
- Character A trusts Character B with hidden flaws → VALID
- Characters reconcile after past conflict → VALID

TEMPORAL AWARENESS:
- Past conflict + Present cooperation = Reconciliation (severity MAX 4)
- Prohibition + Character violates it = Character choice (severity MAX 3)

SEVERITY SCALE:
10: Physically impossible
9: Factual contradiction
3: Character breaks prohibition (valid choice)
2: Reconciliation/growth

Respond with ONLY valid JSON:
- has_violation: boolean
- character: string or null
- severity: integer 1-10 or null
- explanation: string or null
- baseline_constraint: string or null

JSON:"""
        
        print(f"\n      → Sending structured LLM prompt...")
        
        try:
            structured_llm = self.llm.with_structured_output(ViolationCheck)
            result = structured_llm.invoke(prompt)
            
            print(f"      → Structured result: violation={result.has_violation}")
            
            if result.has_violation and result.character and result.severity and result.explanation:
                print(f"      ⚠ VIOLATION FOUND (severity {result.severity}/10): {result.explanation}")
                
                return Violation(
                    character=result.character,
                    violation_type="constraint_violation",
                    severity=result.severity,
                    baseline_constraint=result.baseline_constraint,
                    proposed_event=interaction.description,
                    explanation=result.explanation
                )
            else:
                print(f"      ✓ No violation detected\n")
                return None
                
        except Exception as e:
            print(f"      ⚠ Error in structured extraction: {e}")
            print(f"      Treating as no violation\n")
            return None
    
    def get_validation_report(self, violations: List[Violation]) -> str:
        """Generate a formatted report of violations with overall validity assessment."""
        if not violations:
            return "✓ No violations found. Story is consistent with baseline.\n\nOVERALL VALIDITY: 1 (VALID)"
        
        # Calculate overall validity (1 = valid, 0 = invalid)
        # Invalid if: any severity >= 8, or 2+ violations with severity >= 6, or 3+ violations with severity >= 4
        max_severity = max(v.severity for v in violations)
        high_severity_count = sum(1 for v in violations if v.severity >= 6)
        medium_severity_count = sum(1 for v in violations if v.severity >= 4)
        
        is_valid = 1
        if max_severity >= 8:
            is_valid = 0
        elif high_severity_count >= 2:
            is_valid = 0
        elif medium_severity_count >= 3:
            is_valid = 0
        
        report = f"Found {len(violations)} violation(s):\n\n"
        
        # Sort by severity descending
        sorted_violations = sorted(violations, key=lambda v: v.severity, reverse=True)
        
        for v in sorted_violations:
            severity_emoji = "🔴" if v.severity >= 8 else "🟠" if v.severity >= 6 else "🟡" if v.severity >= 4 else "🟢"
            report += f"{severity_emoji} [{v.severity}/10] {v.character}: {v.explanation}\n"
        
        report += f"\nOVERALL VALIDITY: {is_valid} ({'VALID' if is_valid else 'INVALID'})\n"
        report += f"  Max severity: {max_severity}/10\n"
        report += f"  High severity violations (≥6): {high_severity_count}\n"
        report += f"  Medium severity violations (≥4): {medium_severity_count}\n"
        
        return report
