"""
Test the StoryValidator with baseline and proposed stories.
"""

import json
from constraint_graph import StoryValidator

# Load baseline story
with open('constraint.json', 'r') as f:
    baseline = json.load(f)

print("="*80)
print("BASELINE STORY LOADED")
print("="*80)
print(f"Characters: {baseline['summary']['total_characters']}")
print(f"Constraints: {baseline['summary']['total_constraints']}")
print(f"Interactions: {baseline['summary']['total_interactions']}")

# Create a proposed story with violations
proposed = {
    "summary": {
        "total_characters": 1,
        "total_constraints": 1,
        "total_interactions": 0,
        "total_chunks": 1
    },
    "characters": [
        {
            "name": "Alice",
            "backstory": {
                "constraints": [
                    {
                        "type": "state",
                        "value": "was never a patrol officer",
                        "temporal_tag": "past",
                        "chunk_index": 0,
                        "source_chunk": "Alice was never a patrol officer.",
                        "confidence": 0.8
                    }
                ],
                "events": []
            },
            "current_story": {
                "constraints": [],
                "interactions": {
                    "outgoing": [],
                    "incoming": [],
                    "bidirectional": []
                }
            }
        }
    ]
}

print("\n" + "="*80)
print("TESTING VALIDATOR")
print("="*80)

# Initialize validator with baseline
validator = StoryValidator(baseline)

# Validate proposed story
report = validator.get_validation_report(proposed)

print(f"\nValidation Result: {'✓ VALID' if report['is_valid'] else '✗ INVALID'}")
print(f"Total Violations: {report['total_violations']}")
print(f"  - Critical: {report['critical_count']}")
print(f"  - Warnings: {report['warning_count']}")

if report['violations']:
    print("\nViolations Found:")
    for i, violation in enumerate(report['violations'], 1):
        print(f"\n{i}. [{violation['severity'].upper()}] {violation['violation_type']}")
        print(f"   Character: {violation['character']}")
        print(f"   Baseline: {violation['baseline_constraint']}")
        print(f"   Proposed: {violation['proposed_event']}")
        print(f"   Reason: {violation['explanation']}")

print("\n" + "="*80)
