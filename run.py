#!/usr/bin/env python3
"""
CLI entry point for the constraint extraction pipeline.
Accepts story text via command-line argument or stdin.
"""

import argparse
import sys

from core import run_extraction_pipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract constraints and interactions from story text using LLM-based pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From file
  python run.py --story-file story.txt
  
  # From argument
  python run.py --story "Alice is a detective. Bob met Alice at the precinct."
  
  # With backstory and main story
  python run.py --backstory "Alice was once a police officer." --story "Alice is now a detective."
  
  # With backstory and story files
  python run.py --backstory-file backstory.txt --story-file story.txt
  
  # From stdin
  cat story.txt | python run.py
  echo "Alice is a detective." | python run.py
  
  # Specify output file
  python run.py --story-file story.txt --output results.json
        """
    )
    
    # Input options
    parser.add_argument(
        '--story',
        type=str,
        help='Main story text as a string'
    )
    parser.add_argument(
        '--story-file',
        type=str,
        help='Path to file containing main story text'
    )
    parser.add_argument(
        '--backstory',
        type=str,
        help='Backstory text as a string (combined with main story)'
    )
    parser.add_argument(
        '--backstory-file',
        type=str,
        help='Path to file containing backstory text (combined with main story)'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (default: constraint.json)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get backstory text if provided
    backstory_text = None
    
    if args.backstory:
        backstory_text = args.backstory
    elif args.backstory_file:
        try:
            with open(args.backstory_file, 'r', encoding='utf-8') as f:
                backstory_text = f.read()
        except FileNotFoundError:
            print(f"Error: Backstory file not found: {args.backstory_file}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading backstory file: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Get main story text from various sources
    story_text = None
    
    if args.story:
        # From command-line argument
        story_text = args.story
    elif args.story_file:
        # From file
        try:
            with open(args.story_file, 'r', encoding='utf-8') as f:
                story_text = f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {args.story_file}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    elif not sys.stdin.isatty():
        # From stdin (pipe)
        story_text = sys.stdin.read()
    else:
        # No input provided
        parser.print_help()
        print("\nError: No story text provided. Use --story, --story-file, or pipe via stdin.", file=sys.stderr)
        sys.exit(1)
    
    # Combine backstory and story if both provided
    if backstory_text:
        story_text = f"{backstory_text.strip()}\n\n{story_text.strip()}"
    
    # Validate story text
    if not story_text or not story_text.strip():
        print("Error: Story text is empty", file=sys.stderr)
        sys.exit(1)
    
    # Update output file if specified
    if args.output:
        import config
        config.OUTPUT_FILE = args.output
    
    # Run the pipeline
    try:
        final_state = run_extraction_pipeline(story_text.strip())
        print("\n✓ Extraction completed successfully!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error during extraction: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
