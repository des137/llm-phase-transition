# LLM Comprehension Phase Transition — Experiment Code

## Structure

```
experiment/
├── README.md
├── requirements.txt
├── config.py                  # Central config: alpha levels, models, paths
├── corpus/
│   ├── __init__.py
│   ├── loader.py              # Load and prepare source texts
│   └── sample_texts.py        # Small built-in sample texts for testing
├── corruption/
│   ├── __init__.py
│   ├── character.py           # Character-level corruption
│   ├── word.py                # Word-level corruption
│   └── sentence.py            # Sentence-level corruption
├── probing/
│   ├── __init__.py
│   ├── base.py                # Abstract base class for probing tasks
│   ├── qa.py                  # Task A: Closed-form QA
│   ├── summarization.py       # Task B: Summarization
│   ├── coherence_judge.py     # Task C: LLM self-coherence rating
│   └── instruction_follow.py  # Task D: Embedded instruction following
├── llm/
│   ├── __init__.py
│   └── client.py              # LLM API client (OpenAI-compatible, stub-able)
├── metrics/
│   ├── __init__.py
│   ├── task_metrics.py        # EM, F1, BERTScore, ROUGE, IFR
│   └── phase_transition.py    # Sigmoid fitting, α*, β, AUC
├── runner.py                  # Orchestrates the full experiment sweep
├── analysis.py                # Post-hoc analysis and plotting
└── tests/
    ├── __init__.py
    ├── test_corruption.py
    ├── test_metrics.py
    └── test_runner.py
```

## Quick Start

```bash
pip install -r requirements.txt

# Run with built-in sample texts (no API key needed for dry run)
python runner.py --dry-run

# Run full experiment (requires OPENAI_API_KEY env var)
export OPENAI_API_KEY=sk-...
python runner.py --models gpt-4o --granularities char word sentence --alpha-steps 21

# Analyze results
python analysis.py --results-dir results/