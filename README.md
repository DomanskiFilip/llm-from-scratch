main.py — project entry point

Run with:  py main.py [command] [options]

py -m src.main download
py -m src.main tokenise
py -m src.main embeddings
py -m src.training_and_evaluation.model (optionally and standalone)
py -m src.main train --grid-search
py -m src.main evaluate --ckpt checkpoints/default_run_best.pt

interactive mode:
py -m src.main generate --ckpt checkpoints/default_run_best.pt

you can youse python instead of py

Tip: start prompts with 'def ' or '# ' to get Python code