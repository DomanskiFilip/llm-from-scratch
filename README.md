main.py — project entry point

Run with:  py main.py [command] [options]

py -m src.main download
py -m src.main tokenise
py -m src.main embeddings (skip if you dont want glove and run train --no-glove)
py -m src.training_and_evaluation.model (optionally and standalone)
py -m src.main train --grid-search
python train.py --no-glove - train without glove

evaluate part of the model:
py -m src.main evaluate --ckpt artefacts/checkpoints/default_run_best.pt

evaluate full model:
py -m src.main evaluate --ckpt artefacts/checkpoints/default_run_best.pt --batches 0

interactive mode:
py -m src.main generate --ckpt artefacts/checkpoints/default_run_best.pt

you can youse python instead of py