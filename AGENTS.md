# Running the research models

Start with the root README. Use its CPU smoke commands for a fresh checkout, and its real-data or torchrun commands when a corpus, budget and allocated device are specified. Run the unit suite before training. Use a different output directory for each recipe and keep the JSON training logs with the checkpoint. To continue an existing run, use the same recipe and data with --resume. Record the commit, data manifest hash, step, cumulative targets, validation NLL and checkpoint path in the handoff.
