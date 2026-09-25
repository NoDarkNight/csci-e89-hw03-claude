# Prompt Log

A running log of each request made during this project.

---

## 1. Project setup + `data.py`

> Hello! I'd like your help in building an image classifier for the Fashion MNIST dataset with PyTorch. For each major step, write and commit a separate Python file.
>
> Some things additional things worth noting:
>
> 1. Part of the final goal includes creating a Jupyter Notebook containing all the scripts. Thus, make the scripts compatible with being pasted into Jupyter Notebook cells
> 2. Keep a file, PROMPTS.md, that logs each one of my requests. At the end of the project, I'll have you write a summary of the entire chat. The PROMPTS.md file may be useful for that :)
>
> Okay. The first step is to write "data.py". This should handle downloading Fashion MNIST with torchvision, converting and normalizing vectors to float32 tensors, setting the seed (42), splitting the 60,000 training images into 55,000 for training and 5,000 for validation, creating dataloaders with batch size 32, and whatever else is necessary.

**Result:** Created `data.py` (seeding, download, seeded 55k/5k split, float32 conversion + normalization using train-split mean/std, train/val/test DataLoaders at batch size 32, sanity-check printout), `.gitignore` (excludes `data/`), and this `PROMPTS.md`.
