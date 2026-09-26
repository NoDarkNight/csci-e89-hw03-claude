# Prompt log — Problem 3 (Fashion MNIST MLP classifier in PyTorch)

Every request made during this session is logged here word for word, followed by
the files and commits it produced.

---

## Prompt 1 — initial request

> Hello! I'd like your help building an image classifier for the Fashion MNIST dataset with PyTorch, from start to finish, in this one request. Work only inside a new folder called prob3/ and ignore any other files in this repo; build everything fresh. Write each major step as a separate Python script and commit each script separately, with a clear commit message, in the order below.
>
> General requirements:
> - Every script must also work when pasted into consecutive Jupyter Notebook cells in the order below (no argparse errors from Jupyter's arguments, plots shown inline, later cells reusing names defined by earlier cells when the .py files aren't available).
> - Keep prob3/PROMPTS.md: log this prompt word for word, and any follow-up requests I make, with the files and commit hashes each one produced.
> - Add a .gitignore for the downloaded dataset, \_\_pycache\_\_/, and .ipynb_checkpoints/.
> - When running things, run one command at a time and never start two training runs at once; they compete for the CPU and slow each other down badly.
>
> Steps:
> 1. data.py: download Fashion MNIST with torchvision, convert the images to float32 tensors and normalize them, set the seed (42), split the 60,000 training images into 55,000 for training and 5,000 for validation, and create DataLoaders with batch size 32 (only the training loader shuffled), plus whatever else is necessary.
> 2. model.py: an MLP classifier as an nn.Module. Flatten the 28x28 image, then two hidden layers of 300 and 100 neurons with ReLU, and an output layer of 10 logits (one per class). Make the hidden sizes configurable.
> 3. train.py: train the model with SGD (learning rate 0.1) and cross-entropy loss for 20 epochs, using torchmetrics to compute multiclass accuracy (plain accuracy: correct / total). For each epoch, record the mean training loss, training accuracy, and validation accuracy and print them. Use a GPU if one is available, otherwise CPU. After training, evaluate accuracy on the test set, save the model weights, and save the history (including test accuracy) to a JSON file.
> 4. plot.py: plot the training accuracy per epoch, with validation accuracy on the same chart for comparison. Save it as a PNG and display it inline in Jupyter. Have train.py call the plotting after training.
> 5. predict.py: use the trained model on the first 3 images of the validation set; print the predicted and true class names, the softmax probabilities for all 10 classes (rounded to 3 decimals), and the top 4 most likely classes for each image.
> 6. tune.py: hyperparameter tuning with Optuna (seeded). Search the learning rate (log scale from about 1e-5 to 1e-1), the number of neurons in each hidden layer, and the SGD momentum. Keep it fast: about 10 trials of 5 epochs each. Print a table of the trials and the best parameters, then retrain the best configuration for 20 epochs and report its test accuracy compared with the baseline from train.py.
>
> Then:
> 7. Run every script in order, one at a time, and fix anything that breaks. Commit the outputs (history JSON, plot PNG, tuning results) along with any fixes.
> 8. Write prob3/SUMMARY.md summarizing this whole session: what I asked for, what you built in each step, problems you ran into and how you fixed them, and the final results (train/validation/test accuracy, baseline vs. tuned). Include a table of scripts with the commit that created each one and the order to run them in.
> 9. Wrap all of the code into one self-contained Jupyter notebook, prob3/e89_Li_Ethan_HW03_Prob3.ipynb. The first cell is markdown with my name (Ethan Li), "CSCI E-89 Deep Learning, Assignment 03, Problem 3", and a short description. Quote this prompt in a markdown cell near the top. Before each code cell, add a markdown cell saying which script it came from and what it does, and add comments in the code. It must not import the .py files, and it must run top to bottom with Restart & Run All. Execute it so the outputs are saved, commit it, and add a short section about it to SUMMARY.md and PROMPTS.md.
>
> When you're done, give me a short list of what was committed and the key results.

### Files and commits produced

(Filled in as the work progresses.)
