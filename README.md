# Clickbait spoiling - team "Alexander Knox"
This repository contains code for `Alexander Knox at SemEval-2023 Task 5: The comparison of prompting and standard fine-tuning for spoiler-type detection task`. There are used three different methods:
 - prompt engineering for spoiler-type classification
 - prompt engineering as data augmentation technique (for create spoilers) and finetuning selected models based on clickbait and spoiler (original or generated)
 - finetuning models using clickbait post and main content of web page

 Usage of the code implemented for these methods is shown in `main.ipynb` file. In `data` folder are located files used for training and testing and file containing generated spoilers for dataset `validation.jsonl`.