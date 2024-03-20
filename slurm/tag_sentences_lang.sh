#!/bin/zsh
#SBATCH --job-name=tag_language
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops2
#SBATCH --output=language_tagging.log
#SBATCH --error=language_tagging.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# Add your SLURM directives and options here

# Load any necessary modules
source ~/.zshrc

# Change to the working directory
cd ~/language-tagging

conda activate lang-tagging
python code/tag_with_probs.py --messages_path data/raw-data/lang-ordered_message_history_by_sentence.csv --batch_size 100
