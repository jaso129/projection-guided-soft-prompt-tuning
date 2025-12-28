# Projection-Guided Soft Prompt Tuning

This repository contains the source code for the paper:

**Semantically Interpretable Soft Prompt Tuning via Projection-Guided Alignment**  
Published in *Knowledge-Based Systems* (SCI Q1, IF: 7.6)

## Overview

Soft prompt tuning has shown strong performance in parameter-efficient fine-tuning, 
but its learned representations are often difficult to interpret.

In this work, we propose a projection-guided soft prompt tuning framework that aligns 
continuous soft prompts with task-relevant discrete embedding spaces. 
By projecting soft prompts into interpretable semantic spaces during training, 
our method enables tracking and analyzing semantic evolution without sacrificing task performance.

Experiments on multiple NLP benchmarks demonstrate that the proposed method achieves 
comparable or improved performance while providing improved interpretability.

## Repository Structure

```text
.
├── Dataset/                # Dataset loading and preprocessing
├── Model/                  # Prompt tuning models and projection modules
├── Helper/                 # Utility functions
├── Config/                 # Experiment configurations
├── main.py                 # Main training entry
