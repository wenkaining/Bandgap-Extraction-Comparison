# Bandgap Extraction Comparison: A Comprehensive Evaluation of Information Extraction Tools for Materials Science Literature

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the supplementary code and data for the paper "Optimizing Data Extraction from Materials Science Literature: A Study of Tools Using Large Language Models", which presents a comprehensive comparison of information extraction tools for automatically extracting bandgap values from materials science literature.



## Repository Structure

```
Bandgap-Extraction-Comparison/
├── README.md                    # This file
├── LICENSE.txt                  # MIT License
├── environment.yml              # Conda environment configuration
├── extract.ipynb                # Main extraction pipeline notebook
├── my_post.py                   # Post-processing utilities
├── ESI_files/                   # Electronic Supplementary Information
│   ├── ESI_context_analysis_topk5.json
│   ├── ESI_context_analysis_topk10.json
│   └── *.xlsx                   # Analysis results and comparisons
├── HOME/                        # Main directory for code operations
├── HOME_pub/                    # Actual I/O data of 200 publisher papers
│   ├── (sample)arx...shot.json  # Kaggle dataset
│   ├── arXiv_mtrl-sci_200.json  # Selected 200 papers metadata
│   ├── PDF/                     # PDF files for 200 papers
│   ├── TXT(fromPDF)/            # Processed text files
│   ├── TXT(fromPDF_processed)/  # Processed text files (in sentences)
│   ├── output/                  # Raw extraction results
│   │   ├── 1-ChemDataExtractor/
│   │   ├── 2-BERT-PSIE/
│   │   ├── 3-ChatExtract/
│   │   ├── 4-LangChain/
│   │   └── 5-Kimi/
│   ├── comparison_0922_test/    # Cleaned extraction results
│   ├── manual_pub.xlsx          # Manual curated dataset
│   └── project/                 # External dependencies
├── HOME_arxiv/                  # Another comparison round on arXiv
├── DOIs.xlsx                    # Dataset includes doi and arxiv id.
└── HOME_arxiv/                  # Another comparison round on arXiv

```

## Extraction Tools

### 1. ChemDataExtractor (CDE)
- **Type**: Rule-based NLP system
- **Approach**: Pattern matching with chemical entity recognition
- **Environment**: Docker container (`obrink/chemdataextractor:2.1.2`)

### 2. BERT-PSIE (PSIE)
- **Type**: Deep learning (BERT-based)
- **Approach**: Three-stage pipeline (classification → NER → relation extraction)
- **Model**: Fine-tuned MatSciBERT for materials science

### 3. ChatExtract (CE)
- **Type**: Large Language Models
- **Approach**: Extraction via carefully designed prompts
- **Models**: Llama2-13B, Llama3.1-70B, Qwen2.5-14B

### 4. LangChain (LC)
- **Type**: Retrieval-Augmented Generation (RAG)
- **Components**: Embeddings + LLM inference
- **Embeddings**: nomic-embed-text, bge-m3
- **Models**: Llama2-13B, Llama3.1-70B, Qwen2.5-14B

### 5. Kimi-1.5 (Kimi)
- **Type**: Commercial LLM web interface
- **Approach**: Direct extraction via web interface

## Prerequisites

### Required Models and Data

#### Language Models
```bash
# Ollama models (install via: ollama pull <model>)
ollama pull nomic-embed-text:latest      # Embedding model
ollama pull bge-m3:latest                # Embedding model  
ollama pull llama2:13b                   # Inference model
ollama pull qwen2.5:14b                  # Inference model

# Custom Ollama model (requires Modelfile)
ollama create llama3.1:70b -f Modelfile  # From Llama-3.1-Nemotron-70B-Instruct-HF-GGUF
```

#### Hugging Face Models
- **MatSciBERT**: `m3rg-iitd/matscibert` (version: 24a4e4318dda9bc18bff5e6a45debdcb3e1780e3)
- **Llama 3.1 Nemotron**: `bartowski/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF`

#### External Projects
```bash
# Clone required repositories to HOME/project/
git clone https://github.com/QingyangDong-qd220/BandgapDatabase1.git
git clone https://github.com/StefanoSanvitoGroup/BERT-PSIE-TC.git
```

#### Dataset
- **ArXiv Dataset**: Download from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download)
- **Size**: ~5GB compressed, contains metadata for all ArXiv papers

## Installation

### 1. Environment Setup

Clone the repository and create the conda environment:

```bash
git clone https://github.com/wenkaining/Bandgap-Extraction-Comparison.git
cd Bandgap-Extraction-Comparison
conda env create -f environment.yml
conda activate lc
```

### 2. Directory Structure Setup

Create the required directory structure:

```bash
# Set your HOME directory path
export HOME_DIR="/your/HOME/directory"
```

### 3. Download Dependencies

```bash
# Download ArXiv dataset
# Place arxiv-metadata-oai-snapshot.json in $HOME_DIR

# Clone external projects
cd $HOME_DIR/project
git clone https://github.com/QingyangDong-qd220/BandgapDatabase1.git
git clone https://github.com/StefanoSanvitoGroup/BERT-PSIE-TC.git
```

### 4. Docker Setup (for ChemDataExtractor; if needed)

```bash
# Pull and run ChemDataExtractor container
docker run --name cde \
  --mount type=bind,source='$HOME_DIR',target='/home/chemdataextractor2' \
  -it -p 8888:8888 \
  --entrypoint bash obrink/chemdataextractor:2.1.2
```

## Data Description

### Input Data Format

The pipeline expects:
1. **ArXiv Metadata**: JSON file with paper metadata including abstracts, categories, and DOIs
2. **PDF Files**: Research papers in PDF format, named using DOI pattern (e.g., `10.1038_nmat1234.pdf`)
3. **Manual Annotations**: Excel file with ground truth bandgap extractions

### Data Preprocessing

The preprocessing pipeline includes:
1. **Paper Selection**: Filter arXiv papers by `mtrl-sci` category and date range (2000-2024)
2. **Random Sampling**: Select 200 papers using random seed 42 (initially 50, then additional 150)
3. **Text Extraction**: Convert PDFs to structured text files
4. **Sentence Segmentation**: Process text using spaCy for proper sentence boundaries

### Expected Data Structure

```
HOME/
├── arxiv-metadata-oai-snapshot.json    # Full ArXiv metadata from Kaggle
├── arXiv_mtrl-sci_200.json             # Selected 200 papers metadata
├── PDF/                                # PDF files named as DOI
│   ├── 10.1038_nmat1234.pdf
│   └── ...
├── TXT(fromPDF)/                       # Processed text files
├── TXT(fromPDF_processed)/             # Processed text files (in sentences)
├── output/                             # Raw extraction results
│   ├── 1-ChemDataExtractor/
│   ├── 2-BERT-PSIE/
│   ├── 3-ChatExtract/
│   ├── 4-LangChain/
│   └── 5-Kimi/
├── comparison_{DATE}_{MARK}/           # Cleaned extraction results
├── manual_pub.xlsx                     # Manual curated dataset
└── project/                            # External dependencies
```

## Usage Instructions

### 1. Configuration

Update the `HOME` variable in `extract.ipynb`:

```python
HOME = '/your/HOME/directory'  # Update this path
```

### 2. Data Preparation

Run Section 1 of `extract.ipynb` to prepare the dataset:

```jupyter
# Section 1.1: Prepare 200 Papers
# This will:
# - Filter arXiv papers by materials science category
# - Randomly select 200 papers
# - Generate paper list and metadata files
```

Download the 200 selected papers and place them in the `PDF/` directory.

### 3. Manual Annotation (Ground Truth)

Create manual annotations by extracting bandgap data from papers and saving to `comparison.xlsx` with columns:
- `doi`: Paper DOI
- `material`: Material name
- `value`: Bandgap value with unit
- `sentence`: Source sentence

### 4. Run Extraction Tools

Execute each extraction tool by running the corresponding sections in `extract.ipynb` Section 2:

### 5. Post-processing and Analysis

Run Section 3 of `extract.ipynb` to process results:

```jupyter
# Section 3: Process results from all tools
# This will:
# - Standardize output formats
# - Apply cleaning and normalization
# - Generate comparison analysis
# - Create summary statistics
```

## Reproducibility Notes

### Random Seeds
- **Paper Selection**: Random seed 42 used for selecting 200 papers from Kaggle's arXiv dataset
- **Selection Strategy**: First 50 papers selected, then additional 150 papers to reach 200 total

### Model Versions
- **MatSciBERT**: Version 24a4e4318dda9bc18bff5e6a45debdcb3e1780e3
- **Ollama Models**: 
  - nomic-embed-text:latest (version: 0a109f422b47)
  - bge-m3:latest (version: 790764642607)
  - llama2:13b (version: d475bf4c50bc)
  - qwen2.5:14b (version: 7cdf5a0187d5)
- **Llama 3.1 Nemotron**: Version dfc9cf0b496aea479874ddce703154f07d22ec3d

### Environment Specifications
- **Python**: 3.10.16
- **PyTorch**: 2.2.1 with MPS support (Apple Silicon) or CUDA
- **Transformers**: 4.48.3
- **LangChain**: 0.3.18

## Output Files

### Primary Results
- `FINAL_{TOOL}_{TIMESTAMP}.xlsx`: Processed results for each tool
- `comparison_{DATE}_{MARK}.xlsx`: Consolidated comparison file
- `summary_{DATE}_{MARK}.xlsx`: Final analysis summary
- `manual.xlsx`: Crucial part of the curated dataset.

### Intermediate Files
- `arXiv_mtrl-sci_200.json`: Selected papers metadata
- `TXT(fromPDF_processed)`: Parsed plain text.
- `{METHOD}/runtime_{TIMESTAMP}.txt`: Runtime logs
- Raw extraction outputs in respective tool directories ($HOME_DIR/output)

## Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@article{🔴CITATION_KEY_PLACEHOLDER,
  title={🔴PAPER_TITLE_PLACEHOLDER},
  author={🔴AUTHORS_PLACEHOLDER},
  journal={🔴JOURNAL_PLACEHOLDER},
  year={🔴YEAR_PLACEHOLDER},
  doi={🔴DOI_PLACEHOLDER}
}
```

For additional support, please refer to the paper.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.
