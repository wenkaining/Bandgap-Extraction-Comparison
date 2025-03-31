# Bandgap Extraction Comparison

## Project Description

This project serves as supplementary information for `{Paper Link}`, please refer to it for more details.

```
my-project/
├── environment.yml          # Conda environment configuration
├── README.md                # Project documentation
├── extract.ipynb            # Main program
├── my_post.py               # Function library
├── RESULT                   # Manually curated result data
└── HOME                     # Directory for execution data
```

## Usage Instructions

### Environment Setup

To configure the environment, run:

```bash
conda env create -f environment.yml
```

### **Preparation**

Organize the files and directories as follows:

```
my-project/HOME/
├── comparison.xlsx                # Manually extracted data and subsequent code results
├── output                         # Outputs from various methods
├── PDF                            # Directory for downloaded PDF files
├── project                        # Directory for other project files
└── sample_arxiv...snapshot.json   # Sample arXiv metadata JSON file
```

1. Download the arXiv dataset JSON file from <https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download> and move it to the HOME directory.
2. Run section 1.1 of the extract.ipynb notebook to generate the ARXIV_50_JSON file.
3. Download the 50 papers listed in ARXIV_50_JSON and place them in the PDF directory.
4. Manually extract bandgap data from the 50 papers and save the results to comparison.xlsx.
5. Download the following projects to the project directory:
   - <https://github.com/QingyangDong-qd220/BandgapDatabase1>
   - <https://github.com/StefanoSanvitoGroup/BERT-PSIE-TC>
6. Download LLMs:
   1. <https://huggingface.co/m3rg-iitd/matscibert>
   2. <https://ollama.com/library/nomic-embed-text> (ollama pull nomic-embed-text)
   3. <https://ollama.com/library/bge-m3> (ollama pull bge-m3)
   4. <https://ollama.com/library/llama2> (ollama pull llama2:13b)
   5. <https://huggingface.co/bartowski/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF> (ollama create llama3.1:70b -f Modelfile)
   6. <https://ollama.com/library/qwen2.5> (ollama pull qwen2.5:14b)
7. Use the following prompt to extract data using Kimi and save the output to the KIMI_OUT directory:

> PROMPT:
>
> You are an expert information extraction algorithm.
> Extract all the band gap values in this article and output them in the form of a markdown table, including: Material (name of the material), Value (value with unit), Sentence (the sentence from which this data record comes).
> If data is not present in the article, type "None". 
> Table only, no need for explanation or any other content.
> The output is strictly in the following format.
>
> ```markdown
> | Material | Value | Sentence |
> |----------|-------|---------|
> | Material1 | 0.1 eV | ... Eg of Material1 is 0.1 eV ... |
> | Material1 | 200 meV | Material1 has a band gap of 200 meV, so ... |
> | Material2 | None | Material2 ... |
> ```
>
> If no band gap values mentioned in the article, the following table is acceptable:
>
> ```markdown
> | Material | Value | Sentence |
> |----------|-------|----------|
> | None | None | None |
> ```

### Data Extraction

Follow the instructions in extract.ipynb to run the code in it.

> Please note that the code environment I am using does not support running ChemDataExtractor2 directly, so I used the Docker environment from <https://hub.docker.com/r/obrink/chemdataextractor>.
>
> For more details, please refer to http://chemdataextractor2.org.

