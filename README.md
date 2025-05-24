# Prompt-Tuning-without-Labeled-Samples-for-Zero-Shot-Node-Classification-in-Text-Attributed-Graphs
### Dataset and Pre-trained Graph Language Model:
The datasets and the pretrained graph language model can be downloaded from: [Dataset](https://drive.google.com/drive/folders/1OntIyhrsFh44MmDIeIH804qmCw6LVUkZ?usp=drive_link), [Pre-trained Model](https://drive.google.com/drive/folders/13Y04apgjvrhzpxG9RzzD7dPR-vH6shaf?usp=sharing)
Place the datasets and the pre-trained model weights in G2P2_datasets folder. 
Also the extracted node and text embeddings are provided [here](https://drive.google.com/drive/folders/1moL4FZmCvuXtMKERZgBxRYYLg2JZ1OPH?usp=sharing) which can be placed in the save_folder. The splits are provided [here](https://drive.google.com/drive/folders/1kS50iDzi8Ul2mz_J8VepOm2N9BpbnkDv?usp=sharing) and they can be placed in the org_seed_data folder.

### Running our ZPT model and G2P2 Model with Node and Text Embedding Fusion:
Run the exec_cora.sh and exec_amazon.sh files to get the results. The results will be stored in results_final folder. 

### List of discrete prompts considered for our experiments
**Table:** List of discrete prompts used in `+d` baselines and context used in UBCG model for generating class-specific synthetic samples. The best template for the proposed approach is highlighted in **bold**.

| Cora                             | Arts                                   | Industrial                              | MI                                       |
|----------------------------------|----------------------------------------|-----------------------------------------|------------------------------------------|
| [class]                          | [class]                                | [class]                                 | [class]                                  |
| **a [class]**                    | **a [class]**                          | a [class]                               | a [class]                                |
| an [class]                       | an [class]                             | an [class]                              | an [class]                               |
| of [class]                       | of [class]                             | of [class]                              | of [class]                               |
| paper of [class]                 | art [class]                            | industrial [class]                      | instrument [class]                       |
| research of [class]              | sewing [class]                         | scientific [class]                      | **musical [class]**                      |
| a paper of [class]               | art of [class]                         | an industrial [class]                   | an instrument [class]                    |
| a research of [class]            | sewing of [class]                      | a scientific [class]                    | a musical [class]                        |
| a model of [class]               | arts crafts of [class]                 | industrial and scientific [class]       | an instrument of [class]                 |
| research paper of [class]        | arts crafts or sewing of [class]       | **an industrial and scientific [class]** | musical instrument of [class]            |
| a research of [class]            | an arts crafts or sewing of [class]    | of industrail and scientific [class]    | a musical instrument of [class]          |

