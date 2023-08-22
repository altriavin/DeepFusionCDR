# DeepFusionCDR: Employing Multi-Omics Integration and Molecule-Specific Transformers for Enhanced Prediction of Cancer Drug Responses
DeepFusionCDR is a novel approach employing self-supervised contrastive learning to amalgamate multi-omics features, including mutation, transcriptome, methylome, and copy number variation data, from cell lines. Furthermore, we incorporate molecular SMILES-specific transformers to derive drug features from their chemical structures. The unified multi-omics and drug signatures are combined, and a multi-layer perceptron (MLP) is applied to predict IC50 values for cell line-drug pairs. Moreover, this MLP can discern whether a cell line is resistant or sensitive to a particular drug.
# Requirements
torch1.8
python 3.7
numpy 1.19
pandas 1.1
scikit-learn 1.0
scipy 1.5
#Data
In this study, we assessed DeepFusionCDR using both classification and regression tasks. For the regression task, we primarily used IC50 values for cell line-drug pairs from the GDSC database\cite{yang2012genomics} and CCLE database\cite{barretina2012cancer}. We downloaded multiple omics features of cell lines from both the GDSC and CCLE databases, and then selected cell lines that concurrently possessed CNV, gene mutation, transcriptomics, and methylation features, and had drug SMILE strings available in PubChem.

The transcriptomics analysis involved applying RMA normalization to RNA data, while methylation assessment used pre-processed CpG islands' $\beta$-values. We encoded gene mutation data in binary format and represented copy number variation (CNV) data ternary format, designating gene loss, normal copy number, and gene gain as -1, 0, and 1, respectively. As a result, we compiled a dataset consisting of 68,996 IC50 value pairs between 489 cell lines and 297 drugs.
#Run the demo
