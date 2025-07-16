# Heat Stress Poultry Research Analysis

This repository contains a comprehensive NLP analysis of heat stress research in poultry, analyzing 410 research abstracts from PubMed (406 after cleaning).

## Overview

The analysis includes:
1. **Data preprocessing and cleaning** - Text normalization, stopword removal, and tokenization
2. **TF-IDF vectorization** - Building feature vectors with unigrams and bigrams
3. **Topic modeling** - LDA with 7 topics to identify research themes
4. **Keyword trend analysis** - Tracking key terms over time
5. **Topic trend analysis** - Monitoring topic prevalence by year
6. **Comprehensive visualizations** - PDF figures for all analyses

## Files

- `heat_stress_poultry_pubmed.py` - Script to fetch data from PubMed
- `heat_stress_poultry.json` - Research paper dataset (410 records)
- `analyze_heat_stress.py` - Main analysis script
- `ngram_analysis.py` - N-gram frequency analysis script
- `requirements.txt` - Python dependencies
- `figures/` - Generated visualization PDFs

## Key Findings

### Dataset Statistics
- **Total abstracts analyzed**: 406 (after cleaning from 410 records)
- **Year range**: 1995-2025
- **Average abstracts per year**: 15.0
- **Most productive years**: 2024 (58), 2023 (52), 2022 (50)

### Research Focus (Updated Keywords)
**Main Keywords**: Broiler, Heat stress, Manipulation, Performance/Production  
**Sub-keywords**: Antioxidants, Vitamins (E, C), Probiotics, Phytogenics (rutin, resveratrol, curcumin), Minerals (Na, K, Se, Zn, Mn), Amino acids (citrulline, glutamine, arginine, methionine, tryptophan), Betaine

### Topic Analysis (7 Topics Identified)
1. **Topic 1: Vit & Yeast & Exp** (organic_selenium, smd) - Vitamin and selenium supplementation
2. **Topic 2: Curcumin & Lycopene & Znonps** (ros, oxide) - Curcumin and antioxidant compounds  
3. **Topic 3: Ala & Gly & Rosemary** (oregano, lipoic) - Herbal extracts and lipoic acid
4. **Topic 4: Leu & Thermotolerance & Ovo** (cit, recovery) - Amino acids and thermotolerance
5. **Topic 5: Cit & Pfa & Gln** (gaa, arg) - Muscle physiology and amino acids
6. **Topic 6: Ros & Astaxanthin & Hypothalamic** (line, skeletal) - Oxidative stress and astaxanthin
7. **Topic 7: Broilers & Group & Diet** (birds, broiler) - General broiler performance studies (~87% prevalence)

### Research Trends
- **Topic 7** (General broiler performance) dominates the literature (~87% of abstracts)
- Growing research activity in recent years (2022-2025) 
- Increasing focus on antioxidant supplementation strategies
- Emerging interest in specific phytogenic compounds (curcumin, resveratrol)
- Consistent research on amino acid interventions

## Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Main analysis (topic modeling, trends)
python analyze_heat_stress.py

# N-gram frequency analysis
python ngram_analysis.py
```

### Generated Outputs
- `figures/keyword_trends.pdf` - Keyword occurrence trends over time
- `figures/topic_trends.pdf` - Topic prevalence trends over time (dark theme)
- `figures/bigram_network.pdf` - Bigram network visualization (from N-gram analysis)

## Technical Details

### Data Processing
- Text preprocessing: lowercasing, punctuation removal, stopword filtering
- Bigram construction for phrase detection
- TF-IDF vectorization with 1000 features
- Missing data and year validation

### Analysis Methods
- **LDA Topic Modeling**: Latent Dirichlet Allocation with 7 topics
- **TF-IDF**: Term Frequency-Inverse Document Frequency weighting
- **N-gram Analysis**: Unigrams, bigrams, and trigrams with POS filtering
- **Network Analysis**: NetworkX for bigram relationship visualization

### Visualization
- Time series plots for trends with professional dark theme
- Network graphs for term relationships
- High-quality PDF output (300 DPI)

## Dependencies

- biopython - PubMed data retrieval
- pandas - Data manipulation and analysis
- scikit-learn - Machine learning and TF-IDF
- matplotlib - Plotting and visualization
- nltk - Natural language processing and POS tagging
- numpy - Numerical operations
- seaborn - Statistical visualization
- spacy - Advanced NLP (optional, for ngram_analysis.py)

## Research Insights

The analysis reveals that poultry heat stress research has evolved to focus specifically on **broiler heat stress manipulation** with targeted interventions. Key findings include:

### Research Focus Areas:
- **Nutritional Interventions** dominate (Topic 7: ~87% prevalence)
- **Antioxidant Systems**: Emphasis on vitamins, selenium, and organic compounds
- **Phytogenic Compounds**: Growing interest in curcumin, resveratrol, and herbal extracts
- **Amino Acid Supplementation**: Consistent research on citrulline, glutamine, arginine
- **Thermotolerance Mechanisms**: Understanding heat adaptation pathways

### Research Evolution:
- **Peak Activity**: 2024 (58 abstracts), 2023 (52 abstracts), 2022 (50 abstracts)
- **Methodological Shift**: From general poultry to broiler-specific research
- **Intervention Focus**: From broad "thermoregulatory agents" to specific compounds
- **Mechanistic Understanding**: Increased focus on oxidative stress and molecular pathways

### N-gram Analysis Insights:
- **Top Unigrams**: heat (2.77%), stress (2.43%), broilers (1.65%)
- **Key Bigrams**: heat stress (1.94%), growth performance (0.54%), broiler chickens (0.52%)
- **Research Patterns**: Strong emphasis on "feed conversion ratio", "heat stressed broilers", "antioxidant capacity"

The temporal analysis shows increasing research intensity, reflecting growing awareness of climate change impacts on poultry production and the need for targeted mitigation strategies.

## Author

Analysis scripts developed by GitHub Copilot (July 2025)
