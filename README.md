# Collaborative_fltering_recommendation_system_based_on_improved_Jaccard_similarity
Author : Soon Hyeok Park, Kyoungok Kim(coressponding author)

Paper Link : https://link.springer.com/article/10.1007/s12652-023-04647-0

The first author uploads the experimental code for this study through this repository.

## The main contributions of this study are summarized as follows:
- We proposed strategies to address each limitation of Rating_Jaccard to obtain better extensions of Jaccard similarity.
- We verifed the efectiveness of each proposed strategy in detail using several datasets and determined the best similarity among the proposed Jaccard similarity extensions, considering the prediction performance and computation time.
- RJAC_DUB, which was determined as the best similarity among the proposed ones in this study, considered the rating information and inherently users’ rating preference behavior in the similarity calculation.
- Through extensive experiments based on various datasets, we demonstrated the superiority of the proposed similarity over existing variations of Jaccard similarity and other similarity measures.

## Motivation
The similarity measures proposed in this study are inspired by Rating_Jaccard (Ayub et al. 2020a). This study aims to address the following limitations of Rating_Jaccard. 
1. In general, the similarity between two users is proportional to the number of co-rated items. However, Rating_Jaccard is inversely proportional to the number of co-rated items.
2. Rating_Jaccard only enumerates co-rated items with identical ratings; thus, the similarity between users may be zero in more cases than for Jaccard similarity. Particularly, if two users have a small number of co-rated 
items, the similarity between them is likely to be zero. In this case, it is impossible to distinguish them from a pair of users with zero co-rated items. Moreover, the overabundance of zero similarity values complicates the identifcation of a sufcient number of nearest neighbors, which increases the number of items whose ratings cannot be predicted during the prediction of ratings of unrated items.
3. The rating behaviors of users vary widely. However, Rating_Jaccard does not consider this variability

## Proposed methodology
#### RJAC_U
- Equation

  ![image](https://github.com/soonhp/Collaborative_fltering_recommendation_system_based_on_improved_Jaccard_similarity/assets/73877159/5248f0d7-faa2-4f13-ba38-5e5bd848be0e)
  ![image](https://github.com/soonhp/Collaborative_fltering_recommendation_system_based_on_improved_Jaccard_similarity/assets/73877159/35b6ebc8-f3fd-48bd-a0f3-427eca73e6ee)


#### RJAC_DUB
- Equation
  
    ![image](https://github.com/soonhp/Collaborative_fltering_recommendation_system_based_on_improved_Jaccard_similarity/assets/73877159/d78611a0-4839-4759-9bfd-4c95e856250b)
  

## Comparison results of the proposed similarity measures : MAE

![image](https://github.com/soonhp/Collaborative_fltering_recommendation_system_based_on_improved_Jaccard_similarity/assets/73877159/2e7cc9dd-d000-469d-a1ce-55c31f46bd16)


## Comparison results of the proposed similarity measures : F1

![image](https://github.com/soonhp/Collaborative_fltering_recommendation_system_based_on_improved_Jaccard_similarity/assets/73877159/db29959d-4a5d-4f88-b723-254a63391b52)


## The scores to determine the best similarity

![image](https://github.com/soonhp/Collaborative_fltering_recommendation_system_based_on_improved_Jaccard_similarity/assets/73877159/7f5d4262-940c-482a-9016-61ee8f770b00)

##  Comparison results of Proposed Methodology and other similarities : MAE

![image](https://github.com/soonhp/Collaborative_fltering_recommendation_system_based_on_improved_Jaccard_similarity/assets/73877159/9d70197a-8bec-4acf-8bcc-9497fdaf6cb7)


##  Comparison results of Proposed Methodology and other similarities : F1

![image](https://github.com/soonhp/Collaborative_fltering_recommendation_system_based_on_improved_Jaccard_similarity/assets/73877159/5f817bac-397c-45b5-8cec-223d13950e24)

## Conclusion
In terms of the MAE, F1-score, and computation time, RJAC_DUB was determined to be the best metric among the proposed candidates. 

RJAC_DUB generally outperformed the other similarity measures in terms of MAE and F1-score.

Moreover, RJAC_DUB was slower than the traditional similarity but the computation time did not increase signifcantly compared to  the other improved similarities.

Furthermore, RJAC_DUB was superior to JacLMH for all the datasets and RJaccard for the large-size datasets in terms of calculation speed.
