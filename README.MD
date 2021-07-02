Medical Concept Normalization in Clinical Trials with Drug and Disease Representation Learning
---
This repository contains additional materials of our paper ["Medical Concept Normalization in Clinical Trials with Drug and Disease Representation Learning"](https://arxiv.org/pdf/2101.09311.pdf). 
We investigate the effectiveness of transferring concept normalization from the general biomedical domain to the 
clinical trials domain in a zero-shot setting with an absence of labeled data. We propose a simple and effective 
two-stage neural approach based on fine-tuned BERT architectures. In the first stage, we train a metric learning model 
that optimizes relative similarity of mentions and concepts via triplet loss. The model is trained on available labeled 
corpora of scientific abstracts to obtain vector embeddings of concept names and entity mentions from texts. In the 
second stage, we find the closest concept name representation in an embedding space to a given clinical mention. We 
evaluated several models, including state-of-the-art architectures, on a dataset of abstracts and a real-world dataset
 of trial records with interventions and conditions mapped to drug and disease terminologies. Extensive experiments
  validate the effectiveness of our approach in knowledge transfer from the scientific literature to clinical trials.
  
Evaluation & Results
---
Table 1
Out-of-domain performance of the proposed DILBERT model andbaselines in terms of Acc@1 on the 
filtered test set of clinical trials (CT)

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="2">CT Condition</th>
    <th colspan="2">CT Intervention</th>
  </tr>
  <tr>
    <td>single concept</td>
    <td>full set</td>
    <td>single concept</td>
    <td>full set</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>BioBERT ranking</td>
    <td>72.6</td>
    <td>71.74</td>
    <td>77.83</td>
    <td>56.97</td>
  </tr>
  <tr>
    <td>BioSyn</td>
    <td>86.36</td>
    <td>-</td>
    <td>79.58</td>
    <td>-</td>
  </tr>
  <tr>
    <td colspan="5">DILBERT with different ranking strategies</td>
  </tr>
  <tr>
    <td>random sampling</td>
    <td>85.73</td>
    <td>84.85</td>
    <td>82.54</td>
    <td>81.16</td>
  </tr>
  <tr>
    <td>random + 2 parents</td>
    <td>86.74</td>
    <td>86.36</td>
    <td>81.84</td>
    <td>79.14</td>
  </tr>
  <tr>
    <td>random + 5 parents</td>
    <td>87.12</td>
    <td>86.74</td>
    <td>81.67</td>
    <td>79.14</td>
  </tr>
  <tr>
    <td>resampling</td>
    <td>85.22</td>
    <td>84.63</td>
    <td>81.67</td>
    <td>80.21</td>
  </tr>
  <tr>
    <td>resampling + 5 siblings</td>
    <td>84.84</td>
    <td>84.26</td>
    <td>80.62</td>
    <td>76.16</td>
  </tr>
</tbody>
</table>

Table 2
In-domain performance of the proposed DILBERT model interms of Acc@1 on the refined test set of 
the Biocreative V CDR corpus

<table>
<thead>
  <tr>
    <th>Model</th>
    <th>CDR Disease</th>
    <th>CDR Chemical</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>BioBERT ranking</td>
    <td>66.4</td>
    <td>80.7</td>
  </tr>
  <tr>
    <td>BioSyn</td>
    <td>74.1</td>
    <td>83.8</td>
  </tr>
  <tr>
    <td>DILBERT, random sampling</td>
    <td>75.5</td>
    <td>81.4</td>
  </tr>
  <tr>
    <td>DILBERT, random + 2 parents</td>
    <td>75.0</td>
    <td>81.2</td>
  </tr>
  <tr>
    <td>DILBERT, random + 5 parents</td>
    <td>73.5</td>
    <td>81.4</td>
  </tr>
  <tr>
    <td>DILBERT, resampling</td>
    <td>75.8</td>
    <td>83.3</td>
  </tr>
  <tr>
    <td>DILBERT, resampling + 5 siblings</td>
    <td>75.3</td>
    <td>82.1</td>
  </tr>
</tbody>
</table>
  
 Figure 1
 In-domain performance  of the proposed DILBERT model in terms of Acc@1 on the refined test set of the Biocreative V 
 CDR corpus using reduced dictionaries.
 ![](images/cdr_combined.png)
 
 
 Citing & Authors
 ---