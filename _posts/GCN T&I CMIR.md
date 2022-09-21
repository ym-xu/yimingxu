---
layout: post
title:  Modeling Text with Graph Convolutional Network for Cross-Modal Information Retrieval.
date:   2022-03-21 16:40:16
description: 
tags: Cross-modal-Retrieval GCN
categories: Literature-Notes
---

## Overview

**What is Cross-modal Information Retrieval?**

- CMIR enables users to take a query of one modality to retrieve data in relevant content in other modalities.

This paper model texts by graphs using similarity measure based on word2vec. The authors build a dual-path neural network to learn the couple features in cross-modal information retrival.

In text modeling, they use GCN, and in image modeling, they use neural network with layers of nonlinearities  based on off-the-shelf features.

## Motivation and Contributions

**To solve the Challenges:** 

- Heterogeneous gap and semantic gap between different modality features, no natural correspondence between different modalities. 
- Although existed text feature extraction methods like word2vec is enriched by learning from neighboring words, it still ignores the global structural information. 

**Contributions**

- Model text data by GCN, which realizes the cross-modal retrieval between irregular graph-structured data and regular grid-structured data;
- Jointly learn the textual and visual representations as well as text-image similarity metric, providing an end-to-end training model;
- Experimental on Eng-Wiki, NUS-WIDE, Pascal, TVGraz and Ch-Wiki.

## Methodology

### Text Modeling

Words: $$W = [w_1,w_2,...,w_N]$$

Graph is a k-nearest neighbor graph - $$G = (V,E)$$ . vertex $$v_i \in V$$ is corresponding to a unique word and each edge $$e_{ij} \in E$$ is defined by the word2vec similarity between two words.

$$  e_{ij} =    \begin{cases}      1,  & \text{if $w_i \in N_{k}(w_{j}) $ or $w_j \in N_{k}(w_{i}) $ } \\      0, & \text{otherwise}    \end{cases} $$

k = 8, graph structure is $$A \in \mathbb{R}^{N \times N}$$ , graph features is $$bag-of-words$$ vector and the frequency value of word $$w_i$$ serves as the 1-dimensional feature on vertex $$v_i$$.

## Graph Convolutional

Given a text's input graph feature vector $$F_{in}$$ , output $$F_{out}$$.

**First**, $$F_{in}$$ is transformed to the spectral domain via graph Fourier transform, based on the normalized graph Laplacian:

$$L = I_N - D^{-1/2}AD^{-1/2}$$

$$I_N$$ and $D$ are respectively the identity matrix and diagonal degree matrix of the graph structure $G$.

**Then** , $$L$$ can be eigendecomposed as $$L = U \Lambda U^{T}$$, $$U$$ is a set of eigenvectors and $$\Lambda$$ is a set of real, non-negative eigenvalues. The Fourier transform of $$F_{in}$$ is a function $$U$$:

$$\widehat{F_{in}} = U^T F_{in}$$

Inverse Transform:

$$F_{in} = U \widehat{F_{in}}$$

Convolution of $$F_{in}$$ with a spectral filter $$g_{\theta}$$ is:

$$F_{out} = g_{\theta} * F_{in} = U g_{\theta}U^{T}F_{in}$$

$$\theta$$ is a vector to learn:

$$g_{\theta} = \sum_{k = 0}^{K-1} \theta_k T_k (\widetilde{L})$$

where $$T_{k}(x) = 2xT_{k-1}(x) - T_{k-2}(x)$$ , $$T_{0}(x) = 1$$ , $$T_{1}(x) = x $$, $$\widetilde{L} = 2/\lambda_{max}L - I_N$$ , $$\lambda_{max}$$ is the lengest enginvalue of $$L$$ .

**Then**, $$F_{out} = g_{\theta}F_{in}$$ .