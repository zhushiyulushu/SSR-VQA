# Learning Structured Scene Representations for Visual Question Answering

This repository provides the implementation of the Structured Scene Reasoning (SSR) framework for dense structured visual question answering. The method converts unordered object detections into a learnable topological grid representation and performs missing-object reasoning for structured shelf scenes.

## Overview

SSR follows a structured pipeline:

1. Object detection and local semantic-spatial aggregation.
2. Sinusoidal Spatial Embedding (SSE) for spatial representation.
3. Neural Grid Inference (NGI) for row-column topology learning.
4. Directional Structure Induction (DSI) for missing-region reasoning.
5. Symbolic answer synthesis for structured VQA.

The repository contains the source code used for data processing, learnable grid completion, baseline evaluation, ablation studies, cross-domain evaluation, and visualization.

## Repository Structure

```text
configs/       Configuration templates
scripts/       Helper scripts
src/           Source code

