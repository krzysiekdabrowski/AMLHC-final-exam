# Report

## Introduction

### Motivation and Background

The report explores the suitability of dysphonia measurements for telemonitoring Parkinson's disease, with a focus on voice disorders and dysphonia.

### Reference

**Title**: "Suitability of Dysphonia Measurements for Telemonitoring of Parkinson's Disease"

**Author**: Max Little

**Email**: [littlem@robots.ox.ac.uk](http://robots.ox.ac.uk/)

Dysphonia measurements refer to a set of vocal features or measurements that are used to assess and characterize voice disorders, particularly dysphonia. Dysphonia is a medical term that refers to difficulty or impairment in voice production, often resulting in changes in voice quality, pitch, loudness, or resonance. Dysphonia measurements are used in clinical settings to diagnose voice disorders, monitor treatment progress, and evaluate voice rehabilitation outcomes.

## Methods

### 1. Fetch the Dataset

The dataset is fetched using `fetch_ucirepo(id=174)`.

### 2. Explore the Data

Metadata and variable information are printed to understand the dataset. First few rows of features and targets are printed to inspect the data.

## Type of Machine Learning

Supervised learning is the chosen approach because the targets are clearly labeled as 0 (no Parkinson) or 1 (Parkinson's positive). The data is labeled, making it suitable for supervised learning techniques.
