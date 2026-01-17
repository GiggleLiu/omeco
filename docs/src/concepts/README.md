# Concepts

Understanding these core concepts will help you use omeco effectively.

## Overview

Tensor network contraction is a fundamental operation in many scientific and machine learning applications. This section explains:

- What tensor networks are and why they matter
- Why contraction order makes an exponential difference
- How to interpret complexity metrics

## Topics

- [Tensor Networks](./tensor-networks.md) - Einstein summation and tensor operations
- [Contraction Order Problem](./contraction-order.md) - Why order matters
- [Complexity Metrics](./complexity-metrics.md) - Understanding tc, sc, and rwc

## Quick Summary

**The Problem**: Contracting N tensors can be done in many orders, with costs differing exponentially.

**The Solution**: Heuristic algorithms find near-optimal orders in polynomial time.

**The Trade-off**: Time vs quality - greedy is fast, simulated annealing is better.
