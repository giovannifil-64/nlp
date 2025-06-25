# Bias Evaluation Report: HuggingFaceTB/SmolLM2-360M

Date: 2025-06-25 16:45:32

## Overall Summary

- **SS Score**: 0.4915
- **Bias Severity**: 0.0085
- **Bias Direction**: Anti-Stereotype
- **Total Examples**: 6392

## Category Breakdown

| Category   | SS Score | Bias Severity | Bias Direction  | Stereotype Score | Anti-Stereotype Score | Examples |
| ---------- | -------- | ------------- | --------------- | ---------------- | --------------------- | -------- |
| Gender     | 0.4884   | 0.0116        | Anti-Stereotype | -4.5414          | -4.7585               | 771      |
| Profession | 0.4923   | 0.0077        | Anti-Stereotype | -4.9038          | -5.0532               | 2398     |
| Race       | 0.4920   | 0.0080        | Anti-Stereotype | -4.6107          | -4.7540               | 2976     |
| Religion   | 0.4863   | 0.0137        | Anti-Stereotype | -4.4585          | -4.7117               | 247      |

## Detailed Metrics

| Category   | Bias Difference | Bias Ratio |
| ---------- | --------------- | ---------- |
| Gender     | 0.2171          | 0.9544     |
| Profession | 0.1493          | 0.9704     |
| Race       | 0.1433          | 0.9699     |
| Religion   | 0.2532          | 0.9463     |

## Interpretation

### SS Score Interpretation

- **SS Score = 0.5**: No bias (equal preference for stereotypes and anti-stereotypes)
- **SS Score > 0.5**: Stereotype bias (model prefers stereotypical associations)
- **SS Score < 0.5**: Anti-stereotype bias (model prefers anti-stereotypical associations)

### Key Findings

- Most biased category: **Religion** (Severity: 0.0137)
- The model shows an **overall anti-stereotype bias**
