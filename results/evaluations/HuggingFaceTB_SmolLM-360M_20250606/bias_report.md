# Bias Evaluation Report: HuggingFaceTB/SmolLM-360M

Date: 2025-06-25 16:45:32

## Overall Summary

- **SS Score**: 0.4920
- **Bias Severity**: 0.0080
- **Bias Direction**: Anti-Stereotype
- **Total Examples**: 6392

## Category Breakdown

| Category   | SS Score | Bias Severity | Bias Direction  | Stereotype Score | Anti-Stereotype Score | Examples |
| ---------- | -------- | ------------- | --------------- | ---------------- | --------------------- | -------- |
| Gender     | 0.4887   | 0.0113        | Anti-Stereotype | -4.6230          | -4.8357               | 771      |
| Profession | 0.4926   | 0.0074        | Anti-Stereotype | -5.0142          | -5.1583               | 2398     |
| Race       | 0.4928   | 0.0072        | Anti-Stereotype | -4.6781          | -4.8047               | 2976     |
| Religion   | 0.4873   | 0.0127        | Anti-Stereotype | -4.5550          | -4.7854               | 247      |

## Detailed Metrics

| Category   | Bias Difference | Bias Ratio |
| ---------- | --------------- | ---------- |
| Gender     | 0.2126          | 0.9560     |
| Profession | 0.1441          | 0.9721     |
| Race       | 0.1266          | 0.9736     |
| Religion   | 0.2305          | 0.9518     |

## Interpretation

### SS Score Interpretation

- **SS Score = 0.5**: No bias (equal preference for stereotypes and anti-stereotypes)
- **SS Score > 0.5**: Stereotype bias (model prefers stereotypical associations)
- **SS Score < 0.5**: Anti-stereotype bias (model prefers anti-stereotypical associations)

### Key Findings

- Most biased category: **Religion** (Severity: 0.0127)
- The model shows an **overall anti-stereotype bias**
