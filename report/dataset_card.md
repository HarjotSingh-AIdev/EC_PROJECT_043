# Dataset card

- Source: `DataSet for Project-EC 101.pdf` (1 page).
- Rows after cleaning: 50; duplicate rows removed: 0; rows removed for missingness: 0.
- Inputs: 10 design variables; outputs: 35 performance values.
- Splits: train 35, validation 8, test 7.
- Missing-data treatment: drop rows above 2% missingness, then use training medians for remaining cells.
- Scalers: X and y StandardScaler fitted on train only; split CSV files remain in original units.
- Constant columns: none.

## Performance output ranges

| Output | Min | Max |
|---|---:|---:|
| P1_F1 | -32.715 | -10.642 |
| P1_F2 | -31.098 | -10.521 |
| P1_F3 | -30.707 | -9.503 |
| P1_F4 | -30.315 | -9.189 |
| P1_F5 | -31.12 | -9.791 |
| P2_F1 | 0.04905 | 0.11635 |
| P2_F2 | 0.12392 | 0.26611 |
| P2_F3 | 0.21733 | 0.44102 |
| P2_F4 | 0.37047 | 0.76126 |
| P2_F5 | 0.71065 | 1.45593 |
| P3_F1 | 2.213 | 3.89 |
| P3_F2 | 3.205 | 4.584 |
| P3_F3 | 3.945 | 5.488 |
| P3_F4 | 4.942 | 6.323 |
| P3_F5 | 5.717 | 7.242 |
| P4_F1 | 3.511 | 5.096 |
| P4_F2 | 4.226 | 5.976 |
| P4_F3 | 4.971 | 6.579 |
| P4_F4 | 6.081 | 7.512 |
| P4_F5 | 6.779 | 8.314 |
| P5_F1 | 72.49 | 81.18 |
| P5_F2 | 72.55 | 80.94 |
| P5_F3 | 72.45 | 81.26 |
| P5_F4 | 72.66 | 81.22 |
| P5_F5 | 72.74 | 81.23 |
| P6_F1 | 15.94 | 20.97 |
| P6_F2 | 18.25 | 23.63 |
| P6_F3 | 21.11 | 25.91 |
| P6_F4 | 22.96 | 30.15 |
| P6_F5 | 26.56 | 31.66 |
| P7_F1 | 13.64 | 18.17 |
| P7_F2 | 15.89 | 20.81 |
| P7_F3 | 18.9 | 23.27 |
| P7_F4 | 20.39 | 25.1 |
| P7_F5 | 22.64 | 27.85 |
