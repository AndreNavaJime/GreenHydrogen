export type Technology = "AWE" | "PEM" | "O-SOEC";
export type CohortKey = "2022" | "2030" | "post2031";

export type TechSummary = {
  technology: Technology;
  median: number;
  p10: number;
  p90: number;
  minimum: number;
  maximum: number;
  winners: number;
};

export type CohortSummary = {
  key: CohortKey;
  label: string;
  subtitle: string;
  rows: TechSummary[];
};

export const cohorts: CohortSummary[] = [
  {
    key: "2022",
    label: "2022 status",
    subtitle: "DOE status technology assumptions with 2022 state energy prices",
    rows: [
      { technology: "AWE", median: 5.574, p10: 4.954, p90: 8.561, minimum: 4.644, maximum: 11.401, winners: 47 },
      { technology: "PEM", median: 6.700, p10: 6.078, p90: 9.697, minimum: 5.767, maximum: 12.548, winners: 0 },
      { technology: "O-SOEC", median: 7.176, p10: 6.715, p90: 9.343, minimum: 6.550, maximum: 11.324, winners: 1 }
    ]
  },
  {
    key: "2030",
    label: "2030 interpolated target",
    subtitle: "Four-fifths interpolation between DOE 2026 and 2031 targets",
    rows: [
      { technology: "AWE", median: 4.236, p10: 3.682, p90: 6.904, minimum: 3.405, maximum: 9.441, winners: 0 },
      { technology: "PEM", median: 4.056, p10: 3.526, p90: 6.610, minimum: 3.261, maximum: 9.037, winners: 0 },
      { technology: "O-SOEC", median: 3.383, p10: 2.952, p90: 5.390, minimum: 2.800, maximum: 7.232, winners: 48 }
    ]
  },
  {
    key: "post2031",
    label: "2031 target held",
    subtitle: "2031 target technology held constant for 2040/2050/2060 screening cohorts",
    rows: [
      { technology: "AWE", median: 4.138, p10: 3.593, p90: 6.765, minimum: 3.320, maximum: 9.264, winners: 0 },
      { technology: "PEM", median: 3.934, p10: 3.416, p90: 6.431, minimum: 3.157, maximum: 8.805, winners: 0 },
      { technology: "O-SOEC", median: 3.287, p10: 2.858, p90: 5.281, minimum: 2.706, maximum: 7.113, winners: 48 }
    ]
  }
];

export const robustness = {
  capexEnvelope: {
    reversibleStates: 47,
    robustStates: 1,
    robustState: "Rhode Island",
    note: "All 47 baseline AWE-winning states admit a winner reversal somewhere within the DOE 2022 installed-CAPEX endpoint box."
  },
  electricity2025: {
    winnerChanges: 1,
    state: "California",
    change: "AWE → O-SOEC",
    note: "Observed 2025 state industrial electricity prices, converted to real-2022 USD."
  },
  gas2024: {
    winnerChanges: 0,
    rhodeIslandGapBefore: 0.0777,
    rhodeIslandGapAfter: 0.0390,
    note: "Observed 2024 industrial natural-gas price case changes no winner."
  }
};

export const doeBenchmarks = [
  { technology: "AWE", v3Median: 5.574, doeLow: 6.227, doeHigh: 6.911 },
  { technology: "PEM", v3Median: 6.700, doeLow: 6.799, doeHigh: 8.098 },
  { technology: "O-SOEC", v3Median: 7.176, doeLow: 8.008, doeHigh: 9.117 }
] as const;
