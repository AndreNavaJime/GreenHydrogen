"use client";

import { useMemo, useState } from "react";
import { cohorts, doeBenchmarks, robustness, type CohortKey } from "@/data/revisedV3";

const techClass: Record<string, string> = {
  AWE: "tech-awe",
  PEM: "tech-pem",
  "O-SOEC": "tech-soec",
};

function Money({ value }: { value: number }) {
  return <>{value.toFixed(3)}</>;
}

export default function Home() {
  const [cohortKey, setCohortKey] = useState<CohortKey>("2022");
  const cohort = useMemo(
    () => cohorts.find((item) => item.key === cohortKey) ?? cohorts[0],
    [cohortKey]
  );

  return (
    <main>
      <section className="hero">
        <div className="eyebrow">REVISED DETERMINISTIC V3</div>
        <h1>U.S. Electrolysis Cost Explorer</h1>
        <p className="hero-copy">
          Grid-connected, plant-gate levelized cost of hydrogen across the 48 contiguous U.S. states
          for liquid alkaline water electrolysis (AWE), PEM, and oxide-ion solid-oxide electrolysis
          (O-SOEC).
        </p>
        <div className="scope-strip">
          <span>Real 2022 USD</span>
          <span>8% real WACC</span>
          <span>20-year project</span>
          <span>90% utilization</span>
          <span>Plant-gate only</span>
        </div>
      </section>

      <section className="notice">
        <strong>Scientific boundary:</strong> this is a grid-connected screening baseline. It does not
        establish renewable matching, emissions intensity, or green-hydrogen certification.
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <div className="eyebrow">DETERMINISTIC RESULTS</div>
            <h2>Technology cost by screening cohort</h2>
          </div>
          <div className="segmented" role="tablist" aria-label="Cohort">
            {cohorts.map((item) => (
              <button
                key={item.key}
                className={item.key === cohortKey ? "active" : ""}
                onClick={() => setCohortKey(item.key)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        <p className="subtle">{cohort.subtitle}</p>

        <div className="tech-grid">
          {cohort.rows.map((row) => (
            <article className="tech-card" key={row.technology}>
              <div className="tech-card-top">
                <span className={"tech-pill " + techClass[row.technology]}>{row.technology}</span>
                <span className="winner-count">{row.winners}/48 state wins</span>
              </div>
              <div className="big-number">
                $<Money value={row.median} /><span>/kg H₂</span>
              </div>
              <div className="metric-row">
                <span>P10–P90 across states</span>
                <strong>${row.p10.toFixed(3)}–${row.p90.toFixed(3)}</strong>
              </div>
              <div className="metric-row">
                <span>Full state range</span>
                <strong>${row.minimum.toFixed(3)}–${row.maximum.toFixed(3)}</strong>
              </div>
            </article>
          ))}
        </div>
        <p className="footnote">
          P10–P90 values are spatial quantiles across 48 equally weighted states, not uncertainty intervals.
        </p>
      </section>

      <section className="panel">
        <div className="eyebrow">SOURCE-BACKED ROBUSTNESS</div>
        <h2>How stable are the technology rankings?</h2>
        <div className="robust-grid">
          <article className="robust-card">
            <div className="stat">{robustness.capexEnvelope.reversibleStates}/48</div>
            <h3>states can reverse within the DOE CAPEX box</h3>
            <p>{robustness.capexEnvelope.note}</p>
            <div className="tag">Only {robustness.capexEnvelope.robustState} is robust</div>
          </article>
          <article className="robust-card">
            <div className="stat">{robustness.electricity2025.winnerChanges}</div>
            <h3>winner change under observed 2025 electricity prices</h3>
            <p>{robustness.electricity2025.state}: {robustness.electricity2025.change}</p>
            <div className="tag">Observed-price case, not a forecast</div>
          </article>
          <article className="robust-card">
            <div className="stat">{robustness.gas2024.winnerChanges}</div>
            <h3>winner changes under observed 2024 gas prices</h3>
            <p>
              Rhode Island remains O-SOEC; its AWE−O-SOEC gap narrows from $
              {robustness.gas2024.rhodeIslandGapBefore.toFixed(4)} to $
              {robustness.gas2024.rhodeIslandGapAfter.toFixed(4)}/kg.
            </p>
            <div className="tag">Gas-only named case</div>
          </article>
        </div>
      </section>

      <section className="panel">
        <div className="eyebrow">DOE 2024 CONSISTENCY BENCHMARK</div>
        <h2>Published current-grid comparison</h2>
        <p className="subtle">
          DOE values are shown in real-2022 USD for display only. The comparison is not empirical
          validation: DOE and v3 use different electricity prices, utilization, project life, financing,
          and other boundaries.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Technology</th>
                <th>v3 state median</th>
                <th>DOE current-grid range</th>
              </tr>
            </thead>
            <tbody>
              {doeBenchmarks.map((row) => (
                <tr key={row.technology}>
                  <td><span className={"tech-pill " + techClass[row.technology]}>{row.technology}</span></td>
                  <td>${row.v3Median.toFixed(3)}/kg</td>
                  <td>${row.doeLow.toFixed(3)}–${row.doeHigh.toFixed(3)}/kg</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel boundary">
        <div>
          <div className="eyebrow">MODEL BOUNDARY</div>
          <h2>What enters the accepted baseline?</h2>
        </div>
        <div className="boundary-grid">
          <div>
            <h3>Included</h3>
            <ul>
              <li>EIA 2022 state industrial electricity prices</li>
              <li>DOE technology SEC, installed CAPEX, OPEX, and stack lifetime</li>
              <li>Discounted stack replacement</li>
              <li>O-SOEC natural-gas thermal input</li>
            </ul>
          </div>
          <div>
            <h3>Outside baseline LCOH</h3>
            <ul>
              <li>Transport and storage</li>
              <li>Direct water-price charge</li>
              <li>CO₂ surcharge</li>
              <li>Land/infrastructure CAPEX multipliers</li>
            </ul>
          </div>
        </div>
      </section>

      <footer>
        <strong>Repository transition:</strong> this interface represents the revised v3 scientific
        architecture. The historical Streamlit/ML dashboard remains in the repository for provenance
        but is not the active scientific interface for the revised manuscript.
      </footer>
    </main>
  );
}
