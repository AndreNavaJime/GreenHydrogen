"use client";

import { useEffect, useMemo, useState } from "react";
import { cohorts, doeBenchmarks, robustness, type CohortKey } from "@/data/revisedV3";
import { fetchCsv, num, type CsvRow } from "@/lib/csv";
import USStateMap from "@/components/USStateMap";

const techClass: Record<string, string> = {
  AWE: "tech-awe",
  PEM: "tech-pem",
  "O-SOEC": "tech-soec",
};

const technologies = ["AWE", "PEM", "O-SOEC"] as const;
const years = [2022, 2030, 2040, 2050, 2060] as const;

function Money({ value, digits = 3 }: { value: number; digits?: number }) {
  return <>{Number.isFinite(value) ? value.toFixed(digits) : "—"}</>;
}

function ComponentBar({ row }: { row: CsvRow }) {
  const components = [
    ["CAPEX", num(row.C_capex_usd_kg)],
    ["Fixed OPEX", num(row.C_fixed_opex_usd_kg)],
    ["Electricity", num(row.C_electric_usd_kg)],
    ["Heat", num(row.C_thermal_usd_kg)],
    ["Replacement", num(row.C_replacement_usd_kg)],
  ] as const;
  const total = num(row.LCOH_plantgate_usd_kg);

  return (
    <div className="component-wrap">
      <div className="component-bar" aria-label="LCOH component shares">
        {components.map(([label, value]) => {
          const width = total > 0 && value > 0 ? (value / total) * 100 : 0;
          return (
            <span
              key={label}
              className={"component-segment component-" + label.toLowerCase().replace(" ", "-")}
              style={{ width: width + "%" }}
              title={label + ": $" + (Number.isFinite(value) ? value.toFixed(3) : "—") + "/kg"}
            />
          );
        })}
      </div>
      <div className="component-legend">
        {components.map(([label, value]) => (
          <span key={label}>
            <i className={"legend-dot component-" + label.toLowerCase().replace(" ", "-")} />
            {label}: $<Money value={value} />
          </span>
        ))}
      </div>
    </div>
  );
}

export default function Home() {
  const [cohortKey, setCohortKey] = useState<CohortKey>("2022");
  const [stateName, setStateName] = useState("Rhode Island");
  const [year, setYear] = useState<number>(2022);
  const [baseline, setBaseline] = useState<CsvRow[]>([]);
  const [rankRobustness, setRankRobustness] = useState<CsvRow[]>([]);
  const [electricityScenario, setElectricityScenario] = useState<CsvRow[]>([]);
  const [gasScenario, setGasScenario] = useState<CsvRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [baselineRows, robustnessRows, electricityRows, gasRows] = await Promise.all([
          fetchCsv("/data/canonical/deterministic_state_year.csv"),
          fetchCsv("/data/canonical/state_rank_robustness_2022.csv"),
          fetchCsv("/data/canonical/eia_2025_observed_price_sensitivity.csv"),
          fetchCsv("/data/canonical/o_soec_2024_ng_price_sensitivity.csv"),
        ]);
        if (cancelled) return;
        setBaseline(baselineRows);
        setRankRobustness(robustnessRows);
        setElectricityScenario(electricityRows);
        setGasScenario(gasRows);
      } catch (error) {
        if (!cancelled) {
          setLoadError(error instanceof Error ? error.message : "Unable to load canonical data.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const cohort = useMemo(
    () => cohorts.find((item) => item.key === cohortKey) ?? cohorts[0],
    [cohortKey]
  );

  const states = useMemo(
    () => Array.from(new Set(baseline.map((row) => row.State))).filter(Boolean).sort(),
    [baseline]
  );

  useEffect(() => {
    if (states.length && !states.includes(stateName)) setStateName(states[0]);
  }, [states, stateName]);

  const selectedRows = useMemo(
    () =>
      technologies
        .map((technology) =>
          baseline.find(
            (row) =>
              row.State === stateName &&
              row.Technology === technology &&
              Number(row.Year) === year
          )
        )
        .filter((row): row is CsvRow => Boolean(row)),
    [baseline, stateName, year]
  );

  const selectedWinner = useMemo(() => {
    if (!selectedRows.length) return "—";
    return [...selectedRows].sort(
      (a, b) => num(a.LCOH_plantgate_usd_kg) - num(b.LCOH_plantgate_usd_kg)
    )[0]?.Technology ?? "—";
  }, [selectedRows]);

  const selectedRobustness = useMemo(
    () => rankRobustness.find((row) => row.State === stateName),
    [rankRobustness, stateName]
  );

  const selectedElectricityScenario = useMemo(
    () =>
      electricityScenario.filter(
        (row) => row.State === stateName && Number(row.Year) === year
      ),
    [electricityScenario, stateName, year]
  );

  const selectedGasScenario = useMemo(
    () =>
      gasScenario.find(
        (row) => row.State === stateName && Number(row.Year) === year
      ),
    [gasScenario, stateName, year]
  );

  const electricityWinner =
    selectedElectricityScenario[0]?.Named_scenario_winner ?? "—";
  const gasWinner = selectedGasScenario?.Named_scenario_winner ?? "—";

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
                <strong>{"$"}{row.p10.toFixed(3)}–{"$"}{row.p90.toFixed(3)}</strong>
              </div>
              <div className="metric-row">
                <span>Full state range</span>
                <strong>{"$"}{row.minimum.toFixed(3)}–{"$"}{row.maximum.toFixed(3)}</strong>
              </div>
            </article>
          ))}
        </div>
        <p className="footnote">
          P10–P90 values are spatial quantiles across 48 equally weighted states, not uncertainty intervals.
        </p>
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <div className="eyebrow">FULL 48-STATE EXPLORER</div>
            <h2>Inspect accepted v3 costs and components</h2>
          </div>
          <div className="explorer-controls">
            <label className="state-picker">
              <span>State</span>
              <select value={stateName} onChange={(event) => setStateName(event.target.value)}>
                {(states.length ? states : ["Rhode Island"]).map((state) => (
                  <option key={state} value={state}>{state}</option>
                ))}
              </select>
            </label>
            <label className="state-picker">
              <span>Screening year</span>
              <select value={year} onChange={(event) => setYear(Number(event.target.value))}>
                {years.map((value) => (
                  <option key={value} value={value}>{value}</option>
                ))}
              </select>
            </label>
          </div>
        </div>

        {loading && <p className="subtle">Loading canonical v3 state data…</p>}
        {loadError && <p className="error-box">{loadError}</p>}

        {!loading && !loadError && (
          <>
            <div className="winner-banner">
              <span>{stateName} · {year}</span>
              <strong>Lowest deterministic LCOH: {selectedWinner}</strong>
            </div>

            <USStateMap
              rows={baseline}
              year={year}
              selectedState={stateName}
              onSelectState={setStateName}
            />
            <div className="tech-grid">
              {selectedRows.map((row) => (
                <article className="tech-card explorer-card" key={row.Technology}>
                  <div className="tech-card-top">
                    <span className={"tech-pill " + techClass[row.Technology]}>{row.Technology}</span>
                    <span className="winner-count">
                      {row.Technology === selectedWinner ? "Lowest cost" : "Comparator"}
                    </span>
                  </div>
                  <div className="big-number">
                    $<Money value={num(row.LCOH_plantgate_usd_kg)} /><span>/kg H₂</span>
                  </div>
                  <ComponentBar row={row} />
                  <div className="mini-grid">
                    <div><span>Avg. SEC</span><strong><Money value={num(row.SEC_average_system_kWh_kg)} digits={1} /> kWh/kg</strong></div>
                    <div><span>Installed CAPEX</span><strong>$<Money value={num(row.Installed_CAPEX_2022real_usd_system_kW)} digits={0} />/kW</strong></div>
                    <div><span>Stack life</span><strong><Money value={num(row.Stack_lifetime_hours_at_commissioning)} digits={0} /> h</strong></div>
                    <div><span>Electricity</span><strong>{(num(row.Electricity_price_2022real_usd_kWh) * 100).toFixed(2)}¢/kWh</strong></div>
                  </div>
                </article>
              ))}
            </div>

            <div className="scenario-grid">
              <article className="scenario-card">
                <div className="eyebrow">DOE CAPEX ENVELOPE · 2022 ONLY</div>
                {year === 2022 && selectedRobustness ? (
                  <>
                    <h3>
                      {selectedRobustness.Ranking_class === "ROBUST_TO_DOE_CAPEX_RANGE"
                        ? "Winner remains robust"
                        : "Winner can reverse"}
                    </h3>
                    <p>
                      Baseline winner: <strong>{selectedRobustness.Deterministic_winner_v3}</strong>.
                      Named winners across the DOE box: <strong>{selectedRobustness.Distinct_named_scenario_winners}</strong>.
                    </p>
                    {selectedRobustness.Reversal_witness_scenario && (
                      <p className="footnote">Witness: <code>{selectedRobustness.Reversal_witness_scenario}</code></p>
                    )}
                  </>
                ) : (
                  <p className="subtle">The DOE status-CAPEX envelope is intentionally applied only to the 2022 status cohort.</p>
                )}
              </article>

              <article className="scenario-card">
                <div className="eyebrow">OBSERVED 2025 ELECTRICITY PRICE</div>
                <h3>Named-scenario winner: {electricityWinner}</h3>
                <p>
                  {selectedElectricityScenario[0]?.Winner_changed_from_v3 === "True"
                    ? "The technology winner changes relative to v3."
                    : "The technology winner is unchanged relative to v3."}
                </p>
                <div className="scenario-list">
                  {selectedElectricityScenario.map((row) => (
                    <span key={row.Technology}>
                      {row.Technology}: $<Money value={num(row.Named_scenario_LCOH_plantgate_usd_kg)} />/kg
                    </span>
                  ))}
                </div>
              </article>

              <article className="scenario-card">
                <div className="eyebrow">OBSERVED 2024 NATURAL GAS PRICE</div>
                <h3>Named-scenario winner: {gasWinner}</h3>
                {selectedGasScenario ? (
                  <p>
                    O-SOEC: $<Money value={num(selectedGasScenario.V3_O_SOEC_LCOH_real2022_USD_kg)} /> → $
                    <Money value={num(selectedGasScenario.Scenario_O_SOEC_LCOH_real2022_USD_kg)} />/kg
                    {" "}({num(selectedGasScenario.Delta_O_SOEC_LCOH_real2022_USD_kg) >= 0 ? "+" : ""}
                    <Money value={num(selectedGasScenario.Delta_O_SOEC_LCOH_real2022_USD_kg)} />/kg).
                  </p>
                ) : (
                  <p className="subtle">No gas-case row found for this state/year.</p>
                )}
              </article>
            </div>

            <div className="download-row">
              <a href="/data/canonical/deterministic_state_year.csv" download>Download deterministic v3 CSV</a>
              <a href="/data/canonical/state_rank_robustness_2022.csv" download>Download ranking robustness CSV</a>
              <a href="/data/canonical/MIGRATION_MANIFEST.json" download>Download provenance manifest</a>
            </div>
          </>
        )}
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
                  <td>{"$"}{row.v3Median.toFixed(3)}/kg</td>
                  <td>{"$"}{row.doeLow.toFixed(3)}–{"$"}{row.doeHigh.toFixed(3)}/kg</td>
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
        <strong>Canonical data source:</strong> the interactive state explorer reads the migrated,
        hash-verified revised-v3 CSVs in <code>public/data/canonical/</code>. The historical
        Streamlit/ML dashboard remains in the repository for provenance but is not used here.
      </footer>
    </main>
  );
}
