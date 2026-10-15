"use client";

import { useEffect, useMemo, useState } from "react";
import { cohorts, doeBenchmarks, robustness, type CohortKey } from "@/data/revisedV3";
import { assetUrl, fetchCsv, num, type CsvRow } from "@/lib/csv";
import USStateMap from "@/components/USStateMap";

type TabKey = "overview" | "explorer" | "robustness" | "benchmark" | "methods";

const techClass: Record<string, string> = {
  AWE: "tech-awe",
  PEM: "tech-pem",
  "O-SOEC": "tech-soec",
};

const technologies = ["AWE", "PEM", "O-SOEC"] as const;
const years = [2022, 2030, 2040, 2050, 2060] as const;

const tabs: { key: TabKey; label: string; icon: string }[] = [
  { key: "overview", label: "Overview", icon: "◉" },
  { key: "explorer", label: "State Explorer", icon: "⌖" },
  { key: "robustness", label: "Robustness", icon: "◇" },
  { key: "benchmark", label: "Benchmark", icon: "≋" },
  { key: "methods", label: "Methods & Data", icon: "▦" },
];

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

function MedianBand({ cohortKey }: { cohortKey: CohortKey }) {
  const cohort = cohorts.find((item) => item.key === cohortKey) ?? cohorts[0];
  const values = cohort.rows.map((row) => row.median);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = Math.max(max - min, 0.001);

  return (
    <div className="median-band">
      {cohort.rows.map((row) => {
        const left = 10 + ((row.median - min) / spread) * 80;
        return (
          <div className="median-dot-row" key={row.technology}>
            <span className={"tech-dot " + techClass[row.technology]} />
            <span className="median-label">{row.technology}</span>
            <div className="median-track">
              <span className={"median-marker " + techClass[row.technology]} style={{ left: left + "%" }} />
            </div>
            <strong>{"$"}{row.median.toFixed(3)}</strong>
          </div>
        );
      })}
    </div>
  );
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabKey>("overview");
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

  const electricityWinner = selectedElectricityScenario[0]?.Named_scenario_winner ?? "—";
  const gasWinner = selectedGasScenario?.Named_scenario_winner ?? "—";

  const moveToExplorer = (state?: string) => {
    if (state) setStateName(state);
    setActiveTab("explorer");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => setActiveTab("overview")} aria-label="Go to overview">
          <span className="brand-mark" aria-hidden="true"><i /><b /></span>
          <span>
            <strong>HYDROGEN USA</strong>
            <small>State-resolved electrolysis cost screening</small>
          </span>
        </button>

        <nav className="tab-nav" aria-label="Primary navigation">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              className={activeTab === tab.key ? "nav-tab active" : "nav-tab"}
              onClick={() => setActiveTab(tab.key)}
            >
              <span aria-hidden="true">{tab.icon}</span>{tab.label}
            </button>
          ))}
        </nav>

        <div className="nav-status"><span className="status-dot" />revised v3</div>
      </header>

      {activeTab === "overview" && (
        <main className="page-view page-enter">
          <section className="hero-redesign">
            <div className="hero-ribbon ribbon-one" />
            <div className="hero-ribbon ribbon-two" />
            <div className="hero-copy-block">
              <div className="eyebrow">U.S. ELECTROLYSIS COST EXPLORER</div>
              <h1>A cleaner view of hydrogen cost competitiveness across America.</h1>
              <p>
                Explore grid-connected, plant-gate LCOH for AWE, PEM, and O-SOEC across the
                48 contiguous U.S. states using the accepted revised deterministic v3 model.
              </p>
              <div className="hero-actions">
                <button className="primary-action" onClick={() => moveToExplorer()}>
                  Explore the states <span>→</span>
                </button>
                <button className="secondary-action" onClick={() => setActiveTab("methods")}>
                  Model boundary
                </button>
              </div>
            </div>

            <div className="hero-metrics">
              <article><span className="metric-icon metric-teal">48</span><div><strong>48 states</strong><small>Contiguous U.S. analytical domain</small></div></article>
              <article><span className="metric-icon metric-blue">3</span><div><strong>3 technologies</strong><small>AWE · PEM · O-SOEC</small></div></article>
              <article><span className="metric-icon metric-amber">47/48</span><div><strong>AWE wins in 2022</strong><small>Status technology assumptions</small></div></article>
              <article><span className="metric-icon metric-cobalt">48/48</span><div><strong>O-SOEC wins in 2030</strong><small>Interpolated target cohort</small></div></article>
            </div>
          </section>

          <section className="scientific-strip">
            <strong>Scientific boundary</strong>
            <span>Grid-connected screening baseline · real 2022 USD · 8% real WACC · 20 years · 90% utilization · plant-gate only</span>
          </section>

          <section className="overview-grid">
            <article className="surface-card map-overview-card">
              <div className="section-heading">
                <div>
                  <div className="eyebrow dark-eyebrow">STATE TECHNOLOGY MAP</div>
                  <h2>Lowest deterministic LCOH</h2>
                  <p className="light-subtle">Click a state to open its detailed explorer.</p>
                </div>
                <label className="light-select">
                  <span>Screening year</span>
                  <select value={year} onChange={(event) => setYear(Number(event.target.value))}>
                    {years.map((value) => <option key={value} value={value}>{value}</option>)}
                  </select>
                </label>
              </div>
              {loading && <p className="light-subtle">Loading canonical state data…</p>}
              {loadError && <p className="error-box">{loadError}</p>}
              {!loading && !loadError && (
                <USStateMap rows={baseline} year={year} selectedState={stateName} onSelectState={moveToExplorer} variant="light" />
              )}
            </article>

            <article className="surface-card cohort-card">
              <div className="eyebrow dark-eyebrow">NATIONAL COST PROFILE</div>
              <h2>Median LCOH by screening cohort</h2>
              <div className="segmented light-segmented" role="tablist" aria-label="Cohort">
                {cohorts.map((item) => (
                  <button key={item.key} className={item.key === cohortKey ? "active" : ""} onClick={() => setCohortKey(item.key)}>
                    {item.label}
                  </button>
                ))}
              </div>
              <p className="light-subtle">{cohort.subtitle}</p>
              <MedianBand cohortKey={cohortKey} />
              <div className="overview-winner">
                <span>Lowest median</span>
                <strong>{[...cohort.rows].sort((a, b) => a.median - b.median)[0].technology}</strong>
              </div>
            </article>
          </section>

          <section className="technology-showcase">
            <div className="section-heading">
              <div>
                <div className="eyebrow dark-eyebrow">TECHNOLOGY COMPARISON</div>
                <h2>Three electrolysis pathways, one consistent cost boundary.</h2>
              </div>
              <button className="text-action" onClick={() => setActiveTab("benchmark")}>View DOE benchmark →</button>
            </div>

            <div className="overview-tech-grid">
              {cohort.rows.map((row) => (
                <article className={"overview-tech-card " + techClass[row.technology]} key={row.technology}>
                  <div className="tech-card-header">
                    <span className={"tech-symbol " + techClass[row.technology]}>{row.technology.slice(0, 1)}</span>
                    <div><h3>{row.technology}</h3><span>{row.winners}/48 state wins</span></div>
                  </div>
                  <div className="overview-cost">{"$"}{row.median.toFixed(3)}<small>/kg H₂</small></div>
                  <div className="overview-range">
                    <span>P10–P90 across states</span>
                    <strong>{"$"}{row.p10.toFixed(3)}–{"$"}{row.p90.toFixed(3)}</strong>
                  </div>
                </article>
              ))}
            </div>
            <p className="light-footnote">P10–P90 values are spatial quantiles, not uncertainty intervals.</p>
          </section>

          <section className="robustness-callout">
            <div>
              <div className="eyebrow">SOURCE-BACKED ROBUSTNESS</div>
              <h2><strong>47/48</strong> status rankings can reverse within the DOE 2022 CAPEX envelope.</h2>
            </div>
            <div className="callout-copy">
              <p>Rhode Island is the only state whose O-SOEC winner remains robust across all eight endpoint combinations.</p>
              <button className="primary-action warm" onClick={() => setActiveTab("robustness")}>Explore robustness →</button>
            </div>
          </section>
        </main>
      )}

      {activeTab === "explorer" && (
        <main className="page-view page-enter">
          <section className="page-heading">
            <div>
              <div className="eyebrow">STATE EXPLORER</div>
              <h1>Inspect state-level cost structure.</h1>
              <p>Canonical v3 plant-gate LCOH and component accounting for each technology and screening cohort.</p>
            </div>
            <div className="explorer-controls">
              <label className="dark-select"><span>State</span><select value={stateName} onChange={(event) => setStateName(event.target.value)}>{(states.length ? states : ["Rhode Island"]).map((state) => <option key={state} value={state}>{state}</option>)}</select></label>
              <label className="dark-select"><span>Screening year</span><select value={year} onChange={(event) => setYear(Number(event.target.value))}>{years.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
            </div>
          </section>

          {loading && <p className="dark-subtle">Loading canonical v3 state data…</p>}
          {loadError && <p className="error-box">{loadError}</p>}

          {!loading && !loadError && (
            <>
              <section className="explorer-layout">
                <article className="dark-card explorer-map-card">
                  <div className="card-kicker">INTERACTIVE MAP</div>
                  <USStateMap rows={baseline} year={year} selectedState={stateName} onSelectState={setStateName} variant="dark" />
                </article>

                <aside className="state-insight-card">
                  <span className="state-year">{stateName} · {year}</span>
                  <h2>{selectedWinner}</h2>
                  <p className="state-winner-label">Lowest deterministic plant-gate LCOH</p>
                  <div className="state-tech-list">
                    {[...selectedRows].sort((a, b) => num(a.LCOH_plantgate_usd_kg) - num(b.LCOH_plantgate_usd_kg)).map((row, index) => (
                      <div className="state-tech-row" key={row.Technology}>
                        <span className={"rank-badge " + techClass[row.Technology]}>{index + 1}</span>
                        <div><strong>{row.Technology}</strong><span>{"$"}{num(row.LCOH_plantgate_usd_kg).toFixed(3)}/kg H₂</span></div>
                      </div>
                    ))}
                  </div>
                  <div className="state-note"><span>Electricity price</span><strong>{selectedRows[0] ? (num(selectedRows[0].Electricity_price_2022real_usd_kWh) * 100).toFixed(2) : "—"}¢/kWh</strong></div>
                  <button className="secondary-action full" onClick={() => setActiveTab("robustness")}>Test ranking robustness</button>
                </aside>
              </section>

              <section className="technology-detail-grid">
                {selectedRows.map((row) => (
                  <article className="detail-card" key={row.Technology}>
                    <div className="tech-card-top">
                      <span className={"tech-pill " + techClass[row.Technology]}>{row.Technology}</span>
                      <span className="winner-count">{row.Technology === selectedWinner ? "Lowest cost" : "Comparator"}</span>
                    </div>
                    <div className="big-number dark-number">$<Money value={num(row.LCOH_plantgate_usd_kg)} /><span>/kg H₂</span></div>
                    <ComponentBar row={row} />
                    <div className="mini-grid light-mini">
                      <div><span>Avg. SEC</span><strong><Money value={num(row.SEC_average_system_kWh_kg)} digits={1} /> kWh/kg</strong></div>
                      <div><span>Installed CAPEX</span><strong>$<Money value={num(row.Installed_CAPEX_2022real_usd_system_kW)} digits={0} />/kW</strong></div>
                      <div><span>Stack life</span><strong><Money value={num(row.Stack_lifetime_hours_at_commissioning)} digits={0} /> h</strong></div>
                      <div><span>Electricity</span><strong>{(num(row.Electricity_price_2022real_usd_kWh) * 100).toFixed(2)}¢/kWh</strong></div>
                    </div>
                  </article>
                ))}
              </section>
            </>
          )}
        </main>
      )}

      {activeTab === "robustness" && (
        <main className="page-view page-enter">
          <section className="page-heading robustness-heading">
            <div>
              <div className="eyebrow">ROBUSTNESS</div>
              <h1>How stable is the technology ranking?</h1>
              <p>Finite, source-backed scenarios replace the previous arbitrary probabilistic uncertainty treatment.</p>
            </div>
            <div className="explorer-controls">
              <label className="dark-select"><span>State</span><select value={stateName} onChange={(event) => setStateName(event.target.value)}>{(states.length ? states : ["Rhode Island"]).map((state) => <option key={state}>{state}</option>)}</select></label>
              <label className="dark-select"><span>Screening year</span><select value={year} onChange={(event) => setYear(Number(event.target.value))}>{years.map((value) => <option key={value}>{value}</option>)}</select></label>
            </div>
          </section>

          <section className="robustness-hero-grid">
            <article className="major-stat-card reversible"><span className="major-stat">47/48</span><h2>Status rankings can reverse</h2><p>{robustness.capexEnvelope.note}</p></article>
            <article className="major-stat-card electricity"><span className="major-stat">1</span><h2>Winner change under observed 2025 electricity prices</h2><p>California 2022: {robustness.electricity2025.change}</p></article>
            <article className="major-stat-card gas"><span className="major-stat">0</span><h2>Winner changes under observed 2024 gas prices</h2><p>Rhode Island remains O-SOEC; its lead narrows under the named gas case.</p></article>
          </section>

          <section className="scenario-detail-grid">
            <article className="surface-card scenario-detail">
              <div className="eyebrow dark-eyebrow">DOE CAPEX ENVELOPE</div><h2>{stateName}</h2>
              {year === 2022 && selectedRobustness ? (
                <>
                  <div className="scenario-status"><span>Baseline winner</span><strong>{selectedRobustness.Deterministic_winner_v3}</strong></div>
                  <div className="scenario-status"><span>Classification</span><strong>{selectedRobustness.Ranking_class === "ROBUST_TO_DOE_CAPEX_RANGE" ? "Robust" : "Winner can reverse"}</strong></div>
                  <div className="scenario-status"><span>Named scenario winners</span><strong>{selectedRobustness.Distinct_named_scenario_winners}</strong></div>
                  {selectedRobustness.Reversal_witness_scenario && <p className="light-footnote">Witness endpoint: <code>{selectedRobustness.Reversal_witness_scenario}</code></p>}
                </>
              ) : <p className="light-subtle">The DOE status-CAPEX envelope is intentionally applied only to the 2022 status cohort.</p>}
            </article>

            <article className="surface-card scenario-detail">
              <div className="eyebrow dark-eyebrow">OBSERVED 2025 ELECTRICITY PRICE</div><h2>{electricityWinner}</h2>
              <p className="light-subtle">Named-scenario winner for {stateName} · {year}</p>
              <div className="scenario-list light-list">
                {selectedElectricityScenario.map((row) => <span key={row.Technology}><i className={"tech-dot " + techClass[row.Technology]} />{row.Technology}<strong>$<Money value={num(row.Named_scenario_LCOH_plantgate_usd_kg)} />/kg</strong></span>)}
              </div>
            </article>

            <article className="surface-card scenario-detail">
              <div className="eyebrow dark-eyebrow">OBSERVED 2024 NATURAL GAS PRICE</div><h2>{gasWinner}</h2>
              {selectedGasScenario ? (
                <>
                  <p className="light-subtle">O-SOEC named-scenario adjustment for {stateName} · {year}</p>
                  <div className="gas-shift"><span>$<Money value={num(selectedGasScenario.V3_O_SOEC_LCOH_real2022_USD_kg)} /></span><b>→</b><span>$<Money value={num(selectedGasScenario.Scenario_O_SOEC_LCOH_real2022_USD_kg)} /></span></div>
                </>
              ) : <p className="light-subtle">No gas-case row found for this state/year.</p>}
            </article>
          </section>
        </main>
      )}

      {activeTab === "benchmark" && (
        <main className="page-view page-enter">
          <section className="page-heading">
            <div>
              <div className="eyebrow">DOE BENCHMARK</div>
              <h1>Published-model consistency check.</h1>
              <p>DOE current-grid values are normalized to real-2022 USD for display. This is a published-model consistency benchmark, not empirical validation.</p>
            </div>
          </section>

          <section className="benchmark-grid">
            {doeBenchmarks.map((row) => {
              const range = row.doeHigh - row.doeLow;
              const gap = row.doeLow - row.v3Median;
              return (
                <article className={"benchmark-card " + techClass[row.technology]} key={row.technology}>
                  <div className="tech-card-header"><span className={"tech-symbol " + techClass[row.technology]}>{row.technology.slice(0, 1)}</span><h2>{row.technology}</h2></div>
                  <div className="benchmark-number">{"$"}{row.v3Median.toFixed(3)}<small>/kg</small></div>
                  <span className="benchmark-caption">v3 state median</span>
                  <div className="benchmark-range"><span>DOE current-grid range</span><strong>{"$"}{row.doeLow.toFixed(3)}–{"$"}{row.doeHigh.toFixed(3)}/kg</strong><div className="range-line"><i style={{ width: Math.max(20, Math.min(100, range * 60)) + "%" }} /></div></div>
                  <p className="light-footnote">v3 median is {"$"}{Math.abs(gap).toFixed(3)}/kg {gap >= 0 ? "below" : "above"} the DOE interval floor.</p>
                </article>
              );
            })}
          </section>

          <section className="surface-card benchmark-note">
            <h2>Boundary differences matter.</h2>
            <p>DOE and v3 use different electricity prices, utilization, project life, financing, and other assumptions. The comparison therefore checks directional consistency, not calibration or empirical validation.</p>
          </section>
        </main>
      )}

      {activeTab === "methods" && (
        <main className="page-view page-enter">
          <section className="page-heading">
            <div>
              <div className="eyebrow">METHODS & DATA</div>
              <h1>Transparent boundaries and canonical provenance.</h1>
              <p>Everything shown in the interface is tied to the accepted revised-v3 outputs.</p>
            </div>
          </section>

          <section className="methods-grid">
            <article className="surface-card method-card"><div className="method-index">01</div><h2>Included in baseline</h2><ul><li>EIA 2022 state industrial electricity prices</li><li>DOE technology SEC, installed CAPEX, OPEX, and stack lifetime</li><li>Discounted stack replacement</li><li>O-SOEC natural-gas thermal input</li></ul></article>
            <article className="surface-card method-card"><div className="method-index">02</div><h2>Outside baseline LCOH</h2><ul><li>Transport and storage</li><li>Direct water-price charge</li><li>CO₂ surcharge</li><li>Land/infrastructure CAPEX multipliers</li></ul></article>
            <article className="surface-card method-card"><div className="method-index">03</div><h2>Financial basis</h2><ul><li>Real 2022 USD</li><li>8% real WACC</li><li>20-year project life</li><li>90% utilization</li></ul></article>
          </section>

          <section className="downloads-panel">
            <div><div className="eyebrow">CANONICAL DOWNLOADS</div><h2>Reproducible data release</h2></div>
            <div className="download-grid">
              <a href={assetUrl("/data/canonical/deterministic_state_year.csv")} download><span>CSV</span><strong>Deterministic v3</strong><small>720 state/year/technology rows</small></a>
              <a href={assetUrl("/data/canonical/state_rank_robustness_2022.csv")} download><span>CSV</span><strong>Ranking robustness</strong><small>48-state DOE CAPEX envelope</small></a>
              <a href={assetUrl("/data/canonical/MIGRATION_MANIFEST.json")} download><span>JSON</span><strong>Provenance manifest</strong><small>SHA-256 file verification</small></a>
            </div>
          </section>
        </main>
      )}

      <nav className="mobile-tabbar" aria-label="Mobile navigation">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            className={activeTab === tab.key ? "mobile-tab active" : "mobile-tab"}
            onClick={() => {
              setActiveTab(tab.key);
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
          >
            <span aria-hidden="true">{tab.icon}</span>
            <small>{tab.key === "explorer" ? "States" : tab.key === "robustness" ? "Robust" : tab.key === "benchmark" ? "DOE" : tab.key === "methods" ? "Data" : "Overview"}</small>
          </button>
        ))}
      </nav>

      <footer className="site-footer"><span>U.S. Electrolysis Cost Explorer · revised deterministic v3</span><span>Canonical data only · historical Streamlit/ML outputs excluded</span></footer>
    </div>
  );
}
