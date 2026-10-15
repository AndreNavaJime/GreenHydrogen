"use client";

import { useEffect, useMemo, useState } from "react";
import { num, type CsvRow } from "@/lib/csv";

type Position = [number, number];

type GeoFeature = {
  type: "Feature";
  properties: { NAME: string; STUSPS: string };
  geometry:
    | { type: "Polygon"; coordinates: Position[][] }
    | { type: "MultiPolygon"; coordinates: Position[][][] };
};

type GeoCollection = {
  type: "FeatureCollection";
  features: GeoFeature[];
};

type Props = {
  rows: CsvRow[];
  year: number;
  selectedState: string;
  onSelectState: (state: string) => void;
};

const fillByTechnology: Record<string, string> = {
  AWE: "#1bb7a7",
  PEM: "#6f78e8",
  "O-SOEC": "#e9894a",
};

function mercator([lon, lat]: Position): Position {
  const x = (lon * Math.PI) / 180;
  const limitedLat = Math.max(-85, Math.min(85, lat));
  const phi = (limitedLat * Math.PI) / 180;
  const y = Math.log(Math.tan(Math.PI / 4 + phi / 2));
  return [x, y];
}

function allPositions(feature: GeoFeature): Position[] {
  if (feature.geometry.type === "Polygon") {
    return feature.geometry.coordinates.flat();
  }
  return feature.geometry.coordinates.flat(2);
}

export default function USStateMap({
  rows,
  year,
  selectedState,
  onSelectState,
}: Props) {
  const [geo, setGeo] = useState<GeoCollection | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetch("/data/us_states_contiguous_48.geojson", { cache: "force-cache" })
      .then((response) => {
        if (!response.ok) throw new Error("Unable to load the 48-state map.");
        return response.json();
      })
      .then((data: GeoCollection) => {
        if (!cancelled) setGeo(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Map load failed.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const winnerByState = useMemo(() => {
    const grouped = new Map<string, CsvRow[]>();
    for (const row of rows) {
      if (Number(row.Year) !== year) continue;
      const list = grouped.get(row.State) ?? [];
      list.push(row);
      grouped.set(row.State, list);
    }

    const winners = new Map<string, { technology: string; lcoh: number }>();
    for (const [state, stateRows] of grouped.entries()) {
      const sorted = [...stateRows].sort(
        (a, b) => num(a.LCOH_plantgate_usd_kg) - num(b.LCOH_plantgate_usd_kg)
      );
      if (sorted[0]) {
        winners.set(state, {
          technology: sorted[0].Technology,
          lcoh: num(sorted[0].LCOH_plantgate_usd_kg),
        });
      }
    }
    return winners;
  }, [rows, year]);

  const mapModel = useMemo(() => {
    if (!geo) return null;

    const width = 960;
    const height = 560;
    const padding = 18;
    const projected = geo.features.flatMap((feature) =>
      allPositions(feature).map(mercator)
    );

    const xs = projected.map(([x]) => x);
    const ys = projected.map(([, y]) => y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const usableW = width - padding * 2;
    const usableH = height - padding * 2;
    const scale = Math.min(usableW / (maxX - minX), usableH / (maxY - minY));
    const drawnW = (maxX - minX) * scale;
    const drawnH = (maxY - minY) * scale;
    const offsetX = padding + (usableW - drawnW) / 2;
    const offsetY = padding + (usableH - drawnH) / 2;

    const point = (position: Position): Position => {
      const [x, y] = mercator(position);
      return [
        offsetX + (x - minX) * scale,
        offsetY + (maxY - y) * scale,
      ];
    };

    const ringPath = (ring: Position[]) =>
      ring
        .map((position, index) => {
          const [x, y] = point(position);
          return (index === 0 ? "M" : "L") + x.toFixed(2) + "," + y.toFixed(2);
        })
        .join(" ") + " Z";

    const featurePath = (feature: GeoFeature) => {
      if (feature.geometry.type === "Polygon") {
        return feature.geometry.coordinates.map(ringPath).join(" ");
      }
      return feature.geometry.coordinates
        .flatMap((polygon) => polygon.map(ringPath))
        .join(" ");
    };

    return {
      width,
      height,
      features: geo.features.map((feature) => ({
        state: feature.properties.NAME,
        abbreviation: feature.properties.STUSPS,
        path: featurePath(feature),
      })),
    };
  }, [geo]);

  if (error) return <p className="error-box">{error}</p>;
  if (!mapModel) return <p className="subtle">Loading 48-state map…</p>;

  return (
    <div className="map-shell">
      <div className="map-legend">
        <span><i style={{ background: fillByTechnology.AWE }} />AWE</span>
        <span><i style={{ background: fillByTechnology.PEM }} />PEM</span>
        <span><i style={{ background: fillByTechnology["O-SOEC"] }} />O-SOEC</span>
      </div>
      <svg
        className="us-map"
        viewBox={"0 0 " + mapModel.width + " " + mapModel.height}
        role="img"
        aria-label={"Lowest-cost technology by state for " + year}
      >
        {mapModel.features.map((feature) => {
          const result = winnerByState.get(feature.state);
          const selected = feature.state === selectedState;
          const fill = result ? fillByTechnology[result.technology] ?? "#5d6a78" : "#5d6a78";
          return (
            <path
              key={feature.state}
              d={feature.path}
              fill={fill}
              className={selected ? "state-shape selected" : "state-shape"}
              onClick={() => onSelectState(feature.state)}
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelectState(feature.state);
                }
              }}
            >
              <title>
                {feature.state + " (" + feature.abbreviation + "): " +
                  (result
                    ? result.technology + ", $" + result.lcoh.toFixed(3) + "/kg H₂"
                    : "no canonical result")}
              </title>
            </path>
          );
        })}
      </svg>
      <p className="footnote">
        Click a state to update the explorer below. Map color indicates the lowest deterministic
        plant-gate LCOH technology for the selected screening year.
      </p>
    </div>
  );
}
