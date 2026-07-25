"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type { EvidenceGraph as GraphData, GraphNode } from "@/lib/types";
import { cn } from "@/lib/ui";

/**
 * Force-directed evidence graph.
 *
 * Hand-rolled rather than pulled from d3-force: the simulation is about forty
 * lines, and it keeps the bundle free of a dependency whose only job here is
 * moving circles around.
 */

interface SimNode extends GraphNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

const WIDTH = 900;
const HEIGHT = 520;
const REPULSION = 5200;
const SPRING = 0.012;
const DAMPING = 0.86;
const CENTRE_PULL = 0.0016;

const VERDICT_FILL: Record<string, string> = {
  SUPPORTED: "#34d399",
  REFUTED: "#fb7185",
  NEI: "#fbbf24",
};

const TIER_FILL: Record<string, string> = {
  PRIMARY: "#22d3ee",
  HIGH: "#38bdf8",
  MEDIUM: "#818cf8",
  LOW: "#a78bfa",
  UNKNOWN: "#71717a",
  UNRELIABLE: "#f43f5e",
};

function nodeFill(node: GraphNode): string {
  if (node.type === "claim") return VERDICT_FILL[node.verdict ?? "NEI"];
  return TIER_FILL[node.tier ?? "UNKNOWN"] ?? "#71717a";
}

function nodeRadius(node: GraphNode): number {
  return node.type === "claim" ? 9 + (node.confidence ?? 0) * 7 : 6;
}

export function EvidenceGraph({ graph }: { graph: GraphData }) {
  const [nodes, setNodes] = useState<SimNode[]>([]);
  const [hovered, setHovered] = useState<SimNode | null>(null);
  const frameRef = useRef<number>(0);

  // Only keep sources that are actually connected — orphans add clutter
  // without adding information.
  const { visibleNodes, visibleEdges } = useMemo(() => {
    const connected = new Set<string>();
    for (const edge of graph.edges) {
      connected.add(edge.source);
      connected.add(edge.target);
    }
    const kept = graph.nodes.filter(
      (n) => n.type === "claim" || connected.has(n.id),
    );
    const keptIds = new Set(kept.map((n) => n.id));
    return {
      visibleNodes: kept,
      visibleEdges: graph.edges.filter(
        (e) => keptIds.has(e.source) && keptIds.has(e.target),
      ),
    };
  }, [graph]);

  useEffect(() => {
    // Seed on a circle so the layout unfolds outward instead of exploding from
    // a single degenerate point.
    const seeded: SimNode[] = visibleNodes.map((node, index) => {
      const angle = (index / Math.max(1, visibleNodes.length)) * Math.PI * 2;
      const radius = node.type === "claim" ? 110 : 210;
      return {
        ...node,
        x: WIDTH / 2 + Math.cos(angle) * radius,
        y: HEIGHT / 2 + Math.sin(angle) * radius,
        vx: 0,
        vy: 0,
      };
    });
    setNodes(seeded);

    const index = new Map(seeded.map((n, i) => [n.id, i]));
    let ticks = 0;

    const step = () => {
      ticks += 1;
      for (let i = 0; i < seeded.length; i++) {
        const a = seeded[i];
        for (let j = i + 1; j < seeded.length; j++) {
          const b = seeded[j];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const distanceSq = Math.max(120, dx * dx + dy * dy);
          const force = REPULSION / distanceSq;
          const distance = Math.sqrt(distanceSq);
          const fx = (dx / distance) * force;
          const fy = (dy / distance) * force;
          a.vx -= fx;
          a.vy -= fy;
          b.vx += fx;
          b.vy += fy;
        }
      }

      for (const edge of visibleEdges) {
        const ai = index.get(edge.source);
        const bi = index.get(edge.target);
        if (ai === undefined || bi === undefined) continue;
        const a = seeded[ai];
        const b = seeded[bi];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const strength = SPRING * (edge.type === "contradiction" ? 0.4 : 1);
        a.vx += dx * strength;
        a.vy += dy * strength;
        b.vx -= dx * strength;
        b.vy -= dy * strength;
      }

      for (const node of seeded) {
        node.vx += (WIDTH / 2 - node.x) * CENTRE_PULL;
        node.vy += (HEIGHT / 2 - node.y) * CENTRE_PULL;
        node.vx *= DAMPING;
        node.vy *= DAMPING;
        node.x = Math.min(WIDTH - 20, Math.max(20, node.x + node.vx));
        node.y = Math.min(HEIGHT - 20, Math.max(20, node.y + node.vy));
      }

      setNodes([...seeded]);
      if (ticks < 260) {
        frameRef.current = requestAnimationFrame(step);
      }
    };

    frameRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frameRef.current);
  }, [visibleNodes, visibleEdges]);

  const positions = useMemo(
    () => new Map(nodes.map((n) => [n.id, n])),
    [nodes],
  );

  if (visibleNodes.length === 0) {
    return (
      <div className="panel flex h-[520px] items-center justify-center text-sm text-zinc-600">
        No graph data yet.
      </div>
    );
  }

  return (
    <div className="panel overflow-hidden">
      <div className="panel-header">
        <span className="panel-title">Evidence graph</span>
        <div className="flex flex-wrap items-center gap-3 text-[11px] text-zinc-500">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-emerald-400" /> supported
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-rose-400" /> refuted
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-amber-400" /> not established
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-cyan-400" /> source
          </span>
          <span className="flex items-center gap-1.5">
            <svg width="18" height="6">
              <line
                x1="0"
                y1="3"
                x2="18"
                y2="3"
                stroke="#fb923c"
                strokeWidth="2"
                strokeDasharray="3 2"
              />
            </svg>
            conflict
          </span>
        </div>
      </div>

      <div className="relative">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="h-[520px] w-full"
          role="img"
          aria-label="Force-directed graph of claims and their evidence sources"
        >
          <g>
            {visibleEdges.map((edge, index) => {
              const a = positions.get(edge.source);
              const b = positions.get(edge.target);
              if (!a || !b) return null;

              const isConflict = edge.type === "contradiction";
              const stroke = isConflict
                ? "#fb923c"
                : edge.stance === "SUPPORTS"
                  ? "#34d399"
                  : edge.stance === "REFUTES"
                    ? "#fb7185"
                    : "#3f3f46";

              return (
                <line
                  key={index}
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  stroke={stroke}
                  strokeWidth={isConflict ? 1.8 : 0.4 + edge.weight * 2}
                  strokeOpacity={edge.derivative ? 0.15 : isConflict ? 0.8 : 0.45}
                  strokeDasharray={
                    isConflict ? "4 3" : edge.derivative ? "2 3" : undefined
                  }
                />
              );
            })}
          </g>

          <g>
            {nodes.map((node) => (
              <circle
                key={node.id}
                cx={node.x}
                cy={node.y}
                r={nodeRadius(node)}
                fill={nodeFill(node)}
                fillOpacity={node.retracted ? 0.3 : 0.85}
                stroke={hovered?.id === node.id ? "#fafafa" : "#18181b"}
                strokeWidth={hovered?.id === node.id ? 2 : 1}
                className="cursor-pointer transition-all"
                onMouseEnter={() => setHovered(node)}
                onMouseLeave={() => setHovered(null)}
              />
            ))}
          </g>
        </svg>

        {hovered ? (
          <div className="pointer-events-none absolute left-4 top-4 max-w-md rounded-lg border border-zinc-700 bg-zinc-900/95 p-3 text-xs shadow-xl backdrop-blur">
            <div
              className={cn(
                "mb-1 font-medium",
                hovered.type === "claim" ? "text-zinc-200" : "text-cyan-400",
              )}
            >
              {hovered.type === "claim" ? "Claim" : hovered.domain}
              {hovered.type === "claim" && hovered.verdict ? (
                <span className="ml-2 text-zinc-500">
                  {hovered.verdict} · {(hovered.confidence ?? 0).toFixed(2)}
                </span>
              ) : null}
              {hovered.type === "source" && hovered.tier ? (
                <span className="ml-2 text-zinc-500">{hovered.tier}</span>
              ) : null}
            </div>
            <p className="text-zinc-400">{hovered.label}</p>
          </div>
        ) : null}
      </div>
    </div>
  );
}
