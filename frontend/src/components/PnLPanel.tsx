import { useMemo } from 'react'
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Scatter,
} from 'recharts'

interface Props {
  pnls: number[]
  eventTimes: number[]
}

const PNL_TOOLTIP = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-card border border-border rounded px-3 py-2 text-[11px] font-mono shadow-xl">
      {payload.map((p: any) => (
        <div key={p.name} style={{ color: p.color }} className="flex justify-between gap-4">
          <span>{p.name}</span>
          <span className="tabular-nums">{typeof p.value === 'number' ? p.value.toFixed(4) : '—'}</span>
        </div>
      ))}
    </div>
  )
}

// Hawkes intensity: λ(t) = μ + α·Σ exp(−β·(t−tᵢ)) for tᵢ < t
const MU = 0.5, ALPHA = 0.3, BETA = 1.0

function computeHawkes(evTimes: number[], nPoints = 120) {
  if (evTimes.length < 2) return []
  const tMax  = evTimes[evTimes.length - 1]
  const tMin  = Math.max(0, tMax - 50)
  const dt    = (tMax - tMin) / nPoints
  return Array.from({ length: nPoints }, (_, i) => {
    const t = tMin + i * dt
    const prev = evTimes.filter(ti => ti < t)
    const lambda = MU + ALPHA * prev.reduce((s, ti) => s + Math.exp(-BETA * (t - ti)), 0)
    return { t: parseFloat(t.toFixed(2)), λ: parseFloat(lambda.toFixed(4)) }
  })
}

export function PnLPanel({ pnls, eventTimes }: Props) {
  const pnlData = useMemo(() =>
    pnls.map((p, i) => ({ i, 'P&L': p })),
    [pnls]
  )

  const hawkesData = useMemo(() => computeHawkes(eventTimes), [eventTimes])

  const arrivalMarkers = useMemo(() => {
    if (!eventTimes.length) return []
    const tMax = eventTimes[eventTimes.length - 1]
    return eventTimes
      .filter(t => t > tMax - 50)
      .map(t => ({ t: parseFloat(t.toFixed(2)), y: 0 }))
  }, [eventTimes])

  const lastPnl = pnls[pnls.length - 1] ?? 0
  const isPositive = lastPnl >= 0
  const pnlColor  = isPositive ? '#00c853' : '#ff3b3b'
  const fillColor = isPositive ? 'rgba(0,200,83,0.12)' : 'rgba(255,59,59,0.12)'

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 pt-3 pb-0 flex items-center justify-between">
        <span className="text-[11px] font-mono uppercase tracking-widest text-muted">
          P&amp;L + Hawkes Intensity
        </span>
        <span className={`text-sm font-mono font-semibold tabular-nums ${isPositive ? 'text-green' : 'text-red'}`}>
          {lastPnl >= 0 ? '+' : ''}{lastPnl.toFixed(4)}
        </span>
      </div>

      {/* P&L chart — top 55% */}
      <div className="flex-[55] min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={pnlData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
            <CartesianGrid stroke="#1a2744" strokeWidth={0.5} vertical={false} />
            <XAxis hide dataKey="i" />
            <YAxis
              tick={{ fill: '#4a5568', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
              axisLine={false}
              tickLine={false}
              tickFormatter={v => v.toFixed(2)}
            />
            <ReferenceLine y={0} stroke="#2d3748" strokeWidth={1} />
            <Tooltip content={<PNL_TOOLTIP />} />
            <Area
              dataKey="P&L"
              stroke={pnlColor}
              strokeWidth={1.5}
              fill={fillColor}
              dot={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Separator */}
      <div className="mx-4 border-t border-border" />

      {/* Hawkes intensity — bottom 45% */}
      <div className="px-4 py-0.5 flex items-center justify-between">
        <span className="text-[9px] font-mono uppercase tracking-widest text-dim">
          Hawkes λ(t) — order arrival intensity
        </span>
        <span className="text-[9px] font-mono text-dim">
          μ={MU} α={ALPHA} β={BETA}
        </span>
      </div>
      <div className="flex-[45] min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={hawkesData} margin={{ top: 2, right: 16, bottom: 16, left: 8 }}>
            <CartesianGrid stroke="#1a2744" strokeWidth={0.5} vertical={false} />
            <XAxis
              dataKey="t"
              tick={{ fill: '#4a5568', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
              axisLine={{ stroke: '#1a2744' }}
              tickLine={false}
              label={{ value: 'Time', position: 'insideBottom', offset: -6, fill: '#4a5568', fontSize: 9 }}
            />
            <YAxis
              tick={{ fill: '#4a5568', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
              axisLine={false}
              tickLine={false}
              tickFormatter={v => v.toFixed(1)}
            />
            <ReferenceLine y={MU} stroke="#2d3748" strokeDasharray="3 3" strokeWidth={1} />
            <Tooltip
              content={({ active, payload }: any) => {
                if (!active || !payload?.length) return null
                return (
                  <div className="bg-card border border-border rounded px-2 py-1 text-[10px] font-mono">
                    <span className="text-accent">λ(t) = {payload[0]?.value?.toFixed(3)}</span>
                  </div>
                )
              }}
            />
            <Line dataKey="λ" stroke="#0088ff" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            {arrivalMarkers.length > 0 && (
              <Scatter
                data={arrivalMarkers}
                dataKey="y"
                fill="#ffab00"
                opacity={0.7}
                shape={(props: any) => (
                  <circle cx={props.cx} cy={props.cy} r={2} fill="#ffab00" />
                )}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
