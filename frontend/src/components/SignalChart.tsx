import { useMemo } from 'react'
import {
  ComposedChart, Line, ReferenceArea, ReferenceLine,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'

interface Props {
  mids:  number[]
  ofis:  number[]
  vpins: number[]
  kyles: number[]
}

function zscore(arr: number[]) {
  if (arr.length === 0) return arr
  const mu  = arr.reduce((a, b) => a + b, 0) / arr.length
  const std = Math.sqrt(arr.reduce((a, b) => a + (b - mu) ** 2, 0) / arr.length)
  return std > 1e-10 ? arr.map(v => (v - mu) / std) : arr.map(v => v - mu)
}

const CHART_STYLE = {
  backgroundColor: 'transparent',
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-card border border-border rounded px-3 py-2 text-[11px] font-mono shadow-xl">
      <div className="text-muted mb-1">tick {label}</div>
      {payload.map((p: any) => (
        <div key={p.name} style={{ color: p.color }} className="flex justify-between gap-4">
          <span>{p.name}</span>
          <span className="tabular-nums">{typeof p.value === 'number' ? p.value.toFixed(3) : '—'}</span>
        </div>
      ))}
    </div>
  )
}

export function SignalChart({ mids, ofis, vpins, kyles }: Props) {
  const data = useMemo(() => {
    const zOfi  = zscore(ofis)
    const zVpin = zscore(vpins)
    const zKyle = zscore(kyles)

    // Normalize mid to [0, range] for secondary axis alignment
    const midMin = Math.min(...mids)
    const midMax = Math.max(...mids)
    const midRange = midMax - midMin || 1

    return mids.map((mid, i) => ({
      t:    i,
      OFI:  zOfi[i]  ?? null,
      VPIN: zVpin[i] ?? null,
      Kyle: zKyle[i] ?? null,
      Mid:  mid,
    }))
  }, [mids, ofis, vpins, kyles])

  const [midMin, midMax] = useMemo(() => {
    if (!mids.length) return [0, 1]
    return [Math.min(...mids) - 0.001, Math.max(...mids) + 0.001]
  }, [mids])

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 pt-3 pb-1 flex items-center justify-between">
        <span className="text-[11px] font-mono uppercase tracking-widest text-muted">
          Signal Time Series
        </span>
        <span className="text-[10px] font-mono text-dim">
          Z-scored · last {data.length} ticks
        </span>
      </div>
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} style={CHART_STYLE} margin={{ top: 4, right: 48, bottom: 16, left: 8 }}>
            <CartesianGrid stroke="#1a2744" strokeWidth={0.5} vertical={false} />

            {/* Threshold bands */}
            <ReferenceArea y1={0.7}  y2={3}   fill="rgba(0,136,255,0.06)"  yAxisId="z" />
            <ReferenceArea y1={-3}   y2={-0.7} fill="rgba(255,59,59,0.06)"  yAxisId="z" />
            <ReferenceLine y={0.7}  stroke="rgba(0,136,255,0.35)" strokeDasharray="3 3" yAxisId="z" />
            <ReferenceLine y={-0.7} stroke="rgba(255,59,59,0.35)" strokeDasharray="3 3" yAxisId="z" />

            <XAxis
              dataKey="t"
              tick={{ fill: '#4a5568', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
              axisLine={{ stroke: '#1a2744' }}
              tickLine={false}
              interval={Math.floor(data.length / 5)}
            />
            <YAxis
              yAxisId="z"
              tick={{ fill: '#4a5568', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
              axisLine={false}
              tickLine={false}
              domain={[-3.5, 3.5]}
              label={{ value: 'Z-score', angle: -90, position: 'insideLeft', fill: '#4a5568', fontSize: 9 }}
            />
            <YAxis
              yAxisId="mid"
              orientation="right"
              domain={[midMin, midMax]}
              tick={{ fill: '#4a5568', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
              axisLine={false}
              tickLine={false}
              tickFormatter={v => v.toFixed(3)}
            />

            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 10, fontFamily: 'JetBrains Mono, monospace', paddingTop: 4 }}
            />

            <Line yAxisId="z"   dataKey="OFI"  stroke="#0088ff" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line yAxisId="z"   dataKey="VPIN" stroke="#ffab00" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line yAxisId="z"   dataKey="Kyle" stroke="#00c853" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line yAxisId="mid" dataKey="Mid"  stroke="#4a5568" strokeWidth={1}   dot={false} isAnimationActive={false} strokeDasharray="4 2" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
