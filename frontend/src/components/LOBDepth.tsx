import { useMemo } from 'react'
import type { LOBLevel, HJBData } from '../types'

interface Props {
  bids: LOBLevel[]
  asks: LOBLevel[]
  mid: number
  hjb: HJBData | null
  inventory: number
}

const PANEL_W = 420
const PANEL_H = 300
const MARGIN  = { top: 20, right: 16, bottom: 32, left: 72 }

export function LOBDepth({ bids, asks, mid, hjb, inventory }: Props) {
  const { paths, yScale, xScale, maxVol, priceRange } = useMemo(() => {
    if (!bids.length && !asks.length) return { paths: null, yScale: null, xScale: null, maxVol: 0, priceRange: [0, 1] as [number, number] }

    const allPxs = [...bids.map(b => b.px), ...asks.map(a => a.px)]
    const pMin = Math.min(...allPxs) - 0.0005
    const pMax = Math.max(...allPxs) + 0.0005
    const maxVol = Math.max(...bids.map(b => b.vol), ...asks.map(a => a.vol))

    const W = PANEL_W - MARGIN.left - MARGIN.right
    const H = PANEL_H - MARGIN.top - MARGIN.bottom

    const yScale = (p: number) => MARGIN.top + H - ((p - pMin) / (pMax - pMin)) * H
    // Bids: bars extend LEFT from center (center = W/2)
    // Asks: bars extend RIGHT from center
    const cx = MARGIN.left + W / 2
    const xBid = (v: number) => cx - (v / maxVol) * (W / 2 - 4)
    const xAsk = (v: number) => cx + (v / maxVol) * (W / 2 - 4)

    const BAR_H = Math.max(2, (H / Math.max(bids.length, asks.length, 1)) * 0.72)

    return { paths: { bids, asks, BAR_H, cx, xBid, xAsk }, yScale, xScale: cx, maxVol, priceRange: [pMin, pMax] as [number, number] }
  }, [bids, asks])

  const W = PANEL_W - MARGIN.left - MARGIN.right
  const H = PANEL_H - MARGIN.top - MARGIN.bottom

  const hjbBidPx = hjb && hjb.cur_bid_sp != null ? mid - hjb.cur_bid_sp : null
  const hjbAskPx = hjb && hjb.cur_ask_sp != null ? mid + hjb.cur_ask_sp : null

  // Y tick marks
  const yTicks = useMemo(() => {
    if (!priceRange) return []
    const [pMin, pMax] = priceRange
    const range = pMax - pMin
    const step  = range / 5
    return Array.from({ length: 6 }, (_, i) => pMin + i * step)
  }, [priceRange])

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 pt-3 pb-1 flex items-center justify-between">
        <span className="text-[11px] font-mono uppercase tracking-widest text-muted">
          LOB Depth
        </span>
        <span className="text-xs font-mono text-text">
          mid <span className="text-accent">{mid.toFixed(4)}</span>
          <span className="ml-3 text-muted">q=</span>
          <span className={inventory >= 0 ? 'text-green' : 'text-red'}>{inventory >= 0 ? `+${inventory}` : inventory}</span>
        </span>
      </div>

      <div className="flex-1 flex items-center justify-center">
        <svg width={PANEL_W} height={PANEL_H} className="overflow-visible">
          {/* Grid lines */}
          {yTicks.map((p, i) => {
            const y = yScale ? yScale(p) : 0
            return (
              <g key={i}>
                <line x1={MARGIN.left} x2={PANEL_W - MARGIN.right} y1={y} y2={y}
                  stroke="#1a2744" strokeWidth={0.5} />
                <text x={MARGIN.left - 4} y={y + 4} textAnchor="end"
                  fill="#4a5568" fontSize={9} fontFamily="JetBrains Mono, monospace">
                  {p.toFixed(3)}
                </text>
              </g>
            )
          })}

          {/* Center divider */}
          {paths && (
            <line x1={paths.cx} x2={paths.cx} y1={MARGIN.top} y2={MARGIN.top + H}
              stroke="#1a2744" strokeWidth={1} strokeDasharray="3 3" />
          )}

          {/* Bid bars */}
          {paths && yScale && paths.bids.map((b, i) => {
            const y   = yScale(b.px)
            const x1  = paths.xBid(b.vol)
            const opa = 0.9 - i * 0.12
            return (
              <g key={`bid-${i}`}>
                <rect
                  x={x1} y={y - paths.BAR_H / 2}
                  width={paths.cx - x1} height={paths.BAR_H}
                  fill={`rgba(0,200,83,${opa})`}
                  rx={1}
                />
                <text x={x1 - 3} y={y + 3.5} textAnchor="end"
                  fill="#4a5568" fontSize={8} fontFamily="JetBrains Mono, monospace">
                  {b.vol > 999 ? `${(b.vol / 1000).toFixed(1)}k` : b.vol.toFixed(0)}
                </text>
              </g>
            )
          })}

          {/* Ask bars */}
          {paths && yScale && paths.asks.map((a, i) => {
            const y   = yScale(a.px)
            const x2  = paths.xAsk(a.vol)
            const opa = 0.9 - i * 0.12
            return (
              <g key={`ask-${i}`}>
                <rect
                  x={paths.cx} y={y - paths.BAR_H / 2}
                  width={x2 - paths.cx} height={paths.BAR_H}
                  fill={`rgba(255,59,59,${opa})`}
                  rx={1}
                />
              </g>
            )
          })}

          {/* Mid price line */}
          {yScale && (
            <line x1={MARGIN.left} x2={PANEL_W - MARGIN.right}
              y1={yScale(mid)} y2={yScale(mid)}
              stroke="#4a5568" strokeWidth={1} strokeDasharray="4 2" />
          )}

          {/* HJB bid quote */}
          {hjbBidPx && yScale && Math.abs(hjbBidPx - mid) < (priceRange[1] - priceRange[0]) && (
            <g>
              <line x1={MARGIN.left} x2={PANEL_W - MARGIN.right}
                y1={yScale(hjbBidPx)} y2={yScale(hjbBidPx)}
                stroke="#00c853" strokeWidth={1.5} strokeDasharray="5 3" opacity={0.8} />
              <text x={PANEL_W - MARGIN.right + 2} y={yScale(hjbBidPx) + 3}
                fill="#00c853" fontSize={8} fontFamily="JetBrains Mono, monospace">
                δ*bid
              </text>
            </g>
          )}

          {/* HJB ask quote */}
          {hjbAskPx && yScale && Math.abs(hjbAskPx - mid) < (priceRange[1] - priceRange[0]) && (
            <g>
              <line x1={MARGIN.left} x2={PANEL_W - MARGIN.right}
                y1={yScale(hjbAskPx)} y2={yScale(hjbAskPx)}
                stroke="#ffab00" strokeWidth={1.5} strokeDasharray="5 3" opacity={0.8} />
              <text x={PANEL_W - MARGIN.right + 2} y={yScale(hjbAskPx) + 3}
                fill="#ffab00" fontSize={8} fontFamily="JetBrains Mono, monospace">
                δ*ask
              </text>
            </g>
          )}

          {/* Axis labels */}
          <text x={MARGIN.left + W / 4} y={PANEL_H - 4} textAnchor="middle"
            fill="#4a5568" fontSize={9} fontFamily="JetBrains Mono, monospace">
            BID
          </text>
          <text x={MARGIN.left + 3 * W / 4} y={PANEL_H - 4} textAnchor="middle"
            fill="#4a5568" fontSize={9} fontFamily="JetBrains Mono, monospace">
            ASK
          </text>
        </svg>
      </div>

      {/* HJB spread info */}
      {hjb && hjb.cur_bid_sp != null && hjb.cur_ask_sp != null && (
        <div className="px-4 pb-2 flex gap-4 text-[10px] font-mono">
          <span className="text-muted">HJB quotes:</span>
          <span className="text-green">δ*bid={hjb.cur_bid_sp.toFixed(5)}</span>
          <span className="text-yellow">δ*ask={hjb.cur_ask_sp.toFixed(5)}</span>
          <span className="text-muted ml-auto">{(hjb.t_frac * 100).toFixed(0)}% of day</span>
        </div>
      )}
    </div>
  )
}
