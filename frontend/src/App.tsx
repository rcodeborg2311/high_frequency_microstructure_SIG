import { useState } from 'react'
import { useMarketWebSocket } from './hooks/useWebSocket'
import { useHJB } from './hooks/useHJB'
import { Header } from './components/Header'
import { StatsBar } from './components/StatsBar'
import { AlertBanner } from './components/AlertBanner'
import { Controls } from './components/Controls'
import { LOBDepth } from './components/LOBDepth'
import { SignalChart } from './components/SignalChart'
import { HJBSpread } from './components/HJBSpread'
import { PnLPanel } from './components/PnLPanel'

function CardPanel({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-card border border-border rounded-lg overflow-hidden flex flex-col min-h-0">
      {children}
    </div>
  )
}

export default function App() {
  const { state, connected } = useMarketWebSocket()
  const [inventory, setInventory] = useState(0)
  const [gamma, setGamma] = useState(0.01)
  const { hjb } = useHJB(inventory, gamma)

  if (!state || !state.ready) {
    return (
      <div className="min-h-screen bg-bg flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="flex gap-1.5">
            {[0, 1, 2].map(i => (
              <div
                key={i}
                className="w-2 h-2 rounded-full bg-accent animate-bounce"
                style={{ animationDelay: `${i * 0.15}s` }}
              />
            ))}
          </div>
          <span className="text-muted text-xs font-mono">
            {connected ? 'Loading market data…' : 'Connecting to server…'}
          </span>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-bg flex flex-col font-sans text-text">
      <Header
        connected={connected}
        source={state.source}
        tick={state.tick}
      />

      <StatsBar state={state} />

      {state.alerts.length > 0 && (
        <AlertBanner alerts={state.alerts} />
      )}

      <Controls
        inventory={inventory}
        gamma={gamma}
        onInventoryChange={setInventory}
        onGammaChange={setGamma}
      />

      {/* Main chart grid */}
      <main className="flex-1 min-h-0 grid grid-cols-[420px_1fr] grid-rows-2 gap-2 p-2">
        <CardPanel>
          <LOBDepth
            bids={state.lob.bids}
            asks={state.lob.asks}
            mid={state.mid}
            hjb={hjb}
            inventory={inventory}
          />
        </CardPanel>

        <CardPanel>
          <SignalChart
            mids={state.mids}
            ofis={state.ofis}
            vpins={state.vpins}
            kyles={state.kyles}
          />
        </CardPanel>

        <CardPanel>
          <HJBSpread
            hjb={hjb}
            inventory={inventory}
            gamma={gamma}
          />
        </CardPanel>

        <CardPanel>
          <PnLPanel
            pnls={state.pnls}
            eventTimes={state.event_times}
          />
        </CardPanel>
      </main>
    </div>
  )
}
