import { useEffect, useRef, useState, useCallback } from 'react'
import type { MarketState } from '../types'

// Dev: Vite proxy forwards /ws → localhost:8000
// Prod (Vercel): VITE_API_URL must be set to the Railway backend URL
const WS_URL = (() => {
  if (import.meta.env.DEV) return 'ws://localhost:8000/ws'
  const api = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
  if (!api) {
    console.error('[WS] VITE_API_URL is not set — set it in Vercel environment variables')
    return ''
  }
  const proto = api.startsWith('https') ? 'wss' : 'ws'
  return `${proto}://${api.replace(/^https?:\/\//, '')}/ws`
})()

export function useMarketWebSocket() {
  const [state, setState] = useState<MarketState | null>(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const connect = useCallback(() => {
    if (!WS_URL) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => setConnected(true)

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as MarketState
        setState(data)
      } catch {
        // ignore malformed frames
      }
    }

    ws.onclose = () => {
      setConnected(false)
      retryRef.current = setTimeout(connect, 2000)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      if (retryRef.current) clearTimeout(retryRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { state, connected }
}
