import { useEffect, useRef, useState } from 'react'
import type { HJBData } from '../types'

export function useHJB(q: number, gamma: number) {
  const [hjb, setHjb] = useState<HJBData | null>(null)
  const [loading, setLoading] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(async () => {
      setLoading(true)
      try {
        const base = import.meta.env.DEV
          ? 'http://localhost:8000'
          : (import.meta.env.VITE_API_URL ?? '')
        const res = await fetch(`${base}/api/hjb?q=${q}&gamma=${gamma}`)
        const data = await res.json() as HJBData
        setHjb(data)
      } catch {
        // server not ready yet
      } finally {
        setLoading(false)
      }
    }, 80)
  }, [q, gamma])

  return { hjb, loading }
}
