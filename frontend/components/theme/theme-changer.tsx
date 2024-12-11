"use client"
import { useTheme } from 'next-themes'
import { useEffect, useState } from 'react'

// Need to handle hydration mismatch
const ThemeChanger = () => {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) return null

  return (
    <div>
      <button onClick={() => setTheme('light')}>Light</button>
      <button onClick={() => setTheme('dark')}>Dark</button>
    </div>
  )
}

export default ThemeChanger